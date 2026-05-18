"""
GPU worker for batched LLM inference with Qwen 2.5.

Handles model loading, batched classification, response validation,
retries, and checkpointing
"""

import gc
import json
import logging
import os
import sys
import time
import traceback
import multiprocessing as mp
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Valid Tier 1 LLM categories
VALID_CATEGORIES = {
    "NOUN_PROPER",
    "NOUN_COMMON",
    "VERB",
    "ADJECTIVE",
    "ADVERB",
    "PRONOUN",
    "FUNCTION",
    "NUMERAL",
    "INTERJECTION",
    "PREFIX",
    "SUFFIX",
    "STEM",
    "CONTRACTION",
}


def setup_logging(output_dir: Path, name: str = "main") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter(f"%(asctime)s [{name}] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(console)

    fh = logging.FileHandler(
        output_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    return logger


def cleanup_logger(logger: logging.Logger):
    for handler in logger.handlers[:]:
        try:
            handler.flush()
            handler.close()
            logger.removeHandler(handler)
        except Exception:
            pass


# ============================================================================
# PROMPT BUILDING
# ============================================================================


def build_user_message(record: Dict) -> str:
    """Build the user message for a single token."""
    cleaned = record["cleaned_token"]
    signal = record["primary_signal"]
    confidence = record["confidence_level"]
    contexts = record["contexts"]

    if signal in ("WORD", "SUBTOKEN", "AMBIGUOUS"):
        type_hint = f"{signal} ({confidence} confidence)"
    else:
        type_hint = signal or "UNKNOWN"

    parts = [
        f'Token: "{cleaned}"',
        f"Token type hint: {type_hint}",
    ]

    if contexts:
        parts.append("")
        parts.append("Usage contexts from web text:")
        for i, ctx in enumerate(contexts, 1):
            parts.append(f"{i}. {ctx}")

    parts.append("")
    parts.append("Classify this token.")

    return "\n".join(parts)


# ============================================================================
# GPU WORKER
# ============================================================================


class GPUWorker:
    """Processes a chunk of tokens on a single GPU."""

    def __init__(
        self,
        gpu_id: int,
        output_dir: Path,
        model_name: str,
        batch_size: int,
        max_retries: int,
        timeout: int,
    ):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout

        self.logger = setup_logging(output_dir, f"gpu_{gpu_id}")
        self.logger.info("=" * 70)
        self.logger.info(f"GPU {gpu_id} WORKER INITIALIZING")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Model: {model_name}")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info("=" * 70)

        self.logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.logger.info("Model loaded successfully")

        self.stats = {
            "total_processed": 0,
            "successful": 0,
            "retries": 0,
            "retry_successes": 0,
            "fallbacks": 0,
            "total_batches": 0,
            "total_inference_time": 0.0,
        }

    def cleanup(self):
        try:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

    def validate_response(self, response: str) -> Optional[str]:
        """Extract a valid category from the model response. Returns None if invalid."""
        cleaned = response.strip().upper()
        if cleaned in VALID_CATEGORIES:
            return cleaned
        # Check longest categories first to avoid substring collisions
        # (e.g., "VERB" is a substring of "ADVERB")
        for cat in sorted(VALID_CATEGORIES, key=len, reverse=True):
            if cat in cleaned:
                return cat
        return None

    def classify_batch(
        self,
        batch: List[Dict],
        system_prompt: str,
        is_retry: bool = False,
    ) -> List[Tuple[Optional[str], float]]:
        """
        Classify a batch of tokens. Returns list of (category, inference_time) tuples.
        category is None if validation failed.
        """
        all_texts = []
        for record in batch:
            user_msg = build_user_message(record)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_texts.append(text)

        model_inputs = self.tokenizer(
            all_texts,
            padding="longest",
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)

        t0 = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        inference_time = time.time() - t0
        per_token_time = inference_time / len(batch)

        results = []
        for i, (gen_ids, input_ids) in enumerate(
            zip(generated_ids, model_inputs.input_ids)
        ):
            response_ids = gen_ids[len(input_ids) :]
            response = self.tokenizer.decode(
                response_ids, skip_special_tokens=True
            ).strip()
            category = self.validate_response(response)

            if category is None:
                self.logger.debug(
                    f"Invalid response for token_id={batch[i]['token_id']}: "
                    f"'{response}' (token='{batch[i]['cleaned_token']}')"
                )

            results.append((category, per_token_time))

        self.stats["total_batches"] += 1
        self.stats["total_inference_time"] += inference_time

        return results

    def process_chunk(
        self,
        tokens: List[Dict],
        system_prompt: str,
        checkpoint_dir: Path,
        init_queue: mp.Queue,
        progress_queue: mp.Queue,
    ) -> List[Dict]:
        """Process a chunk of tokens with batching, retries, and checkpointing."""

        init_queue.put(self.gpu_id)

        self.logger.info(
            f"Processing {len(tokens):,} tokens in batches of {self.batch_size}"
        )
        results = []
        processed_ids: Set[int] = set()

        # Load checkpoint
        checkpoint_file = checkpoint_dir / f"gpu_{self.gpu_id}_checkpoint.json"
        start_idx = 0

        if checkpoint_file.exists():
            try:
                with open(checkpoint_file) as f:
                    ckpt = json.load(f)
                results = ckpt["results"]
                start_idx = ckpt["last_index"] + 1
                processed_ids = {r["token_id"] for r in results}
                self.logger.info(
                    f"Resumed from checkpoint: {len(results)} results, "
                    f"starting at index {start_idx}"
                )
            except Exception as e:
                self.logger.warning(f"Checkpoint load failed: {e}, starting fresh")
                results = []
                start_idx = 0

        last_log = time.time()
        last_ckpt = time.time()

        for batch_start in range(start_idx, len(tokens), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(tokens))
            batch = [
                t
                for t in tokens[batch_start:batch_end]
                if t["token_id"] not in processed_ids
            ]
            if not batch:
                continue

            try:
                batch_results = self.classify_batch(batch, system_prompt)
            except Exception as e:
                self.logger.error(f"Batch {batch_start}-{batch_end} failed: {e}")
                self.logger.debug(traceback.format_exc())
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for record in batch:
                    results.append(
                        {
                            "token_id": record["token_id"],
                            "llm_category": "ERROR",
                            "gpu_id": self.gpu_id,
                        }
                    )
                    processed_ids.add(record["token_id"])
                    self.stats["fallbacks"] += 1
                self.stats["total_processed"] += len(batch)
                progress_queue.put(len(batch))
                continue

            # Separate successes and failures
            failed_indices = []
            for i, (category, inf_time) in enumerate(batch_results):
                record = batch[i]
                if category is not None:
                    results.append(
                        {
                            "token_id": record["token_id"],
                            "llm_category": category,
                            "gpu_id": self.gpu_id,
                        }
                    )
                    processed_ids.add(record["token_id"])
                    self.stats["successful"] += 1
                else:
                    failed_indices.append(i)

            # Retry failed tokens
            if failed_indices and self.max_retries > 0:
                failed_batch = [batch[i] for i in failed_indices]
                self.stats["retries"] += len(failed_batch)

                for attempt in range(self.max_retries):
                    if not failed_batch:
                        break
                    still_failing = []
                    retry_size = max(1, min(8, self.batch_size // 4))
                    for rb_start in range(0, len(failed_batch), retry_size):
                        rb = failed_batch[rb_start : rb_start + retry_size]
                        try:
                            retry_results = self.classify_batch(
                                rb, system_prompt, is_retry=True
                            )
                            for j, (cat, _) in enumerate(retry_results):
                                if cat is not None:
                                    results.append(
                                        {
                                            "token_id": rb[j]["token_id"],
                                            "llm_category": cat,
                                            "gpu_id": self.gpu_id,
                                        }
                                    )
                                    processed_ids.add(rb[j]["token_id"])
                                    self.stats["retry_successes"] += 1
                                else:
                                    still_failing.append(rb[j])
                        except Exception:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            still_failing.extend(rb)
                    failed_batch = still_failing

                # Fallback for tokens that failed all retries
                for record in failed_batch:
                    signal = record.get("primary_signal", "")
                    fallback = "NOUN_COMMON" if signal == "WORD" else "STEM"
                    results.append(
                        {
                            "token_id": record["token_id"],
                            "llm_category": fallback,
                            "gpu_id": self.gpu_id,
                        }
                    )
                    processed_ids.add(record["token_id"])
                    self.stats["fallbacks"] += 1
                    self.logger.warning(
                        f"Fallback for token_id={record['token_id']} "
                        f"('{record['cleaned_token']}'): {fallback}"
                    )

            self.stats["total_processed"] += len(batch)
            progress_queue.put(len(batch))

            # Periodic logging
            if time.time() - last_log > 60:
                pct = 100 * batch_end / len(tokens)
                self.logger.info(
                    f"Progress: {batch_end}/{len(tokens)} ({pct:.1f}%) | "
                    f"Success: {self.stats['successful']:,} | "
                    f"Retries: {self.stats['retries']} | "
                    f"Fallbacks: {self.stats['fallbacks']}"
                )
                last_log = time.time()

            # Checkpoint every 5 minutes
            if time.time() - last_ckpt > 300:
                try:
                    with open(checkpoint_file, "w") as f:
                        json.dump(
                            {
                                "last_index": batch_end - 1,
                                "results": results,
                                "stats": self.stats,
                                "timestamp": datetime.now().isoformat(),
                            },
                            f,
                        )
                    self.logger.debug(f"Checkpoint saved at index {batch_end - 1}")
                except Exception as e:
                    self.logger.warning(f"Checkpoint save failed: {e}")
                last_ckpt = time.time()

        # Final checkpoint
        try:
            with open(checkpoint_file, "w") as f:
                json.dump(
                    {
                        "last_index": len(tokens) - 1,
                        "results": results,
                        "stats": self.stats,
                        "timestamp": datetime.now().isoformat(),
                        "completed": True,
                    },
                    f,
                )
        except Exception as e:
            self.logger.warning(f"Final checkpoint failed: {e}")

        # Save per-GPU CSV
        gpu_csv = self.output_dir / f"gpu_{self.gpu_id}_results.csv"
        pd.DataFrame(results).to_csv(gpu_csv, index=False)
        self.logger.info(f"Saved {len(results):,} results to {gpu_csv}")

        self.logger.info("=" * 70)
        self.logger.info(f"GPU {self.gpu_id} COMPLETE")
        self.logger.info(f"  Processed: {self.stats['total_processed']:,}")
        self.logger.info(f"  Successful: {self.stats['successful']:,}")
        self.logger.info(f"  Retries: {self.stats['retries']}")
        self.logger.info(f"  Retry successes: {self.stats['retry_successes']}")
        self.logger.info(f"  Fallbacks: {self.stats['fallbacks']}")
        if self.stats["total_batches"] > 0:
            avg = self.stats["total_inference_time"] / self.stats["total_batches"]
            self.logger.info(f"  Avg batch time: {avg:.2f}s")
        self.logger.info("=" * 70)

        return results


# ============================================================================
# WORKER PROCESS ENTRY POINT
# ============================================================================


def worker_process(
    gpu_id: int,
    tokens_chunk: List[Dict],
    system_prompt: str,
    output_dir: str,
    checkpoint_dir: str,
    model_name: str,
    batch_size: int,
    max_retries: int,
    timeout: int,
    result_queue: mp.Queue,
    init_queue: mp.Queue,
    progress_queue: mp.Queue,
):
    """Entry point for each GPU worker process."""
    worker = None
    try:
        worker = GPUWorker(
            gpu_id=gpu_id,
            output_dir=Path(output_dir),
            model_name=model_name,
            batch_size=batch_size,
            max_retries=max_retries,
            timeout=timeout,
        )
        results = worker.process_chunk(
            tokens_chunk,
            system_prompt,
            Path(checkpoint_dir),
            init_queue,
            progress_queue,
        )
        # Signal completion with count only : don't put large result lists
        # on the queue; the main process reads per-GPU CSV files instead.
        result_queue.put((gpu_id, len(results)))

    except Exception as e:
        logger = logging.getLogger(f"gpu_{gpu_id}")
        logger.error(f"FATAL: {e}")
        logger.error(traceback.format_exc())
        result_queue.put((gpu_id, 0))

    finally:
        if worker:
            worker.cleanup()
            cleanup_logger(worker.logger)
        gc.collect()
        # Give the queue feeder thread time to flush result data
        # before os._exit kills all threads
        time.sleep(5)
        os._exit(0)
