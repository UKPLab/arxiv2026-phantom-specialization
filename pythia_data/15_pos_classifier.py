#!/usr/bin/env python3
"""
POS Classifier (English Words & Subwords).

Two-tier classification for word_en and subword_en tokens:
  Tier 1: LLM (Qwen 2.5-14B) classifies into 13 categories using Pile contexts
  Tier 2: Rule-based post-processing refines closed classes (20 final categories)

LLM categories (Tier 1):
  Words:    NOUN_PROPER, NOUN_COMMON, VERB, ADJECTIVE, ADVERB, PRONOUN,
            FUNCTION, NUMERAL, INTERJECTION
  Subwords: PREFIX, SUFFIX, STEM, CONTRACTION

Refined categories (Tier 2 adds):
  VERB_LEXICAL, VERB_AUXILIARY, VERB_MODAL
  FUNCTION_DET, FUNCTION_PREP, FUNCTION_CONJ, FUNCTION_PARTICLE
  PRONOUN_PERSONAL, PRONOUN_POSSESSIVE, PRONOUN_DEMONSTRATIVE,
  PRONOUN_REFLEXIVE, PRONOUN_INTERROGATIVE, PRONOUN_INDEFINITE

Input:
  - pythia_data/token_categories/token_categories.csv     (Script 08)
  - pythia_data/context_windows/token_contexts.pkl         (Script 14, Pile-based)

Output:
  - pythia_data/pos_classification/pos_classification.csv
  - pythia_data/pos_classification/pos_classification_stats.json
"""

import sys
import json
import pickle
import random
import argparse
import time
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from tqdm import tqdm

# Add script directory to path for utils import
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from pos_classification_utils.gpu_worker import (
    worker_process,
    setup_logging,
    cleanup_logger,
)
from pos_classification_utils.postprocessor import (
    load_closed_class_lists,
    apply_tier2_refinement,
)


UTILS_DIR = SCRIPT_DIR / "pos_classification_utils"

DEFAULT_CATEGORIES_CSV = SCRIPT_DIR / "token_categories" / "token_categories.csv"
DEFAULT_CONTEXTS_PKL = SCRIPT_DIR / "context_windows" / "token_contexts.pkl"
DEFAULT_PROMPT_FILE = UTILS_DIR / "prompt.txt"
DEFAULT_CLOSED_LISTS = UTILS_DIR / "closed_class_lists.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "pos_classification"

DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct-1M"
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_CONTEXTS = 10
DEFAULT_SEED = 42
DEFAULT_MAX_RETRIES = 2
DEFAULT_TIMEOUT = 180


def load_data(
    categories_csv: Path,
    contexts_pkl: Path,
    n_contexts: int,
    seed: int,
    logger,
) -> list:
    """
    Load and merge data from Script 08 categories and Script 14 contexts.
    Returns list of token dicts ready for classification.
    """
    logger.info("Loading token categories (Script 08)...")
    cat_df = pd.read_csv(categories_csv)
    target = cat_df[cat_df["token_label"].isin(["word_en", "subword_en"])].copy()
    logger.info(
        f"  {len(target):,} tokens (word_en + subword_en) from {len(cat_df):,} total"
    )

    logger.info("Loading context windows (Script 14, Pile-based)...")
    with open(contexts_pkl, "rb") as f:
        all_contexts = pickle.load(f)
    logger.info(f"  {len(all_contexts):,} tokens with contexts")

    # Build final records using Script 08 columns directly
    rng = random.Random(seed)
    records = []

    for _, row in target.iterrows():
        token_string = str(row["token_string"]) if pd.notna(row["token_string"]) else ""

        # Use Script 08's content_text as cleaned token
        cleaned = (
            str(row.get("content_text", ""))
            if pd.notna(row.get("content_text"))
            else ""
        )
        if not cleaned:
            cleaned = token_string.replace("\u0120", "").replace("\u010a", "").strip()

        pool = all_contexts.get(token_string, [])
        sampled = rng.sample(pool, n_contexts) if len(pool) > n_contexts else list(pool)

        # Parse has_space_prefix
        hsp = row.get("has_space_prefix", False)
        if isinstance(hsp, str):
            hsp = hsp.lower() == "true"
        elif pd.isna(hsp):
            hsp = False

        records.append(
            {
                "token_id": int(row["token_id"]),
                "token_string": token_string,
                "cleaned_token": cleaned,
                "token_label": row["token_label"],
                "has_leading_space": bool(hsp),
                "is_roman_numeral": bool(row.get("is_roman_numeral", False)),
                "is_abbreviation": bool(row.get("is_abbreviation", False)),
                "contexts": sampled,
                "n_contexts": len(sampled),
            }
        )

    with_ctx = sum(1 for r in records if r["n_contexts"] > 0)
    full_ctx = sum(1 for r in records if r["n_contexts"] >= n_contexts)
    logger.info(
        f"  With contexts: {with_ctx:,}/{len(records):,} ({100 * with_ctx / len(records):.1f}%)"
    )
    logger.info(
        f"  Full ({n_contexts}): {full_ctx:,}/{len(records):,} ({100 * full_ctx / len(records):.1f}%)"
    )

    return records


def main():
    parser = argparse.ArgumentParser(
        description="POS Classifier (English Words & Subwords)"
    )
    parser.add_argument("--categories-csv", type=Path, default=DEFAULT_CATEGORIES_CSV)
    parser.add_argument("--contexts-pkl", type=Path, default=DEFAULT_CONTEXTS_PKL)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--closed-lists", type=Path, default=DEFAULT_CLOSED_LISTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--gpus", type=int, default=None, help="Number of GPUs (default: all available)"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n-contexts", type=int, default=DEFAULT_NUM_CONTEXTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument(
        "--no-resume", action="store_true", help="Don't resume from checkpoint"
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.no_resume:
        for f in checkpoint_dir.glob("*.json"):
            f.unlink()

    logger = setup_logging(output_dir)

    logger.info("=" * 70)
    logger.info("POS CLASSIFIER")
    logger.info("=" * 70)
    logger.info(f"  Model:      {args.model}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Contexts:   {args.n_contexts} per token")
    logger.info(f"  Seed:       {args.seed}")
    logger.info(f"  Retries:    {args.max_retries}")
    logger.info("")

    if not torch.cuda.is_available():
        logger.error("CUDA not available!")
        sys.exit(1)

    num_gpus = args.gpus or torch.cuda.device_count()
    num_gpus = min(num_gpus, torch.cuda.device_count())
    logger.info(f"GPUs: {num_gpus}")
    for i in range(num_gpus):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # ---- Load data ----
    logger.info("")
    records = load_data(
        args.categories_csv,
        args.contexts_pkl,
        args.n_contexts,
        args.seed,
        logger,
    )
    total_tokens = len(records)

    # ---- Load prompt ----
    logger.info(f"Loading prompt from {args.prompt_file}...")
    system_prompt = args.prompt_file.read_text(encoding="utf-8")
    logger.info(f"  Prompt: {len(system_prompt):,} chars")

    # ---- Split across GPUs ----
    chunk_size = len(records) // num_gpus
    chunks = []
    for i in range(num_gpus):
        start = i * chunk_size
        end = start + chunk_size if i < num_gpus - 1 else len(records)
        chunks.append(records[start:end])
        logger.info(f"  GPU {i}: {len(chunks[-1]):,} tokens")
    del records

    # ---- Launch workers ----
    logger.info("")
    logger.info(f"Launching {num_gpus} worker processes...")
    mp.set_start_method("spawn", force=True)
    result_queue = mp.Queue()
    init_queue = mp.Queue()
    progress_queue = mp.Queue()

    start_time = time.time()
    processes = []

    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=worker_process,
            args=(
                gpu_id,
                chunks[gpu_id],
                system_prompt,
                str(output_dir),
                str(checkpoint_dir),
                args.model,
                args.batch_size,
                args.max_retries,
                args.timeout,
                result_queue,
                init_queue,
                progress_queue,
            ),
            name=f"GPU-{gpu_id}",
            daemon=True,
        )
        p.start()
        processes.append(p)
        logger.info(f"  GPU {gpu_id} started (PID: {p.pid})")

    # ---- Wait for initialization ----
    logger.info("Waiting for model loading...")
    initialized = set()
    init_timeout = time.time() + 600

    while len(initialized) < num_gpus and time.time() < init_timeout:
        try:
            gpu_id = init_queue.get(timeout=15)
            initialized.add(gpu_id)
            logger.info(f"  GPU {gpu_id} ready ({len(initialized)}/{num_gpus})")
        except Exception:
            for i, p in enumerate(processes):
                if not p.is_alive() and i not in initialized:
                    logger.error(f"  GPU {i} died during init (exit: {p.exitcode})")

    if len(initialized) < num_gpus:
        logger.error(f"Only {len(initialized)}/{num_gpus} GPUs initialized!")

    # ---- Monitor progress ----
    logger.info("")
    logger.info("Processing...")
    processed = 0
    last_health = time.time()
    completed_gpus = set()

    with tqdm(total=total_tokens, desc="Classifying", unit="tok") as pbar:
        while True:
            try:
                while True:
                    n = progress_queue.get(timeout=0.1)
                    processed += n
                    pbar.update(n)
            except Exception:
                pass

            if time.time() - last_health > 30:
                elapsed = time.time() - start_time
                if processed > 0:
                    throughput = processed / elapsed
                    remaining = (total_tokens - processed) / throughput
                    pbar.set_postfix(
                        {
                            "tok/s": f"{throughput:.1f}",
                            "ETA": f"{remaining / 60:.1f}m",
                        }
                    )

                for i, p in enumerate(processes):
                    if not p.is_alive() and i not in completed_gpus:
                        completed_gpus.add(i)
                        if p.exitcode == 0:
                            logger.info(f"GPU {i} completed")
                        else:
                            logger.error(f"GPU {i} crashed (exit: {p.exitcode})")

                if len(completed_gpus) == num_gpus:
                    try:
                        while True:
                            n = progress_queue.get(timeout=0.5)
                            processed += n
                            pbar.update(n)
                    except Exception:
                        pass
                    break

                last_health = time.time()

            time.sleep(0.5)

    # ---- Wait for workers ----
    for i, p in enumerate(processes):
        if i not in completed_gpus:
            p.join(timeout=120)
            if p.is_alive():
                logger.warning(f"GPU {i} still alive, terminating...")
                p.terminate()
                p.join(timeout=10)

    # ---- Collect results from per-GPU CSV files ----
    # Workers save results to disk (gpu_N_results.csv) and only send a
    # lightweight completion signal through the queue.  Reading from CSV
    # avoids the multiprocessing Queue pipe-buffer issue that caused hangs
    # when transferring large result sets.
    logger.info("")
    logger.info("Collecting results from per-GPU CSV files...")
    all_results = []

    for gpu_id in range(num_gpus):
        gpu_csv = output_dir / f"gpu_{gpu_id}_results.csv"
        if gpu_csv.exists():
            try:
                gpu_df = pd.read_csv(gpu_csv)
                all_results.extend(gpu_df.to_dict("records"))
                logger.info(f"  GPU {gpu_id}: {len(gpu_df):,} results from CSV")
                continue
            except Exception as e:
                logger.warning(f"  GPU {gpu_id}: CSV read failed: {e}")

        # Fallback to checkpoint
        ckpt_file = checkpoint_dir / f"gpu_{gpu_id}_checkpoint.json"
        if ckpt_file.exists():
            try:
                with open(ckpt_file) as f:
                    ckpt = json.load(f)
                all_results.extend(ckpt["results"])
                logger.info(
                    f"  GPU {gpu_id}: recovered {len(ckpt['results']):,} from checkpoint"
                )
            except Exception as e:
                logger.error(f"  GPU {gpu_id}: checkpoint recovery failed: {e}")
        else:
            logger.error(f"  GPU {gpu_id}: no CSV or checkpoint found!")

    if not all_results:
        logger.error("No results collected!")
        sys.exit(1)

    logger.info(f"Total results: {len(all_results):,}")

    # ---- Merge with input data ----
    logger.info("Merging with input data...")
    results_df = pd.DataFrame(all_results)

    cat_df = pd.read_csv(args.categories_csv)
    target = cat_df[cat_df["token_label"].isin(["word_en", "subword_en"])].copy()

    # Rename Script 08 columns to match expected output format
    base = target.rename(
        columns={
            "content_text": "cleaned_token",
            "has_space_prefix": "has_leading_space",
        }
    )

    with open(args.contexts_pkl, "rb") as f:
        all_contexts = pickle.load(f)
    n_ctx = args.n_contexts
    base["n_contexts"] = base["token_string"].apply(
        lambda ts: min(
            n_ctx, len(all_contexts.get(str(ts) if pd.notna(ts) else "", []))
        )
    )
    base["has_contexts"] = base["n_contexts"] > 0
    del all_contexts

    final = base.merge(
        results_df[["token_id", "llm_category", "gpu_id"]],
        on="token_id",
        how="left",
    )

    missing_mask = final["llm_category"].isna()
    if missing_mask.any():
        logger.warning(f"{missing_mask.sum():,} tokens missing LLM classification")
        final.loc[missing_mask, "llm_category"] = "NOUN_COMMON"
        final.loc[missing_mask, "gpu_id"] = -1

    # ---- Tier 2: closed-class refinement ----
    logger.info("")
    closed_lists = load_closed_class_lists(args.closed_lists)
    final = apply_tier2_refinement(final, closed_lists, logger)

    final = final.sort_values("token_id").reset_index(drop=True)

    output_cols = [
        "token_id",
        "token_string",
        "cleaned_token",
        "token_label",
        "has_leading_space",
        "is_roman_numeral",
        "is_abbreviation",
        "llm_category",
        "final_category",
        "n_contexts",
        "has_contexts",
        "gpu_id",
    ]
    output_cols = [c for c in output_cols if c in final.columns]
    final = final[output_cols]

    # ---- Save outputs ----
    logger.info("")
    logger.info("Saving outputs...")

    output_csv = output_dir / "pos_classification.csv"
    final.to_csv(output_csv, index=False)
    logger.info(f"  {output_csv} ({len(final):,} rows)")

    elapsed = time.time() - start_time
    stats = {
        "timestamp": datetime.now().isoformat(),
        "params": {
            "model": args.model,
            "num_gpus": num_gpus,
            "batch_size": args.batch_size,
            "n_contexts": args.n_contexts,
            "seed": args.seed,
            "max_retries": args.max_retries,
        },
        "summary": {
            "total_tokens": len(final),
            "word_en": int((final["token_label"] == "word_en").sum()),
            "subword_en": int((final["token_label"] == "subword_en").sum()),
            "with_contexts": int(final["has_contexts"].sum()),
            "elapsed_minutes": round(elapsed / 60, 2),
            "tokens_per_second": round(len(final) / elapsed, 2) if elapsed > 0 else 0,
        },
        "llm_category_distribution": final["llm_category"].value_counts().to_dict(),
        "final_category_distribution": final["final_category"].value_counts().to_dict(),
        "by_token_label": {
            "word_en": final[final["token_label"] == "word_en"]["final_category"]
            .value_counts()
            .to_dict(),
            "subword_en": final[final["token_label"] == "subword_en"]["final_category"]
            .value_counts()
            .to_dict(),
        },
    }

    stats_path = output_dir / "pos_classification_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  {stats_path}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("DONE")
    logger.info(f"  Tokens classified: {len(final):,}")
    logger.info(f"  Elapsed: {elapsed / 60:.1f} minutes")
    logger.info(f"  Output: {output_csv}")
    logger.info("=" * 70)

    cleanup_logger(logger)


if __name__ == "__main__":
    main()
