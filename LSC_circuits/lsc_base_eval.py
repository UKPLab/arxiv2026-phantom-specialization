#!/usr/bin/env python3
"""
LSC Base Model Evaluation (Pre-computation)
=============================================

Pre-compute base model metrics for all (model, band) combinations on VAL and TEST splits.
These cached results are used by:
  - lsc_pareto_sweep.py (Phase 1): uses control band VAL metrics for threshold selection
  - lsc_acdc_circuit.py (Phase 2): uses all bands TEST metrics for final evaluation

DATA SPLIT RATIONALE:
  - TRAIN split: Used ONLY by ACDC for circuit pruning (not evaluated here)
  - VAL split: Used for threshold selection in Phase 1 Pareto sweep
  - TEST split: Used for final circuit evaluation in Phase 2 (held out)

This script evaluates on FULL val/test splits (no sampling) to ensure:
  1. Consistent comparison across all downstream evaluations
  2. No variance from random sampling
  3. Reproducibility of all results

Output structure:
    base_metrics/
    ├── registry.json
    ├── pythia_70m/
    │   ├── draw_1/
    │   │   ├── low.json  (val/test metrics + example indices)
    │   │   ├── control.json
    │   │   └── ...
    │   ├── draw_2/
    │   │   └── ...
    │   └── draw_3/
    │       └── ...
    ├── pythia_160m/
    │   └── ...
    └── summary/
        └── base_eval_summary.json

Usage:
    python lsc_base_eval.py
    python lsc_base_eval.py --models pythia-70m pythia-160m
    python lsc_base_eval.py --bands control low medium high
    python lsc_base_eval.py --draws draw_1 draw_2
    python lsc_base_eval.py --gpus 0 1 2 3
    python lsc_base_eval.py --force  # recompute all
"""

import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import random
import argparse
import traceback
import fcntl
import gc
import time
import threading
import logging
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import OrderedDict

import numpy as np
import torch as t
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
ISC_ROOT = SCRIPT_DIR.parent  # repo root


ALL_BANDS = [
    "low",
    "medium",
    "high",
    "very_high",
    "control",
]

DEFAULT_MODELS = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
]

# Safe batch sizes per model for A100 80GB (conservative for base eval)
MODEL_BATCH_SIZES = {
    "pythia-70m": 256,
    "pythia-160m": 256,
    "pythia-410m": 128,
    "pythia-1b": 128,
    "pythia-1.4b": 64,
}

# LSC sequence constants
N_SOURCE = 5
N_DISTRACT = 10
RAW_SEQ_LEN = N_SOURCE + 1 + N_DISTRACT + N_SOURCE  # 21
SEQ_LEN_WITH_BOS = RAW_SEQ_LEN + 1  # 22


@dataclass
class BaseEvalConfig:
    """Configuration for base model evaluation."""

    # Paths (relative to ISC_ROOT by default)
    data_dir: str = field(default_factory=lambda: str(ISC_ROOT / "LSC_data"))
    pool_dir: str = field(
        default_factory=lambda: str(
            ISC_ROOT / "LSC_data" / "lsc_token_pools" / "matched"
        )
    )
    output_dir: str = field(default_factory=lambda: str(SCRIPT_DIR / "base_metrics"))
    # Dataset structure: datasets/{variant}/{draw}/{band}/{split}.json
    variant: str = "matched"
    draws: List[str] = field(default_factory=lambda: ["draw_1", "draw_2", "draw_3"])

    # Experiment grid
    models: List[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    bands: List[str] = field(default_factory=lambda: list(ALL_BANDS))
    # Only val and test - train is used exclusively by ACDC for circuit pruning
    splits: List[str] = field(default_factory=lambda: ["val", "test"])

    # Evaluation settings
    # Note: We evaluate on FULL splits (no sampling) for consistency
    # eval_seed is used for deterministic batch ordering
    eval_seed: int = 123

    # GPU
    gpus: List[int] = field(default_factory=list)
    registry_file: str = "registry.json"


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False


def cleanup_gpu():
    gc.collect()
    if t.cuda.is_available():
        t.cuda.synchronize()
        t.cuda.empty_cache()


def safe_delete_model(model):
    if model is not None:
        try:
            model.cpu()
        except Exception:
            pass
        del model
    cleanup_gpu()


def setup_logging(output_dir: Path, debug: bool = False) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"base_eval_{timestamp}.log"

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(process)d] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("base_eval")
    logger.info(f"Log file: {log_file}")
    return logger


def model_safe_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def get_batch_size(model_name: str) -> int:
    return MODEL_BATCH_SIZES.get(model_name, 64)


def load_pool(band: str, pool_dir: Path) -> dict:
    """Load token pool JSON for a frequency band."""
    with open(pool_dir / f"lsc_pool_{band}.json") as f:
        return json.load(f)


def load_dataset(
    band: str, split: str, data_dir: Path, variant: str, draw: str
) -> dict:
    """Load LSC dataset JSON for a band and split."""
    with open(data_dir / "datasets" / variant / draw / band / f"{split}.json") as f:
        return json.load(f)


def prepare_evaluation_data(
    dataset: dict,
    pool: dict,
    bos_token_id: int,
    seed: int,
    device: str,
) -> Tuple[t.Tensor, List[int], List[int]]:
    """
    Prepare FULL evaluation data without AutoCircuit dependencies.
    Uses all examples in the dataset (no sampling) for consistency.

    Returns:
        input_ids: Tensor of shape (N, 22) - full input sequences
        answer_ids: List of target token IDs
        example_indices: List of original indices (for reproducibility tracking)
    """
    examples = dataset["examples"]
    n_examples = len(examples)

    # Use seed for deterministic ordering (though we use all examples)
    rng = random.Random(seed)
    indices = list(range(n_examples))
    rng.shuffle(indices)

    input_ids = []
    answer_ids = []
    example_indices = []

    for idx in indices:
        ex = examples[idx]
        token_ids = ex["token_ids"]  # length 21, no BOS
        clean = [bos_token_id] + token_ids
        input_ids.append(clean)
        answer_ids.append(ex["target_token_id"])
        example_indices.append(idx)

    input_tensor = t.tensor(input_ids, dtype=t.long, device=device)
    return input_tensor, answer_ids, example_indices


def load_model(model_name: str, device: str):
    """
    Load Pythia model via HuggingFace transformers.

    For base evaluation we only need forward passes, no hooks or circuit manipulation.
    HuggingFace is simpler, faster, and more memory-efficient than TransformerLens
    for this use case.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Map short names to HuggingFace model IDs
    hf_name = f"EleutherAI/{model_name}"

    logger = logging.getLogger("base_eval")
    logger.info(f"[{device}] Loading {hf_name} via HuggingFace...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    # Load model in FP32 for consistency with circuit evaluation pipeline
    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=t.float32,
        device_map=device,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Attach tokenizer for easy access
    model.tokenizer = tokenizer

    logger.info(
        f"[{device}] Loaded {model_name} ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)"
    )
    return model


def compute_metrics(
    logits: t.Tensor,
    answer_ids: List[int],
) -> Dict[str, Any]:
    """
    Compute accuracy (top-1/5/10), mean P(correct).
    """
    if len(logits.shape) == 3:
        logits = logits[:, -1, :]

    probs = F.softmax(logits, dim=-1)
    n = min(len(answer_ids), logits.shape[0])

    top1 = top5 = top10 = 0
    correct_probs = []

    for i in range(n):
        topk = t.topk(logits[i], k=10).indices.tolist()
        if topk[0] == answer_ids[i]:
            top1 += 1
        if answer_ids[i] in topk[:5]:
            top5 += 1
        if answer_ids[i] in topk:
            top10 += 1
        correct_probs.append(probs[i, answer_ids[i]].item())

    return {
        "accuracy": top1 / n if n else 0.0,
        "top5_accuracy": top5 / n if n else 0.0,
        "top10_accuracy": top10 / n if n else 0.0,
        "mean_correct_prob": float(np.mean(correct_probs)) if correct_probs else 0.0,
        "n_samples": n,
    }


def run_base_eval_task(
    model_name: str,
    band: str,
    draw: str,
    config: BaseEvalConfig,
    device: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Evaluate base model on VAL and TEST splits for a single (model, band, draw).
    Uses FULL splits (no sampling) for consistent comparison with circuit evaluations.
    """
    logger = logging.getLogger("base_eval")
    batch_size = get_batch_size(model_name)

    logger.info(f"[{device}] Starting: {model_name} / {band} / {draw}")

    model = None

    try:
        set_all_seeds(config.eval_seed)

        # Load model
        model = load_model(model_name, device)
        bos_id = model.tokenizer.bos_token_id

        # Load pool and prepare for all splits
        pool_dir = Path(config.pool_dir)
        data_dir = Path(config.data_dir)
        pool = load_pool(band, pool_dir)

        split_metrics = {}
        split_indices = {}  # Track example indices for reproducibility
        t_start = time.time()

        for split in config.splits:
            try:
                dataset = load_dataset(band, split, data_dir, config.variant, draw)
            except FileNotFoundError:
                logger.warning(f"[{device}] Missing: {band}/{split}.json")
                split_metrics[split] = {"error": "file_not_found"}
                continue

            # Prepare FULL data (no sampling)
            input_ids, answer_ids, example_indices = prepare_evaluation_data(
                dataset,
                pool,
                bos_id,
                seed=config.eval_seed,
                device=device,
            )
            split_indices[split] = example_indices

            # Forward pass in batches with OOM handling
            all_logits = []
            current_batch_size = batch_size
            i = 0

            with t.no_grad():
                while i < len(input_ids):
                    batch = input_ids[i : i + current_batch_size]
                    try:
                        # HuggingFace returns CausalLMOutput with .logits attribute
                        outputs = model(batch)
                        logits = (
                            outputs.logits if hasattr(outputs, "logits") else outputs
                        )
                        if len(logits.shape) == 3:
                            logits = logits[:, -1, :]
                        all_logits.append(
                            logits.float().cpu()
                        )  # Convert to FP32 and move to CPU
                        i += current_batch_size
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            # Reduce batch size and retry
                            cleanup_gpu()
                            current_batch_size = max(1, current_batch_size // 2)
                            logger.warning(
                                f"[{device}] OOM on {model_name}/{band}/{split}, "
                                f"reducing batch size to {current_batch_size}"
                            )
                            if current_batch_size < 1:
                                raise RuntimeError(
                                    f"Cannot fit even batch_size=1 for {model_name}"
                                )
                        else:
                            raise

            logits = t.cat(all_logits, dim=0)
            metrics = compute_metrics(logits, answer_ids)
            split_metrics[split] = metrics

            logger.debug(
                f"[{device}] {band}/{split}: n={metrics['n_samples']}, acc={metrics['accuracy']:.1%}"
            )

        eval_time = time.time() - t_start

        # Save per-band metrics (draw-aware path)
        m_safe = model_safe_name(model_name)
        model_dir = output_dir / m_safe / draw
        model_dir.mkdir(parents=True, exist_ok=True)

        band_result = {
            "model": model_name,
            "band": band,
            "variant": config.variant,
            "draw": draw,
            "eval_seed": config.eval_seed,
            "full_split_evaluation": True,  # Flag indicating no sampling
            "splits": split_metrics,
            "example_indices": split_indices,  # For reproducibility tracking
            "eval_time_seconds": eval_time,
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
        }

        with open(model_dir / f"{band}.json", "w") as f:
            json.dump(band_result, f, indent=2)

        # Log summary
        val_acc = split_metrics.get("val", {}).get("accuracy", 0)
        val_n = split_metrics.get("val", {}).get("n_samples", 0)
        test_acc = split_metrics.get("test", {}).get("accuracy", 0)
        test_n = split_metrics.get("test", {}).get("n_samples", 0)

        logger.info(
            f"[{device}] Done: {model_name}/{band}/{draw} -> "
            f"val={val_acc:.1%} (n={val_n}), test={test_acc:.1%} (n={test_n}), "
            f"{eval_time:.1f}s"
        )

        return band_result

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"[{device}] FAILED: {model_name}/{band}/{draw}: {error_msg}")
        return {
            "model": model_name,
            "band": band,
            "draw": draw,
            "status": "failed",
            "error": error_msg,
            "failed_at": datetime.now().isoformat(),
        }

    finally:
        safe_delete_model(model)
        cleanup_gpu()


class Registry:
    """Atomic JSON registry with file locking for concurrent GPU workers."""

    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_suffix(".lock")

    def _lock(self, timeout=30.0):
        lf = open(self.lock_path, "w")
        t0 = time.time()
        while True:
            try:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lf
            except (IOError, OSError):
                if time.time() - t0 > timeout:
                    lf.close()
                    raise TimeoutError(f"Lock timeout: {self.lock_path}")
                time.sleep(0.1)

    def _unlock(self, lf):
        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        lf.close()

    def load(self) -> dict:
        lf = self._lock()
        try:
            if self.path.exists():
                with open(self.path) as f:
                    return json.load(f)
            return {"metadata": {"created_at": datetime.now().isoformat()}, "tasks": {}}
        finally:
            self._unlock(lf)

    def add_task(self, task_id: str, data: dict):
        lf = self._lock()
        try:
            if self.path.exists():
                with open(self.path) as _f:
                    reg = json.load(_f)
            else:
                reg = {
                    "metadata": {"created_at": datetime.now().isoformat()},
                    "tasks": {},
                }
            reg["tasks"][task_id] = data
            reg["metadata"]["last_updated"] = datetime.now().isoformat()
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(reg, f, indent=2, default=str)
            tmp.rename(self.path)
        finally:
            self._unlock(lf)

    def get_completed_ids(self) -> Set[str]:
        reg = self.load()
        return {
            tid
            for tid, d in reg.get("tasks", {}).items()
            if d.get("status") == "completed"
        }

    @staticmethod
    def make_task_id(model: str, band: str, draw: str) -> str:
        m = model_safe_name(model)
        return f"{m}__{band}__{draw}"


def gpu_worker(
    gpu_id,
    task_queue,
    result_queue,
    config_dict,
    output_dir_str,
    progress_dict,
    worker_id,
):
    """Process tasks from queue on a single GPU."""
    config = BaseEvalConfig(**config_dict)
    output_dir = Path(output_dir_str)
    device = f"cuda:{gpu_id}"
    t.cuda.set_device(gpu_id)
    registry = Registry(output_dir / config.registry_file)

    while True:
        try:
            task = task_queue.get(timeout=2)
        except Exception:
            if task_queue.empty():
                break
            continue
        if task is None:  # poison pill
            break

        progress_dict[worker_id] = (
            f"GPU {gpu_id}: {task['model']}/{task['band']}/{task['draw']}"
        )
        try:
            result = run_base_eval_task(
                model_name=task["model"],
                band=task["band"],
                draw=task["draw"],
                config=config,
                device=device,
                output_dir=output_dir,
            )
            registry.add_task(task["id"], result)
            result_queue.put((task, result, None))
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            err = {
                "model": task["model"],
                "band": task["band"],
                "draw": task["draw"],
                "status": "failed",
                "error": error_msg,
                "failed_at": datetime.now().isoformat(),
            }
            registry.add_task(task["id"], err)
            result_queue.put((task, err, error_msg))
            cleanup_gpu()

        progress_dict[worker_id] = f"GPU {gpu_id}: idle"


def run_all_tasks(
    config: BaseEvalConfig,
    output_dir: Path,
    gpus: List[int],
    force: bool = False,
):
    """Generate and execute all (model, band, draw) tasks."""
    logger = logging.getLogger("base_eval")
    registry = Registry(output_dir / config.registry_file)

    # Generate tasks: draws (outermost) x models (ascending size) x bands
    # Completes all models/bands for draw_1 before moving to draw_2, etc.
    all_tasks = []
    for draw in config.draws:
        for model in config.models:
            for band in config.bands:
                all_tasks.append(
                    {
                        "model": model,
                        "band": band,
                        "draw": draw,
                        "id": Registry.make_task_id(model, band, draw),
                    }
                )

    logger.info(f"Total tasks: {len(all_tasks)}")

    # Filter completed
    if force:
        tasks = all_tasks
    else:
        completed = registry.get_completed_ids()
        tasks = [tk for tk in all_tasks if tk["id"] not in completed]
        logger.info(f"Already completed: {len(all_tasks) - len(tasks)}")

    logger.info(f"Remaining: {len(tasks)}")
    if not tasks:
        return

    n_gpus = len(gpus)

    # ---- Single GPU: sequential ----
    if n_gpus <= 1:
        device = f"cuda:{gpus[0]}" if gpus else "cpu"
        logger.info(f"Sequential execution on {device}")
        t_start = time.time()

        for i, task in enumerate(tasks):
            if i > 0:
                elapsed = time.time() - t_start
                avg = elapsed / i
                eta = timedelta(seconds=int(avg * (len(tasks) - i)))
                logger.info(
                    f"Progress: {i}/{len(tasks)} ({100 * i / len(tasks):.0f}%) ETA: {eta}"
                )

            result = run_base_eval_task(
                model_name=task["model"],
                band=task["band"],
                draw=task["draw"],
                config=config,
                device=device,
                output_dir=output_dir,
            )
            registry.add_task(task["id"], result)
        return

    # ---- Multi-GPU: parallel ----
    logger.info(f"Parallel execution on {n_gpus} GPUs: {gpus}")
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    manager = ctx.Manager()
    progress_dict = manager.dict()

    for task in tasks:
        task_queue.put(task)
    for _ in gpus:
        task_queue.put(None)  # poison pills

    config_dict = asdict(config)
    workers = []
    for i, gpu_id in enumerate(gpus):
        progress_dict[i] = f"GPU {gpu_id}: starting"
        p = ctx.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                task_queue,
                result_queue,
                config_dict,
                str(output_dir),
                progress_dict,
                i,
            ),
        )
        p.start()
        workers.append(p)

    # Progress monitor
    def monitor():
        while any(p.is_alive() for p in workers):
            time.sleep(15)
            try:
                done = max(0, len(tasks) - task_queue.qsize() - n_gpus)
            except Exception:
                done = 0
            if done > 0:
                logger.info(
                    f"Progress: ~{done}/{len(tasks)} ({100 * done / len(tasks):.0f}%)"
                )
                for wid, st in progress_dict.items():
                    logger.info(f"  {st}")

    threading.Thread(target=monitor, daemon=True).start()

    # Collect results
    for _ in range(len(tasks)):
        try:
            result_queue.get(timeout=86400)  # 24h timeout
        except Exception:
            break

    for p in workers:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()


def generate_summary(config: BaseEvalConfig, output_dir: Path) -> dict:
    """Aggregate all results into summary with mean +/- std across draws."""
    logger = logging.getLogger("base_eval")
    registry = Registry(output_dir / config.registry_file)
    reg = registry.load()

    completed = {
        tid: d
        for tid, d in reg.get("tasks", {}).items()
        if d.get("status") == "completed"
    }
    failed = {
        tid: d for tid, d in reg.get("tasks", {}).items() if d.get("status") == "failed"
    }

    logger.info(f"Registry: {len(completed)} completed, {len(failed)} failed")

    summary = OrderedDict()
    summary["created_at"] = datetime.now().isoformat()
    summary["n_completed"] = len(completed)
    summary["n_failed"] = len(failed)
    summary["config"] = {
        "models": config.models,
        "bands": config.bands,
        "draws": config.draws,
        "splits": config.splits,
        "variant": config.variant,
        "eval_seed": config.eval_seed,
        "full_split_evaluation": True,  # No sampling - full splits
    }

    # Build results table: per-draw results + aggregated mean +/- std
    results = OrderedDict()
    for model in config.models:
        model_results = OrderedDict()
        for band in config.bands:
            # Collect per-draw results
            per_draw = OrderedDict()
            for draw in config.draws:
                task_id = Registry.make_task_id(model, band, draw)
                if task_id in completed:
                    data = completed[task_id]
                    per_draw[draw] = {
                        split: data.get("splits", {}).get(split, {})
                        for split in config.splits
                    }

            if not per_draw:
                continue

            # Aggregate across draws: mean +/- std for each metric
            aggregated = OrderedDict()
            for split in config.splits:
                split_vals = {}
                for draw_data in per_draw.values():
                    if split not in draw_data:
                        continue
                    for metric, val in draw_data[split].items():
                        if isinstance(val, (int, float)):
                            split_vals.setdefault(metric, []).append(val)

                agg = OrderedDict()
                for metric, vals in split_vals.items():
                    agg[f"{metric}_mean"] = float(np.mean(vals))
                    agg[f"{metric}_std"] = float(np.std(vals))
                agg["n_draws"] = len(per_draw)
                aggregated[split] = agg

            model_results[band] = {
                "per_draw": per_draw,
                "aggregated": aggregated,
            }

        if model_results:
            results[model] = model_results

    summary["results"] = results

    # Save summary
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / "base_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def print_summary_table(summary: dict, config: BaseEvalConfig):
    """Print concise results table with mean +/- std across draws."""
    logger = logging.getLogger("base_eval")

    logger.info("\n" + "=" * 100)
    logger.info(
        f"BASE MODEL EVALUATION SUMMARY (Full VAL/TEST splits, {len(config.draws)} draws)"
    )
    logger.info("=" * 100)

    results = summary.get("results", {})

    for model in config.models:
        model_data = results.get(model)
        if not model_data:
            continue

        logger.info(f"\n--- {model} ---")
        logger.info(
            f"  {'Band':<14s} {'Val Acc (mean+/-std)':>22s} {'Test Acc (mean+/-std)':>22s} {'Draws':>6s}"
        )
        logger.info("  " + "-" * 68)

        for band in config.bands:
            entry = model_data.get(band)
            if not entry:
                continue

            agg = entry.get("aggregated", {})
            val = agg.get("val", {})
            test = agg.get("test", {})
            n_draws = val.get("n_draws", test.get("n_draws", 0))

            val_mean = val.get("accuracy_mean", 0)
            val_std = val.get("accuracy_std", 0)
            test_mean = test.get("accuracy_mean", 0)
            test_std = test.get("accuracy_std", 0)

            logger.info(
                f"  {band:<14s} {val_mean:>6.1%} +/- {val_std:>5.1%}      "
                f"{test_mean:>6.1%} +/- {test_std:>5.1%}      {n_draws:>3d}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="LSC Base Model Evaluation -- pre-compute metrics for all bands",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to evaluate (default: all Pythia 70m-12b)",
    )
    parser.add_argument(
        "--bands", nargs="+", default=None, help="Bands to evaluate (default: all 8)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to evaluate (default: val test)",
    )
    parser.add_argument(
        "--gpus", nargs="+", default=["auto"], help="GPU devices: 'auto' or list of IDs"
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--pool-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--variant",
        type=str,
        default="matched",
        help="Dataset variant: 'matched' or 'unmatched'",
    )
    parser.add_argument(
        "--draws",
        nargs="+",
        type=str,
        default=None,
        help="Draw indices (default: draw_1 draw_2 draw_3)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute all tasks (ignore registry)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only generate summary from existing results",
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # ---- Build config ----
    config = BaseEvalConfig()
    if args.models:
        config.models = args.models
    if args.bands:
        config.bands = args.bands
    if args.splits:
        config.splits = args.splits
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.pool_dir:
        config.pool_dir = args.pool_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    config.variant = args.variant
    if args.draws:
        config.draws = args.draws

    if args.gpus == ["auto"]:
        config.gpus = (
            list(range(t.cuda.device_count())) if t.cuda.is_available() else []
        )
    else:
        config.gpus = [int(g) for g in args.gpus]

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir, args.debug)

    n_tasks = len(config.models) * len(config.bands) * len(config.draws)

    logger.info("=" * 70)
    logger.info("LSC BASE MODEL EVALUATION (Full VAL/TEST splits)")
    logger.info("=" * 70)
    logger.info(f"Models:     {config.models}")
    logger.info(f"Bands:      {config.bands}")
    logger.info(f"Draws:      {config.draws}")
    logger.info(f"Splits:     {config.splits} (FULL - no sampling)")
    logger.info(f"GPUs:       {config.gpus}")
    logger.info(f"Data:       {config.data_dir}")
    logger.info(f"Output:     {config.output_dir}")
    logger.info(f"Tasks:      {n_tasks}")
    logger.info("=" * 70)

    # ---- Summary-only mode ----
    if args.summary_only:
        summary = generate_summary(config, output_dir)
        print_summary_table(summary, config)
        logger.info(f"\nSummary: {output_dir / 'summary' / 'base_eval_summary.json'}")
        return 0

    # ---- Validate data exists ----
    pool_dir = Path(config.pool_dir)
    data_dir = Path(config.data_dir)
    missing = []
    for band in config.bands:
        if not (pool_dir / f"lsc_pool_{band}.json").exists():
            missing.append(f"lsc_pool_{band}.json")
        for draw in config.draws:
            for split in config.splits:
                split_path = (
                    data_dir
                    / "datasets"
                    / config.variant
                    / draw
                    / band
                    / f"{split}.json"
                )
                if not split_path.exists():
                    missing.append(
                        f"datasets/{config.variant}/{draw}/{band}/{split}.json"
                    )
    if missing:
        for m in missing[:10]:  # show first 10
            logger.error(f"Missing: {m}")
        if len(missing) > 10:
            logger.error(f"... and {len(missing) - 10} more")
        return 1

    # ---- Run tasks ----
    run_all_tasks(config, output_dir, config.gpus, force=args.force)

    # ---- Generate summary ----
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    summary = generate_summary(config, output_dir)
    print_summary_table(summary, config)

    # ---- Final ----
    logger.info("\n" + "=" * 70)
    logger.info("BASE EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output:  {output_dir}")
    logger.info(f"Summary: {output_dir / 'summary' / 'base_eval_summary.json'}")
    logger.info(f"\nPer-draw results: {output_dir}/{{model}}/{{draw}}/{{band}}.json")
    logger.info(f"\nData pipeline:")
    logger.info(
        f"  - VAL split:  Used by Phase 1 (Pareto sweep) for threshold selection"
    )
    logger.info(
        f"  - TEST split: Used by Phase 2 (circuit discovery) for final evaluation"
    )
    logger.info(f"  - TRAIN split: Used by ACDC directly (not pre-computed here)")
    logger.info(f"\nTo use in Phase 1/2, load metrics with:")
    logger.info(f"  base_dir = Path('{output_dir}')")
    logger.info(
        f"  with open(base_dir / '{{model}}' / '{{draw}}' / '{{band}}.json') as f:"
    )
    logger.info(f"      metrics = json.load(f)['splits']['{{split}}']")

    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main() or 0)
