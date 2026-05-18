#!/usr/bin/env python3
"""
LSC Per-Example Circuit Evaluation
====================================
For each of the 60 circuits evaluated on all 5 test bands (300 conditions),
compute per-example metrics to characterize failure patterns.

This enables:
- Bimodal vs uniform failure analysis (dip test)
- Per-example robustness scores
- Failure pattern cross-correlation across conditions

Algorithm:
  For each circuit (model, band, draw):
    For each test_band (5 bands):
      1. Run circuit on test_band's FULL TEST data
      2. For each example: compute correct, top5_correct, correct_prob, top1_prediction
      3. Save per-example results to JSON

Output:
  per_example_eval/
    {model}/{band}/{draw}/{test_band}.json   # Per-example metrics (300 files)
    per_example_summary.csv                   # Distribution stats per condition

Usage:
    python lsc_per_example_eval.py
    python lsc_per_example_eval.py --models pythia-70m --bands low --draws draw_1
"""

import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import pickle
import argparse
import logging
import time
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch as t
import torch.nn.functional as F

# Import shared utilities
SCRIPT_DIR = Path(__file__).resolve().parent
ISC_ROOT = SCRIPT_DIR.parent

AUTOCIRCUIT_PATH = os.environ.get("AUTOCIRCUIT_PATH") or str(
    ISC_ROOT / "circuit_discovery" / "auto-circuit"
)
sys.path.insert(0, AUTOCIRCUIT_PATH)

from lsc_acdc_circuit import (
    load_model,
    load_pool,
    load_dataset,
    prepare_full_dataloader,
    run_circuit_and_collect,
    model_safe_name,
    get_batch_size,
    set_all_seeds,
    cleanup_gpu,
    safe_delete_model,
    ALL_BANDS,
    DEFAULT_MODELS,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "per_example_eval"
CIRCUITS_DIR = SCRIPT_DIR / "circuit_discovery" / "circuits"
DATA_DIR = ISC_ROOT / "LSC_data"
POOL_DIR = ISC_ROOT / "LSC_data" / "lsc_token_pools" / "matched"
EVAL_SEED = 123
VARIANT = "matched"
DRAWS = ["draw_1", "draw_2", "draw_3"]


def compute_per_example_metrics(
    logits: t.Tensor,
    answer_ids: List[int],
) -> List[dict]:
    """
    Compute per-example metrics from circuit logits.

    Args:
        logits: [n_samples, vocab_size] tensor
        answer_ids: list of correct token IDs

    Returns:
        List of dicts, one per example, with:
          - example_idx: int
          - correct: bool (top-1)
          - top5_correct: bool
          - correct_prob: float
          - top1_prediction: int
          - answer_id: int
    """
    if len(logits.shape) == 3:
        logits = logits[:, -1, :]

    probs = F.softmax(logits, dim=-1)
    n = min(len(answer_ids), logits.shape[0])

    results = []
    for i in range(n):
        topk = t.topk(logits[i], k=10).indices.tolist()
        correct_prob = probs[i, answer_ids[i]].item()

        results.append(
            {
                "example_idx": i,
                "correct": topk[0] == answer_ids[i],
                "top5_correct": answer_ids[i] in topk[:5],
                "top10_correct": answer_ids[i] in topk,
                "correct_prob": correct_prob,
                "top1_prediction": topk[0],
                "answer_id": answer_ids[i],
            }
        )

    return results


def run_per_example_eval(
    models: List[str],
    bands: List[str],
    draws: List[str],
    test_bands: List[str],
    output_dir: Path,
    device: str,
):
    """Run per-example evaluation for all specified circuits on all test bands."""
    from auto_circuit.utils.graph_utils import patchable_model

    logger = logging.getLogger("per_example_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_conditions = len(models) * len(bands) * len(draws) * len(test_bands)
    condition_idx = 0
    summary_rows = []

    for model_name in models:
        m_safe = model_safe_name(model_name)
        batch_size = get_batch_size(model_name)
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Loading model: {model_name}")
        logger.info(f"{'=' * 70}")

        model = load_model(model_name, device)
        bos_id = model.tokenizer.bos_token_id

        patchable = patchable_model(
            model=model,
            factorized=True,
            slice_output="last_seq",
            seq_len=None,
            separate_qkv=False,
            device=device,
        )

        # Pre-load all pools and test datasets
        pools = {}
        for b in test_bands:
            pools[b] = load_pool(b, POOL_DIR)

        for band in bands:
            for draw in draws:
                # Load circuit prune_scores
                scores_path = CIRCUITS_DIR / m_safe / band / draw / "prune_scores.pkl"
                if not scores_path.exists():
                    logger.warning(f"  prune_scores not found: {scores_path}")
                    continue

                with open(scores_path, "rb") as f:
                    prune_scores_cpu = pickle.load(f)

                n_edges = sum(
                    t.isinf(s).sum().item() for s in prune_scores_cpu.values()
                )
                prune_scores_dev = {
                    k: v.to(device) for k, v in prune_scores_cpu.items()
                }

                logger.info(f"\n  Circuit: {m_safe}/{band}/{draw} ({n_edges} edges)")

                for test_band in test_bands:
                    condition_idx += 1
                    result_dir = output_dir / m_safe / band / draw
                    result_file = result_dir / f"{test_band}.json"

                    # Check for existing result (resumability)
                    if result_file.exists():
                        try:
                            with open(result_file) as f:
                                existing = json.load(f)
                            if len(existing.get("examples", [])) > 0:
                                logger.info(
                                    f"  [{condition_idx}/{total_conditions}] "
                                    f"SKIP {band}->{test_band} (already done)"
                                )
                                # Add to summary from existing
                                n_ex = len(existing["examples"])
                                n_correct = sum(
                                    1 for e in existing["examples"] if e["correct"]
                                )
                                probs = [
                                    e["correct_prob"] for e in existing["examples"]
                                ]
                                summary_rows.append(
                                    {
                                        "model": model_name,
                                        "train_band": band,
                                        "draw": draw,
                                        "test_band": test_band,
                                        "n_examples": n_ex,
                                        "accuracy": n_correct / n_ex if n_ex else 0,
                                        "mean_correct_prob": np.mean(probs),
                                        "std_correct_prob": np.std(probs),
                                        "median_correct_prob": np.median(probs),
                                    }
                                )
                                continue
                        except (json.JSONDecodeError, KeyError):
                            pass

                    logger.info(
                        f"  [{condition_idx}/{total_conditions}] {band}->{test_band}"
                    )
                    t_start = time.time()

                    # Load test data
                    test_data = load_dataset(test_band, "test", DATA_DIR, VARIANT, draw)
                    pool = pools[test_band]

                    # Run circuit
                    if n_edges > 0:
                        logits, answer_ids = run_circuit_and_collect(
                            patchable,
                            prune_scores_dev,
                            n_edges,
                            test_data,
                            pool,
                            bos_id,
                            batch_size,
                            EVAL_SEED,
                            device,
                        )
                        per_example = compute_per_example_metrics(logits, answer_ids)
                    else:
                        per_example = []

                    elapsed = time.time() - t_start

                    # Save per-example results
                    result = {
                        "model": model_name,
                        "train_band": band,
                        "draw": draw,
                        "test_band": test_band,
                        "n_edges": n_edges,
                        "n_examples": len(per_example),
                        "elapsed_seconds": elapsed,
                        "completed_at": datetime.now().isoformat(),
                        "examples": per_example,
                    }

                    result_dir.mkdir(parents=True, exist_ok=True)
                    with open(result_file, "w") as f:
                        json.dump(result, f, indent=2)

                    # Summary stats
                    if per_example:
                        n_correct = sum(1 for e in per_example if e["correct"])
                        probs = [e["correct_prob"] for e in per_example]
                        accuracy = n_correct / len(per_example)
                        summary_rows.append(
                            {
                                "model": model_name,
                                "train_band": band,
                                "draw": draw,
                                "test_band": test_band,
                                "n_examples": len(per_example),
                                "accuracy": accuracy,
                                "mean_correct_prob": np.mean(probs),
                                "std_correct_prob": np.std(probs),
                                "median_correct_prob": np.median(probs),
                            }
                        )
                        logger.info(
                            f"    acc={accuracy:.4f}, "
                            f"mean_p={np.mean(probs):.4f}, "
                            f"time={elapsed:.1f}s"
                        )

                # Clean up circuit scores
                del prune_scores_dev, prune_scores_cpu
                cleanup_gpu()

        # Free model
        del patchable
        safe_delete_model(model)
        logger.info(f"\nModel {model_name} done, GPU memory freed")

    # Save summary
    if summary_rows:
        import pandas as pd

        df_summary = pd.DataFrame(summary_rows)
        summary_path = output_dir / "per_example_summary.csv"
        df_summary.to_csv(summary_path, index=False)
        logger.info(f"\nSummary saved: {summary_path} ({len(df_summary)} rows)")

    logger.info(f"\nTotal conditions evaluated: {condition_idx}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-example circuit evaluation for failure analysis",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, help=f"Models (default: {DEFAULT_MODELS})"
    )
    parser.add_argument(
        "--bands",
        nargs="+",
        default=None,
        help=f"Training bands (default: {ALL_BANDS})",
    )
    parser.add_argument(
        "--draws", nargs="+", default=None, help=f"Draws (default: {DRAWS})"
    )
    parser.add_argument(
        "--test-bands",
        nargs="+",
        default=None,
        help=f"Test bands (default: same as --bands)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (default: 0)")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    models = args.models or DEFAULT_MODELS
    bands = args.bands or ALL_BANDS
    draws = args.draws or DRAWS
    test_bands = args.test_bands or ALL_BANDS
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

    # Setup logging
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"per_example_eval_{timestamp}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("per_example_eval")

    device = f"cuda:{args.gpu}" if t.cuda.is_available() else "cpu"

    logger.info("=" * 70)
    logger.info("LSC PER-EXAMPLE CIRCUIT EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Models:      {models}")
    logger.info(f"Train bands: {bands}")
    logger.info(f"Test bands:  {test_bands}")
    logger.info(f"Draws:       {draws}")
    logger.info(f"Device:      {device}")
    logger.info(f"Output:      {output_dir}")
    n_conditions = len(models) * len(bands) * len(draws) * len(test_bands)
    logger.info(f"Conditions:  {n_conditions}")
    logger.info("=" * 70)

    run_per_example_eval(
        models=models,
        bands=bands,
        draws=draws,
        test_bands=test_bands,
        output_dir=output_dir,
        device=device,
    )

    logger.info("\nPer-example evaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
