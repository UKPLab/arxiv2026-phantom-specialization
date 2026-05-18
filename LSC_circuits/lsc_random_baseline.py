#!/usr/bin/env python3
"""
LSC Random Baseline Evaluation
================================
For each of the 60 ACDC-discovered circuits, compare its accuracy against
K random edge sets of the same size. This validates that ACDC found
meaningful edges; not just any set of edges performs well.

Algorithm:
  For each circuit (model, band, draw):
    1. Load prune_scores.pkl -> count n_edges (inf entries)
    2. Evaluate real circuit on same-band test data -> real_accuracy
    3. For k in 0..K-1:
       a. Generate random prune_scores with exactly n_edges random edges
       b. Evaluate random circuit -> random_accuracy[k]
    4. Compute z_score = (real - mean_random) / std_random
    5. Compute percentile_rank

Output:
  random_baseline/
    random_baseline_results.json          # All circuits aggregated
    {model}/{band}/{draw}/result.json     # Per-circuit (for resumability)

Usage:
    python lsc_random_baseline.py
    python lsc_random_baseline.py --models pythia-70m --bands low --draws draw_1
    python lsc_random_baseline.py --K 50  # fewer random trials (faster)
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

# Import shared utilities from the main circuit script
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
    compute_accuracy_metrics,
    model_safe_name,
    get_batch_size,
    set_all_seeds,
    cleanup_gpu,
    safe_delete_model,
    ALL_BANDS,
    DEFAULT_MODELS,
    SEQ_LEN_WITH_BOS,
    DIVERGE_IDX,
)

DEFAULT_K = 100
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "random_baseline"
CIRCUITS_DIR = SCRIPT_DIR / "circuit_discovery" / "circuits"
DATA_DIR = ISC_ROOT / "LSC_data"
POOL_DIR = ISC_ROOT / "LSC_data" / "lsc_token_pools" / "matched"
EVAL_SEED = 123  # Same as lsc_acdc_circuit.py for consistency
VARIANT = "matched"
DRAWS = ["draw_1", "draw_2", "draw_3"]


def generate_random_prune_scores(
    prune_scores: Dict[str, t.Tensor],
    n_edges: int,
    seed: int,
) -> Dict[str, t.Tensor]:
    """
    Generate random prune_scores with exactly n_edges edges selected.

    Strategy:
    1. Flatten all tensor positions into a single index space
    2. Randomly sample exactly n_edges positions
    3. Build new prune_scores: sampled positions get inf, rest get 0.0

    Args:
        prune_scores: Original prune_scores dict (used for shape/structure only)
        n_edges: Exact number of edges to include in random circuit
        seed: Random seed for reproducibility

    Returns:
        New prune_scores dict with exactly n_edges random inf entries
    """
    rng = np.random.RandomState(seed)

    # Build index map: list of (module_name, flat_position)
    index_map = []
    for name, scores in prune_scores.items():
        n_positions = scores.numel()
        for pos in range(n_positions):
            index_map.append((name, pos))

    total_positions = len(index_map)
    if n_edges > total_positions:
        n_edges = total_positions

    # Sample random positions
    selected_indices = set(
        rng.choice(total_positions, size=n_edges, replace=False).tolist()
    )

    # Build position sets per module for efficient construction
    module_positions = {}
    for flat_idx in selected_indices:
        name, pos = index_map[flat_idx]
        if name not in module_positions:
            module_positions[name] = set()
        module_positions[name].add(pos)

    # Construct new prune_scores
    random_scores = {}
    for name, scores in prune_scores.items():
        new_scores = t.zeros_like(scores)
        if name in module_positions:
            flat = new_scores.view(-1)
            for pos in module_positions[name]:
                flat[pos] = float("inf")
            new_scores = flat.view(scores.shape)
        random_scores[name] = new_scores

    return random_scores


def evaluate_circuit_accuracy(
    patchable,
    prune_scores_dev: Dict[str, t.Tensor],
    n_edges: int,
    dataset: dict,
    pool: dict,
    bos_id: int,
    batch_size: int,
    device: str,
) -> float:
    """Evaluate a circuit and return top-1 accuracy."""
    if n_edges == 0:
        return 0.0

    logits, answer_ids = run_circuit_and_collect(
        patchable,
        prune_scores_dev,
        n_edges,
        dataset,
        pool,
        bos_id,
        batch_size,
        EVAL_SEED,
        device,
    )
    metrics = compute_accuracy_metrics(logits, answer_ids)
    return metrics["accuracy"]


def run_random_baseline(
    models: List[str],
    bands: List[str],
    draws: List[str],
    K: int,
    output_dir: Path,
    device: str,
):
    """Run random baseline evaluation for all specified circuits."""
    from auto_circuit.utils.graph_utils import patchable_model

    logger = logging.getLogger("random_baseline")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    total_circuits = len(models) * len(bands) * len(draws)
    circuit_idx = 0

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

        for band in bands:
            pool = load_pool(band, POOL_DIR)
            test_data = load_dataset(band, "test", DATA_DIR, VARIANT, draws[0])
            # All draws share the same test data structure, but we reload per draw
            # to be safe (draws affect train split, not test)

            for draw in draws:
                circuit_idx += 1
                task_id = f"{m_safe}/{band}/{draw}"
                result_dir = output_dir / m_safe / band / draw
                result_file = result_dir / "result.json"

                # Check for existing result (resumability)
                if result_file.exists():
                    try:
                        with open(result_file) as f:
                            existing = json.load(f)
                        if existing.get("K", 0) >= K:
                            logger.info(
                                f"[{circuit_idx}/{total_circuits}] SKIP {task_id} (already done)"
                            )
                            all_results.append(existing)
                            continue
                    except (json.JSONDecodeError, KeyError):
                        pass

                logger.info(f"\n[{circuit_idx}/{total_circuits}] {task_id}")
                t_start = time.time()

                # Load real prune_scores
                scores_path = CIRCUITS_DIR / m_safe / band / draw / "prune_scores.pkl"
                if not scores_path.exists():
                    logger.warning(f"  prune_scores not found: {scores_path}")
                    continue

                with open(scores_path, "rb") as f:
                    prune_scores_cpu = pickle.load(f)

                n_edges = sum(
                    t.isinf(s).sum().item() for s in prune_scores_cpu.values()
                )
                total_possible = sum(s.numel() for s in prune_scores_cpu.values())
                logger.info(
                    f"  Circuit: {n_edges}/{total_possible} edges ({n_edges / total_possible:.1%})"
                )

                # Reload test data for this draw
                test_data = load_dataset(band, "test", DATA_DIR, VARIANT, draw)

                # Evaluate real circuit
                prune_scores_dev = {
                    k: v.to(device) for k, v in prune_scores_cpu.items()
                }
                real_acc = evaluate_circuit_accuracy(
                    patchable,
                    prune_scores_dev,
                    n_edges,
                    test_data,
                    pool,
                    bos_id,
                    batch_size,
                    device,
                )
                logger.info(f"  Real accuracy: {real_acc:.4f}")

                # Evaluate K random circuits
                random_accs = []
                for k in range(K):
                    random_seed = k * 1000 + hash(task_id) % 10000
                    random_scores_cpu = generate_random_prune_scores(
                        prune_scores_cpu,
                        n_edges,
                        seed=random_seed,
                    )
                    random_scores_dev = {
                        name: s.to(device) for name, s in random_scores_cpu.items()
                    }
                    rand_acc = evaluate_circuit_accuracy(
                        patchable,
                        random_scores_dev,
                        n_edges,
                        test_data,
                        pool,
                        bos_id,
                        batch_size,
                        device,
                    )
                    random_accs.append(rand_acc)

                    # Clean up random scores from GPU
                    del random_scores_dev, random_scores_cpu

                    if (k + 1) % 10 == 0:
                        logger.info(
                            f"  Random trial {k + 1}/{K}: acc={rand_acc:.4f} "
                            f"(running mean={np.mean(random_accs):.4f})"
                        )

                # Clean up real scores
                del prune_scores_dev

                # Compute statistics
                mean_random = float(np.mean(random_accs))
                std_random = float(np.std(random_accs))
                z_score = (
                    (real_acc - mean_random) / std_random
                    if std_random > 0
                    else float("inf")
                )
                percentile_rank = float(
                    np.mean([1 if real_acc > ra else 0 for ra in random_accs]) * 100
                )

                elapsed = time.time() - t_start

                result = {
                    "model": model_name,
                    "band": band,
                    "draw": draw,
                    "n_edges": n_edges,
                    "total_possible_edges": total_possible,
                    "K": K,
                    "real_accuracy": real_acc,
                    "random_accuracies": random_accs,
                    "mean_random_accuracy": mean_random,
                    "std_random_accuracy": std_random,
                    "z_score": z_score,
                    "percentile_rank": percentile_rank,
                    "elapsed_seconds": elapsed,
                    "completed_at": datetime.now().isoformat(),
                }

                # Save per-circuit result
                result_dir.mkdir(parents=True, exist_ok=True)
                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2)

                all_results.append(result)
                logger.info(
                    f"  z-score={z_score:.2f}, percentile={percentile_rank:.1f}%, "
                    f"time={elapsed:.1f}s"
                )

                cleanup_gpu()

        # Free model between models
        del patchable
        safe_delete_model(model)
        logger.info(f"\nModel {model_name} done, GPU memory freed")

    # Save aggregated results
    agg_path = output_dir / "random_baseline_results.json"
    # Strip per-example random_accuracies for the aggregate file to keep it small
    agg_results = []
    for r in all_results:
        agg = {k: v for k, v in r.items() if k != "random_accuracies"}
        agg_results.append(agg)

    with open(agg_path, "w") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "K": K,
                "n_circuits": len(agg_results),
                "results": agg_results,
            },
            f,
            indent=2,
        )
    logger.info(f"\nAggregated results: {agg_path}")
    logger.info(f"Total circuits evaluated: {len(all_results)}")

    # Print summary table
    logger.info(f"\n{'=' * 80}")
    logger.info("RANDOM BASELINE SUMMARY")
    logger.info(f"{'=' * 80}")
    logger.info(
        f"{'Model':<15s} {'Band':<12s} {'Draw':<8s} {'Edges':>6s} "
        f"{'Real':>7s} {'Random':>7s} {'Z-score':>8s} {'Pctile':>7s}"
    )
    logger.info("-" * 80)
    for r in all_results:
        logger.info(
            f"{r['model']:<15s} {r['band']:<12s} {r['draw']:<8s} "
            f"{r['n_edges']:>6d} {r['real_accuracy']:>7.4f} "
            f"{r['mean_random_accuracy']:>7.4f} {r['z_score']:>8.2f} "
            f"{r['percentile_rank']:>6.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Random baseline evaluation for ACDC circuits",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, help=f"Models (default: {DEFAULT_MODELS})"
    )
    parser.add_argument(
        "--bands", nargs="+", default=None, help=f"Bands (default: {ALL_BANDS})"
    )
    parser.add_argument(
        "--draws", nargs="+", default=None, help=f"Draws (default: {DRAWS})"
    )
    parser.add_argument(
        "--K",
        type=int,
        default=DEFAULT_K,
        help=f"Number of random trials per circuit (default: {DEFAULT_K})",
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
            logging.FileHandler(log_dir / f"random_baseline_{timestamp}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("random_baseline")

    device = f"cuda:{args.gpu}" if t.cuda.is_available() else "cpu"

    logger.info("=" * 70)
    logger.info("LSC RANDOM BASELINE EVALUATION")
    logger.info("=" * 70)
    logger.info(f"Models:     {models}")
    logger.info(f"Bands:      {bands}")
    logger.info(f"Draws:      {draws}")
    logger.info(f"K:          {args.K}")
    logger.info(f"Device:     {device}")
    logger.info(f"Output:     {output_dir}")
    logger.info(f"Circuits:   {CIRCUITS_DIR}")
    n_circuits = len(models) * len(bands) * len(draws)
    n_forward = n_circuits * (1 + args.K)
    logger.info(f"Circuits:   {n_circuits}")
    logger.info(f"Forward passes: {n_forward}")
    logger.info("=" * 70)

    run_random_baseline(
        models=models,
        bands=bands,
        draws=draws,
        K=args.K,
        output_dir=output_dir,
        device=device,
    )

    logger.info("\nRandom baseline evaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
