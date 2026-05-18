#!/usr/bin/env python3
"""
lsc_threshold_robustness_eval.py
=================================
D2: τ Threshold Robustness Evaluation

For each model, load control-band circuits at three thresholds (τ_low, τ*, τ_high)
from the pareto sweep, evaluate each on all 5 test bands, and save results.

τ_low/τ_high = immediate neighbors of τ* in the 11-point log-uniform sweep.
All circuits are control-band circuits (extracted during pareto sweep).

Outputs: CSV with 5 models x 3 τ variants x 5 test bands = 75 rows.

Usage:
    python lsc_threshold_robustness_eval.py
    python lsc_threshold_robustness_eval.py --models pythia-70m pythia-160m
    python lsc_threshold_robustness_eval.py --output /path/to/results.csv
"""

import os as _os
from pathlib import Path as _Path


def _find_project_root() -> _Path:
    env = _os.environ.get("PROJECT_ROOT")
    if env:
        return _Path(env).resolve()
    for p in _Path(__file__).resolve().parents:
        if (p / "src" / "config.py").exists():
            return p
    return _Path(__file__).resolve().parents[1]


PROJECT_ROOT = _find_project_root()

import os
import sys
import gc
import json
import math
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch as t


SCRIPT_DIR = PROJECT_ROOT / "LSC_circuits"
ISC_ROOT = PROJECT_ROOT

# AutoCircuit must be on path (same as lsc_acdc_circuit.py)
AUTOCIRCUIT_PATH = os.environ.get("AUTOCIRCUIT_PATH") or str(
    ISC_ROOT / "circuit_discovery" / "auto-circuit"
)
sys.path.insert(0, AUTOCIRCUIT_PATH)
sys.path.insert(0, str(SCRIPT_DIR))

# Import evaluation infrastructure from existing scripts
from lsc_acdc_circuit import (
    evaluate_on_band,
    DiscoveryConfig,
    load_model,
    ALL_BANDS,
    DEFAULT_MODELS,
)
from auto_circuit.utils.graph_utils import patchable_model


# 11 log-uniform thresholds from lsc_pareto_sweep.py (sorted ascending)
DEFAULT_THRESHOLDS_SORTED = sorted(
    [
        1e-2,
        3.98e-3,
        1.58e-3,
        6.31e-4,
        2.51e-4,
        1e-4,
        3.98e-5,
        1.58e-5,
        6.31e-6,
        2.51e-6,
        1e-6,
    ]
)

SWEEP_DIR = SCRIPT_DIR / "pareto_sweep"
THRESHOLD_SUMMARY = SWEEP_DIR / "sweep_results" / "threshold_summary.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("threshold_robustness")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tau_to_filename(tau: float) -> str:
    """Convert tau float to pkl filename stem matching lsc_pareto_sweep naming.

    Examples:
        1.58e-3  -> tau_1_58em03
        6.31e-4  -> tau_6_31em04
        1.00e-4  -> tau_1_00em04
    """
    exp = math.floor(math.log10(tau))
    mantissa = tau / (10**exp)
    mantissa_str = f"{mantissa:.2f}".replace(".", "_")
    return f"tau_{mantissa_str}em{abs(exp):02d}"


def get_tau_neighbors(tau_star: float) -> dict:
    """Return {low, star, high} τ values as immediate neighbors in DEFAULT_THRESHOLDS_SORTED.

    τ_low  = smaller τ -> less pruning -> more edges (larger circuit)
    τ_high = larger τ  -> more pruning -> fewer edges (smaller circuit)
    """
    idx = min(
        range(len(DEFAULT_THRESHOLDS_SORTED)),
        key=lambda i: abs(DEFAULT_THRESHOLDS_SORTED[i] - tau_star),
    )
    return {
        "low": DEFAULT_THRESHOLDS_SORTED[max(0, idx - 1)],
        "star": DEFAULT_THRESHOLDS_SORTED[idx],
        "high": DEFAULT_THRESHOLDS_SORTED[
            min(len(DEFAULT_THRESHOLDS_SORTED) - 1, idx + 1)
        ],
    }


def model_dir_name(model_name: str) -> str:
    """Convert model name to pareto sweep directory name (hyphens -> underscores)."""
    return model_name.replace("-", "_")


def get_prune_scores_path(model_name: str, tau: float) -> Path:
    """Return path to the pareto sweep prune scores pkl for given model and tau."""
    fname = tau_to_filename(tau) + ".pkl"
    return (
        SWEEP_DIR
        / "sweep_results"
        / model_dir_name(model_name)
        / "prune_scores"
        / fname
    )


def count_edges(prune_scores: dict) -> tuple:
    """Return (n_edges, total_edges) from prune_scores dict (inf = kept edge)."""
    n_edges = sum(t.isinf(s).sum().item() for s in prune_scores.values())
    total_edges = sum(s.numel() for s in prune_scores.values())
    return n_edges, total_edges


def cleanup_gpu():
    """Free GPU memory."""
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()
        t.cuda.synchronize()


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def run_evaluation(models: list, output_path: Path, device: str):
    """Evaluate control-band circuits at τ_low/τ*/τ_high for each model x test band."""

    with open(THRESHOLD_SUMMARY) as f:
        threshold_summary = json.load(f)["selections"]

    results = []
    total_models = len(models)

    for model_idx, model_name in enumerate(models):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Model {model_idx + 1}/{total_models}: {model_name}")
        logger.info(f"{'=' * 60}")

        tau_star = threshold_summary[model_name]["threshold"]
        taus = get_tau_neighbors(tau_star)
        logger.info(
            f"τ variants: low={taus['low']:.2e}  star={taus['star']:.2e}  high={taus['high']:.2e}"
        )

        # Verify all pkl files exist before loading model
        missing = []
        for variant, tau_val in taus.items():
            pkl_path = get_prune_scores_path(model_name, tau_val)
            if not pkl_path.exists():
                missing.append(f"{variant} ({tau_val:.2e}) -> {pkl_path}")
        if missing:
            logger.error(
                f"Missing pkl files for {model_name}:\n  " + "\n  ".join(missing)
            )
            continue

        # Load model (once per model)
        logger.info(f"Loading {model_name} ...")
        model = load_model(model_name, device)

        # Build patchable model
        logger.info("Building patchable model ...")
        patchable = patchable_model(
            model=model,
            factorized=True,
            slice_output="last_seq",
            seq_len=None,
            separate_qkv=False,
            device=device,
        )

        # Default config; paths match lsc_acdc_circuit.py defaults
        config = DiscoveryConfig(
            variant="matched",
            eval_seed=123,
            use_bf16_eval=False,
        )

        for tau_variant, tau_val in taus.items():
            pkl_path = get_prune_scores_path(model_name, tau_val)
            logger.info(
                f"\nτ_{tau_variant} = {tau_val:.2e}  ({tau_to_filename(tau_val)}.pkl)"
            )

            with open(pkl_path, "rb") as f:
                prune_scores_cpu = pickle.load(f)

            n_edges, total_edges = count_edges(prune_scores_cpu)
            size_fraction = n_edges / total_edges if total_edges else 0.0
            logger.info(
                f"  Circuit: {n_edges}/{total_edges} edges ({size_fraction:.1%})"
            )

            prune_scores_dev = {k: v.to(device) for k, v in prune_scores_cpu.items()}
            del prune_scores_cpu

            for test_band in ALL_BANDS:
                logger.info(f"  Evaluating on band: {test_band} ...")
                try:
                    result = evaluate_on_band(
                        model=model,
                        patchable=patchable,
                        prune_scores_dev=prune_scores_dev,
                        n_edges=n_edges,
                        band=test_band,
                        model_name=model_name,
                        config=config,
                        device=device,
                        draw="draw_1",
                    )
                    if "error" in result:
                        logger.warning(f"    Error: {result['error']}")
                        circuit_acc = float("nan")
                        circuit_kl = float("nan")
                        base_acc = float("nan")
                    else:
                        circuit_acc = result["circuit"]["accuracy"]
                        circuit_kl = result["circuit"].get("kl_div", float("nan"))
                        base_acc = result["base"]["accuracy"]
                        logger.info(
                            f"    circuit_acc={circuit_acc:.4f}  kl={circuit_kl:.4f}  base={base_acc:.4f}"
                        )

                except Exception as e:
                    logger.error(f"    FAILED: {e}")
                    circuit_acc = float("nan")
                    circuit_kl = float("nan")
                    base_acc = float("nan")

                results.append(
                    {
                        "model": model_name,
                        "tau_variant": tau_variant,
                        "tau_value": tau_val,
                        "n_edges": n_edges,
                        "total_edges": total_edges,
                        "size_fraction": size_fraction,
                        "test_band": test_band,
                        "circuit_accuracy": circuit_acc,
                        "circuit_kl_div": circuit_kl,
                        "base_accuracy": base_acc,
                    }
                )

            del prune_scores_dev
            cleanup_gpu()

        # Free model memory before next model
        del model, patchable
        cleanup_gpu()
        logger.info(f"Freed GPU memory after {model_name}")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved {len(df)} rows to {output_path}")
    logger.info(f"\nPreview:\n{df.to_string()}")
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="D2: τ threshold robustness evaluation"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Models to evaluate (default: all 5)",
    )
    parser.add_argument(
        "--output",
        default="LSC_circuit_analysis/05_Phase_Targeted/outputs/threshold_robustness/eval_results.csv",
        help="Output CSV path",
    )
    parser.add_argument("--device", default="cuda:0", help="GPU device")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_path = Path(args.output)

    logger.info(f"D2 Threshold Robustness Evaluation")
    logger.info(f"Models:  {args.models}")
    logger.info(f"Output:  {output_path}")
    logger.info(f"Device:  {args.device}")
    logger.info(f"Started: {datetime.now().isoformat()}")

    df = run_evaluation(args.models, output_path, args.device)

    logger.info(f"\nDone: {datetime.now().isoformat()}")
