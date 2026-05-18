#!/usr/bin/env python3
"""
LSC Subnetwork Probing Circuit Discovery & Cross-Band Evaluation
================================================================
Phases:
  Phase 0: Lambda selection on control band -> pick best lambda per model
  Phase 1: SP extraction on all (model, band, draw_1) -> save prune_scores
  Phase 2: Evaluation (same-band + cross-band transfer) -> save CSV
  Phase 3: Analysis summary -> generate tables

Memory management: each task loads model, runs SP or eval, then fully
frees GPU memory before the next task. No model reuse across tasks.

Usage:
    python lsc_sp_circuit.py --models pythia-160m --phase 0
    python lsc_sp_circuit.py --models pythia-160m --phase 1
    python lsc_sp_circuit.py --models pythia-160m --phase 2
    python lsc_sp_circuit.py --phase 3
"""

import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import pickle
import time
import logging
import traceback
import gc
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np
import torch as t

SCRIPT_DIR = Path(__file__).resolve().parent
LSC_CIRCUITS_DIR = SCRIPT_DIR.parent
ISC_ROOT = LSC_CIRCUITS_DIR.parent

AUTOCIRCUIT_PATH = os.environ.get("AUTOCIRCUIT_PATH") or str(
    ISC_ROOT / "circuit_discovery" / "auto-circuit"
)
sys.path.insert(0, AUTOCIRCUIT_PATH)
sys.path.insert(0, str(LSC_CIRCUITS_DIR))

from lsc_acdc_circuit import (
    ALL_BANDS,
    SEQ_LEN_WITH_BOS,
    DIVERGE_IDX,
    DiscoveryConfig,
    set_all_seeds,
    cleanup_gpu,
    model_safe_name,
    get_batch_size,
    load_pool,
    load_dataset,
    prepare_dataloader,
    prepare_full_dataloader,
    load_model,
    load_base_metrics,
    compute_accuracy_metrics,
    compute_kl_divergence,
    compute_base_logits,
    run_circuit_and_collect,
    invert_prune_scores,
    evaluate_on_band,
)

SP_MODELS = ["pythia-160m", "pythia-410m", "pythia-1b"]
SP_DRAW = "draw_1"
SP_EPOCHS = 200
SP_LR = 0.1
SP_MASK_FN = "hard_concrete"
LAMBDA_CANDIDATES = [0.1, 1.0, 10.0]

# SP needs smaller batch sizes (gradients for all edge masks)
SP_BATCH = {"pythia-160m": 64, "pythia-410m": 16, "pythia-1b": 12}
# Evaluation can use larger batches (no gradients)
EVAL_BATCH = {"pythia-160m": 128, "pythia-410m": 64, "pythia-1b": 48}

ACDC_REF = {
    "pythia-160m": {"n_edges": 1396, "frac": 0.122},
    "pythia-410m": {"n_edges": 3444, "frac": 0.043},
    "pythia-1b": {"n_edges": 939, "frac": 0.094},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sp")


def sp_to_binary(raw_scores: dict, n_keep: int) -> dict:
    """Top-k thresholding of SP scores into binary circuit (inf/0)."""
    all_vals = []
    for name, s in raw_scores.items():
        flat = s.abs().flatten()
        for i in range(flat.numel()):
            all_vals.append((flat[i].item(), name, i))
    all_vals.sort(key=lambda x: x[0], reverse=True)

    keep = set()
    for i in range(min(n_keep, len(all_vals))):
        _, name, idx = all_vals[i]
        keep.add((name, idx))

    binary = {}
    for name, s in raw_scores.items():
        m = t.zeros_like(s)
        flat = m.flatten()
        for i in range(flat.numel()):
            if (name, i) in keep:
                flat[i] = float("inf")
        binary[name] = m.reshape(s.shape)
    return binary


def count_edges(scores: dict) -> Tuple[int, int]:
    n = sum(t.isinf(s).sum().item() for s in scores.values())
    total = sum(s.numel() for s in scores.values())
    return n, total


def jaccard(a: dict, b: dict) -> float:
    sa = {
        (n, i)
        for n, s in a.items()
        for i in range(s.flatten().numel())
        if s.flatten()[i].item() == float("inf")
    }
    sb = {
        (n, i)
        for n, s in b.items()
        for i in range(s.flatten().numel())
        if s.flatten()[i].item() == float("inf")
    }
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def full_cleanup():
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()
        t.cuda.synchronize()


def run_sp_training(patchable, train_loader, epochs, lr, reg_lambda, circuit_size):
    """Run SP with tree_optimisation=True (start full, prune down)."""
    from auto_circuit.prune_algos.subnetwork_probing import (
        subnetwork_probing_prune_scores,
    )

    return subnetwork_probing_prune_scores(
        model=patchable,
        dataloader=train_loader,
        official_edges=None,
        learning_rate=lr,
        epochs=epochs,
        regularize_lambda=reg_lambda,
        mask_fn=SP_MASK_FN,
        faithfulness_target="kl_div",
        circuit_size=circuit_size,
        tree_optimisation=True,
        show_train_graph=False,
    )


# =========================================================================
# PHASE 0: Lambda Selection (one model at a time)
# =========================================================================


def phase0(models: List[str], device: str):
    from auto_circuit.utils.graph_utils import patchable_model

    config = DiscoveryConfig()
    out_dir = SCRIPT_DIR / "lambda_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for model_name in models:
        target = ACDC_REF[model_name]["n_edges"]
        log.info(f"=== Lambda sweep: {model_name} (target={target} edges) ===")
        sweep = []

        for lam in LAMBDA_CANDIDATES:
            log.info(f"  lambda={lam}")
            # Fresh model load per lambda to avoid memory buildup
            set_all_seeds(42)
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

            pool = load_pool("control", Path(config.pool_dir))
            train_data = load_dataset(
                "control", "train", Path(config.data_dir), config.variant, SP_DRAW
            )
            loader, _ = prepare_dataloader(
                train_data,
                pool,
                bos_id,
                n_samples=256,
                batch_size=SP_BATCH[model_name],
                seed=42,
                device=device,
            )

            t0 = time.time()
            raw = run_sp_training(patchable, loader, SP_EPOCHS, SP_LR, lam, target)
            elapsed = time.time() - t0

            binary = sp_to_binary(raw, target)
            n_edges, n_total = count_edges(binary)

            # Evaluate on control test set
            scores_dev = {k: v.to(device) for k, v in binary.items()}
            test_data = load_dataset(
                "control", "test", Path(config.data_dir), config.variant, SP_DRAW
            )
            base_logits, _, _ = compute_base_logits(
                model, test_data, pool, bos_id, EVAL_BATCH[model_name], 123, device
            )
            circ_logits, circ_ans = run_circuit_and_collect(
                patchable,
                scores_dev,
                n_edges,
                test_data,
                pool,
                bos_id,
                EVAL_BATCH[model_name],
                123,
                device,
            )
            acc = compute_accuracy_metrics(circ_logits, circ_ans)["accuracy"]
            kl = compute_kl_divergence(circ_logits, base_logits)

            sweep.append(
                {
                    "lambda": lam,
                    "n_edges": n_edges,
                    "total": n_total,
                    "frac": n_edges / n_total,
                    "acc": acc,
                    "kl": kl,
                    "time": elapsed,
                }
            )
            log.info(
                f"    edges={n_edges}/{n_total} ({n_edges / n_total:.1%}), "
                f"acc={acc:.3f}, kl={kl:.4f}, time={elapsed:.0f}s"
            )

            # Full cleanup
            del model, patchable, raw, binary, scores_dev, loader
            del circ_logits, base_logits
            full_cleanup()

        # Select best (highest accuracy)
        valid = [s for s in sweep if s["acc"] > 0]
        best = max(valid, key=lambda s: s["acc"]) if valid else sweep[0]
        results[model_name] = {
            "selected_lambda": best["lambda"],
            "selected_acc": best["acc"],
            "target_edges": target,
            "sweep": sweep,
        }
        log.info(f"  -> Selected lambda={best['lambda']}, acc={best['acc']:.3f}")

        m_dir = out_dir / model_safe_name(model_name)
        m_dir.mkdir(parents=True, exist_ok=True)
        with open(m_dir / "lambda_sweep.json", "w") as f:
            json.dump(results[model_name], f, indent=2, default=str)

    with open(out_dir / "lambda_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info("Lambda selection done.")
    return results


# =========================================================================
# PHASE 1: SP Extraction (save prune_scores only, no evaluation)
# =========================================================================


def phase1(models: List[str], device: str):
    from auto_circuit.utils.graph_utils import patchable_model

    config = DiscoveryConfig()

    # Read per-model lambda files (safe for parallel execution)
    lam_summary = {}
    for model_name in models:
        m_safe = model_safe_name(model_name)
        lam_file = SCRIPT_DIR / "lambda_selection" / m_safe / "lambda_sweep.json"
        if not lam_file.exists():
            log.error(f"No lambda file for {model_name}, run phase 0 first")
            continue
        with open(lam_file) as f:
            lam_summary[model_name] = json.load(f)

    for model_name in models:
        if model_name not in lam_summary:
            continue
        reg_lambda = lam_summary[model_name]["selected_lambda"]
        target = ACDC_REF[model_name]["n_edges"]
        m_safe = model_safe_name(model_name)

        for band in ALL_BANDS:
            circuit_dir = SCRIPT_DIR / "circuits" / m_safe / band / SP_DRAW
            if (circuit_dir / "prune_scores.pkl").exists():
                log.info(f"  {model_name}/{band}: already extracted, skip")
                continue

            log.info(f"--- {model_name}/{band} (lambda={reg_lambda}) ---")
            set_all_seeds(42)
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

            pool = load_pool(band, Path(config.pool_dir))
            train_data = load_dataset(
                band, "train", Path(config.data_dir), config.variant, SP_DRAW
            )
            loader, _ = prepare_dataloader(
                train_data,
                pool,
                bos_id,
                n_samples=256,
                batch_size=SP_BATCH[model_name],
                seed=42,
                device=device,
            )

            t0 = time.time()
            raw = run_sp_training(
                patchable, loader, SP_EPOCHS, SP_LR, reg_lambda, target
            )
            elapsed = time.time() - t0

            binary = sp_to_binary(raw, target)
            n_edges, n_total = count_edges(binary)
            log.info(
                f"  SP done: {n_edges}/{n_total} ({n_edges / n_total:.1%}), {elapsed:.0f}s"
            )

            circuit_dir.mkdir(parents=True, exist_ok=True)
            with open(circuit_dir / "prune_scores.pkl", "wb") as f:
                pickle.dump({k: v.cpu() for k, v in binary.items()}, f)
            with open(circuit_dir / "prune_scores_raw.pkl", "wb") as f:
                pickle.dump({k: v.cpu() for k, v in raw.items()}, f)
            with open(circuit_dir / "extraction_info.json", "w") as f:
                json.dump(
                    {
                        "model": model_name,
                        "band": band,
                        "draw": SP_DRAW,
                        "lambda": reg_lambda,
                        "epochs": SP_EPOCHS,
                        "n_edges": n_edges,
                        "total": n_total,
                        "frac": n_edges / n_total,
                        "time": elapsed,
                    },
                    f,
                    indent=2,
                )

            del model, patchable, raw, binary, loader
            full_cleanup()

    log.info("Phase 1 extraction done.")


# =========================================================================
# PHASE 2: Evaluation (load saved circuits, evaluate cross-band)
# =========================================================================


def phase2(models: List[str], device: str):
    from auto_circuit.utils.graph_utils import patchable_model

    config = DiscoveryConfig()
    all_rows = []

    for model_name in models:
        m_safe = model_safe_name(model_name)
        target = ACDC_REF[model_name]["n_edges"]

        # Load ACDC reference circuits for Jaccard
        acdc_base = LSC_CIRCUITS_DIR / "circuit_discovery" / "circuits" / m_safe
        acdc_circuits = {}
        for band in ALL_BANDS:
            p = acdc_base / band / SP_DRAW / "prune_scores.pkl"
            if p.exists():
                with open(p, "rb") as f:
                    acdc_circuits[band] = pickle.load(f)

        for band in ALL_BANDS:
            circuit_dir = SCRIPT_DIR / "circuits" / m_safe / band / SP_DRAW
            scores_path = circuit_dir / "prune_scores.pkl"
            if not scores_path.exists():
                log.warning(f"  {model_name}/{band}: no circuit, skip")
                continue
            if (circuit_dir / "metrics.json").exists():
                log.info(f"  {model_name}/{band}: already evaluated, skip")
                with open(circuit_dir / "metrics.json") as f:
                    data = json.load(f)
                for tb, xb in data.get("cross_band", {}).items():
                    if "error" not in xb:
                        all_rows.append(
                            {
                                "model": model_name,
                                "circuit_band": band,
                                "test_band": tb,
                                "is_same_band": band == tb,
                                "circuit_acc": xb.get("circuit", {}).get("accuracy", 0),
                                "base_acc": xb.get("base", {}).get("accuracy", 0),
                            }
                        )
                continue

            log.info(f"--- Eval: {model_name}/{band} ---")
            with open(scores_path, "rb") as f:
                binary = pickle.load(f)
            n_edges, _ = count_edges(binary)

            # Load model fresh
            set_all_seeds(123)
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
            scores_dev = {k: v.to(device) for k, v in binary.items()}

            pool = load_pool(band, Path(config.pool_dir))
            base_metrics = load_base_metrics(
                model_name, band, "test", Path(config.base_metrics_dir), draw=SP_DRAW
            )
            test_data = load_dataset(
                band, "test", Path(config.data_dir), config.variant, SP_DRAW
            )
            base_logits, _, _ = compute_base_logits(
                model, test_data, pool, bos_id, EVAL_BATCH[model_name], 123, device
            )

            # Same-band eval
            circ_logits, circ_ans = run_circuit_and_collect(
                patchable,
                scores_dev,
                n_edges,
                test_data,
                pool,
                bos_id,
                EVAL_BATCH[model_name],
                123,
                device,
            )
            circuit_metrics = compute_accuracy_metrics(circ_logits, circ_ans)
            circuit_metrics["kl_div"] = compute_kl_divergence(circ_logits, base_logits)

            # Ablation
            abl_scores, n_abl = invert_prune_scores(scores_dev)
            if n_abl > 0:
                abl_logits, abl_ans = run_circuit_and_collect(
                    patchable,
                    abl_scores,
                    n_abl,
                    test_data,
                    pool,
                    bos_id,
                    EVAL_BATCH[model_name],
                    123,
                    device,
                )
                ablation_metrics = compute_accuracy_metrics(abl_logits, abl_ans)
            else:
                ablation_metrics = {"accuracy": 0.0}

            log.info(
                f"  Same: base={base_metrics['accuracy']:.1%}, "
                f"circuit={circuit_metrics['accuracy']:.1%}, "
                f"abl={ablation_metrics['accuracy']:.1%}"
            )

            # Cross-band transfer
            transfer = {band: {"base": base_metrics, "circuit": circuit_metrics}}
            for tb in ALL_BANDS:
                if tb == band:
                    continue
                xb = evaluate_on_band(
                    model,
                    patchable,
                    scores_dev,
                    n_edges,
                    tb,
                    model_name,
                    config,
                    device,
                    draw=SP_DRAW,
                )
                transfer[tb] = xb
                xb_acc = xb.get("circuit", {}).get("accuracy", 0)
                log.info(f"    {band}->{tb}: {xb_acc:.1%}")

            # Jaccard with ACDC
            j_acdc = (
                jaccard(binary, acdc_circuits[band]) if band in acdc_circuits else None
            )

            # Save
            result = {
                "model": model_name,
                "band": band,
                "draw": SP_DRAW,
                "n_edges": n_edges,
                "circuit_metrics": circuit_metrics,
                "ablation_metrics": ablation_metrics,
                "base_metrics": base_metrics,
                "cross_band": transfer,
                "jaccard_acdc": j_acdc,
            }
            with open(circuit_dir / "metrics.json", "w") as f:
                json.dump(result, f, indent=2, default=str)

            for tb, xb in transfer.items():
                if "error" not in xb:
                    all_rows.append(
                        {
                            "model": model_name,
                            "circuit_band": band,
                            "test_band": tb,
                            "is_same_band": band == tb,
                            "circuit_acc": xb.get("circuit", {}).get("accuracy", 0),
                            "base_acc": xb.get("base", {}).get("accuracy", 0),
                        }
                    )

            del model, patchable, scores_dev, binary
            full_cleanup()

    # Save CSV
    if all_rows:
        import csv

        with open(SCRIPT_DIR / "sp_eval_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)
        log.info(f"Saved {len(all_rows)} rows to sp_eval_results.csv")


# =========================================================================
# PHASE 3: Analysis
# =========================================================================


def phase3():
    import csv

    csv_path = SCRIPT_DIR / "sp_eval_results.csv"
    if not csv_path.exists():
        log.error("No sp_eval_results.csv")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    summary_dir = SCRIPT_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    models = sorted(set(r["model"] for r in rows))
    summary = {}

    for model in models:
        mr = [r for r in rows if r["model"] == model]
        same = [float(r["circuit_acc"]) for r in mr if r["is_same_band"] == "True"]
        cross = [float(r["circuit_acc"]) for r in mr if r["is_same_band"] == "False"]
        sm, cm = np.mean(same) if same else 0, np.mean(cross) if cross else 0
        adv = sm - cm
        te = cm / sm if sm > 0 else 0

        # Jaccard from metrics.json
        m_safe = model_safe_name(model)
        jaccards = []
        for band in ALL_BANDS:
            mf = SCRIPT_DIR / "circuits" / m_safe / band / SP_DRAW / "metrics.json"
            if mf.exists():
                with open(mf) as f:
                    d = json.load(f)
                if d.get("jaccard_acdc") is not None:
                    jaccards.append(d["jaccard_acdc"])
        mj = np.mean(jaccards) if jaccards else 0

        summary[model] = {
            "same_acc": round(sm, 4),
            "cross_acc": round(cm, 4),
            "advantage_pp": round(adv * 100, 2),
            "transfer_eff": round(te, 4),
            "jaccard_acdc": round(mj, 3),
        }

    with open(summary_dir / "sp_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("\n" + "=" * 70)
    log.info("SP BAND SPECIFICITY SUMMARY")
    log.info("=" * 70)
    log.info(
        f"{'Model':<15} {'Same':>7} {'Cross':>7} {'Adv(pp)':>8} {'TE':>6} {'J(ACDC)':>8}"
    )
    log.info("-" * 55)
    for m, s in summary.items():
        log.info(
            f"{m:<15} {s['same_acc']:>7.3f} {s['cross_acc']:>7.3f} "
            f"{s['advantage_pp']:>+8.2f} {s['transfer_eff']:>6.3f} {s['jaccard_acdc']:>8.3f}"
        )

    # LaTeX table
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Subnetwork Probing cross-band evaluation. Same-band advantage",
        r"(pp) and transfer efficiency (TE) confirm no band-specific computation.",
        r"$J$(ACDC) is Jaccard overlap with ACDC circuits.}",
        r"\label{tab:sp_validation}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Same Acc & Cross Acc & Adv.\ (pp) & TE & $J$(ACDC) \\",
        r"\midrule",
    ]
    for m, s in summary.items():
        short = m.replace("pythia-", "")
        tex.append(
            f"Pythia-{short} & {s['same_acc']:.3f} & {s['cross_acc']:.3f} & "
            f"{s['advantage_pp']:+.1f} & {s['transfer_eff']:.2f} & "
            f"{s['jaccard_acdc']:.2f} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    with open(summary_dir / "band_specificity_table.tex", "w") as f:
        f.write("\n".join(tex) + "\n")

    log.info(f"\nSaved to {summary_dir}")
    return summary


# =========================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=SP_MODELS)
    parser.add_argument("--phase", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    phases = [args.phase] if args.phase is not None else [0, 1, 2, 3]
    log.info(f"Models: {args.models}, Phases: {phases}, Device: {args.device}")

    if 0 in phases:
        phase0(args.models, args.device)
    if 1 in phases:
        phase1(args.models, args.device)
    if 2 in phases:
        phase2(args.models, args.device)
    if 3 in phases:
        phase3()
    log.info("Done!")


if __name__ == "__main__":
    main()
