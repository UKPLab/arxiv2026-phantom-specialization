#!/usr/bin/env python3
"""LSC threshold selection.

Reads pareto_summary.json, displays the per-model Pareto frontier and
recommendations, accepts the user's chosen thresholds (via --select or
interactively), and writes threshold_summary.json for the next stage.

    python lsc_threshold_select.py
    python lsc_threshold_select.py --select pythia-70m=0.001 pythia-160m=0.0001
    python lsc_threshold_select.py --view-only
    python lsc_threshold_select.py --sweep-dir <dir> --output-dir <dir>
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
ISC_ROOT = SCRIPT_DIR.parent


def load_pareto_summary(sweep_dir: Path) -> dict:
    """Load pareto_summary.json from Phase 1."""
    summary_path = sweep_dir / "sweep_results" / "pareto_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"pareto_summary.json not found at {summary_path}\n"
            f"Run lsc_pareto_sweep.py first."
        )
    with open(summary_path) as f:
        return json.load(f)


# =============================================================================
# KL DIVERGENCE INTERPRETATION
# =============================================================================
#
# Guidelines from literature and empirical observation:
#   KL < 0.1:  Good faithfulness - circuit closely reproduces full model
#   KL 0.1-0.5: Moderate - circuit captures main behavior with noticeable deviation
#   KL > 0.5:  Poor - circuit significantly diverges from full model
#   KL > 1.0:  Bad - circuit behavior is quite different
#


def interpret_kl(kl: float) -> str:
    """Return human-readable KL interpretation."""
    if kl < 0.1:
        return "good"
    elif kl < 0.3:
        return "moderate"
    elif kl < 0.5:
        return "acceptable"
    elif kl < 1.0:
        return "poor"
    else:
        return "bad"


# =============================================================================
# EXPECTED CIRCUIT SIZE BY MODEL CAPACITY
# =============================================================================
#
# circuit size (edge fraction) has NEGATIVE relationship with model capacity.
# Smaller models are denser - they must use most edges efficiently.
# Larger models have redundancy - the core circuit is a smaller fraction.
#
# The absolute edge count for a task's circuit may be similar across model sizes,
# but the FRACTION decreases with model size.
#

EXPECTED_CIRCUIT_FRACTION = {
    # model: (min_fraction, max_fraction, typical_fraction)
    "pythia-70m": (0.15, 0.35, 0.25),
    "pythia-160m": (0.06, 0.15, 0.10),
    "pythia-410m": (0.03, 0.08, 0.05),
    "pythia-1b": (0.02, 0.05, 0.03),
    "pythia-1.4b": (0.015, 0.04, 0.025),
}


def get_expected_fraction(model: str) -> Tuple[float, float, float]:
    """Return (min, max, typical) expected circuit fraction for model."""
    return EXPECTED_CIRCUIT_FRACTION.get(model, (0.05, 0.20, 0.10))


def assess_circuit_size(model: str, fraction: float) -> str:
    """Assess if circuit size is appropriate for this model."""
    min_f, max_f, typical_f = get_expected_fraction(model)
    if fraction < min_f:
        return "too_small"
    elif fraction <= max_f:
        return "good"
    elif fraction <= max_f * 2:
        return "large"
    else:
        return "too_large"  # This is basically the whole model, not a circuit


# =============================================================================
# RECOMMENDATION HEURISTICS
# =============================================================================


def compute_recommendations(pareto_data: dict, model: str) -> Dict[str, Any]:
    """
    Compute recommendation heuristics for a single model.
    These are SUGGESTIONS only.

    principle: A circuit must be MINIMAL and INTERPRETABLE.
    A 40-76% edge fraction is basically the whole model, not a circuit.

    Heuristics (prioritizing minimality):
      1. minimal_acceptable: Smallest circuit with retention >= 80% and KL < 0.5
      2. interpretable: Circuit in expected size range with best KL
      3. same_threshold: τ = 0.00158 (works across model sizes, produces appropriate fractions)
    """
    sweep_points = pareto_data.get("sweep_points", [])
    base_accuracy = pareto_data.get("base_accuracy", 0)
    if not sweep_points:
        return {}

    # Sort by size (smallest first) - we want minimal circuits
    sorted_by_size = sorted(sweep_points, key=lambda p: p["size_fraction"])

    recommendations = {}
    min_f, max_f, typical_f = get_expected_fraction(model)

    # 1. Minimal acceptable: smallest circuit with retention >= 80% AND KL < 0.5
    for pt in sorted_by_size:
        retention = pt.get(
            "retention", pt["accuracy"] / base_accuracy if base_accuracy > 0 else 0
        )
        if retention >= 0.80 and pt["kl_div"] < 0.5:
            recommendations["minimal_acceptable"] = {
                "threshold": pt["threshold"],
                "size_fraction": pt["size_fraction"],
                "n_edges": pt["n_edges"],
                "kl_div": pt["kl_div"],
                "accuracy": pt["accuracy"],
                "retention": retention,
                "reason": f"Smallest circuit with >=80% retention and KL<0.5",
            }
            break

    # 2. Interpretable: within expected size range, best retention
    in_range = [p for p in sorted_by_size if min_f <= p["size_fraction"] <= max_f]
    if in_range:
        # Among circuits in expected range, pick one with best retention
        best_in_range = max(
            in_range,
            key=lambda p: p.get(
                "retention", p["accuracy"] / base_accuracy if base_accuracy > 0 else 0
            ),
        )
        retention = best_in_range.get(
            "retention",
            best_in_range["accuracy"] / base_accuracy if base_accuracy > 0 else 0,
        )
        recommendations["interpretable"] = {
            "threshold": best_in_range["threshold"],
            "size_fraction": best_in_range["size_fraction"],
            "n_edges": best_in_range["n_edges"],
            "kl_div": best_in_range["kl_div"],
            "accuracy": best_in_range["accuracy"],
            "retention": retention,
            "reason": f"Best retention within expected range ({min_f * 100:.0f}-{max_f * 100:.0f}%)",
        }

    # 3. Standard threshold τ = 0.00158 (empirically works across model sizes)
    # Find closest threshold in sweep
    standard_tau = 0.00158
    closest = min(sorted_by_size, key=lambda p: abs(p["threshold"] - standard_tau))
    if abs(closest["threshold"] - standard_tau) / standard_tau < 0.5:  # Within 50%
        retention = closest.get(
            "retention", closest["accuracy"] / base_accuracy if base_accuracy > 0 else 0
        )
        size_assessment = assess_circuit_size(model, closest["size_fraction"])
        recommendations["standard_tau"] = {
            "threshold": closest["threshold"],
            "size_fraction": closest["size_fraction"],
            "n_edges": closest["n_edges"],
            "kl_div": closest["kl_div"],
            "accuracy": closest["accuracy"],
            "retention": retention,
            "size_assessment": size_assessment,
            "reason": f"τ≈0.00158 (standard threshold, size: {size_assessment})",
        }

    # 4. Fallback: if nothing meets criteria, find best compromise
    if not recommendations:
        # Just pick smallest with KL < 1.0
        for pt in sorted_by_size:
            if pt["kl_div"] < 1.0:
                retention = pt.get(
                    "retention",
                    pt["accuracy"] / base_accuracy if base_accuracy > 0 else 0,
                )
                recommendations["fallback"] = {
                    "threshold": pt["threshold"],
                    "size_fraction": pt["size_fraction"],
                    "n_edges": pt["n_edges"],
                    "kl_div": pt["kl_div"],
                    "accuracy": pt["accuracy"],
                    "retention": retention,
                    "reason": "Fallback: smallest with KL<1.0",
                }
                break

    return recommendations


def get_primary_recommendation(recommendations: dict) -> Optional[dict]:
    """
    Get primary recommendation.
    Priority: interpretable > minimal_acceptable > standard_tau > fallback
    """
    for key in ["interpretable", "minimal_acceptable", "standard_tau", "fallback"]:
        if key in recommendations:
            return recommendations[key]
    return None


def print_separator(char: str = "=", width: int = 100):
    print(char * width)


def print_model_table(model: str, pareto_data: dict, recommendations: dict):
    """Print detailed table for a single model."""
    print_separator()
    print(f"MODEL: {model}")

    base_acc = pareto_data.get("base_accuracy", 0)
    print(f"Base accuracy: {base_acc:.1%}")
    print(f"Thresholds tested: {pareto_data.get('n_thresholds_tested', 0)}")
    print(f"Pareto-optimal points: {pareto_data.get('n_pareto_optimal', 0)}")

    # Expected circuit size range
    min_f, max_f, typical_f = get_expected_fraction(model)
    print(
        f"Expected circuit size: {min_f * 100:.0f}-{max_f * 100:.0f}% (typical: {typical_f * 100:.0f}%)"
    )

    print_separator("-")

    sweep_points = pareto_data.get("sweep_points", [])
    if not sweep_points:
        print("  No sweep points available.")
        return

    # KL interpretation guide
    print("  KL Guide: <0.1=good, 0.1-0.3=moderate, 0.3-0.5=acceptable, >0.5=poor")
    print()

    # Header - sorted by edges (smallest first = most minimal)
    print(
        f"  {'':>2} {'Threshold':>12} {'Edges':>7} {'Size%':>7} "
        f"{'KL':>8} {'KL?':>6} {'Acc%':>7} {'Ret%':>7} {'SizeOK':>7}"
    )
    print("  " + "-" * 80)

    # Get recommended thresholds for marking
    rec_thresholds = {
        r["threshold"] for r in recommendations.values() if "threshold" in r
    }

    # Sort by number of edges (smallest first - we want minimal circuits)
    for pt in sorted(sweep_points, key=lambda p: p["n_edges"]):
        pareto_mark = "*" if pt.get("is_pareto_optimal") else " "
        rec_mark = "R" if pt["threshold"] in rec_thresholds else " "
        marker = f"{pareto_mark}{rec_mark}"

        # Compute retention if not present
        retention = pt.get(
            "retention", pt["accuracy"] / base_acc if base_acc > 0 else 0
        )

        # KL interpretation
        kl_interp = interpret_kl(pt["kl_div"])

        # Size assessment
        size_ok = assess_circuit_size(model, pt["size_fraction"])
        size_mark = {"good": "+", "large": "~", "too_large": "-", "too_small": "!"}.get(
            size_ok, "?"
        )

        print(
            f"  {marker:>2} {pt['threshold']:>12.5f} {pt['n_edges']:>7d} "
            f"{pt['size_fraction'] * 100:>6.1f}% {pt['kl_div']:>8.3f} "
            f"{kl_interp:>6} {pt['accuracy'] * 100:>6.1f}% {retention * 100:>6.1f}% "
            f"{size_mark:>4} {size_ok}"
        )

    print()
    print("  Legend: * = Pareto-optimal, R = Recommended")
    print("  SizeOK: +=good, ~=large, -=too_large (not a circuit), !=too_small")
    print()

    # Recommendations
    if recommendations:
        print("  RECOMMENDATIONS (for reference only; you decide):")
        print("  " + "-" * 70)
        for name, rec in recommendations.items():
            retention = rec.get("retention", 0)
            kl_interp = interpret_kl(rec["kl_div"])
            print(f"    {name}:")
            print(f"      τ = {rec['threshold']:.5f}")
            print(
                f"      {rec['n_edges']} edges ({rec['size_fraction'] * 100:.1f}%), "
                f"KL={rec['kl_div']:.3f} ({kl_interp}), retention={retention * 100:.0f}%"
            )
            print(f"      {rec['reason']}")
        print()


def print_summary_table(pareto_summary: dict, all_recommendations: dict):
    """Print compact summary across all models."""
    print_separator("=")
    print("SUMMARY: PRIMARY RECOMMENDATIONS")
    print("(Priority: interpretable > minimal_acceptable > standard_tau)")
    print_separator("-")

    print(
        f"  {'Model':<14} {'Threshold':>10} {'Edges':>7} {'Size%':>7} "
        f"{'KL':>7} {'KL?':>6} {'Ret%':>6} {'Base%':>6} {'Type':<12}"
    )
    print("  " + "-" * 90)

    for model, recs in all_recommendations.items():
        primary = get_primary_recommendation(recs)
        pareto_data = pareto_summary.get("pareto_results", {}).get(model, {})
        base_acc = pareto_data.get("base_accuracy", 0)

        # Find which recommendation type was selected
        rec_type = "none"
        for key in ["interpretable", "minimal_acceptable", "standard_tau", "fallback"]:
            if key in recs and recs[key] == primary:
                rec_type = key
                break

        if primary:
            kl_interp = interpret_kl(primary["kl_div"])
            retention = primary.get("retention", 0)
            print(
                f"  {model:<14} {primary['threshold']:>10.5f} "
                f"{primary['n_edges']:>7d} {primary['size_fraction'] * 100:>6.1f}% "
                f"{primary['kl_div']:>7.3f} {kl_interp:>6} "
                f"{retention * 100:>5.0f}% {base_acc * 100:>5.0f}% {rec_type:<12}"
            )
        else:
            print(
                f"  {model:<14} {'N/A':>10} {'N/A':>7} {'N/A':>7} "
                f"{'N/A':>7} {'N/A':>6} {'N/A':>6} {base_acc * 100:>5.0f}% {'none':<12}"
            )

    print()
    print("  Note: Smaller models need larger edge fractions (denser circuits).")
    print("        70m~25%, 160m~10%, 410m~5%, 1b+~2-3%")
    print()


def plot_all_models(pareto_summary: dict, all_recommendations: dict, output_dir: Path):
    """Generate comparison plots for all models."""
    plot_dir = output_dir / "threshold_selection_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    pareto_results = pareto_summary.get("pareto_results", {})
    models = list(pareto_results.keys())

    if not models:
        print("No models to plot.")
        return

    # Plot 1: Per-model Pareto frontiers (subplots)
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_models == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, model in enumerate(models):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        pareto_data = pareto_results[model]
        sweep_points = pareto_data.get("sweep_points", [])
        min_f, max_f, _ = get_expected_fraction(model)

        if not sweep_points:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(model)
            continue

        sizes = [p["size_fraction"] * 100 for p in sweep_points]
        kls = [p["kl_div"] for p in sweep_points]
        pareto_mask = [p.get("is_pareto_optimal", False) for p in sweep_points]

        # Expected size range (shaded region)
        ax.axvspan(
            min_f * 100, max_f * 100, alpha=0.2, color="green", label="Expected range"
        )

        # KL thresholds (horizontal lines)
        ax.axhline(0.1, color="green", ls="--", alpha=0.5, lw=1)
        ax.axhline(0.5, color="orange", ls="--", alpha=0.5, lw=1)
        ax.text(
            ax.get_xlim()[1] * 0.98,
            0.1,
            "good",
            fontsize=7,
            ha="right",
            va="bottom",
            color="green",
        )
        ax.text(
            ax.get_xlim()[1] * 0.98,
            0.5,
            "poor",
            fontsize=7,
            ha="right",
            va="bottom",
            color="orange",
        )

        # All points
        ax.scatter(sizes, kls, c="lightgray", s=50, zorder=2, label="All")

        # Pareto points
        p_sizes = [s for s, m in zip(sizes, pareto_mask) if m]
        p_kls = [k for k, m in zip(kls, pareto_mask) if m]
        if p_sizes:
            # Sort for line
            sorted_pareto = sorted(zip(p_sizes, p_kls))
            ax.plot(
                [x[0] for x in sorted_pareto],
                [x[1] for x in sorted_pareto],
                "ro-",
                markersize=8,
                zorder=3,
                label="Pareto",
            )

        # Recommendation
        recs = all_recommendations.get(model, {})
        primary = get_primary_recommendation(recs)
        if primary:
            ax.scatter(
                [primary["size_fraction"] * 100],
                [primary["kl_div"]],
                c="blue",
                s=200,
                marker="*",
                zorder=4,
                label="Recommended",
                edgecolors="black",
                linewidths=1,
            )

        ax.set_xlabel("Circuit Size (%)")
        ax.set_ylabel("KL Divergence")
        ax.set_title(f"{model}\n(base acc: {pareto_data.get('base_accuracy', 0):.1%})")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_models, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(plot_dir / "pareto_frontiers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_dir / 'pareto_frontiers.png'}")

    # Plot 2: Cross-model comparison (recommended thresholds)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    rec_models = []
    rec_sizes = []
    rec_kls = []
    rec_accs = []

    for model in models:
        recs = all_recommendations.get(model, {})
        primary = get_primary_recommendation(recs)
        if primary:
            rec_models.append(model.replace("pythia-", ""))
            rec_sizes.append(primary["size_fraction"] * 100)
            rec_kls.append(primary["kl_div"])
            rec_accs.append(primary["accuracy"] * 100)

    if rec_models:
        x = np.arange(len(rec_models))

        # Size comparison
        ax = axes[0]
        ax.bar(x, rec_sizes, color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(rec_models, rotation=45, ha="right")
        ax.set_ylabel("Circuit Size (%)")
        ax.set_title("Recommended Circuit Size by Model")
        ax.grid(axis="y", alpha=0.3)

        # KL comparison
        ax = axes[1]
        ax.bar(x, rec_kls, color="coral", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(rec_models, rotation=45, ha="right")
        ax.set_ylabel("KL Divergence")
        ax.set_title("Recommended KL Divergence by Model")
        ax.grid(axis="y", alpha=0.3)

        # Accuracy comparison
        ax = axes[2]
        ax.bar(x, rec_accs, color="forestgreen", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(rec_models, rotation=45, ha="right")
        ax.set_ylabel("Circuit Accuracy (%)")
        ax.set_title("Recommended Circuit Accuracy by Model")
        ax.axhline(100, color="red", ls="--", alpha=0.5, label="Perfect")
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(plot_dir / "cross_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_dir / 'cross_model_comparison.png'}")


def parse_cli_selections(selections: List[str]) -> Dict[str, float]:
    """Parse CLI selections in format 'model=threshold'."""
    result = {}
    for sel in selections:
        if "=" not in sel:
            raise ValueError(
                f"Invalid selection format: {sel}. Expected 'model=threshold'"
            )
        model, threshold_str = sel.split("=", 1)
        result[model.strip()] = float(threshold_str.strip())
    return result


def interactive_selection(
    pareto_summary: dict,
    all_recommendations: dict,
    existing_selections: Dict[str, float] = None,
) -> Dict[str, float]:
    """Interactive threshold selection for each model."""
    pareto_results = pareto_summary.get("pareto_results", {})
    models = list(pareto_results.keys())
    selections = existing_selections.copy() if existing_selections else {}

    print_separator("=")
    print("INTERACTIVE THRESHOLD SELECTION")
    print("Enter threshold for each model, or:")
    print("  - Press Enter to accept recommendation")
    print("  - Type 'skip' to skip this model")
    print("  - Type 'list' to show available thresholds with details")
    print("  - Type 'quit' to exit without saving")
    print_separator("-")

    for model in models:
        pareto_data = pareto_results[model]
        sweep_points = pareto_data.get("sweep_points", [])
        base_acc = pareto_data.get("base_accuracy", 0)
        recs = all_recommendations.get(model, {})
        primary = get_primary_recommendation(recs)

        available_thresholds = sorted(set(p["threshold"] for p in sweep_points))
        min_f, max_f, typical_f = get_expected_fraction(model)

        print(
            f"\n{model} (base acc: {base_acc * 100:.1f}%, expected size: {min_f * 100:.0f}-{max_f * 100:.0f}%):"
        )
        if primary:
            kl_interp = interpret_kl(primary["kl_div"])
            retention = primary.get("retention", 0)
            print(f"  Recommended: τ={primary['threshold']:.5f}")
            print(
                f"    {primary['n_edges']} edges ({primary['size_fraction'] * 100:.1f}%), "
                f"KL={primary['kl_div']:.3f} ({kl_interp}), retention={retention * 100:.0f}%"
            )
        else:
            print("  No recommendation available.")

        if model in selections:
            print(f"  Current selection: τ={selections[model]:.5f}")

        while True:
            default_str = f"{primary['threshold']:.5f}" if primary else "none"
            prompt = f"  Enter threshold [default: {default_str}]: "
            user_input = input(prompt).strip().lower()

            if user_input == "quit":
                print("Exiting without saving.")
                return None

            if user_input == "skip":
                print(f"  Skipped {model}")
                break

            if user_input == "list":
                print(f"\n  Available thresholds for {model}:")
                print(
                    f"    {'Threshold':>12} {'Edges':>7} {'Size%':>7} {'KL':>8} {'Ret%':>7} {'SizeOK'}"
                )
                for pt in sorted(sweep_points, key=lambda p: p["n_edges"]):
                    retention = pt.get(
                        "retention", pt["accuracy"] / base_acc if base_acc > 0 else 0
                    )
                    size_ok = assess_circuit_size(model, pt["size_fraction"])
                    mark = (
                        "->"
                        if primary and pt["threshold"] == primary["threshold"]
                        else " "
                    )
                    print(
                        f"  {mark} {pt['threshold']:>12.5f} {pt['n_edges']:>7d} "
                        f"{pt['size_fraction'] * 100:>6.1f}% {pt['kl_div']:>8.3f} "
                        f"{retention * 100:>6.0f}% {size_ok}"
                    )
                print()
                continue

            if user_input == "":
                if primary:
                    selections[model] = primary["threshold"]
                    print(f"  Selected: τ={primary['threshold']:.5f} (recommendation)")
                    break
                else:
                    print("  No default available. Please enter a threshold.")
                    continue

            try:
                threshold = float(user_input)
                # Check if it's close to any available threshold
                close_match = None
                for t in available_thresholds:
                    if abs(t - threshold) / max(t, 1e-10) < 0.01:  # Within 1%
                        close_match = t
                        break

                if close_match:
                    threshold = close_match  # Use exact value from sweep

                if threshold not in available_thresholds:
                    print(
                        f"  Warning: {threshold:.5f} not in Pareto sweep. "
                        f"Will need fresh ACDC run in Phase 2."
                    )
                    confirm = input("  Continue anyway? [y/N]: ").strip().lower()
                    if confirm != "y":
                        continue

                selections[model] = threshold
                print(f"  Selected: τ={threshold:.5f}")
                break
            except ValueError:
                print(f"  Invalid threshold: {user_input}")
                continue

    return selections


def write_threshold_summary(
    selections: Dict[str, float],
    pareto_summary: dict,
    all_recommendations: dict,
    output_dir: Path,
):
    """Write threshold_summary.json for Phase 2."""
    output_path = output_dir / "sweep_results" / "threshold_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pareto_results = pareto_summary.get("pareto_results", {})

    # Build selections with metadata
    selections_with_meta = {}
    for model, threshold in selections.items():
        pareto_data = pareto_results.get(model, {})
        sweep_points = pareto_data.get("sweep_points", [])

        # Find matching sweep point
        matching = [p for p in sweep_points if abs(p["threshold"] - threshold) < 1e-12]
        if matching:
            pt = matching[0]
            selections_with_meta[model] = {
                "threshold": threshold,
                "size_fraction": pt["size_fraction"],
                "n_edges": pt["n_edges"],
                "kl_div": pt["kl_div"],
                "accuracy": pt["accuracy"],
                "retention": pt.get("retention", 0),
                "ablation_accuracy": pt.get("ablation_accuracy", 0),
                "is_pareto_optimal": pt.get("is_pareto_optimal", False),
            }
        else:
            # Threshold not from sweep (user override)
            selections_with_meta[model] = {
                "threshold": threshold,
                "note": "Custom threshold (not from Pareto sweep)",
            }

    summary = OrderedDict()
    summary["created_at"] = datetime.now().isoformat()
    summary["source"] = "lsc_threshold_select.py"
    summary["note"] = "Human-selected thresholds for Phase 2 circuit discovery"
    summary["selections"] = selections_with_meta

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="LSC Threshold Selection; human-in-the-loop bridge between Phase 1 and 2",
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default=None,
        help="Directory with pareto_summary.json from Phase 1",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for threshold_summary.json (default: same as sweep-dir)",
    )
    parser.add_argument(
        "--select",
        nargs="+",
        default=None,
        help="CLI selections: model=threshold (e.g., pythia-70m=0.001)",
    )
    parser.add_argument(
        "--view-only",
        action="store_true",
        help="Only display data, don't prompt for selection",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")

    args = parser.parse_args()

    # Paths
    sweep_dir = Path(args.sweep_dir) if args.sweep_dir else SCRIPT_DIR / "pareto_sweep"
    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir

    # Load data
    print(f"Loading Pareto summary from: {sweep_dir}")
    try:
        pareto_summary = load_pareto_summary(sweep_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 1

    pareto_results = pareto_summary.get("pareto_results", {})
    if not pareto_results:
        print("ERROR: No Pareto results found in summary.")
        return 1

    models = list(pareto_results.keys())
    print(f"Found {len(models)} models: {models}")

    # Compute recommendations
    all_recommendations = {}
    for model in models:
        all_recommendations[model] = compute_recommendations(
            pareto_results[model], model
        )

    # Display detailed tables
    for model in models:
        print_model_table(model, pareto_results[model], all_recommendations[model])

    # Summary table
    print_summary_table(pareto_summary, all_recommendations)

    # Generate plots
    if not args.no_plots:
        print("Generating plots...")
        plot_all_models(pareto_summary, all_recommendations, output_dir)

    # View-only mode
    if args.view_only:
        print("\n[View-only mode; no selection saved]")
        return 0

    # Selection
    if args.select:
        # CLI mode
        try:
            selections = parse_cli_selections(args.select)
            print(f"\nCLI selections: {selections}")
        except ValueError as e:
            print(f"ERROR: {e}")
            return 1
    else:
        # Interactive mode
        selections = interactive_selection(pareto_summary, all_recommendations)
        if selections is None:
            return 0  # User quit

    if not selections:
        print("\nNo selections made. Exiting.")
        return 0

    # Confirm
    print_separator("-")
    print("FINAL SELECTIONS:")
    for model, threshold in selections.items():
        print(f"  {model}: τ={threshold:.2e}")

    confirm = input("\nSave these selections? [Y/n]: ").strip().lower()
    if confirm == "n":
        print("Selections not saved.")
        return 0

    # Save
    output_path = write_threshold_summary(
        selections, pareto_summary, all_recommendations, output_dir
    )

    print_separator("=")
    print("THRESHOLD SELECTION COMPLETE")
    print(f"Output: {output_path}")
    print("\nNext step: Run lsc_acdc_circuit.py for Phase 2 circuit discovery")
    print(f"  python lsc_acdc_circuit.py --sweep-dir {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
