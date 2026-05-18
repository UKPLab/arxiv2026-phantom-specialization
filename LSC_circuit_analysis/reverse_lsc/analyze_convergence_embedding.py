#!/usr/bin/env python3
"""
Link embedding geometry to logit-lens convergence depth.

Tests the hypothesis from Section 7.4: "low-frequency tokens occupy
less-separated regions of embedding space (having received fewer
gradient updates during training)," which could explain why they
converge later in the logit lens.

Analyses:
1. Per-band embedding norms vs convergence depth
2. Component attribution (attention vs MLP fraction) by band and layer
3. Convergence gap scaling with model size
4. Correlation between embedding properties and convergence delay
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

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PHASE3_DIR = PROJECT_ROOT / "LSC_circuit_analysis/03_Phase_Representational"
LOGIT_LENS_DIR = PHASE3_DIR / "outputs" / "logit_lens" / "base" / "analysis"
EMBEDDING_DIR = PHASE3_DIR / "outputs" / "embedding" / "base" / "analysis"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

BANDS = ["low", "medium", "high", "very_high"]
BAND_FREQ_ORDER = {"low": 1, "medium": 2, "high": 3, "very_high": 4, "control": 5}


def main():
    print("=" * 70)
    print("Embedding Geometry vs Logit-Lens Convergence Analysis")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    master_ll = pd.read_csv(LOGIT_LENS_DIR / "03_master_logit_lens.csv")
    component_attr = pd.read_csv(LOGIT_LENS_DIR / "03_component_attribution.csv")
    embedding_norms = pd.read_csv(EMBEDDING_DIR / "01_embedding_norms.csv")
    convergence_stats = pd.read_csv(LOGIT_LENS_DIR / "03_convergence_stats.csv")
    prob_trajectory = pd.read_csv(LOGIT_LENS_DIR / "03_prob_correct_trajectory.csv")

    # Filter to core bands (exclude control for cleaner frequency gradient)
    core_bands = ["low", "medium", "high", "very_high"]

    # Per-band embedding norms vs convergence depth
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Embedding Norms vs Convergence Depth")
    print("=" * 70)

    # Merge embedding norms with convergence data
    # Average convergence across draws
    conv_avg = (
        master_ll.groupby(["model", "band"])
        .agg(
            {
                "frac_convergence": "mean",
                "convergence_layer": "mean",
                "n_layers": "first",
                "final_prob_correct": "mean",
            }
        )
        .reset_index()
    )

    merged = conv_avg.merge(
        embedding_norms, on=["model", "band"], how="inner", suffixes=("", "_norm")
    )

    # Add frequency rank
    merged["freq_rank"] = merged["band"].map(BAND_FREQ_ORDER)

    print(f"\n  Per-model embedding norm vs convergence depth:")
    print(
        f"  {'Model':15s} | {'Band':12s} | {'Emb Norm':>10s} | {'Conv Depth':>10s} | {'Final P':>8s}"
    )
    print(f"  {'-' * 15}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 8}")

    models = sorted(
        merged["model"].unique(),
        key=lambda m: int(
            m.split("-")[1].replace("m", "").replace("b", "000").replace(".", "")
        ),
    )

    for model in models:
        m_data = merged[merged["model"] == model].sort_values("freq_rank")
        for _, row in m_data.iterrows():
            if row["band"] in core_bands:
                print(
                    f"  {model:15s} | {row['band']:12s} | {row['mean']:10.4f} | "
                    f"{row['frac_convergence']:10.4f} | {row['final_prob_correct']:8.4f}"
                )

    # Within-model correlation: embedding norm vs convergence depth
    print(
        f"\n  Within-model Spearman correlation (embedding norm vs frac_convergence):"
    )
    for model in models:
        m_data = merged[(merged["model"] == model) & (merged["band"].isin(core_bands))]
        if len(m_data) >= 4:
            rho, p = stats.spearmanr(m_data["mean"], m_data["frac_convergence"])
            print(f"    {model:15s}: rho={rho:+.3f}, p={p:.4f} (n={len(m_data)})")

    # Component attribution by band
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Attention vs MLP Contribution by Band")
    print("=" * 70)

    # Average across draws, look at per-layer attribution by band
    attr_avg = (
        component_attr.groupby(["model", "band", "layer"])
        .agg(
            {
                "attn_frac": "mean",
                "mlp_frac": "mean",
                "attn_logit_mean": "mean",
                "mlp_logit_mean": "mean",
            }
        )
        .reset_index()
    )

    # For each model, compare low vs very_high attribution profiles
    print(f"\n  Attribution difference (low - very_high) at key layers:")
    for model in models:
        m_attr = attr_avg[attr_avg["model"] == model]
        low_attr = m_attr[m_attr["band"] == "low"].sort_values("layer")
        vh_attr = m_attr[m_attr["band"] == "very_high"].sort_values("layer")

        if len(low_attr) == 0 or len(vh_attr) == 0:
            continue

        n_layers = len(low_attr)
        # Show early, middle, late layers
        key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

        print(f"\n  {model} ({n_layers} layers):")
        print(
            f"    {'Layer':>5s} | {'Low attn%':>9s} | {'VH attn%':>9s} | {'Δ attn%':>8s} | "
            f"{'Low mlp%':>9s} | {'VH mlp%':>9s} | {'Δ mlp%':>8s}"
        )
        print(
            f"    {'-' * 5}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 8}-+-{'-' * 9}-+-{'-' * 9}-+-{'-' * 8}"
        )

        for l_idx in key_layers:
            if l_idx < len(low_attr) and l_idx < len(vh_attr):
                l_row = low_attr.iloc[l_idx]
                v_row = vh_attr.iloc[l_idx]
                d_attn = l_row["attn_frac"] - v_row["attn_frac"]
                d_mlp = l_row["mlp_frac"] - v_row["mlp_frac"]
                print(
                    f"    {int(l_row['layer']):5d} | {l_row['attn_frac']:9.1%} | {v_row['attn_frac']:9.1%} | "
                    f"{d_attn:+8.1%} | {l_row['mlp_frac']:9.1%} | {v_row['mlp_frac']:9.1%} | {d_mlp:+8.1%}"
                )

    # Overall mean attribution by band (averaged across all layers)
    print(f"\n  Mean attribution fraction across all layers:")
    mean_attr = (
        attr_avg.groupby(["model", "band"])
        .agg(
            {
                "attn_frac": "mean",
                "mlp_frac": "mean",
            }
        )
        .reset_index()
    )

    for model in models:
        m_data = mean_attr[
            (mean_attr["model"] == model) & (mean_attr["band"].isin(core_bands))
        ]
        m_data = m_data.sort_values("band", key=lambda x: x.map(BAND_FREQ_ORDER))
        attn_range = f"{m_data['attn_frac'].min():.1%}-{m_data['attn_frac'].max():.1%}"
        attn_spread = m_data["attn_frac"].max() - m_data["attn_frac"].min()
        print(f"    {model:15s}: attn range {attn_range} (spread: {attn_spread:.1%})")

    # Convergence gap scaling
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Convergence Gap Scaling with Model Size")
    print("=" * 70)

    for model in models:
        m_data = conv_avg[
            (conv_avg["model"] == model) & (conv_avg["band"].isin(core_bands))
        ]
        low_conv = m_data[m_data["band"] == "low"]["frac_convergence"].values
        vh_conv = m_data[m_data["band"] == "very_high"]["frac_convergence"].values
        if len(low_conv) > 0 and len(vh_conv) > 0:
            gap = low_conv[0] - vh_conv[0]
            n_layers = m_data["n_layers"].iloc[0]
            print(
                f"  {model:15s}: low={low_conv[0]:.3f}, very_high={vh_conv[0]:.3f}, "
                f"gap={gap:+.3f} ({gap * n_layers:+.1f} layers)"
            )

    # Correlation summary
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Cross-Model Correlation Summary")
    print("=" * 70)

    # Pooled correlation (embedding norm vs convergence): but check for Simpson's paradox
    core_merged = merged[merged["band"].isin(core_bands)]

    # Pooled
    rho_pooled, p_pooled = stats.spearmanr(
        core_merged["mean"], core_merged["frac_convergence"]
    )
    print(f"\n  Pooled: rho={rho_pooled:+.3f}, p={p_pooled:.4f}")

    # Within-model average
    within_rhos = []
    for model in models:
        m_data = core_merged[core_merged["model"] == model]
        if len(m_data) >= 4:
            rho, _ = stats.spearmanr(m_data["mean"], m_data["frac_convergence"])
            within_rhos.append(rho)
    if within_rhos:
        print(
            f"  Within-model mean: rho={np.mean(within_rhos):+.3f} +/- {np.std(within_rhos):.3f}"
        )
        if np.sign(rho_pooled) != np.sign(np.mean(within_rhos)):
            print(f"  Simpson's paradox: pooled sign differs from within-model mean")

    # Convergence depth table for appendix
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Convergence Depth Table (for appendix)")
    print("=" * 70)

    print(
        f"\n  {'Model':15s} | {'low':>8s} | {'medium':>8s} | {'high':>8s} | {'v_high':>8s} | {'gap':>8s}"
    )
    print(f"  {'-' * 15}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}")

    gap_data = []
    for model in models:
        m_data = conv_avg[conv_avg["model"] == model]
        row = {}
        for band in core_bands:
            b_data = m_data[m_data["band"] == band]
            if len(b_data) > 0:
                row[band] = b_data["frac_convergence"].values[0]
        if "low" in row and "very_high" in row:
            gap = row["low"] - row["very_high"]
            gap_data.append({"model": model, "gap": gap})
            print(
                f"  {model:15s} | {row.get('low', 0):8.3f} | {row.get('medium', 0):8.3f} | "
                f"{row.get('high', 0):8.3f} | {row.get('very_high', 0):8.3f} | {gap:+8.3f}"
            )

    # Save results
    results = {
        "embedding_norm_vs_convergence": {
            "pooled_spearman": {"rho": float(rho_pooled), "p": float(p_pooled)},
            "within_model_mean_rho": float(np.mean(within_rhos))
            if within_rhos
            else None,
        },
        "convergence_gaps": gap_data,
    }

    output_path = OUTPUT_DIR / "convergence_embedding_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
