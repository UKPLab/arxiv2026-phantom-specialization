"""Regenerate T2_02_boost_heatmaps.png from cross_band_eval_results.csv and universal_core_comparison.csv (CPU-only)."""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ANALYSIS_ROOT = Path("LSC_circuit_analysis")
PHASE5_DIR = ANALYSIS_ROOT / "05_Phase_Targeted"
ANALYSIS_DIR = PHASE5_DIR / "outputs" / "analysis"
VIZ_DIR = PHASE5_DIR / "outputs" / "viz"

MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]
BAND_ORDER = ["low", "medium", "high", "very_high", "control"]

# Load data
df_all = pd.read_csv(ANALYSIS_DIR / "cross_band_eval_results.csv")
df_nb01 = pd.read_csv(ANALYSIS_DIR / "universal_core_comparison.csv")

print(f"Loaded {len(df_all)} rows from cross_band_eval_results.csv")
print(f"Models in CSV: {sorted(df_all['model'].unique())}")

# Build universal accuracy reference
univ_ref = df_nb01[["model", "test_band", "universal_acc"]].copy()

# Cross-band aggregation
df_cross = df_all[df_all["circuit_type"] == "cross_band"].copy()
df_cross = df_cross.merge(univ_ref, on=["model", "test_band"], how="left")
df_cross["boost"] = df_cross["accuracy"] - df_cross["universal_acc"]

df_cross_agg = (
    df_cross.groupby(["model", "source_band", "test_band"])
    .agg(
        mean_boost=("boost", "mean"),
        std_boost=("boost", "std"),
    )
    .reset_index()
)

# Generate figure
fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 5.5))

for ax, model_name in zip(axes, MODELS):
    sub = df_cross_agg[df_cross_agg["model"] == model_name]
    pivot_mean = sub.pivot(
        index="source_band", columns="test_band", values="mean_boost"
    )
    pivot_std = sub.pivot(index="source_band", columns="test_band", values="std_boost")
    pivot_mean = pivot_mean.reindex(index=BAND_ORDER, columns=BAND_ORDER)
    pivot_std = pivot_std.reindex(index=BAND_ORDER, columns=BAND_ORDER)

    annot = pivot_mean.copy().astype(str)
    for r in BAND_ORDER:
        for c in BAND_ORDER:
            m = pivot_mean.loc[r, c]
            s = pivot_std.loc[r, c]
            annot.loc[r, c] = f"{m:.2f}\n\u00b1{s:.2f}"

    vmax = pivot_mean.max().max() * 1.1
    sns.heatmap(
        pivot_mean,
        annot=annot,
        fmt="",
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        ax=ax,
        square=True,
        linewidths=0,
        linecolor="none",
        annot_kws={"fontsize": 8},
    )
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.set_xlabel("Tested on band ...")
    ax.set_ylabel("Circuit from band ...")

fig.suptitle(
    "Boost over Universal Core (cross_acc \u2212 universal_acc)\n"
    "Mean \u00b1 std across 3 draws",
    fontsize=14,
    fontweight="bold",
    y=1.03,
)
fig.tight_layout()

out = VIZ_DIR / "T2_02_boost_heatmaps.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
