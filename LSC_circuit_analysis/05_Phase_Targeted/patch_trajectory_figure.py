"""Regenerate the logit lens trajectory overlay figure from pre-computed CSVs (CPU-only)."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -- Data paths --
BASE_TRAJ = "../03_Phase_Representational/outputs/logit_lens/base/analysis/03_prob_correct_trajectory.csv"
CIRCUIT_TRAJ = "../03_Phase_Representational/outputs/logit_lens/circuit/analysis/03c_circuit_prob_correct.csv"
CORE_TRAJ = "outputs/analysis/universal_core_logit_trajectory.csv"

OUT_PATH = "tmlr_draft_compressed/figures/targeted/T9_01_trajectory_overlay.png"

# -- Load data --
df_base = pd.read_csv(BASE_TRAJ)
df_circuit = pd.read_csv(CIRCUIT_TRAJ)
df_core = pd.read_csv(CORE_TRAJ)


# -- Average across draws and bands --
def avg_trajectory(df):
    return df.groupby(["model", "layer"])["mean_prob_correct"].mean().reset_index()


base_avg = avg_trajectory(df_base)
circuit_avg = avg_trajectory(df_circuit)
core_avg = avg_trajectory(df_core)

# -- Model order and layer counts --
MODEL_ORDER = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]
MODEL_LABELS = ["Pythia-70m", "Pythia-160m", "Pythia-410m", "Pythia-1b", "Pythia-1.4b"]
LAYER_COUNTS = {
    "pythia-70m": 6,
    "pythia-160m": 12,
    "pythia-410m": 24,
    "pythia-1b": 16,
    "pythia-1.4b": 24,
}


# -- Compute per-model correlation --
def pearson_r(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return float("nan")
    return np.corrcoef(x, y)[0, 1]


# -- Plot --
fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=True)

for idx, (model, label) in enumerate(zip(MODEL_ORDER, MODEL_LABELS)):
    ax = axes[idx]
    n_layers = LAYER_COUNTS[model]

    b = base_avg[base_avg["model"] == model].sort_values("layer")
    c = circuit_avg[circuit_avg["model"] == model].sort_values("layer")
    k = core_avg[core_avg["model"] == model].sort_values("layer")

    layers_b = b["layer"].values.astype(int)
    layers_c = c["layer"].values.astype(int)
    layers_k = k["layer"].values.astype(int)

    # Compute correlation between full circuit and universal core
    # Merge on layer
    merged = pd.merge(
        c[["layer", "mean_prob_correct"]],
        k[["layer", "mean_prob_correct"]],
        on="layer",
        suffixes=("_circuit", "_core"),
    )
    r = pearson_r(
        merged["mean_prob_correct_circuit"].values,
        merged["mean_prob_correct_core"].values,
    )

    ax.plot(
        layers_b,
        b["mean_prob_correct"].values,
        color="gray",
        linewidth=1.5,
        alpha=0.5,
        label="Base model",
    )
    ax.plot(
        layers_c,
        c["mean_prob_correct"].values,
        color="#2a7f2a",
        linewidth=2,
        label="Full circuit",
    )
    ax.plot(
        layers_k,
        k["mean_prob_correct"].values,
        color="#2a7f2a",
        linewidth=2,
        linestyle="--",
        label="Universal core",
    )

    ax.set_title(f"{label}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Layer", fontsize=9)

    # -- Fix x-axis: use sensible tick spacing --
    max_layer = max(layers_b.max(), layers_c.max(), layers_k.max())
    if max_layer <= 8:
        tick_step = 1
    elif max_layer <= 16:
        tick_step = 2
    else:
        tick_step = 4
    ticks = np.arange(0, max_layer + 1, tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks.astype(int), fontsize=8)

    # Add correlation annotation
    ax.text(
        0.95,
        0.05,
        f"r = {r:.2f}",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8
        ),
    )

    if idx == 0:
        ax.set_ylabel("Mean P(correct)", fontsize=10)
        ax.legend(fontsize=7, loc="upper left", framealpha=0.8)

    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.2)

plt.suptitle(
    "P(correct) Trajectory: Base Model vs Full Circuit vs Universal Core\n(averaged across all draws and bands)",
    fontsize=12,
    y=1.02,
)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved to {OUT_PATH}")
