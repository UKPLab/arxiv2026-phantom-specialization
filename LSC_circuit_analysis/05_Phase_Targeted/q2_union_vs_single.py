"""Q2: Compare majority-vote (union) circuits vs single-draw circuits for cross-band transfer."""

import pandas as pd
import numpy as np

DATA = "LSC_circuit_analysis/05_Phase_Targeted/outputs/analysis"

# Load data
union = pd.read_csv(f"{DATA}/draw_union_eval.csv")
cross = pd.read_csv(f"{DATA}/cross_band_eval_results.csv")

# Filter single-draw circuits (real, not random)
single = cross[
    (cross["circuit_type"] == "cross_band") & (cross["k_sample"] == 0)
].copy()

# Mean accuracy per (model, source_band, test_band) across 3 draws
single_mean = (
    single.groupby(["model", "source_band", "test_band"])
    .agg(accuracy=("accuracy", "mean"), n_edges=("n_edges", "mean"))
    .reset_index()
)

# Merge with union
merged = union.merge(
    single_mean,
    on=["model", "source_band", "test_band"],
    suffixes=("_union", "_single"),
)

# Compute delta
merged["delta"] = merged["accuracy_union"] - merged["accuracy_single"]
merged["is_same_band"] = merged["source_band"] == merged["test_band"]
merged["size_ratio"] = merged["n_edges_union"] / merged["n_edges_single"]

# Summary per model
print("=" * 80)
print("Q2: Union (majority-vote) vs Single-Draw Circuit Accuracy")
print("=" * 80)

for model in sorted(merged["model"].unique()):
    m = merged[merged["model"] == model]
    same = m[m["is_same_band"]]
    cross_band = m[~m["is_same_band"]]

    print(f"\n{model}:")
    print(f"  Size ratio (union/single): {m['size_ratio'].mean():.2f}x")
    print(f"  Overall delta:    {m['delta'].mean():+.4f} (std {m['delta'].std():.4f})")
    print(f"  Same-band delta:  {same['delta'].mean():+.4f}")
    print(f"  Cross-band delta: {cross_band['delta'].mean():+.4f}")

    # Transfer efficiency: mean(cross-band acc) / same-band acc, per source_band
    for src in sorted(m["source_band"].unique()):
        ms = m[m["source_band"] == src]
        same_acc_u = ms[ms["is_same_band"]]["accuracy_union"].values[0]
        same_acc_s = ms[ms["is_same_band"]]["accuracy_single"].values[0]
        cross_acc_u = ms[~ms["is_same_band"]]["accuracy_union"].mean()
        cross_acc_s = ms[~ms["is_same_band"]]["accuracy_single"].mean()
        te_u = cross_acc_u / same_acc_u if same_acc_u > 0 else float("nan")
        te_s = cross_acc_s / same_acc_s if same_acc_s > 0 else float("nan")
        # skip printing per-source detail

# Aggregate summary
print("\n" + "=" * 80)
print("Aggregate Summary (excluding 70m)")
print("=" * 80)
big = merged[~merged["model"].str.contains("70m")]
same = big[big["is_same_band"]]
cross_band = big[~big["is_same_band"]]
print(
    f"  Mean delta (all):        {big['delta'].mean():+.4f} +/- {big['delta'].std():.4f}"
)
print(f"  Mean delta (same-band):  {same['delta'].mean():+.4f}")
print(f"  Mean delta (cross-band): {cross_band['delta'].mean():+.4f}")
print(f"  Mean size ratio:         {big['size_ratio'].mean():.2f}x")

# Transfer efficiency comparison
print("\n" + "=" * 80)
print("Transfer Efficiency: Union vs Single-Draw (per model)")
print("=" * 80)
for model in sorted(merged["model"].unique()):
    m = merged[merged["model"] == model]
    # Per source_band, compute TE = mean(cross-band acc) / same-band acc
    tes_u, tes_s = [], []
    for src in m["source_band"].unique():
        ms = m[m["source_band"] == src]
        same_u = ms[ms["is_same_band"]]["accuracy_union"].values[0]
        same_s = ms[ms["is_same_band"]]["accuracy_single"].values[0]
        cross_u = ms[~ms["is_same_band"]]["accuracy_union"].mean()
        cross_s = ms[~ms["is_same_band"]]["accuracy_single"].mean()
        if same_u > 0:
            tes_u.append(cross_u / same_u)
        if same_s > 0:
            tes_s.append(cross_s / same_s)
    print(
        f"  {model}: union TE={np.mean(tes_u):.3f}, single TE={np.mean(tes_s):.3f}, diff={np.mean(tes_u) - np.mean(tes_s):+.3f}"
    )
