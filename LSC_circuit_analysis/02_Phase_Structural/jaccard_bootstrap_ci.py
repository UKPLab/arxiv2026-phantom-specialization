"""
Compute bootstrap confidence intervals for the within-band vs between-band
Jaccard gap, per model. Outputs a CSV for inclusion in the paper.

No GPU required: runs on CPU using pre-extracted circuit edge sets.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]
BANDS = ["low", "medium", "high", "very_high", "control"]


def get_edge_set(circuit):
    return set(circuit.get("edge_list", []))


def compute_jaccard(set1, set2):
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


N_BOOTSTRAP = 10_000
SEED = 42
CI_LEVEL = 0.95
CIRCUITS_JSON = Path(__file__).parent / "outputs/extraction/all_circuits_structure.json"
OUTPUT_DIR = Path(__file__).parent / "outputs/analysis"


print("Loading circuits...")
with open(CIRCUITS_JSON) as f:
    data = json.load(f)
circuits = data["circuits"]
print(f"  Loaded {len(circuits)} circuits")


# --------------------------------------------------------------------------
# Compute raw pairwise Jaccard values per model
# --------------------------------------------------------------------------
def get_raw_within_between(circuits, model, bands=BANDS):
    """Return lists of individual pairwise Jaccard values."""
    band_draw_edges = defaultdict(dict)
    for c in circuits.values():
        if c.get("status") != "success":
            continue
        if c["model"] == model and c["band"] in bands:
            band_draw_edges[c["band"]][c["draw"]] = get_edge_set(c)

    within = []
    between = []
    band_list = [b for b in bands if b in band_draw_edges]

    for i, band1 in enumerate(band_list):
        for j, band2 in enumerate(band_list):
            for draw1, edges1 in band_draw_edges[band1].items():
                for draw2, edges2 in band_draw_edges[band2].items():
                    if band1 == band2 and draw1 == draw2:
                        continue
                    jac = compute_jaccard(edges1, edges2)
                    if band1 == band2:
                        within.append(jac)
                    elif i < j:
                        between.append(jac)

    return np.array(within), np.array(between)


# --------------------------------------------------------------------------
# Bootstrap the gap
# --------------------------------------------------------------------------
def bootstrap_gap_ci(within, between, n_bootstrap=N_BOOTSTRAP, seed=SEED, ci=CI_LEVEL):
    """Bootstrap CI for gap = mean(within) - mean(between)."""
    rng = np.random.default_rng(seed)
    n_w, n_b = len(within), len(between)

    gaps = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        w_sample = within[rng.integers(0, n_w, size=n_w)]
        b_sample = between[rng.integers(0, n_b, size=n_b)]
        gaps[i] = w_sample.mean() - b_sample.mean()

    alpha = (1 - ci) / 2
    lo, hi = np.percentile(gaps, [100 * alpha, 100 * (1 - alpha)])
    return gaps.mean(), lo, hi


rows = []
for model in MODELS:
    print(f"\n{model}:")
    within, between = get_raw_within_between(circuits, model)
    print(f"  n_within={len(within)}, n_between={len(between)}")

    gap_obs = within.mean() - between.mean()
    gap_mean, gap_lo, gap_hi = bootstrap_gap_ci(within, between)

    # Also bootstrap CIs for the individual means
    rng = np.random.default_rng(SEED)
    within_means = [
        within[rng.integers(0, len(within), size=len(within))].mean()
        for _ in range(N_BOOTSTRAP)
    ]
    rng = np.random.default_rng(SEED + 1)
    between_means = [
        between[rng.integers(0, len(between), size=len(between))].mean()
        for _ in range(N_BOOTSTRAP)
    ]

    alpha = (1 - CI_LEVEL) / 2
    w_lo, w_hi = np.percentile(within_means, [100 * alpha, 100 * (1 - alpha)])
    b_lo, b_hi = np.percentile(between_means, [100 * alpha, 100 * (1 - alpha)])

    row = {
        "model": model,
        "within_mean": within.mean(),
        "within_ci_lo": w_lo,
        "within_ci_hi": w_hi,
        "between_mean": between.mean(),
        "between_ci_lo": b_lo,
        "between_ci_hi": b_hi,
        "gap": gap_obs,
        "gap_bootstrap_mean": gap_mean,
        "gap_ci_lo": gap_lo,
        "gap_ci_hi": gap_hi,
        "n_within": len(within),
        "n_between": len(between),
    }
    rows.append(row)

    print(f"  within={within.mean():.4f} [{w_lo:.4f}, {w_hi:.4f}]")
    print(f"  between={between.mean():.4f} [{b_lo:.4f}, {b_hi:.4f}]")
    print(f"  gap={gap_obs:.4f} [{gap_lo:.4f}, {gap_hi:.4f}]")

df = pd.DataFrame(rows)
outpath = OUTPUT_DIR / "jaccard_bootstrap_ci.csv"
df.to_csv(outpath, index=False)
print(f"\nSaved to {outpath}")
