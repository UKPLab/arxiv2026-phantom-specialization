"""Per-example agreement analysis: same-band vs cross-band circuit agreement."""

import json
import os
import numpy as np
from itertools import combinations
from collections import defaultdict

DATA = "LSC_circuits/per_example_eval"
MODELS = ["pythia_70m", "pythia_160m", "pythia_410m", "pythia_1b", "pythia_1.4b"]
BANDS = ["low", "medium", "high", "very_high", "control"]
DRAWS = ["draw_1", "draw_2", "draw_3"]


def load_predictions(model, train_band, draw, test_band):
    path = os.path.join(DATA, model, train_band, draw, f"{test_band}.json")
    with open(path) as f:
        data = json.load(f)
    return {e["example_idx"]: e["correct"] for e in data["examples"]}


def agreement_rate(preds_a, preds_b):
    assert set(preds_a.keys()) == set(preds_b.keys())
    agree = sum(1 for i in preds_a if preds_a[i] == preds_b[i])
    return agree / len(preds_a)


def cohens_kappa(preds_a, preds_b):
    n = len(preds_a)
    agree = sum(1 for i in preds_a if preds_a[i] == preds_b[i])
    p_o = agree / n
    # Marginal frequencies
    a_pos = sum(1 for v in preds_a.values() if v) / n
    b_pos = sum(1 for v in preds_b.values() if v) / n
    p_e = a_pos * b_pos + (1 - a_pos) * (1 - b_pos)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


print("=" * 90)
print("Per-Example Agreement: Same-Band vs Cross-Band Circuits")
print("=" * 90)

for model in MODELS:
    same_band_agreements = []
    cross_band_agreements = []
    same_band_kappas = []
    cross_band_kappas = []

    # For each test_band, compare all pairs of (train_band_A, draw_i) vs (train_band_B, draw_j)
    for test_band in BANDS:
        # Collect all circuit predictions on this test_band
        circuit_preds = {}  # (train_band, draw) -> {example_idx: correct}
        for train_band in BANDS:
            for draw in DRAWS:
                try:
                    preds = load_predictions(model, train_band, draw, test_band)
                    circuit_preds[(train_band, draw)] = preds
                except FileNotFoundError:
                    pass

        # Compare all pairs
        keys = list(circuit_preds.keys())
        for i, j in combinations(range(len(keys)), 2):
            tb_a, draw_a = keys[i]
            tb_b, draw_b = keys[j]
            pa = circuit_preds[keys[i]]
            pb = circuit_preds[keys[j]]

            agr = agreement_rate(pa, pb)
            kap = cohens_kappa(pa, pb)

            if tb_a == tb_b:
                same_band_agreements.append(agr)
                same_band_kappas.append(kap)
            else:
                cross_band_agreements.append(agr)
                cross_band_kappas.append(kap)

    model_name = model.replace("_", "-")
    s_agr = np.mean(same_band_agreements)
    c_agr = np.mean(cross_band_agreements)
    s_kap = np.mean(same_band_kappas)
    c_kap = np.mean(cross_band_kappas)

    print(f"\n{model_name}:")
    print(
        f"  Same-band pairs:  n={len(same_band_agreements):>4}, agreement={s_agr:.3f}, kappa={s_kap:.3f}"
    )
    print(
        f"  Cross-band pairs: n={len(cross_band_agreements):>4}, agreement={c_agr:.3f}, kappa={c_kap:.3f}"
    )
    print(
        f"  Difference:       agreement={s_agr - c_agr:+.3f}, kappa={s_kap - c_kap:+.3f}"
    )

    # Check if disagreements cluster on specific examples
    # For cross-band pairs on each test_band, count how often each example is disagreed upon
    for test_band in BANDS[:1]:  # Just show one example
        disagree_counts = defaultdict(int)
        total_pairs = 0
        for i, j in combinations(range(len(list(circuit_preds.keys()))), 2):
            pa = circuit_preds[list(circuit_preds.keys())[i]]
            pb = circuit_preds[list(circuit_preds.keys())[j]]
            total_pairs += 1
            for idx in pa:
                if pa[idx] != pb[idx]:
                    disagree_counts[idx] += 1

        if disagree_counts:
            counts = list(disagree_counts.values())
            n_disagreed = len(counts)
            print(
                f"  Disagreement distribution (test={test_band}): {n_disagreed}/225 examples ever disagreed"
            )
            print(
                f"    Mean disagree count: {np.mean(counts):.1f}, max: {max(counts)}, "
                f"top-10% threshold: {np.percentile(counts, 90):.0f}"
            )

# Overall summary
print("\n" + "=" * 90)
print("Summary")
print("=" * 90)
print(
    "Same-band = both circuits extracted from the same frequency band (different draws)"
)
print("Cross-band = circuits extracted from different frequency bands")
print(
    "High agreement + small same-cross gap confirms phantom specialization at per-example level"
)
