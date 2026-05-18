#!/usr/bin/env python3
"""
Per-Example Disagreement Analysis Stratified by Token Frequency
================================================================
Tests whether the 7-17% disagreement between circuit pairs is distributed
uniformly across example difficulties or concentrated on harder
(lower-frequency) examples.

For each test band, we:
1. Load per-example predictions from all circuit pairs (same-band and cross-band)
2. Load the target token's log-frequency for each example
3. Compute disagreement rate per frequency quartile
4. Test correlation between token frequency and disagreement probability
5. Compare same-band vs cross-band disagreement patterns
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np

ISC_ROOT = PROJECT_ROOT
PER_EXAMPLE_DIR = ISC_ROOT / "LSC_circuits" / "per_example_eval"
DATA_DIR = ISC_ROOT / "LSC_data" / "datasets" / "matched"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

MODELS = [
    "pythia_160m",
    "pythia_410m",
    "pythia_1b",
    "pythia_1.4b",
]  # Skip 70m (low accuracy)
MODEL_NAMES = {
    "pythia_160m": "pythia-160m",
    "pythia_410m": "pythia-410m",
    "pythia_1b": "pythia-1b",
    "pythia_1.4b": "pythia-1.4b",
}
BANDS = ["low", "medium", "high", "very_high"]
DRAWS = ["draw_1", "draw_2", "draw_3"]


def load_per_example_preds(
    model_dir: str, train_band: str, draw: str, test_band: str
) -> list:
    """Load per-example predictions for a circuit evaluated on a test band."""
    path = PER_EXAMPLE_DIR / model_dir / train_band / draw / f"{test_band}.json"
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data["examples"]


def load_test_frequencies(draw: str, band: str) -> list:
    """Load target token log-frequencies for test examples."""
    path = DATA_DIR / draw / band / "test.json"
    with open(path) as f:
        data = json.load(f)
    return [ex["token_log_frequencies"]["target"] for ex in data["examples"]]


# ============================================================================
# ANALYSIS
# ============================================================================


def compute_pairwise_agreement(preds_a: list, preds_b: list) -> dict:
    """Compute per-example agreement between two circuit predictions."""
    n = min(len(preds_a), len(preds_b))
    agreements = []
    for i in range(n):
        a_correct = preds_a[i]["correct"]
        b_correct = preds_b[i]["correct"]
        agree = a_correct == b_correct
        agreements.append(
            {
                "example_idx": i,
                "agree": agree,
                "both_correct": a_correct and b_correct,
                "both_wrong": (not a_correct) and (not b_correct),
                "a_only": a_correct and (not b_correct),
                "b_only": (not a_correct) and b_correct,
            }
        )
    return agreements


def main():
    print("=" * 70)
    print("Per-Example Disagreement Analysis by Token Frequency")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for model_dir in MODELS:
        model_name = MODEL_NAMES[model_dir]
        print(f"\n{'=' * 60}")
        print(f"  {model_name}")
        print(f"{'=' * 60}")

        model_results = {"same_band": [], "cross_band": []}

        for test_band in BANDS:
            # Load frequencies for test examples
            freqs = load_test_frequencies("draw_1", test_band)
            n_examples = len(freqs)
            freq_arr = np.array(freqs)

            # Split into quartiles by frequency (within this band's test set)
            quartile_edges = np.percentile(freq_arr, [0, 25, 50, 75, 100])
            quartile_labels = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]

            def get_quartile(f):
                for q in range(3, -1, -1):
                    if f >= quartile_edges[q]:
                        return q
                return 0

            example_quartiles = [get_quartile(f) for f in freqs]

            # Collect all circuit predictions on this test band
            circuit_preds = {}  # (train_band, draw) -> predictions
            for train_band in BANDS:
                for draw in DRAWS:
                    preds = load_per_example_preds(
                        model_dir, train_band, draw, test_band
                    )
                    if preds is not None:
                        circuit_preds[(train_band, draw)] = preds

            if len(circuit_preds) < 2:
                continue

            # Compute pairwise agreement for same-band and cross-band pairs
            keys = list(circuit_preds.keys())

            for k1, k2 in combinations(keys, 2):
                train1, draw1 = k1
                train2, draw2 = k2

                is_same_band = train1 == train2
                pair_type = "same_band" if is_same_band else "cross_band"

                agreements = compute_pairwise_agreement(
                    circuit_preds[k1], circuit_preds[k2]
                )

                # Stratify by quartile
                quartile_disagree = defaultdict(list)
                for ag in agreements:
                    q = example_quartiles[ag["example_idx"]]
                    quartile_disagree[q].append(not ag["agree"])

                for q in range(4):
                    if q in quartile_disagree:
                        disagree_rate = np.mean(quartile_disagree[q])
                        model_results[pair_type].append(
                            {
                                "test_band": test_band,
                                "quartile": q,
                                "quartile_label": quartile_labels[q],
                                "disagree_rate": disagree_rate,
                                "n_examples": len(quartile_disagree[q]),
                                "freq_mean": np.mean(
                                    [
                                        freq_arr[i]
                                        for i, eq in enumerate(example_quartiles)
                                        if eq == q
                                    ]
                                ),
                            }
                        )

        # Aggregate across all pairs
        print(f"\n  Disagreement rate by within-band frequency quartile:")
        for pair_type in ["same_band", "cross_band"]:
            print(f"\n  {pair_type.replace('_', '-')} pairs:")
            print(
                f"    {'Quartile':15s} | {'Disagree %':>10s} | {'N pairs':>8s} | {'Mean freq':>10s}"
            )
            print(f"    {'-' * 15}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 10}")

            for q in range(4):
                q_data = [r for r in model_results[pair_type] if r["quartile"] == q]
                if q_data:
                    mean_disagree = np.mean([r["disagree_rate"] for r in q_data])
                    mean_freq = np.mean([r["freq_mean"] for r in q_data])
                    n_pairs = len(q_data)
                    print(
                        f"    {quartile_labels[q]:15s} | {mean_disagree:10.1%} | {n_pairs:8d} | {mean_freq:10.4f}"
                    )

        # Correlation: frequency vs disagreement
        # Pool all examples across all test bands and pair types
        all_freq_disagree = {
            "same_band": defaultdict(list),
            "cross_band": defaultdict(list),
        }

        for test_band in BANDS:
            freqs = load_test_frequencies("draw_1", test_band)
            freq_arr = np.array(freqs)

            keys = list(circuit_preds.keys()) if circuit_preds else []
            # Reload for this test band
            circuit_preds_tb = {}
            for train_band in BANDS:
                for draw in DRAWS:
                    preds = load_per_example_preds(
                        model_dir, train_band, draw, test_band
                    )
                    if preds is not None:
                        circuit_preds_tb[(train_band, draw)] = preds

            for k1, k2 in combinations(list(circuit_preds_tb.keys()), 2):
                train1, _ = k1
                train2, _ = k2
                is_same = train1 == train2
                pt = "same_band" if is_same else "cross_band"

                agreements = compute_pairwise_agreement(
                    circuit_preds_tb[k1], circuit_preds_tb[k2]
                )
                for ag in agreements:
                    idx = ag["example_idx"]
                    all_freq_disagree[pt][idx].append((freq_arr[idx], not ag["agree"]))

        # Compute per-example average disagreement rate and correlate with frequency
        from scipy import stats as scipy_stats

        print(f"\n  Correlation: token frequency vs disagreement rate")
        for pt in ["same_band", "cross_band"]:
            if not all_freq_disagree[pt]:
                continue
            freqs_pooled = []
            disagree_pooled = []
            for idx, pairs in all_freq_disagree[pt].items():
                mean_freq = np.mean([p[0] for p in pairs])
                mean_disagree = np.mean([p[1] for p in pairs])
                freqs_pooled.append(mean_freq)
                disagree_pooled.append(mean_disagree)

            if len(freqs_pooled) > 10:
                rho, p = scipy_stats.spearmanr(freqs_pooled, disagree_pooled)
                print(
                    f"    {pt.replace('_', '-'):12s}: rho={rho:+.3f}, p={p:.4f} (n={len(freqs_pooled)})"
                )

        all_results[model_name] = model_results

    # Cross-model summary
    print(f"\n\n{'=' * 70}")
    print("CROSS-MODEL SUMMARY: Same-band vs Cross-band Disagreement by Quartile")
    print(f"{'=' * 70}")

    quartile_labels = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]

    for pair_type in ["same_band", "cross_band"]:
        print(f"\n  {pair_type.replace('_', '-')} pairs:")
        print(
            f"    {'Model':15s} | {'Q1 low':>8s} | {'Q2':>8s} | {'Q3':>8s} | {'Q4 high':>8s} | {'Q4-Q1':>8s}"
        )
        print(
            f"    {'-' * 15}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}"
        )

        for model_dir in MODELS:
            model_name = MODEL_NAMES[model_dir]
            if model_name not in all_results:
                continue

            q_means = {}
            for q in range(4):
                q_data = [
                    r for r in all_results[model_name][pair_type] if r["quartile"] == q
                ]
                if q_data:
                    q_means[q] = np.mean([r["disagree_rate"] for r in q_data])

            if len(q_means) == 4:
                spread = q_means[3] - q_means[0]
                print(
                    f"    {model_name:15s} | {q_means[0]:8.1%} | {q_means[1]:8.1%} | "
                    f"{q_means[2]:8.1%} | {q_means[3]:8.1%} | {spread:+8.1%}"
                )

    # Save
    output_path = OUTPUT_DIR / "disagreement_by_frequency.json"
    # Convert to serializable
    save_data = {}
    for model_name, mr in all_results.items():
        save_data[model_name] = {pt: [dict(r) for r in rows] for pt, rows in mr.items()}
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
