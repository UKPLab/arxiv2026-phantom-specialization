#!/usr/bin/env python3
"""
Analyze k>=3 edge composition per band for Pythia-1.4b (and all models).

Verifies that the majority-shared core (k>=3 edges) is band-agnostic in
terms of which specific edges are included, not merely that it recovers
accuracy.

Analyses:
1. Per-band k>=3 edge sets: which k>=3 edges appear in each band's circuit?
2. Jaccard similarity of per-band k>=3 edge sets (within-band vs between-band)
3. Per-tier (k=1..5) composition breakdown per band
4. Stability: what fraction of k>=3 edges appear in ALL bands vs only 3-4?

"""

import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations

import numpy as np
import torch as t

ISC_ROOT = Path(__file__).resolve().parent.parent.parent
CIRCUIT_DIR = ISC_ROOT / "LSC_circuits" / "circuit_discovery" / "circuits"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

BANDS = ["low", "medium", "high", "very_high", "control"]
DRAWS = ["draw_1", "draw_2", "draw_3"]
MODELS = ["pythia_70m", "pythia_160m", "pythia_410m", "pythia_1b", "pythia_1.4b"]


# ============================================================================
# EDGE EXTRACTION
# ============================================================================


def extract_edge_set(prune_scores: dict) -> set:
    """Extract set of edge identifiers from prune_scores."""
    edges = set()
    for name, scores in prune_scores.items():
        circuit_mask = t.isinf(scores) & (scores > 0)
        for idx in circuit_mask.nonzero(as_tuple=False):
            edge_id = f"{name}[{','.join(str(i.item()) for i in idx)}]"
            edges.add(edge_id)
    return edges


def load_circuit_edges(model_dir: str, band: str, draw: str) -> set:
    """Load prune_scores and extract edge set."""
    path = CIRCUIT_DIR / model_dir / band / draw / "prune_scores.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        scores = pickle.load(f)
    return extract_edge_set(scores)


# ============================================================================
# SHARING ANALYSIS
# ============================================================================


def compute_edge_sharing(band_edges: dict) -> dict:
    """
    For each edge in the union, count how many bands it appears in.
    Returns: {edge_id: count}
    """
    counts = Counter()
    for band, edges in band_edges.items():
        for e in edges:
            counts[e] += 1
    return counts


def get_edges_at_k(edge_counts: dict, k: int) -> set:
    """Get edges appearing in exactly k bands."""
    return {e for e, c in edge_counts.items() if c == k}


def get_edges_gte_k(edge_counts: dict, k: int) -> set:
    """Get edges appearing in >=k bands."""
    return {e for e, c in edge_counts.items() if c >= k}


def compute_jaccard(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


# ============================================================================
# MAIN ANALYSIS
# ============================================================================


def analyze_model(model_dir: str, model_name: str):
    """Run full k>=3 composition analysis for one model."""
    print(f"\n{'=' * 70}")
    print(f"  {model_name}")
    print(f"{'=' * 70}")

    results = {"model": model_name, "draws": {}}

    for draw in DRAWS:
        print(f"\n  --- {draw} ---")

        # Load all band circuits for this draw
        band_edges = {}
        for band in BANDS:
            edges = load_circuit_edges(model_dir, band, draw)
            if edges is None:
                print(f"    WARNING: Missing {band}/{draw}")
                continue
            band_edges[band] = edges

        if len(band_edges) != len(BANDS):
            print(f"    Skipping {draw}: only {len(band_edges)}/{len(BANDS)} bands")
            continue

        # Edge sharing counts
        edge_counts = compute_edge_sharing(band_edges)
        all_edges = set(edge_counts.keys())

        # Sharing spectrum
        spectrum = {}
        for k in range(1, 6):
            edges_at_k = get_edges_at_k(edge_counts, k)
            spectrum[k] = len(edges_at_k)
        print(f"    Sharing spectrum: {spectrum}")
        print(f"    Total unique edges: {len(all_edges)}")

        # k>=3 edges (majority-shared core)
        k3_edges = get_edges_gte_k(edge_counts, 3)
        k5_edges = get_edges_gte_k(edge_counts, 5)  # universal
        print(
            f"    k>=3 edges: {len(k3_edges)} ({len(k3_edges) / len(all_edges):.1%} of union)"
        )
        print(
            f"    k=5 edges: {len(k5_edges)} ({len(k5_edges) / len(all_edges):.1%} of union)"
        )

        # ---- ANALYSIS 1: Per-band k>=3 edge sets ----
        # For each band, which k>=3 edges appear in that band's circuit?
        per_band_k3 = {}
        for band in BANDS:
            band_k3 = band_edges[band] & k3_edges
            per_band_k3[band] = band_k3
            coverage = len(band_k3) / len(k3_edges) if k3_edges else 0
            band_frac = len(band_k3) / len(band_edges[band]) if band_edges[band] else 0
            print(
                f"    {band:12s}: {len(band_k3):5d} k>=3 edges "
                f"({coverage:.1%} of k>=3 pool, {band_frac:.1%} of band circuit)"
            )

        # ---- ANALYSIS 2: Jaccard of per-band k>=3 sets ----
        print(f"\n    Pairwise Jaccard of per-band k>=3 edge sets:")
        within_jaccards = []
        between_jaccards = []
        pair_results = {}

        for b1, b2 in combinations(BANDS, 2):
            j = compute_jaccard(per_band_k3[b1], per_band_k3[b2])
            pair_results[f"{b1}-{b2}"] = j
            between_jaccards.append(j)  # all pairs are "between-band" for k>=3
            print(f"      {b1:12s} vs {b2:12s}: {j:.4f}")

        mean_j = np.mean(between_jaccards)
        std_j = np.std(between_jaccards)
        min_j = np.min(between_jaccards)
        max_j = np.max(between_jaccards)
        print(
            f"    Mean Jaccard: {mean_j:.4f} +/- {std_j:.4f} (range: {min_j:.4f}-{max_j:.4f})"
        )

        # ---- ANALYSIS 3: Per-tier composition per band ----
        print(f"\n    Per-tier composition (fraction of each band's circuit):")
        print(
            f"    {'Band':12s} | {'k=1':>6s} | {'k=2':>6s} | {'k=3':>6s} | {'k=4':>6s} | {'k=5':>6s} | {'Total':>6s}"
        )
        print(
            f"    {'-' * 12}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 6}-+-{'-' * 6}"
        )

        tier_composition = {}
        for band in BANDS:
            band_total = len(band_edges[band])
            tier_counts = {}
            for k in range(1, 6):
                edges_at_k = get_edges_at_k(edge_counts, k)
                band_at_k = band_edges[band] & edges_at_k
                tier_counts[k] = len(band_at_k)

            fracs = {
                k: tier_counts[k] / band_total if band_total else 0 for k in range(1, 6)
            }
            tier_composition[band] = fracs
            print(
                f"    {band:12s} | {fracs[1]:6.1%} | {fracs[2]:6.1%} | {fracs[3]:6.1%} | "
                f"{fracs[4]:6.1%} | {fracs[5]:6.1%} | {band_total:6d}"
            )

        # ---- ANALYSIS 4: k>=3 stability across bands ----
        # Among k>=3 edges, what fraction appears in ALL 5 bands vs only 3 or 4?
        k3_by_count = {3: 0, 4: 0, 5: 0}
        for e in k3_edges:
            c = edge_counts[e]
            k3_by_count[c] += 1

        print(f"\n    k>=3 breakdown:")
        for k in [3, 4, 5]:
            frac = k3_by_count[k] / len(k3_edges) if k3_edges else 0
            print(f"      k={k}: {k3_by_count[k]:5d} edges ({frac:.1%} of k>=3)")

        # ---- ANALYSIS 5: Symmetric vs asymmetric sharing ----
        # For k=3 and k=4 edges, are specific bands systematically over/under-represented?
        print(f"\n    Band membership of k=3 edges (which 3 bands?):")
        k3_exact = get_edges_at_k(edge_counts, 3)
        band_membership_k3 = {band: 0 for band in BANDS}
        for e in k3_exact:
            for band in BANDS:
                if e in band_edges[band]:
                    band_membership_k3[band] += 1

        for band in BANDS:
            frac = band_membership_k3[band] / (len(k3_exact) * 3 / 5) if k3_exact else 0
            expected = (
                len(k3_exact) * 3 / 5
            )  # each k=3 edge in 3 of 5 bands -> expected per band
            print(
                f"      {band:12s}: {band_membership_k3[band]:5d} "
                f"(expected if uniform: {expected:.0f}, ratio: {frac:.2f})"
            )

        # Store draw results
        results["draws"][draw] = {
            "spectrum": spectrum,
            "n_total": len(all_edges),
            "n_k3": len(k3_edges),
            "n_k5": len(k5_edges),
            "per_band_k3_sizes": {b: len(s) for b, s in per_band_k3.items()},
            "pairwise_jaccard_k3": pair_results,
            "mean_jaccard_k3": float(mean_j),
            "std_jaccard_k3": float(std_j),
            "tier_composition": {
                b: {str(k): float(v) for k, v in fracs.items()}
                for b, fracs in tier_composition.items()
            },
            "k3_breakdown": k3_by_count,
            "band_membership_k3_exact": band_membership_k3,
        }

    return results


def main():
    print("=" * 70)
    print("k>=3 Edge Composition Analysis")
    print("Tests whether the majority-shared core is band-agnostic.")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for model_dir in MODELS:
        model_name = model_dir.replace("_", "-")
        results = analyze_model(model_dir, model_name)
        all_results[model_name] = results

    # ---- Cross-model summary ----
    print(f"\n\n{'=' * 70}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'=' * 70}")

    print(
        f"\n{'Model':15s} | {'k>=3 Jaccard':>12s} | {'k=5 frac':>10s} | {'k>=3 frac':>10s} | {'k=3 spread':>10s}"
    )
    print(f"{'-' * 15}-+-{'-' * 12}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 10}")

    for model_name, results in all_results.items():
        # Average across draws
        jaccards = []
        k5_fracs = []
        k3_fracs = []
        spreads = []

        for draw, dr in results["draws"].items():
            jaccards.append(dr["mean_jaccard_k3"])
            k5_fracs.append(dr["n_k5"] / dr["n_total"] if dr["n_total"] else 0)
            k3_fracs.append(dr["n_k3"] / dr["n_total"] if dr["n_total"] else 0)
            # Spread: std of per-band k>=3 sizes
            sizes = list(dr["per_band_k3_sizes"].values())
            spreads.append(np.std(sizes) / np.mean(sizes) if sizes else 0)

        print(
            f"{model_name:15s} | {np.mean(jaccards):12.4f} | {np.mean(k5_fracs):10.1%} | "
            f"{np.mean(k3_fracs):10.1%} | {np.mean(spreads):10.4f}"
        )

    # Save
    output_path = OUTPUT_DIR / "k3_composition_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
