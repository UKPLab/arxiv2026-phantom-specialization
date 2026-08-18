"""Matched-band primary analysis for SVA (frozen spec, CPU-only).

Analysis plan frozen before recomputation (see SVA/README.md,
PRIMARY analysis):

PRIMARY (transfer): per model, mean own-band minus mean cross-band
  2-way accuracy gap over the three length-matched bands
  (low/medium/high), frozen test sets, resample ablation. Test: exact
  draw-blocked label permutation. Within each draw the three
  source-circuit labels are permuted against the fixed test bands
  (3! per draw, 6^3 = 216 relabelings, identity included); statistic =
  mean own-minus-cross gap; directional alternative (same-band
  greater); BH over the 5 models. The 9 band-x-draw cells are
  dependent (cells within a draw share circuits and test data), so a
  cell-level test treating them as independent is not valid as the
  primary test. The universal-core baseline cancels in the gap (the
  same u is subtracted from own and cross on each test band), so the
  permutation runs on raw transfer accuracies.

SECONDARY ("LSC-protocol" readout): cell-level Wilcoxon signed-rank
  over the 9 matched cells via exact sign-flip enumeration (2^9) with
  midranks for tied |diff|, zeros dropped; directional; BH over the 5
  models. Reported for comparability with the LSC protocol, with the
  dependence caveat.

MATCHED TE: transfer efficiency relative to the matched three-band
  universal core (sva_matched_core_eval.py output), matched cells only.

CANONICAL five-condition Wilcoxon: the saved sva_phantom_summary_*
  wilcoxon_p used scipy.stats.wilcoxon method resolution, which does
  not properly midrank ties (1.4b: saved 0.020424, scipy "exact"
  0.020630, true exact-midrank 0.019867). Recomputed here by exact
  sign-flip enumeration (2^15) with midranks on the identical
  five-condition diffs; five-condition results are SENSITIVITY, no new
  inferential claims.

STRUCTURAL (matched scope): between-band vs same-band-across-draws
  Jaccard restricted to low/medium/high; five-condition version kept
  as sensitivity. Descriptive (means), matching the LSC readout.

Output: results/sva_matched_stats.json
"""

import json
import sys
from itertools import permutations, product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sva_stats_util import exact_wilcoxon_greater  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
CIRCUITS = BASE / "circuits"
RESULTS = BASE / "results"

MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b",
          "pythia-1.4b"]
BANDS = ["low", "medium", "high", "very_high", "control"]
MATCHED = ["low", "medium", "high"]
DRAWS = ["draw_1", "draw_2", "draw_3"]


N_TEST = 225


def load_transfer(mname):
    """T[draw][source_band][test_band] = 2-way accuracy on frozen test set,
    plus C[draw][source_band][test_band] = INTEGER success count out of 225.

    Accuracies are integer successes stored as float32 means; ranking or
    comparing the raw floats splits mathematical ties (float error up to
    ~5e-8), so all exact tests below run on the reconstructed integer
    counts.
    """
    T, C = {}, {}
    for dr in DRAWS:
        T[dr], C[dr] = {}, {}
        for sb in BANDS:
            m = json.load(open(CIRCUITS / mname / sb / dr / "metrics.json"))
            T[dr][sb] = {tb: m["transfer"][tb]["acc_2way"] for tb in BANDS}
            C[dr][sb] = {}
            for tb in BANDS:
                raw = m["transfer"][tb]["acc_2way"] * N_TEST
                cnt = round(raw)
                assert abs(raw - cnt) < 1e-3, (mname, sb, dr, tb, raw)
                C[dr][sb][tb] = cnt
    return T, C


def gap_from_counts(C, bands):
    """Mean own-minus-cross gap in accuracy units, from integer counts."""
    k = len(bands) - 1
    vals = []
    for dr in DRAWS:
        for tb in bands:
            colsum = sum(C[dr][sb][tb] for sb in bands)
            vals.append((C[dr][tb][tb] - (colsum - C[dr][tb][tb]) / k)
                        / N_TEST)
    return float(np.mean(vals))


def blocked_permutation(C):
    """Exact 216-relabeling directional p for the matched-band gap.

    Because the per-cell column sum over sources is invariant under the
    relabeling, the gap statistic is a monotone function of the sum of
    'own'-cell counts; the test therefore compares that INTEGER sum, so
    mathematically tied relabelings are exactly tied."""
    def own_sum(assign):
        return sum(C[dr][assign[dr][tb]][tb]
                   for dr in DRAWS for tb in MATCHED)

    identity = {dr: {tb: tb for tb in MATCHED} for dr in DRAWS}
    obs = own_sum(identity)
    perms = list(permutations(MATCHED))
    hits = total = 0
    for combo in product(perms, repeat=len(DRAWS)):
        assign = {dr: dict(zip(MATCHED, p)) for dr, p in zip(DRAWS, combo)}
        total += 1
        if own_sum(assign) >= obs:
            hits += 1
    assert total == 216
    return gap_from_counts(C, MATCHED), hits / total


def bh(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 1.0
    for rank_from_end, idx in enumerate(order[::-1]):
        rank = m - rank_from_end
        running = min(running, p[idx] * m / rank)
        adj[idx] = running
    return adj.tolist()


def cell_diffs_int(C, bands):
    """Per-cell own-minus-mean-cross diffs, scaled by (len(bands)-1) so
    they are exact INTEGERS: d = k*own - (colsum - own) = (k+1)*own -
    colsum. Monotone in the accuracy diff, so ranks and signs are
    identical while ties are exact."""
    k = len(bands) - 1
    out = []
    for dr in DRAWS:
        for tb in bands:
            colsum = sum(C[dr][sb][tb] for sb in bands)
            out.append((k + 1) * C[dr][tb][tb] - colsum)
    return out


def main():
    summary = {"primary_blocked_permutation": [], "secondary_wilcoxon_cells": [],
               "matched_te": [], "canonical_five_condition_wilcoxon": [],
               "structural_jaccard": {}}

    p_primary, p_secondary = [], []
    for mname in MODELS:
        T, C = load_transfer(mname)
        obs, p_blk = blocked_permutation(C)
        per_draw = []
        for dr in DRAWS:
            vals = []
            for tb in MATCHED:
                colsum = sum(C[dr][sb][tb] for sb in MATCHED)
                vals.append((C[dr][tb][tb]
                             - (colsum - C[dr][tb][tb]) / 2) / N_TEST)
            per_draw.append(float(np.mean(vals)))
        summary["primary_blocked_permutation"].append(
            {"model": mname, "gap": obs, "gap_per_draw": per_draw,
             "p_exact_216": p_blk})
        p_primary.append(p_blk)

        diffs9 = cell_diffs_int(C, MATCHED)
        p_w9 = exact_wilcoxon_greater(diffs9)
        summary["secondary_wilcoxon_cells"].append(
            {"model": mname, "n_cells": len(diffs9), "mean_diff":
             float(np.mean(diffs9) / (2 * N_TEST)),
             "p_exact_signflip_midrank": p_w9})
        p_secondary.append(p_w9)

        # matched TE (needs the matched-core forward evaluation)
        f = RESULTS / f"sva_universal_matched_{mname}.csv"
        if f.exists():
            u = pd.read_csv(f).set_index(["draw", "test_band"])
            same, cross = [], []
            for dr in DRAWS:
                for tb in MATCHED:
                    ub = u.loc[(dr, tb), "universal_2way"]
                    same.append(T[dr][tb][tb] - ub)
                    cross.append(np.mean([T[dr][sb][tb] for sb in MATCHED
                                          if sb != tb]) - ub)
            same, cross = np.array(same), np.array(cross)
            te = float(cross.mean() / same.mean()) if same.mean() > 0 else np.nan
            pooled = np.sqrt((np.var(same) + np.var(cross)) / 2)
            d = float((same.mean() - cross.mean()) / pooled) if pooled > 0 else 0.0
            circ_edges = [json.load(open(CIRCUITS / mname / sb / dr
                                         / "metrics.json"))["n_edges"]
                          for sb in MATCHED for dr in DRAWS]
            summary["matched_te"].append(
                {"model": mname, "same_band_boost_mean": float(same.mean()),
                 "cross_band_boost_mean": float(cross.mean()),
                 "transfer_efficiency": te, "cohens_d": d,
                 "mean_core_edges": float(pd.read_csv(f).n_edges.mean()),
                 "mean_matched_circuit_edges": float(np.mean(circ_edges))})
        else:
            summary["matched_te"].append(
                {"model": mname, "status": "matched core CSV missing"})

        # canonical five-condition Wilcoxon (sensitivity; same sign/rank
        # structure as sva_aggregate section 4 -- the universal core
        # cancels in the diff and the integer scaling is monotone)
        diffs15 = cell_diffs_int(C, BANDS)
        legacy = pd.read_csv(
            RESULTS / f"sva_phantom_summary_{mname}.csv").wilcoxon_p.iloc[0]
        summary["canonical_five_condition_wilcoxon"].append(
            {"model": mname, "n_cells": len(diffs15),
             "p_exact_signflip_midrank": exact_wilcoxon_greater(diffs15),
             "p_legacy_scipy": float(legacy)})

    for entry, q in zip(summary["primary_blocked_permutation"], bh(p_primary)):
        entry["q_bh"] = q
    for entry, q in zip(summary["secondary_wilcoxon_cells"], bh(p_secondary)):
        entry["q_bh"] = q

    # structural Jaccard, matched scope + five-condition sensitivity
    matched_pairs = {f"{a}|{b}" for i, a in enumerate(MATCHED)
                     for b in MATCHED[i + 1:]}
    for mname in MODELS:
        jac = pd.read_csv(RESULTS / f"sva_jaccard_{mname}.csv")
        bet = jac[jac.kind == "between_bands"]
        sam = jac[jac.kind == "same_band_across_draws"]
        bet_m = bet[bet.pair.isin(matched_pairs)]
        sam_m = sam[sam.pair.isin(MATCHED)]
        summary["structural_jaccard"][mname] = {
            "matched_between_mean": float(bet_m.jaccard.mean()),
            "matched_within_mean": float(sam_m.jaccard.mean()),
            "matched_n": [len(bet_m), len(sam_m)],
            "all5_between_mean": float(bet.jaccard.mean()),
            "all5_within_mean": float(sam.jaccard.mean())}

    json.dump(summary, open(RESULTS / "sva_matched_stats.json", "w"), indent=1)

    for e in summary["primary_blocked_permutation"]:
        print(f"PRIMARY  {e['model']:12s} gap {e['gap']:+.4f} "
              f"per-draw {'/'.join(f'{g:+.4f}' for g in e['gap_per_draw'])} "
              f"p {e['p_exact_216']:.5f} q {e['q_bh']:.5f}", flush=True)
    for e in summary["secondary_wilcoxon_cells"]:
        print(f"SECOND   {e['model']:12s} mean_diff {e['mean_diff']:+.4f} "
              f"p {e['p_exact_signflip_midrank']:.5f} q {e['q_bh']:.5f}",
              flush=True)
    for e in summary["matched_te"]:
        if "transfer_efficiency" in e:
            print(f"TE_MATCH {e['model']:12s} TE {e['transfer_efficiency']:.4f} "
                  f"d {e['cohens_d']:.3f} core {e['mean_core_edges']:.0f} edges",
                  flush=True)
    for e in summary["canonical_five_condition_wilcoxon"]:
        print(f"WILC15   {e['model']:12s} exact "
              f"{e['p_exact_signflip_midrank']:.6f} "
              f"legacy {e['p_legacy_scipy']:.6f}", flush=True)
    for m, e in summary["structural_jaccard"].items():
        print(f"JACC     {m:12s} matched between {e['matched_between_mean']:.3f} "
              f"vs within {e['matched_within_mean']:.3f} "
              f"(all5: {e['all5_between_mean']:.3f}/{e['all5_within_mean']:.3f})",
              flush=True)
    print("saved sva_matched_stats.json", flush=True)


if __name__ == "__main__":
    main()
