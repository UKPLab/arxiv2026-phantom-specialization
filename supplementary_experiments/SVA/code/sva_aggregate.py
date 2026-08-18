"""SVA phantom readout, mirroring NB12 Parts A+C and cell 16.

Inputs: circuits/{model}/{band}/{draw}/ (prune_scores.pkl + metrics.json with
the full 5-band transfer already computed by sva_discovery.py).

1. Transfer matrix (mean over draws) from metrics.json          [CPU]
2. Structural overlap: Jaccard between band circuits per draw,
   and same-band across draws                                    [CPU]
3. Universal core = AND of the 5 band masks per draw (NB12 cell 5),
   evaluated on every band's test set                            [GPU]
4. Boost = circuit_2way - universal_2way per (draw, test_band);
   same-band vs cross-band -> TE, Cohen's d, Wilcoxon (cell 16)  [CPU]

Boosts use 2-way forced choice (the SVA task metric; LSC used top-1).

Wilcoxon method: exact sign-flip enumeration with midranks
(sva_stats_util.exact_wilcoxon_greater). The p stored in the summary CSVs
comes from scipy's tie handling instead (1.4b: 0.020424 stored vs 0.019867
exact); canonical values live in results/sva_matched_stats.json
(canonical_five_condition_wilcoxon).
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "code"))

from sva_discovery import (  # noqa: E402
    BANDS, DRAWS, base_logits_for, get_patchable, load_split,
    metrics_from_logits, run_circuit,
)
from sva_stats_util import exact_wilcoxon_greater  # noqa: E402

CIRCUITS = BASE / "circuits"
RESULTS = BASE / "results"


def build_universal(per_band):
    """AND of the bands' inf masks (NB12 cell 5 / mean_ablation build_universal)."""
    first = BANDS[0]
    out = {}
    for mod in per_band[first]:
        mask = t.ones_like(per_band[first][mod], dtype=t.bool)
        for b in BANDS:
            mask &= t.isinf(per_band[b][mod])
        tensor = t.zeros_like(per_band[first][mod])
        tensor[mask] = float("inf")
        out[mod] = tensor
    return out


def edge_set(scores):
    return {(m, *idx.tolist()) for m, v in scores.items()
            for idx in t.isinf(v).nonzero()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-160m")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    mname = args.model

    # ---------- load circuits ----------
    scores = {}   # (band, draw) -> prune_scores
    metrics = {}  # (band, draw) -> metrics.json
    for band in BANDS:
        for draw in DRAWS:
            d = CIRCUITS / mname / band / draw
            with open(d / "prune_scores.pkl", "rb") as f:
                scores[(band, draw)] = pickle.load(f)
            metrics[(band, draw)] = json.load(open(d / "metrics.json"))

    # ---------- 1. transfer matrix (mean over draws) ----------
    mat = pd.DataFrame(
        {tb: [np.mean([metrics[(sb, dr)]["transfer"][tb]["acc_2way"]
                       for dr in DRAWS]) for sb in BANDS] for tb in BANDS},
        index=BANDS)
    mat.index.name = "source_band"
    mat.to_csv(RESULTS / f"sva_transfer_matrix_{mname}.csv")
    print(f"=== {mname} transfer matrix (2-way, mean of 3 draws) ===")
    print(mat.round(3).to_string())

    # ---------- 2. structural overlap ----------
    sets = {k: edge_set(v) for k, v in scores.items()}
    rows = []
    for dr in DRAWS:                       # between bands, within draw
        for i, b1 in enumerate(BANDS):
            for b2 in BANDS[i + 1:]:
                a, b = sets[(b1, dr)], sets[(b2, dr)]
                rows.append({"kind": "between_bands", "draw": dr,
                             "pair": f"{b1}|{b2}",
                             "jaccard": len(a & b) / len(a | b),
                             "n_a": len(a), "n_b": len(b)})
    for band in BANDS:                     # same band, between draws
        for i, d1 in enumerate(DRAWS):
            for d2 in DRAWS[i + 1:]:
                a, b = sets[(band, d1)], sets[(band, d2)]
                rows.append({"kind": "same_band_across_draws", "draw": f"{d1}|{d2}",
                             "pair": band,
                             "jaccard": len(a & b) / len(a | b),
                             "n_a": len(a), "n_b": len(b)})
    jac = pd.DataFrame(rows)
    jac.to_csv(RESULTS / f"sva_jaccard_{mname}.csv", index=False)
    print("\n=== structural overlap (Jaccard) ===")
    print(jac.groupby("kind").jaccard.describe()[["mean", "min", "max"]].round(3).to_string())

    # ---------- 3. universal core eval (GPU) ----------
    from lsc_acdc_circuit import load_model, safe_delete_model, cleanup_gpu
    model = load_model(mname, args.device)
    bos = model.tokenizer.bos_token_id
    patchable = get_patchable(model, args.device)
    univ_rows = []
    for dr in DRAWS:
        per_band = {b: scores[(b, dr)] for b in BANDS}
        univ = build_universal(per_band)
        n_univ = sum(int(t.isinf(v).sum()) for v in univ.values())
        univ_dev = {k: v.to(args.device) for k, v in univ.items()}
        for tb in BANDS:
            test = load_split(tb, dr, "test")
            L, T, W = run_circuit(patchable, univ_dev, n_univ, test, bos, args.device)
            m = metrics_from_logits(L, T, W)
            univ_rows.append({"model": mname, "draw": dr, "test_band": tb,
                              "n_edges": n_univ, "universal_2way": m["acc_2way"],
                              "universal_top1": m["acc_top1"]})
        print(f"{dr}: universal core = {n_univ} edges, 2way on bands: "
              + " ".join(f"{r['test_band']}={r['universal_2way']:.3f}"
                         for r in univ_rows[-len(BANDS):]), flush=True)
        del univ_dev
        cleanup_gpu()
    univ_df = pd.DataFrame(univ_rows)
    univ_df.to_csv(RESULTS / f"sva_universal_{mname}.csv", index=False)
    del patchable
    safe_delete_model(model)
    cleanup_gpu()

    # ---------- 4. boosts / TE / d / Wilcoxon (NB12 cell 16) ----------
    univ_acc = univ_df.set_index(["draw", "test_band"]).universal_2way.to_dict()
    same_vals, cross_vals = [], []
    for dr in DRAWS:
        for tb in BANDS:
            u = univ_acc[(dr, tb)]
            same_vals.append(metrics[(tb, dr)]["transfer"][tb]["acc_2way"] - u)
            cross_vals.append(np.mean(
                [metrics[(sb, dr)]["transfer"][tb]["acc_2way"] - u
                 for sb in BANDS if sb != tb]))
    same_arr, cross_arr = np.array(same_vals), np.array(cross_vals)
    te = (np.mean(cross_arr) / np.mean(same_arr)
          if np.mean(same_arr) > 0 else np.nan)
    pooled = np.sqrt((np.var(same_arr) + np.var(cross_arr)) / 2)
    d = (np.mean(same_arr) - np.mean(cross_arr)) / pooled if pooled > 0 else 0.0
    # Exact test on INTEGER-scaled diffs: accuracies are integer successes
    # out of 225 (the universal-core term cancels in same-minus-cross), so
    # d_cell = 5*own_count - colsum_count is exact and monotone in the
    # accuracy diff. Raw float boosts would split count-level ties.
    int_diffs = []
    for dr in DRAWS:
        for tb in BANDS:
            cnts = {sb: round(metrics[(sb, dr)]["transfer"][tb]["acc_2way"]
                              * 225) for sb in BANDS}
            assert all(abs(metrics[(sb, dr)]["transfer"][tb]["acc_2way"] * 225
                           - cnts[sb]) < 1e-3 for sb in BANDS)
            int_diffs.append(5 * cnts[tb] - sum(cnts.values()))
    p = exact_wilcoxon_greater(int_diffs)
    summary = {
        "model": mname, "metric": "acc_2way",
        "same_band_boost_mean": float(np.mean(same_arr)),
        "cross_band_boost_mean": float(np.mean(cross_arr)),
        "transfer_efficiency": float(te), "cohens_d": float(d),
        "wilcoxon_p": float(p), "n_pairs": len(same_arr),
        "mean_universal_edges": float(univ_df.n_edges.mean()),
        "mean_circuit_edges": float(np.mean(
            [metrics[(b, dr)]["n_edges"] for b in BANDS for dr in DRAWS])),
    }
    pd.DataFrame([summary]).to_csv(RESULTS / f"sva_phantom_summary_{mname}.csv",
                                   index=False)
    print("\n=== phantom readout (NB12 cell-16 protocol, 2-way) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
