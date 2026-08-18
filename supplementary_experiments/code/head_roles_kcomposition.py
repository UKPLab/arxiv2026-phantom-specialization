"""Tier-1 QK verification: weight-based K-composition (Elhage et al. 2021).

For every canonical induction-attention candidate h (labels from the
canonical-position classification, head_roles_canonical_induction.py) compute its
maximum K-composition score with previous-token-labeled heads in earlier
layers:

    kcomp(h, p) = ||W_QK^h W_OV^p||_F / (||W_QK^h||_F ||W_OV^p||_F)

with W_QK^h = W_Q^h W_K^h^T and W_OV^p = W_V^p W_O^p (d_model x d_model,
LayerNorm-folded TransformerLens weights). Pure weight algebra; no data.

Test: per model, Mann-Whitney U comparing max-kcomp of candidates vs all
other heads in layers above the first previous-token head (background).
Prediction (Olsson mechanism): candidates show elevated K-composition
with previous-token heads.

Outputs: results/head_roles_kcomposition/kcomp_per_head.csv, summary.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t
from scipy import stats

ISC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ISC / "LSC_circuits"))
from lsc_acdc_circuit import load_model, safe_delete_model, cleanup_gpu  # noqa: E402

SRC = (ISC / "LSC_circuit_analysis/05_Phase_Targeted"
       / "outputs/analysis/head_role_universality.csv")
OUT = ISC / "supplementary_experiments/results/head_roles_kcomposition"
MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]
THR = {"induction": 0.15, "bos_sink": 0.3, "previous_token": 0.2,
       "entropy_diffuse": 3.0, "dominance": 1.5}
DEVICE = "cuda:0"


def classify_canonical(r):
    sc = {"induction": r.target_fraction / THR["induction"],
          "bos_sink": r.bos_fraction / THR["bos_sink"],
          "previous_token": r.prev_fraction / THR["previous_token"]}
    ps = {k: v >= 1 for k, v in sc.items()}
    (top, ts), (_, rs) = sorted(sc.items(), key=lambda kv: -kv[1])[:2]
    if ps[top] and (rs == 0 or ts / max(rs, 1e-10) >= THR["dominance"]):
        return top
    if r.entropy >= THR["entropy_diffuse"]:
        return "diffuse"
    return top if ps[top] else "diffuse"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    roles = pd.read_csv(SRC)
    roles["role_canon"] = roles.apply(classify_canonical, axis=1)

    rows, summary = [], []
    for mname in MODELS:
        rm = roles[roles.model == mname]
        cands = {(r.layer, r.head) for r in rm.itertuples()
                 if r.role_canon == "induction"}
        prevs = [(r.layer, r.head) for r in rm.itertuples()
                 if r.role_canon == "previous_token"]
        if not cands or not prevs:
            print(f"{mname}: skipped (candidates={len(cands)}, prev={len(prevs)})")
            continue
        model = load_model(mname, DEVICE)
        WQ, WK = model.W_Q.float(), model.W_K.float()
        WV, WO = model.W_V.float(), model.W_O.float()
        n_layers, n_heads = WQ.shape[0], WQ.shape[1]
        min_prev_layer = min(l for l, _ in prevs)

        with t.no_grad():
            wov = {(l, h): WV[l, h] @ WO[l, h] for (l, h) in prevs}
            wov_norm = {k: t.linalg.matrix_norm(v) for k, v in wov.items()}
            for L in range(min_prev_layer + 1, n_layers):
                ps = [(l, h) for (l, h) in prevs if l < L]
                if not ps:
                    continue
                for H in range(n_heads):
                    wqk = WQ[L, H] @ WK[L, H].T
                    qn = t.linalg.matrix_norm(wqk)
                    best = max(float(t.linalg.matrix_norm(wqk @ wov[p])
                                     / (qn * wov_norm[p])) for p in ps)
                    rows.append({"model": mname, "layer": L, "head": H,
                                 "is_candidate": (L, H) in cands,
                                 "max_kcomp_prev": best})
        safe_delete_model(model)
        cleanup_gpu()

        dm = pd.DataFrame([r for r in rows if r["model"] == mname])
        cand = dm[dm.is_candidate].max_kcomp_prev
        back = dm[~dm.is_candidate].max_kcomp_prev
        u, p = stats.mannwhitneyu(cand, back, alternative="greater")
        rb = 2 * u / (len(cand) * len(back)) - 1
        summary.append({"model": mname, "n_candidates": len(cand),
                        "n_background": len(back),
                        "median_kcomp_candidates": cand.median(),
                        "median_kcomp_background": back.median(),
                        "mw_p_one_sided": p, "rank_biserial": rb})
        print(f"{mname}: candidates {cand.median():.4f} (n={len(cand)}) vs "
              f"background {back.median():.4f} (n={len(back)}), "
              f"MW p={p:.2e}, r_rb={rb:.2f}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "kcomp_per_head.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    print("done")


if __name__ == "__main__":
    main()
