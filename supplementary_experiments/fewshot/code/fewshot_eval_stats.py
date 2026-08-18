"""Few-shot statistics (analysis-only; no model runs).

Reads results/eval_multidraw_per_example.csv.gz and rewrites
results/eval_multidraw_summary.json with:

1. Gap-family p-values from a NULL-CENTERED, DRAW-BLOCKED permutation
   test (rather than a bootstrap zero-crossing p): the statistic
   delta = mean per-prompt (k0 - k2) gain in the reference band minus
   the same mean in the test band; under the null the per-prompt gains
   are exchangeable between the two bands, so band labels are permuted
   within each draw (Monte Carlo, 100,000 permutations, two-sided,
   resolution floor 1/(n_perm+1)). Bootstrap CIs are recomputed
   identically and kept for interval reporting.
   BH within each declared family (5 models).
2. POST-HOC family (not predeclared): exact McNemar
   k2s vs k0 per cell, BH over 30 cells (does a scrambled prefix
   change performance relative to no prefix at all?).
3. The two predeclared McNemar families unchanged (recomputed for
   completeness).
"""

import gzip
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
BANDS = ["very_low", "low", "medium", "high", "very_high", "control"]
MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]
DRAWS = ["draw_1", "draw_2", "draw_3"]
N_PERM = 100_000
N_BOOT = 10_000
GAP_FAMILIES = {"primary_vh_minus_low": ("very_high", "low"),
                "secondary_control_minus_very_low": ("control", "very_low")}


def mcnemar_exact_p(x, y):
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    n = b + c
    return 1.0 if n == 0 else float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


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


def main():
    df = pd.read_csv(BASE / "results/eval_multidraw_per_example.csv.gz")
    assert len(df) == 60_750

    def arr(m, b, cond, draw=None):
        q = df[(df.model == m) & (df.band == b) & (df.cond == cond)]
        if draw:
            q = q[q.draw == draw]
        return q.sort_values(["draw", "idx"]).top1.to_numpy()

    rng = np.random.default_rng(42)
    summary = {"n_per_cell_pooled": 675, "cells": {},
               "mcnemar_k2_vs_k0": {}, "mcnemar_k2_vs_k2s": {},
               "mcnemar_k2s_vs_k0_posthoc": {}, "gaps": {}}

    keys, p_a, p_s, p_ph = [], [], [], []
    for m, b in product(MODELS, BANDS):
        k0, k2, k2s = (arr(m, b, c) for c in ("k0", "k2", "k2s"))
        summary["cells"][f"{m}/{b}"] = {"k0": float(k0.mean()),
                                        "k2": float(k2.mean()),
                                        "k2s": float(k2s.mean())}
        keys.append(f"{m}/{b}")
        p_a.append(mcnemar_exact_p(k0, k2))
        p_s.append(mcnemar_exact_p(k2s, k2))
        p_ph.append(mcnemar_exact_p(k2s, k0))
    for name, ps in [("mcnemar_k2_vs_k0", p_a), ("mcnemar_k2_vs_k2s", p_s),
                     ("mcnemar_k2s_vs_k0_posthoc", p_ph)]:
        summary[name] = {k: {"p": p, "p_bh": a}
                         for k, p, a in zip(keys, ps, bh(ps))}
    summary["mcnemar_k2s_vs_k0_posthoc"]["NOTE"] = (
        "post-hoc family, not predeclared; added after observing k2s "
        "impairment at larger models")

    for label, (ref, band) in GAP_FAMILIES.items():
        entries = []
        for m in MODELS:
            gains_ref = {d: arr(m, ref, "k0", d) - arr(m, ref, "k2", d)
                         for d in DRAWS}
            gains_band = {d: arr(m, band, "k0", d) - arr(m, band, "k2", d)
                          for d in DRAWS}
            obs = (np.concatenate(list(gains_ref.values())).mean()
                   - np.concatenate(list(gains_band.values())).mean())
            # draw-blocked label permutation (Monte Carlo, two-sided)
            hits = 0
            pooled = {d: np.concatenate([gains_ref[d], gains_band[d]])
                      for d in DRAWS}
            n_half = 225
            for _ in range(N_PERM):
                tr = tb = 0.0
                for d in DRAWS:
                    g = pooled[d]
                    idx = rng.permutation(g.shape[0])
                    tr += g[idx[:n_half]].sum()
                    tb += g[idx[n_half:]].sum()
                if abs((tr - tb) / (3 * n_half)) >= abs(obs) - 1e-12:
                    hits += 1
            p_perm = (hits + 1) / (N_PERM + 1)
            # bootstrap CI (kept for interval reporting)
            r0 = np.concatenate([arr(m, ref, "k0", d) for d in DRAWS])
            r2 = np.concatenate([arr(m, ref, "k2", d) for d in DRAWS])
            b0 = np.concatenate([arr(m, band, "k0", d) for d in DRAWS])
            b2 = np.concatenate([arr(m, band, "k2", d) for d in DRAWS])
            n = len(r0)
            deltas = np.empty(N_BOOT)
            for i in range(N_BOOT):
                ir = rng.integers(0, n, n)
                ib = rng.integers(0, n, n)
                deltas[i] = ((r0[ir].mean() - b0[ib].mean())
                             - (r2[ir].mean() - b2[ib].mean()))
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            delta_per_draw = [
                float(gains_ref[d].mean() - gains_band[d].mean())
                for d in DRAWS]
            entries.append({"model": m,
                            "gap_k0": float(r0.mean() - b0.mean()),
                            "gap_k2": float(r2.mean() - b2.mean()),
                            "delta": float(obs),
                            "delta_per_draw": delta_per_draw,
                            "ci95_bootstrap": [float(lo), float(hi)],
                            "p_perm_blocked": float(p_perm)})
        adj = bh([e["p_perm_blocked"] for e in entries])
        for e, a in zip(entries, adj):
            e["q_bh"] = a
        summary["gaps"][label] = entries
        for e in entries:
            print(f"{label} {e['model']:12s} delta {e['delta']:+.3f} "
                  f"CI [{e['ci95_bootstrap'][0]:+.3f},{e['ci95_bootstrap'][1]:+.3f}] "
                  f"p_perm {e['p_perm_blocked']:.6f} q_bh {e['q_bh']:.6f}",
                  flush=True)

    n_ph = sum(1 for k in keys
               if summary["mcnemar_k2s_vs_k0_posthoc"][k]["p_bh"] < 0.05)
    print(f"post-hoc k2s vs k0: BH-sig {n_ph}/30", flush=True)
    json.dump(summary, open(BASE / "results/eval_multidraw_summary.json",
                            "w"), indent=1)
    print("saved", flush=True)


if __name__ == "__main__":
    main()
