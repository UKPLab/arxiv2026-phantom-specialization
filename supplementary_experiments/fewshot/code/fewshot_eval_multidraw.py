"""Few-shot demonstration variant with scrambled-demonstration control (C8).

Three independent paired prompt draws; (b) a
length-matched scrambled-demonstration condition for attribution; (c) a
pre-declared matched-band primary contrast; (d) BH correction over the
declared test families; (e) two-way accuracies saved.

Conditions per example (identical query and target across all three):
  k0:  c1..c4 c1 c2                                        (7 tokens w/ BOS)
  k2:  a a SEP b b SEP + query                             (25 tokens)
  k2s: same prefix token multiset with the 16 content tokens randomly
       permuted across the demonstration slots (SEP positions fixed),
       destroying the within-demonstration repetition structure while
       matching length, tokens, and query position exactly.

PRE-DECLARED CONTRASTS (before running):
  Primary   (matched bands): gap_m = acc(very_high) - acc(low);
            test delta_m = gap_m(k0) - gap_m(k2) per model, paired
            bootstrap 95% CI; BH over the 5 models.
  Secondary (confound-relaxed): gap_v = acc(control) - acc(very_low);
            same procedure; BH over the 5 models; very_low pool is the
            unmatched 97-token pool (C2 flag).
  Attribution: per-cell exact McNemar k2 vs k2s (does demonstration
            STRUCTURE matter beyond a length-matched prefix?), BH over
            30 cells; likewise k2 vs k0 (does the prefix help at all?),
            BH over 30 cells. Cells pool the three draws (n=675).

Outputs:
  results/eval_multidraw_per_example.csv.gz
  results/eval_multidraw_summary.json
"""

import gzip
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch as t
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
ISC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ISC / "LSC_circuits"))
sys.path.insert(0, str(ISC / "supplementary_experiments" / "code"))
sys.path.insert(0, str(BASE / "code"))

from lsc_acdc_circuit import load_model, safe_delete_model, cleanup_gpu  # noqa: E402
from fewshot_setup import band_tokens, BANDS, SEP  # noqa: E402

DEVICE = "cuda:0"
MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]
N = 225
DRAW_SEEDS = {"draw_1": 51_000, "draw_2": 61_000, "draw_3": 71_000}
CONDS = ("k0", "k2", "k2s")
N_BOOT = 10_000


def make_examples(band, band_idx, draw_base, n=N):
    toks, weights = band_tokens(band)
    rng = random.Random(draw_base + 10 * band_idx)
    out, seen = [], set()
    while len(out) < n:
        if weights:
            pick = []
            while len(set(pick)) < 12:
                pick = rng.choices(toks, weights=weights, k=14)
            pick = list(dict.fromkeys(pick))[:12]
        else:
            pick = rng.sample(toks, 12)
        key = tuple(pick)
        if key in seen:
            continue
        seen.add(key)
        a, b, c = pick[0:4], pick[4:8], pick[8:12]
        query = c + c[:2]
        content = a + a + b + b
        scr = content[:]
        rng.shuffle(scr)
        out.append({
            "k0": query,
            "k2": a + a + [SEP] + b + b + [SEP] + query,
            "k2s": scr[0:8] + [SEP] + scr[8:16] + [SEP] + query,
            "target": c[2], "wrong": c[3],
        })
    return out


def eval_cell(model, prompts, targets, wrongs, bs=128):
    bos = model.tokenizer.bos_token_id
    top1, two_way = [], []
    with t.no_grad():
        for i in range(0, len(prompts), bs):
            ids = t.tensor([[bos] + p for p in prompts[i:i + bs]], device=DEVICE)
            last = model(ids)[:, -1, :]
            tgt = t.tensor(targets[i:i + bs], device=DEVICE)
            wrg = t.tensor(wrongs[i:i + bs], device=DEVICE)
            top1 += (last.argmax(-1) == tgt).int().tolist()
            two_way += (last.gather(1, tgt[:, None])
                        > last.gather(1, wrg[:, None])).int().squeeze(1).tolist()
    return np.array(top1), np.array(two_way)


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
    per_ex, top1 = [], {}
    for draw, base_seed in DRAW_SEEDS.items():
        data = {b: make_examples(b, i, base_seed) for i, b in enumerate(BANDS)}
        for mname in MODELS:
            model = load_model(mname, DEVICE)
            model.cfg.use_attn_result = False
            model.cfg.use_attn_in = False
            model.cfg.use_hook_mlp_in = False
            for band in BANDS:
                ex = data[band]
                tgt = [e["target"] for e in ex]
                wrg = [e["wrong"] for e in ex]
                for cond in CONDS:
                    t1, t2 = eval_cell(model, [e[cond] for e in ex], tgt, wrg)
                    top1[(mname, band, cond, draw)] = t1
                    for j in range(N):
                        per_ex.append(f"{mname},{band},{draw},{j},{cond},"
                                      f"{t1[j]},{t2[j]}")
            print(f"{draw} {mname:12s} "
                  + " ".join(f"{b}:{top1[(mname, b, 'k0', draw)].mean():.2f}/"
                             f"{top1[(mname, b, 'k2', draw)].mean():.2f}/"
                             f"{top1[(mname, b, 'k2s', draw)].mean():.2f}"
                             for b in BANDS), flush=True)
            safe_delete_model(model)
            cleanup_gpu()

    with gzip.open(BASE / "results/eval_multidraw_per_example.csv.gz", "wt") as f:
        f.write("model,band,draw,idx,cond,top1,two_way\n")
        f.write("\n".join(per_ex) + "\n")

    def pooled(m, b, c):
        return np.concatenate([top1[(m, b, c, d)] for d in DRAW_SEEDS])

    rng = np.random.default_rng(42)
    summary = {"n_per_cell_pooled": N * len(DRAW_SEEDS), "cells": {},
               "mcnemar_k2_vs_k0": {}, "mcnemar_k2_vs_k2s": {}, "gaps": {}}

    p_a, p_s, keys = [], [], []
    for m in MODELS:
        for b in BANDS:
            arrs = {c: pooled(m, b, c) for c in CONDS}
            summary["cells"][f"{m}/{b}"] = {c: float(arrs[c].mean())
                                            for c in CONDS}
            keys.append(f"{m}/{b}")
            p_a.append(mcnemar_exact_p(arrs["k0"], arrs["k2"]))
            p_s.append(mcnemar_exact_p(arrs["k2s"], arrs["k2"]))
    for name, ps in [("mcnemar_k2_vs_k0", p_a), ("mcnemar_k2_vs_k2s", p_s)]:
        adj = bh(ps)
        summary[name] = {k: {"p": p, "p_bh": a}
                         for k, p, a in zip(keys, ps, adj)}

    for label, ref, band in [("primary_vh_minus_low", "very_high", "low"),
                             ("secondary_control_minus_very_low",
                              "control", "very_low")]:
        raw_p = []
        entries = []
        for m in MODELS:
            r0, r2 = pooled(m, ref, "k0"), pooled(m, ref, "k2")
            b0, b2 = pooled(m, band, "k0"), pooled(m, band, "k2")
            g0 = float(r0.mean() - b0.mean())
            g2 = float(r2.mean() - b2.mean())
            n = len(r0)
            deltas = np.empty(N_BOOT)
            for i in range(N_BOOT):
                ir = rng.integers(0, n, n)
                ib = rng.integers(0, n, n)
                deltas[i] = ((r0[ir].mean() - b0[ib].mean())
                             - (r2[ir].mean() - b2[ib].mean()))
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            obs = g0 - g2
            p_boot = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
            p_boot = float(min(1.0, max(p_boot, 1.0 / N_BOOT)))
            raw_p.append(p_boot)
            entries.append({"model": m, "gap_k0": g0, "gap_k2": g2,
                            "delta": obs, "ci95": [float(lo), float(hi)],
                            "p_boot": p_boot})
            # per-draw deltas for stability reporting
            entries[-1]["delta_per_draw"] = [
                float((top1[(m, ref, "k0", d)].mean()
                       - top1[(m, band, "k0", d)].mean())
                      - (top1[(m, ref, "k2", d)].mean()
                         - top1[(m, band, "k2", d)].mean()))
                for d in DRAW_SEEDS]
        adj = bh(raw_p)
        for e, a in zip(entries, adj):
            e["p_bh"] = a
        summary["gaps"][label] = entries
        for e in entries:
            print(f"{label} {e['model']:12s} {e['gap_k0']:+.3f} -> "
                  f"{e['gap_k2']:+.3f} delta {e['delta']:+.3f} "
                  f"CI [{e['ci95'][0]:+.3f},{e['ci95'][1]:+.3f}] "
                  f"p_bh {e['p_bh']:.4f} per-draw "
                  + "/".join(f"{d:+.3f}" for d in e["delta_per_draw"]),
                  flush=True)

    # The canonical summary (permutation p-values, post-hoc family) is
    # produced by fewshot_eval_stats.py from the per-example CSV;
    # this in-run summary uses the bootstrap-crossing p and must not
    # overwrite it.
    json.dump(summary,
              open(BASE / "results/eval_multidraw_summary_bootstrap.json",
                   "w"), indent=1)
    print("saved (provisional bootstrap summary; run fewshot_eval_stats.py for the "
          "canonical one)", flush=True)


if __name__ == "__main__":
    main()
