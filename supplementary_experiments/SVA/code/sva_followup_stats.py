"""P1-P3 follow-up readouts (pre-declared; CPU-only).

P1 ZERO + MEAN ABLATION: matched-band own-minus-cross gap per model
   under AblationType.ZERO and TOKENWISE_MEAN_CORRUPT (from
   sva_alt_ablation_{model}.csv), tested with the same exact
   216-relabeling draw-blocked permutation as the resample primary
   (integer success counts; directional; BH over the 5 models WITHIN
   each ablation family). Boosts/TE relative to the matched core under
   the same ablation (TE guarded: reported only when the same-band
   boost is positive, as in the LSC zero-ablation table).
P2 MAJORITY CORE (k>=3-of-5): recovery = core accuracy / same-band
   circuit accuracy per (draw, band) from sva_majority_core_{model}.csv
   and the circuits' metrics.json; min and mean per model; core sizes.
   Descriptive (LSC >=99% recovery claim structure).
P3 ASYMMETRIC TRANSFER (resample, descriptive): PRIMARY statistic is
   the matched LSC analog on RAW transfer accuracies (low-group
   {low, medium} circuits on high tests minus the high circuit on
   low-group tests), matching the LSC Sec 4.5 construction. The
   earlier core-adjusted all-ordinal-pairs version is kept in the JSON
   as a labeled sensitivity (it subtracts a test-band core and is NOT
   LSC's statistic). No inferential claim.

P4 DIFFERENCE-IN-DIFFERENCES diagnostic: does the gap
   differ significantly BETWEEN intervention distributions?
   D = own-sum(resample) - own-sum(alt) under the same 216 relabelings
   applied to both matrices (integer counts, directional resample >
   alt), BH within each pair-family over the 5 models. Significance
   under one ablation and non-significance under another is not itself
   a significant interaction; this tests the interaction directly.

Output: results/sva_followup_stats.json
"""

import json
import sys
from itertools import permutations, product
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
CIRCUITS = BASE / "circuits"
RESULTS = BASE / "results"

MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b",
          "pythia-1.4b"]
BANDS = ["low", "medium", "high", "very_high", "control"]
MATCHED = ["low", "medium", "high"]
DRAWS = ["draw_1", "draw_2", "draw_3"]
N_TEST = 225
ABLATIONS = ["zero", "tokenwise_mean_corrupt"]


def to_count(acc):
    raw = acc * N_TEST
    cnt = round(raw)
    assert abs(raw - cnt) < 1e-3, raw
    return cnt


def blocked_permutation_counts(C):
    """Integer own-sum exact 216-relabeling directional test (identical
    to sva_matched_stats.blocked_permutation)."""
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
    gap = float(np.mean([
        (C[dr][tb][tb] - (sum(C[dr][sb][tb] for sb in MATCHED)
                          - C[dr][tb][tb]) / 2) / N_TEST
        for dr in DRAWS for tb in MATCHED]))
    return gap, hits / total


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


def load_resample(mname):
    """Resample transfer accs + matched-core accs (existing artifacts)."""
    T = {dr: {sb: {} for sb in BANDS} for dr in DRAWS}
    for dr in DRAWS:
        for sb in BANDS:
            m = json.load(open(CIRCUITS / mname / sb / dr / "metrics.json"))
            for tb in BANDS:
                T[dr][sb][tb] = m["transfer"][tb]["acc_2way"]
    u = pd.read_csv(RESULTS / f"sva_universal_matched_{mname}.csv"
                    ).set_index(["draw", "test_band"]).universal_2way.to_dict()
    return T, u


def main():
    out = {"alt_ablation": {a: [] for a in ABLATIONS},
           "majority_core": [], "asymmetry": []}

    # ---------- P1 ----------
    for abl in ABLATIONS:
        p_list = []
        for mname in MODELS:
            df = pd.read_csv(RESULTS / f"sva_alt_ablation_{mname}.csv")
            df = df[df.ablation == abl]
            C = {dr: {sb: {} for sb in MATCHED} for dr in DRAWS}
            core = {}
            for _, r in df.iterrows():
                if r.source == "core":
                    core[(r.draw, r.test_band)] = r.acc_2way
                else:
                    C[r.draw][r.source][r.test_band] = to_count(r.acc_2way)
            gap, p = blocked_permutation_counts(C)
            same = [C[dr][tb][tb] / N_TEST - core[(dr, tb)]
                    for dr in DRAWS for tb in MATCHED]
            cross = [np.mean([C[dr][sb][tb] / N_TEST for sb in MATCHED
                              if sb != tb]) - core[(dr, tb)]
                     for dr in DRAWS for tb in MATCHED]
            sm, cm = float(np.mean(same)), float(np.mean(cross))
            entry = {"model": mname, "gap": gap, "p_exact_216": p,
                     "same_boost_mean": sm, "cross_boost_mean": cm,
                     "te": (cm / sm if sm > 0.01 else None)}
            out["alt_ablation"][abl].append(entry)
            p_list.append(p)
        for e, q in zip(out["alt_ablation"][abl], bh(p_list)):
            e["q_bh"] = q

    # ---------- P2 ----------
    for mname in MODELS:
        mc = pd.read_csv(RESULTS / f"sva_majority_core_{mname}.csv")
        recov = []
        for dr in DRAWS:
            for tb in BANDS:
                circ = json.load(open(
                    CIRCUITS / mname / tb / dr / "metrics.json"
                ))["transfer"][tb]["acc_2way"]
                core = float(mc[(mc.draw == dr)
                                & (mc.test_band == tb)].core_2way.iloc[0])
                recov.append({"draw": dr, "band": tb, "circuit": circ,
                              "core": core,
                              "recovery": core / circ if circ > 0 else None})
        recs = [r["recovery"] for r in recov if r["recovery"] is not None]
        out["majority_core"].append(
            {"model": mname,
             "core_edges": sorted(set(int(x) for x in mc.n_edges)),
             "recovery_mean": float(np.mean(recs)),
             "recovery_min": float(np.min(recs)),
             "cells": recov})

    # ---------- P3 ----------
    for mname in MODELS:
        T, u = load_resample(mname)

        def raw(sb, tb):
            return float(np.mean([T[dr][sb][tb] for dr in DRAWS]))

        # PRIMARY: matched LSC analog, raw accuracies
        up_raw = (raw("low", "high") + raw("medium", "high")) / 2
        down_raw = (raw("high", "low") + raw("high", "medium")) / 2

        # SENSITIVITY: core-adjusted, all ordinal pairs (NOT LSC's stat)
        def boost(sb, tb):
            return float(np.mean([T[dr][sb][tb] - u[(dr, tb)]
                                  for dr in DRAWS]))

        pairs = {f"{a}->{b}": boost(a, b)
                 for a in MATCHED for b in MATCHED if a != b}
        up = np.mean([pairs["low->medium"], pairs["low->high"],
                      pairs["medium->high"]])
        down = np.mean([pairs["medium->low"], pairs["high->low"],
                        pairs["high->medium"]])
        out["asymmetry"].append(
            {"model": mname,
             "lsc_analog_raw": {"up": up_raw, "down": down_raw,
                                "up_minus_down": up_raw - down_raw},
             "core_adjusted_sensitivity": {
                 "pairs": pairs, "up_mean": float(up),
                 "down_mean": float(down),
                 "up_minus_down": float(up - down),
                 "note": "subtracts test-band matched core; not LSC's "
                         "statistic; kept as sensitivity"}})

    # ---------- P4 DiD ----------
    out["did_resample_vs_alt"] = {}
    for abl in ABLATIONS:
        entries = []
        for mname in MODELS:
            # resample counts from metrics.json
            Cres = {dr: {} for dr in DRAWS}
            for dr in DRAWS:
                for sb in MATCHED:
                    m = json.load(open(CIRCUITS / mname / sb / dr
                                       / "metrics.json"))
                    Cres[dr][sb] = {tb: to_count(m["transfer"][tb]["acc_2way"])
                                    for tb in MATCHED}
            df = pd.read_csv(RESULTS / f"sva_alt_ablation_{mname}.csv")
            df = df[(df.ablation == abl) & (df.source != "core")]
            Calt = {dr: {sb: {} for sb in MATCHED} for dr in DRAWS}
            for _, r in df.iterrows():
                Calt[r.draw][r.source][r.test_band] = to_count(r.acc_2way)

            def own_sum(Cm, assign):
                return sum(Cm[dr][assign[dr][tb]][tb]
                           for dr in DRAWS for tb in MATCHED)

            iden = {dr: {tb: tb for tb in MATCHED} for dr in DRAWS}
            d_obs = own_sum(Cres, iden) - own_sum(Calt, iden)
            hits = 0
            for combo in product(list(permutations(MATCHED)),
                                 repeat=len(DRAWS)):
                assign = {dr: dict(zip(MATCHED, p))
                          for dr, p in zip(DRAWS, combo)}
                if own_sum(Cres, assign) - own_sum(Calt, assign) >= d_obs:
                    hits += 1
            entries.append({"model": mname, "p_exact_216": hits / 216})
        for e, q in zip(entries, bh([e["p_exact_216"] for e in entries])):
            e["q_bh"] = q
        out["did_resample_vs_alt"][abl] = entries

    json.dump(out, open(RESULTS / "sva_followup_stats.json", "w"), indent=1)

    for abl in ABLATIONS:
        for e in out["alt_ablation"][abl]:
            te = "n/a(boost<=.01)" if e["te"] is None else f"{e['te']:.3f}"
            print(f"P1 {abl:22s} {e['model']:12s} gap {e['gap']:+.4f} "
                  f"p {e['p_exact_216']:.5f} q {e['q_bh']:.5f} "
                  f"same {e['same_boost_mean']:+.3f} "
                  f"cross {e['cross_boost_mean']:+.3f} TE {te}", flush=True)
    for e in out["majority_core"]:
        print(f"P2 majority(k>=3) {e['model']:12s} edges {e['core_edges']} "
              f"recovery mean {e['recovery_mean']:.3f} "
              f"min {e['recovery_min']:.3f}", flush=True)
    for e in out["asymmetry"]:
        r = e["lsc_analog_raw"]
        s = e["core_adjusted_sensitivity"]
        print(f"P3 asym {e['model']:12s} LSC-analog raw delta "
              f"{r['up_minus_down']:+.4f} (up {r['up']:.3f} down "
              f"{r['down']:.3f}); core-adj sens delta "
              f"{s['up_minus_down']:+.3f}", flush=True)
    for abl in ABLATIONS:
        for e in out["did_resample_vs_alt"][abl]:
            print(f"P4 DiD resample-vs-{abl:22s} {e['model']:12s} "
                  f"p {e['p_exact_216']:.4f} q {e['q_bh']:.4f}", flush=True)
    print("saved sva_followup_stats.json", flush=True)


if __name__ == "__main__":
    main()
