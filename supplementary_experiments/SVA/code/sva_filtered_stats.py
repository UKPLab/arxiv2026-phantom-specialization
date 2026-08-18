"""Filtered-sensitivity family: matched-band blocked permutation
on the brow/disc-FILTERED test sets, all five models (pre-declared).

Filtered cell denominators vary (each (test_band, draw) test set loses
its contaminated prompts), so the statistic is computed with EXACT
RATIONAL arithmetic: per cell the gap contribution is
Fraction(3*own_cnt - colsum_cnt, 2*n_cell), and the permutation
statistic is a sum of Fractions (the per-cell column sum over sources is
invariant under relabeling, so relabelings tie exactly when their
Fraction sums are equal). Directional (same-band greater), all 216
relabelings, BH over the 5 models.

Inputs: results/sva_transfer_filtered_{model}.csv (160m is the legacy
unsuffixed sva_transfer_filtered.csv); filtered n per cell recomputed
from data_final via the BAD_IDS filter.

Output: results/sva_filtered_stats.json
"""

import json
import sys
from fractions import Fraction
from itertools import permutations, product
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b",
          "pythia-1.4b"]
MATCHED = ["low", "medium", "high"]
DRAWS = ["draw_1", "draw_2", "draw_3"]
BAD_IDS = {6479, 22931, 1262, 28217}

FILES = {m: RESULTS / f"sva_transfer_filtered_{m}.csv" for m in MODELS}
FILES["pythia-160m"] = RESULTS / "sva_transfer_filtered.csv"  # legacy name


def filtered_n(band, draw):
    ex = json.load(open(BASE / "data_final" / draw / band / "test.json")
                   )["examples"]
    return sum(1 for e in ex if e["token_ids"][1] not in BAD_IDS
               and e["token_ids"][4] not in BAD_IDS)


def bh(pvals):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 1.0
    for rank_from_end, i in enumerate(idx[::-1]):
        rank = m - rank_from_end
        running = min(running, pvals[i] * m / rank)
        adj[i] = running
    return adj


def main():
    n_cell = {(tb, dr): filtered_n(tb, dr) for tb in MATCHED for dr in DRAWS}
    print("filtered n per matched cell:",
          {f"{tb}/{dr}": n for (tb, dr), n in n_cell.items()}, flush=True)

    entries = []
    for mname in MODELS:
        df = pd.read_csv(FILES[mname])
        C = {dr: {sb: {} for sb in MATCHED} for dr in DRAWS}
        for _, r in df.iterrows():
            if r.source_band in MATCHED and r.test_band in MATCHED:
                n = n_cell[(r.test_band, r.draw)]
                raw = r.acc_2way * n
                cnt = round(raw)
                assert abs(raw - cnt) < 1e-3, (mname, r.source_band,
                                               r.test_band, r.draw, raw)
                C[r.draw][r.source_band][r.test_band] = cnt

        def stat(assign):
            s = Fraction(0)
            for dr in DRAWS:
                for tb in MATCHED:
                    colsum = sum(C[dr][sb][tb] for sb in MATCHED)
                    own = C[dr][assign[dr][tb]][tb]
                    s += Fraction(3 * own - colsum,
                                  2 * n_cell[(tb, dr)])
            return s

        identity = {dr: {tb: tb for tb in MATCHED} for dr in DRAWS}
        obs = stat(identity)
        hits = total = 0
        for combo in product(list(permutations(MATCHED)), repeat=len(DRAWS)):
            assign = {dr: dict(zip(MATCHED, p))
                      for dr, p in zip(DRAWS, combo)}
            total += 1
            if stat(assign) >= obs:
                hits += 1
        assert total == 216
        gap = float(obs) / 9
        entries.append({"model": mname, "gap_filtered": gap,
                        "p_exact_216": hits / 216})
    for e, q in zip(entries, bh([e["p_exact_216"] for e in entries])):
        e["q_bh"] = q

    out = {"n_cell_filtered": {f"{tb}/{dr}": n
                               for (tb, dr), n in n_cell.items()},
           "matched_blocked_permutation_filtered": entries}
    json.dump(out, open(RESULTS / "sva_filtered_stats.json", "w"), indent=1)
    for e in entries:
        print(f"FILTERED {e['model']:12s} gap {e['gap_filtered']:+.4f} "
              f"p {e['p_exact_216']:.5f} q {e['q_bh']:.5f}", flush=True)
    print("saved sva_filtered_stats.json", flush=True)


if __name__ == "__main__":
    main()
