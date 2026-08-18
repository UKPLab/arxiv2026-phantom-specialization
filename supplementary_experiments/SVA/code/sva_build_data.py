"""SVA data construction, LSC-style (pools -> length matching -> generation).

Mirrors the LSC pipeline stages:
  lsc_token_pools.py  -> per-band pools, lowercase, EXACT length matching
                          across matchable bands (low/medium/high); very_high
                          kept unmatched + flagged (like LSC's very_low)
  lsc_generator.py    -> per band x draw datasets, master seed 42, 3 draws,
                          70/15/15, 1875/band surplus (NLL screen trims
                          to 1500 downstream), frequency-weighted control

Prompt (token IDs, fixed length 5, BOS added by loaders):
    [The] [SUBJ] [near] [the] [ATTR]     -> predict " is" / " are"
Subject from the band pool (sg or pl, balanced); attractor = opposite number,
same band, different lemma. Corrupt prompt = subject number swapped (answer
flips), everything else identical -> diverge at subject position.

Inputs:  results/sva_noun_pairs_pos_validated.csv (both forms NOUN_COMMON)
Outputs: pools/sva_pool_{band}.json, data_generated/{draw}/{band}/{split}.json
"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
POOLS = BASE / "pools"
OUT = BASE / "data_generated"
MASTER_SEED = 42
DRAWS = 3
N_SURPLUS = 1875          # trimmed to 1500 by the prompt-NLL screen
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
MATCHED_BANDS = ["low", "medium", "high"]
ALL_BANDS = MATCHED_BANDS + ["very_high"]

# singulars that are awkward subjects for "The X near the Y is/are"
BLOCKLIST = {"scissor", "trouser", "pant", "jean", "spectacle", "outskirt",
             "remain", "surrounding", "belonging", "premise", "thank"}


def main():
    pairs = pd.read_csv(BASE / "results/sva_noun_pairs_pos_validated.csv")
    pairs = pairs[pairs.both_noun & ~pairs.singular.isin(BLOCKLIST)].copy()

    # --- template token ids (verified single tokens) ---
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "LSC_circuits"))
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    tmpl = {}
    for name, s in [("The", "The"), ("near", " near"), ("the", " the"),
                    ("is", " is"), ("are", " are")]:
        ids = tk.encode(s, add_special_tokens=False)
        assert len(ids) == 1, (name, s, ids)
        tmpl[name] = ids[0]
    print("template ids:", tmpl)

    # --- exact length matching across matchable bands (mirror LSC) ---
    by_band = {b: pairs[pairs.sg_band == b] for b in ALL_BANDS}
    len_counts = {b: Counter(by_band[b].sg_len.astype(int)) for b in MATCHED_BANDS}
    common = set.intersection(*[set(c) for c in len_counts.values()])
    quota = {L: min(len_counts[b][L] for b in MATCHED_BANDS) for L in sorted(common)}
    print("matched length quota:", quota, "| total per band:", sum(quota.values()))

    rng = random.Random(MASTER_SEED)
    pool = {}
    for b in MATCHED_BANDS:
        rows = []
        for L, q in quota.items():
            cand = by_band[b][by_band[b].sg_len == L]
            rows += cand.sample(n=q, random_state=MASTER_SEED).to_dict("records")
        pool[b] = rows
    pool["very_high"] = by_band["very_high"].to_dict("records")  # unmatched, flagged
    for b in ALL_BANDS:
        print(f"pool {b}: {len(pool[b])} pairs"
              + ("  [UNMATCHED - report with caveat]" if b == "very_high" else ""))

    POOLS.mkdir(exist_ok=True)
    for b in ALL_BANDS:
        json.dump({"band": b, "matched": b in MATCHED_BANDS,
                   "n_pairs": len(pool[b]), "pairs": pool[b]},
                  open(POOLS / f"sva_pool_{b}.json", "w"), indent=1)

    # --- generation ---
    def make_examples(band, band_pairs, n, seed, weights=None):
        r = random.Random(seed)
        seen = set()
        out = []
        guard = 0
        while len(out) < n and guard < n * 50:
            guard += 1
            if weights is None:
                subj = r.choice(band_pairs)
            else:
                subj = r.choices(band_pairs, weights=weights, k=1)[0]
            attr = r.choice(band_pairs)
            if attr["singular"] == subj["singular"]:
                continue
            number = r.choice(["sg", "pl"])
            s_id = subj["sg_token_id"] if number == "sg" else subj["pl_token_id"]
            s_swap = subj["pl_token_id"] if number == "sg" else subj["sg_token_id"]
            a_id = attr["pl_token_id"] if number == "sg" else attr["sg_token_id"]
            key = (s_id, a_id)
            if key in seen:
                continue
            seen.add(key)
            clean = [tmpl["The"], s_id, tmpl["near"], tmpl["the"], a_id]
            corrupt = [tmpl["The"], s_swap, tmpl["near"], tmpl["the"], a_id]
            out.append({
                "example_id": len(out),
                "token_ids": clean, "corrupt_token_ids": corrupt,
                "subject": subj["singular"], "attractor": attr["singular"],
                "number": number,
                "target_token_id": tmpl["is"] if number == "sg" else tmpl["are"],
                "wrong_token_id": tmpl["are"] if number == "sg" else tmpl["is"],
                "subject_position": 1, "prediction_position": 4,
                "sg_log_freq": subj["sg_log_freq"],
            })
        assert len(out) == n, f"{band}: only {len(out)}/{n}"
        return out

    conditions = {b: (pool[b], None) for b in ALL_BANDS}
    union = [p for b in ALL_BANDS for p in pool[b]]
    w = [10 ** p["sg_log_freq"] for p in union]     # pretraining-frequency weighted
    conditions["control"] = (union, w)

    meta_common = {
        "task": "SVA", "template": "The {SUBJ} near {the} {ATTR} -> is/are",
        "master_seed": MASTER_SEED, "prompt_len": 5, "bos_included": False,
        "prediction_position": 4, "corruption": "subject number swap",
        "answer_ids": {"is": tmpl["is"], "are": tmpl["are"]},
    }
    for draw in range(1, DRAWS + 1):
        for band, (bp, wts) in conditions.items():
            seed = MASTER_SEED + 1000 * draw + hash(band) % 997
            ex = make_examples(band, bp, N_SURPLUS, seed, wts)
            r = random.Random(seed + 7)
            idx = list(range(len(ex)))
            r.shuffle(idx)
            n_tr = int(len(ex) * SPLITS["train"])
            n_va = int(len(ex) * SPLITS["val"])
            parts = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va],
                     "test": idx[n_tr + n_va:]}
            d = OUT / f"draw_{draw}" / band
            d.mkdir(parents=True, exist_ok=True)
            for split, ii in parts.items():
                json.dump({**meta_common, "band": band, "draw": draw,
                           "split": split, "seed": seed,
                           "matched": band in MATCHED_BANDS,
                           "n_examples": len(ii),
                           "examples": [ex[i] for i in ii]},
                          open(d / f"{split}.json", "w"))
        print(f"draw_{draw}: generated {list(conditions)} x {N_SURPLUS}")

    print("\nDONE. Pools ->", POOLS, "| datasets ->", OUT)


if __name__ == "__main__":
    main()
