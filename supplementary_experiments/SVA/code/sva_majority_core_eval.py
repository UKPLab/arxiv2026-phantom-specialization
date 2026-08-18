"""P2 (pre-declared): majority-shared core (k>=3-of-5) sufficiency.

LSC protocol mirror: per draw, keep edges present in at least 3 of the 5
condition masks; forward-evaluate (resample, LSC protocol) on all five
bands' test sets. Readout (computed downstream in sva_followup_stats.py):
recovery = core accuracy / same-band circuit accuracy per (draw, band).
Descriptive; mirrors the LSC k>=3 majority-core >=99% recovery claim
structure. No new discovery.

Output: results/sva_majority_core_{model}.csv
        (draw, test_band, n_edges, core_2way, core_top1)
"""

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch as t

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "code"))

from sva_discovery import (  # noqa: E402
    BANDS, DRAWS, get_patchable, load_split, metrics_from_logits, run_circuit,
)

CIRCUITS = BASE / "circuits"
RESULTS = BASE / "results"
K_MIN = 3


def build_majority(per_band):
    """Edges present in >= K_MIN of the 5 condition masks."""
    first = BANDS[0]
    out = {}
    for mod in per_band[first]:
        votes = t.zeros_like(per_band[first][mod], dtype=t.int64)
        for b in BANDS:
            votes += t.isinf(per_band[b][mod]).to(t.int64)
        tensor = t.zeros_like(per_band[first][mod])
        tensor[votes >= K_MIN] = float("inf")
        out[mod] = tensor
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    mname = args.model

    scores = {}
    for band in BANDS:
        for draw in DRAWS:
            with open(CIRCUITS / mname / band / draw / "prune_scores.pkl",
                      "rb") as f:
                scores[(band, draw)] = pickle.load(f)

    from lsc_acdc_circuit import load_model, safe_delete_model, cleanup_gpu
    model = load_model(mname, args.device)
    bos = model.tokenizer.bos_token_id
    patchable = get_patchable(model, args.device)
    rows = []
    for dr in DRAWS:
        core = build_majority({b: scores[(b, dr)] for b in BANDS})
        n_core = sum(int(t.isinf(v).sum()) for v in core.values())
        dev = {k: v.to(args.device) for k, v in core.items()}
        for tb in BANDS:
            test = load_split(tb, dr, "test")
            L, T, W = run_circuit(patchable, dev, n_core, test, bos,
                                  args.device)
            m = metrics_from_logits(L, T, W)
            rows.append({"draw": dr, "test_band": tb, "n_edges": n_core,
                         "core_2way": m["acc_2way"],
                         "core_top1": m["acc_top1"]})
        print(f"{mname} {dr}: majority core (k>={K_MIN}) = {n_core} edges, "
              "2way: " + " ".join(f"{r['test_band']}={r['core_2way']:.3f}"
                                  for r in rows[-len(BANDS):]), flush=True)
        del dev
        cleanup_gpu()
    pd.DataFrame(rows).to_csv(RESULTS / f"sva_majority_core_{mname}.csv",
                              index=False)
    del patchable
    safe_delete_model(model)
    cleanup_gpu()
    print(f"saved sva_majority_core_{mname}.csv", flush=True)


if __name__ == "__main__":
    main()
