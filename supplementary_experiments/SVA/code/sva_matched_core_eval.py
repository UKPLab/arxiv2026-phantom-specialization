"""Matched three-band universal core, forward evaluation only (frozen spec item 4).

The five-condition universal core (sva_aggregate.py) intersects all five
band masks, including the length-unmatched very_high and control
conditions. The matched-band primary analysis reports transfer
efficiency relative to a core with the same scope: the intersection of
the low, medium, and high masks within each draw, evaluated on those
three bands' frozen test sets. No new discovery; this is a forward pass
over existing prune_scores.pkl masks.

Output: results/sva_universal_matched_{model}.csv
        (model, draw, test_band, n_edges, universal_2way, universal_top1)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch as t

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "code"))

from sva_discovery import (  # noqa: E402
    DRAWS, get_patchable, load_split, metrics_from_logits, run_circuit,
)

MATCHED = ["low", "medium", "high"]
CIRCUITS = BASE / "circuits"
RESULTS = BASE / "results"


def build_matched_universal(per_band):
    """AND of the three matched-band inf masks (same construction as
    sva_aggregate.build_universal, restricted to MATCHED)."""
    first = MATCHED[0]
    out = {}
    for mod in per_band[first]:
        mask = t.ones_like(per_band[first][mod], dtype=t.bool)
        for b in MATCHED:
            mask &= t.isinf(per_band[b][mod])
        tensor = t.zeros_like(per_band[first][mod])
        tensor[mask] = float("inf")
        out[mod] = tensor
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    mname = args.model

    scores = {}
    for band in MATCHED:
        for draw in DRAWS:
            d = CIRCUITS / mname / band / draw
            with open(d / "prune_scores.pkl", "rb") as f:
                scores[(band, draw)] = pickle.load(f)

    from lsc_acdc_circuit import load_model, safe_delete_model, cleanup_gpu
    model = load_model(mname, args.device)
    bos = model.tokenizer.bos_token_id
    patchable = get_patchable(model, args.device)
    rows = []
    for dr in DRAWS:
        univ = build_matched_universal({b: scores[(b, dr)] for b in MATCHED})
        n_univ = sum(int(t.isinf(v).sum()) for v in univ.values())
        univ_dev = {k: v.to(args.device) for k, v in univ.items()}
        for tb in MATCHED:
            test = load_split(tb, dr, "test")
            L, T, W = run_circuit(patchable, univ_dev, n_univ, test, bos,
                                  args.device)
            m = metrics_from_logits(L, T, W)
            rows.append({"model": mname, "draw": dr, "test_band": tb,
                         "n_edges": n_univ, "universal_2way": m["acc_2way"],
                         "universal_top1": m["acc_top1"]})
        print(f"{mname} {dr}: matched core = {n_univ} edges, 2way: "
              + " ".join(f"{r['test_band']}={r['universal_2way']:.3f}"
                         for r in rows[-len(MATCHED):]), flush=True)
        del univ_dev
        cleanup_gpu()
    pd.DataFrame(rows).to_csv(RESULTS / f"sva_universal_matched_{mname}.csv",
                              index=False)
    del patchable
    safe_delete_model(model)
    cleanup_gpu()
    print(f"saved sva_universal_matched_{mname}.csv", flush=True)


if __name__ == "__main__":
    main()
