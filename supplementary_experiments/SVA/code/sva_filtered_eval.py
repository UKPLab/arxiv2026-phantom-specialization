"""Filtered-test transfer for the two models the controls runs
skipped (70m, 410m).

The brow/disc contamination sensitivity (evaluation-only: contaminated
TEST prompts excluded, circuits unchanged) had been run only for
160m/1b/1.4b, which is backwards for the primary claim: 410m is the
only BH-significant model and 70m the nearest miss. This is the
transfer-matrix part of sva_controls.py section 1 (no universal-core
eval needed: the core cancels in the own-minus-cross gap), forward
evaluation only, no ACDC rerun.

Output: results/sva_transfer_filtered_{model}.csv
        (draw, source_band, test_band, acc_2way) -- same schema as the
        existing filtered CSVs.
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
BAD_IDS = {6479, 22931, 1262, 28217}  # Ġbrow, Ġbrows, Ġdisc, Ġdiscs


def filt(examples):
    return [e for e in examples
            if e["token_ids"][1] not in BAD_IDS
            and e["token_ids"][4] not in BAD_IDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    mname = args.model

    scores, n_edges = {}, {}
    for band in BANDS:
        for draw in DRAWS:
            with open(CIRCUITS / mname / band / draw / "prune_scores.pkl",
                      "rb") as f:
                scores[(band, draw)] = pickle.load(f)
            n_edges[(band, draw)] = sum(int(t.isinf(v).sum())
                                        for v in scores[(band, draw)].values())

    from lsc_acdc_circuit import load_model, safe_delete_model, cleanup_gpu
    model = load_model(mname, args.device)
    bos = model.tokenizer.bos_token_id
    patchable = get_patchable(model, args.device)
    rows = []
    for dr in DRAWS:
        test_f = {tb: filt(load_split(tb, dr, "test")) for tb in BANDS}
        for sb in BANDS:
            dev = {k: v.to(args.device)
                   for k, v in scores[(sb, dr)].items()}
            for tb in BANDS:
                L, T, W = run_circuit(patchable, dev, n_edges[(sb, dr)],
                                      test_f[tb], bos, args.device)
                rows.append({"draw": dr, "source_band": sb, "test_band": tb,
                             "acc_2way":
                             metrics_from_logits(L, T, W)["acc_2way"]})
            del dev
        cleanup_gpu()
        print(f"{mname} {dr} done (filtered test sizes: "
              + " ".join(f"{tb}={len(test_f[tb])}" for tb in BANDS) + ")",
              flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / f"sva_transfer_filtered_{mname}.csv",
                              index=False)
    del patchable
    safe_delete_model(model)
    cleanup_gpu()
    print(f"saved sva_transfer_filtered_{mname}.csv", flush=True)


if __name__ == "__main__":
    main()
