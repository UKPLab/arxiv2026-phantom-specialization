"""P1 (pre-declared): matched-band transfer + matched core under ZERO and
TOKENWISE_MEAN_CORRUPT ablation.

Mirrors the LSC zero-ablation robustness check (app:zero_ablation) and
the LSC mean-ablation protocol (tokenwise_mean_corrupt) on the
SVA matched bands. Forward evaluation only; circuits unchanged. The
question: does the Pythia-410m matched-band resample advantage
(gap +0.019, q=0.023) persist when the intervention distribution is not
resample ablation?

Per (model, draw, ablation): the 3x3 matched transfer cells plus the
matched three-band core on the 3 matched test bands.

Output: results/sva_alt_ablation_{model}.csv
        (draw, source, test_band, ablation, n_edges, acc_2way, acc_top1)
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
    BATCH, EVAL_SEED, get_patchable, load_split, make_loader,
    metrics_from_logits, set_all_seeds,
)
from sva_matched_core_eval import build_matched_universal  # noqa: E402

MATCHED = ["low", "medium", "high"]
DRAWS = ["draw_1", "draw_2", "draw_3"]
CIRCUITS = BASE / "circuits"
RESULTS = BASE / "results"


def run_circuit_abl(patchable, scores_dev, n_edges, examples, bos, device,
                    ablation_type):
    """sva_discovery.run_circuit with the ablation type parameterized."""
    from auto_circuit.prune import run_circuits
    from auto_circuit.types import PatchType
    set_all_seeds(EVAL_SEED)
    loader1, _ = make_loader(examples, bos, BATCH, EVAL_SEED, device)
    with t.no_grad():
        outs = run_circuits(model=patchable, dataloader=loader1,
                            test_edge_counts=[n_edges],
                            prune_scores=scores_dev,
                            patch_type=PatchType.TREE_PATCH,
                            ablation_type=ablation_type)
    set_all_seeds(EVAL_SEED)
    loader2, _ = make_loader(examples, bos, BATCH, EVAL_SEED, device)
    L, T, W = [], [], []
    for b in loader2:
        lg = outs[n_edges][b.key]
        if lg.dim() == 3:
            lg = lg[:, -1, :]
        L.append(lg.float())
        T.append(b.answers.squeeze(-1))
        W.append(b.wrong_answers.squeeze(-1))
    return t.cat(L), t.cat(T), t.cat(W)


def main():
    from auto_circuit.types import AblationType
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    mname = args.model
    ablations = {"zero": AblationType.ZERO,
                 "tokenwise_mean_corrupt": AblationType.TOKENWISE_MEAN_CORRUPT}

    scores, n_edges = {}, {}
    for band in MATCHED:
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
        tests = {tb: load_split(tb, dr, "test") for tb in MATCHED}
        univ = build_matched_universal({b: scores[(b, dr)] for b in MATCHED})
        n_univ = sum(int(t.isinf(v).sum()) for v in univ.values())
        sources = {sb: (scores[(sb, dr)], n_edges[(sb, dr)])
                   for sb in MATCHED}
        sources["core"] = (univ, n_univ)
        for abl_name, abl in ablations.items():
            for src, (sc, ne) in sources.items():
                dev = {k: v.to(args.device) for k, v in sc.items()}
                for tb in MATCHED:
                    L, T, W = run_circuit_abl(patchable, dev, ne, tests[tb],
                                              bos, args.device, abl)
                    m = metrics_from_logits(L, T, W)
                    rows.append({"draw": dr, "source": src, "test_band": tb,
                                 "ablation": abl_name, "n_edges": ne,
                                 "acc_2way": m["acc_2way"],
                                 "acc_top1": m["acc_top1"]})
                del dev
            cleanup_gpu()
        print(f"{mname} {dr} done "
              + " ".join(f"{r['ablation'][:4]}/{r['source'][:4]}->"
                         f"{r['test_band'][:4]}={r['acc_2way']:.2f}"
                         for r in rows[-6:]), flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / f"sva_alt_ablation_{mname}.csv",
                              index=False)
    del patchable
    safe_delete_model(model)
    cleanup_gpu()
    print(f"saved sva_alt_ablation_{mname}.csv ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
