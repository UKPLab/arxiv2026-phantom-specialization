"""SVA circuit discovery driver (sweep + discover), mirroring lsc_acdc_circuit.

Differences from LSC handled here: prompts are fixed 6 tokens (BOS + 5),
corruption is the stored number-swapped subject (diverge_idx=2), answers are
the stored is/are ids. ACDC invocation, patchable config, TREE_PATCH resample
eval, and threshold handling copied from the canonical script.

--sweep:    thresholds 1e-2..1e-6 (11, log-uniform; LSC grid) on the control
            band draw_1: ACDC on 256 train, eval on val (225). Selection rule
            = LSC's minimal_acceptable (lsc_threshold_select.py line 169):
            smallest circuit with retention >= 0.80 AND KL < 0.5. Retention
            is on 2-way forced choice (SVA task metric; LSC used top-1 on its
            copy target). ablated_2way recorded as a sanity column (LSC's
            selected circuits all had ablation_accuracy == 0.0 too).
--discover: with selected tau*, ACDC per band x draw + cross-band transfer.
"""

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch as t

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data_final"
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "LSC_circuits"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "circuit_discovery" / "auto-circuit"))

from lsc_acdc_circuit import (  # noqa: E402
    cleanup_gpu, invert_prune_scores, load_model, safe_delete_model,
    set_all_seeds, threshold_to_tao,
)

BANDS = ["low", "medium", "high", "very_high", "control"]
DRAWS = ["draw_1", "draw_2", "draw_3"]
SEQ_LEN, DIVERGE_IDX = 6, 2
ACDC_SEED, EVAL_SEED, TRAIN_SIZE, BATCH = 42, 123, 256, 225
SWEEP_THRESHOLDS = np.logspace(-2, -6, 11)


def load_split(band, draw, split):
    return json.load(open(DATA / draw / band / f"{split}.json"))["examples"]


def make_loader(examples, bos, batch_size, seed, device, n_samples=None):
    from auto_circuit.data import PromptDataset, PromptDataLoader
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    if n_samples:
        idx = idx[:n_samples]
    cp, xp, ans, wrg = [], [], [], []
    for i in idx:
        e = examples[i]
        cp.append(t.tensor([bos] + e["token_ids"], device=device))
        xp.append(t.tensor([bos] + e["corrupt_token_ids"], device=device))
        ans.append(t.tensor([e["target_token_id"]], device=device))
        wrg.append(t.tensor([e["wrong_token_id"]], device=device))
    ds = PromptDataset(clean_prompts=cp, corrupt_prompts=xp,
                       answers=ans, wrong_answers=wrg)
    n = len(idx)
    bs = min(batch_size, n)
    while bs > 1 and n % bs != 0:
        bs -= 1
    return PromptDataLoader(prompt_dataset=ds, seq_len=SEQ_LEN,
                            diverge_idx=DIVERGE_IDX, batch_size=bs), idx


def run_circuit(patchable, scores_dev, n_edges, examples, bos, device):
    """Edge-level TREE_PATCH resample eval; returns (logits, tgt, wrg) aligned."""
    from auto_circuit.prune import run_circuits
    from auto_circuit.types import PatchType, AblationType
    set_all_seeds(EVAL_SEED)
    loader1, _ = make_loader(examples, bos, BATCH, EVAL_SEED, device)
    with t.no_grad():
        outs = run_circuits(model=patchable, dataloader=loader1,
                            test_edge_counts=[n_edges], prune_scores=scores_dev,
                            patch_type=PatchType.TREE_PATCH,
                            ablation_type=AblationType.RESAMPLE)
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


def metrics_from_logits(logits, tgt, wrg, base_logits=None):
    two = float((logits.gather(1, tgt[:, None]) > logits.gather(1, wrg[:, None]))
                .float().mean())
    top1 = float((logits.argmax(-1) == tgt).float().mean())
    m = {"acc_2way": two, "acc_top1": top1}
    if base_logits is not None:
        pb = t.log_softmax(base_logits, -1)
        pc = t.log_softmax(logits, -1)
        m["kl_div"] = float((pb.exp() * (pb - pc)).sum(-1).mean())
    return m


def base_logits_for(model, examples, bos, device):
    set_all_seeds(EVAL_SEED)
    loader, _ = make_loader(examples, bos, BATCH, EVAL_SEED, device)
    L, T, W = [], [], []
    with t.no_grad():
        for b in loader:
            L.append(model(b.clean)[:, -1, :].float())
            T.append(b.answers.squeeze(-1))
            W.append(b.wrong_answers.squeeze(-1))
    return t.cat(L), t.cat(T), t.cat(W)


def run_acdc(patchable, examples, bos, threshold, device):
    from auto_circuit.prune_algos.ACDC import acdc_prune_scores
    tao_exp, tao_base = threshold_to_tao(threshold)
    set_all_seeds(ACDC_SEED)
    loader, _ = make_loader(examples, bos, BATCH, ACDC_SEED, device,
                            n_samples=TRAIN_SIZE)
    t0 = time.time()
    scores = acdc_prune_scores(model=patchable, dataloader=loader,
                               official_edges=None, tao_exps=[tao_exp],
                               tao_bases=[tao_base],
                               faithfulness_target="kl_div")
    n_edges = sum(int(t.isinf(v).sum()) for v in scores.values())
    return scores, n_edges, time.time() - t0


def get_patchable(model, device):
    from auto_circuit.utils.graph_utils import patchable_model
    return patchable_model(model=model, factorized=True,
                           slice_output="last_seq", seq_len=None,
                           separate_qkv=False, device=device)


def sweep(models, device, out_dir, early_stop=False, thresholds=None,
          csv_suffix=""):
    """early_stop: thresholds run largest-first, so the first point passing
    the LSC rule (retention >= 0.80, KL < 0.5) is the smallest passing
    circuit; lower thresholds only grow the circuit and cannot change the
    selection. Stops there and notes the omitted tail in the CSV."""
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    for mname in models:
        model = load_model(mname, device)
        bos = model.tokenizer.bos_token_id
        patchable = get_patchable(model, device)
        total = len(patchable.edges)
        train = load_split("control", "draw_1", "train")
        val = load_split("control", "draw_1", "val")
        bL, bT, bW = base_logits_for(patchable, val, bos, device)
        base = metrics_from_logits(bL, bT, bW)
        print(f"\n{mname}: {total} edges, base val 2way={base['acc_2way']:.3f}",
              flush=True)
        rows = []
        for thr in (thresholds or SWEEP_THRESHOLDS):
            scores, n_edges, dt = run_acdc(patchable, train, bos, float(thr), device)
            dev = {k: v.to(device) for k, v in scores.items()}
            cL, cT, cW = run_circuit(patchable, dev, n_edges, val, bos, device)
            cm = metrics_from_logits(cL, cT, cW, bL)
            inv, n_inv = invert_prune_scores(scores)
            inv_dev = {k: v.to(device) for k, v in inv.items()}
            aL, aT, aW = run_circuit(patchable, inv_dev, n_inv, val, bos, device)
            am = metrics_from_logits(aL, aT, aW)
            row = {"model": mname, "threshold": float(thr), "n_edges": n_edges,
                   "edge_frac": n_edges / total, "base_2way": base["acc_2way"],
                   "circuit_2way": cm["acc_2way"], "kl": cm["kl_div"],
                   "ablated_2way": am["acc_2way"], "acdc_seconds": dt}
            rows.append(row)
            print(f"  thr={thr:.2e} edges={n_edges} ({row['edge_frac']:.1%}) "
                  f"circ2way={cm['acc_2way']:.3f} kl={cm['kl_div']:.3f} "
                  f"abl2way={am['acc_2way']:.3f} [{dt:.0f}s]", flush=True)
            del dev, inv_dev
            cleanup_gpu()
            if (early_stop and cm["acc_2way"] / base["acc_2way"] >= 0.80
                    and cm["kl_div"] < 0.5):
                print("  early stop: smallest passing circuit found "
                      "(lower thresholds cannot change the selection)",
                      flush=True)
                break
        df = pd.DataFrame(rows)
        df["retention"] = df.circuit_2way / df.base_2way
        # LSC minimal_acceptable rule (lsc_threshold_select.py): smallest
        # circuit with retention >= 0.80 AND KL < 0.5
        ok = df[(df.retention >= 0.80) & (df.kl < 0.5)]
        sel = ok.sort_values("edge_frac").iloc[0] if len(ok) else None
        df["selected"] = df.threshold == (sel.threshold if sel is not None else -1)
        df.to_csv(out_dir / f"sweep_{mname}{csv_suffix}.csv", index=False)
        print(f"{mname} SELECTED tau*="
              f"{sel.threshold if sel is not None else 'NONE'}", flush=True)
        del patchable
        safe_delete_model(model)
        cleanup_gpu()


def discover(models, thresholds, device, out_dir, draws=None, bands=None):
    for mname in models:
        thr = thresholds[mname]
        model = load_model(mname, device)
        bos = model.tokenizer.bos_token_id
        patchable = get_patchable(model, device)
        total = len(patchable.edges)
        for draw in (draws or DRAWS):
            test_cache = {b: load_split(b, draw, "test") for b in BANDS}
            base_cache = {b: base_logits_for(patchable, test_cache[b], bos, device)
                          for b in BANDS}
            for band in (bands or BANDS):
                d = out_dir / mname / band / draw
                if (d / "metrics.json").exists():
                    print(f"skip {mname}/{band}/{draw}", flush=True)
                    continue
                d.mkdir(parents=True, exist_ok=True)
                train = load_split(band, draw, "train")
                scores, n_edges, dt = run_acdc(patchable, train, bos, thr, device)
                with open(d / "prune_scores.pkl", "wb") as f:
                    pickle.dump({k: v.cpu() for k, v in scores.items()}, f)
                dev = {k: v.to(device) for k, v in scores.items()}
                transfer = {}
                for tb in BANDS:
                    L, T, W = run_circuit(patchable, dev, n_edges,
                                          test_cache[tb], bos, device)
                    transfer[tb] = metrics_from_logits(L, T, W, base_cache[tb][0])
                inv, n_inv = invert_prune_scores(scores)
                inv_dev = {k: v.to(device) for k, v in inv.items()}
                aL, aT, aW = run_circuit(patchable, inv_dev, n_inv,
                                         test_cache[band], bos, device)
                bm = metrics_from_logits(*base_cache[band])
                json.dump({
                    "model": mname, "band": band, "draw": draw,
                    "threshold": thr, "n_edges": n_edges, "total_edges": total,
                    "size_fraction": n_edges / total,
                    "training_time_seconds": dt,
                    "base_metrics": bm,
                    "circuit_metrics": transfer[band],
                    "ablation_metrics": metrics_from_logits(aL, aT, aW),
                    "transfer": transfer,
                    "acdc_seed": ACDC_SEED, "eval_seed": EVAL_SEED,
                }, open(d / "metrics.json", "w"), indent=1)
                print(f"{mname}/{band}/{draw}: edges={n_edges} "
                      f"({n_edges/total:.1%}) own2way="
                      f"{transfer[band]['acc_2way']:.3f} [{dt:.0f}s]", flush=True)
                del dev, inv_dev
                cleanup_gpu()
        del patchable
        safe_delete_model(model)
        cleanup_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "discover"], required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--thresholds", type=str, default=None,
                    help='discover: JSON {"model": tau} or "auto" to read sweeps')
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--early-stop", action="store_true")
    ap.add_argument("--draws", nargs="+", default=None,
                    help="subset of draws (shard discovery across GPUs)")
    ap.add_argument("--bands", nargs="+", default=None,
                    help="subset of bands (per-circuit sharding across GPUs)")
    ap.add_argument("--sweep-thresholds", nargs="+", type=float, default=None,
                    help="run only these thresholds (parallel sweep shards)")
    ap.add_argument("--csv-suffix", default="",
                    help="suffix for the sweep CSV (avoid clobber in shards)")
    args = ap.parse_args()

    if args.mode == "sweep":
        sweep(args.models, args.device, BASE / "sweep", args.early_stop,
              args.sweep_thresholds, args.csv_suffix)
    else:
        import pandas as pd
        if args.thresholds and args.thresholds != "auto":
            thr = json.loads(args.thresholds)
        else:
            thr = {}
            for m in args.models:
                df = pd.read_csv(BASE / "sweep" / f"sweep_{m}.csv")
                sel = df[df.selected]
                assert len(sel) == 1, f"no selected threshold for {m}"
                thr[m] = float(sel.threshold.iloc[0])
        print("thresholds:", thr, flush=True)
        discover(args.models, thr, args.device, BASE / "circuits", args.draws,
                 args.bands)


if __name__ == "__main__":
    main()
