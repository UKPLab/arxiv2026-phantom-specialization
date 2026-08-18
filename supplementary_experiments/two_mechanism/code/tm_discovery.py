"""Pipeline run on the gated learned two-route toy (DESIGN.md).

Mirrors the sva_discovery protocol for the segment under test - ACDC
extraction -> structural comparison -> cross-condition transfer (NO IIA here;
interchange sensitivity is covered by the paper's layer-sweep control):
--sweep:    LSC 11-point grid on mixed/draw_1 (ACDC on 256 train, eval val),
            selection = LSC minimal_acceptable (retention >= 0.80, KL < 0.5,
            smallest circuit) on 2-way.
--discover: tau* per condition x draw; full cross-condition transfer.
--aggregate: universal core, NB12 cell-16 stats, Jaccard structure.
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
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "LSC_circuits"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "circuit_discovery" / "auto-circuit"))

from lsc_acdc_circuit import (  # noqa: E402
    cleanup_gpu, invert_prune_scores, set_all_seeds, threshold_to_tao,
)
from tm_data import BOS, CONDITIONS, DATA_DIR, DRAWS  # noqa: E402

SEQ_LEN, DIVERGE_IDX = 9, 2   # [BOS CUE] shared prefix
ACDC_SEED, EVAL_SEED, TRAIN_SIZE, BATCH = 42, 123, 256, 225
SWEEP_THRESHOLDS = np.logspace(-2, -6, 11)
DEVICE = "cuda:0" if t.cuda.is_available() else "cpu"


def load_tm_model(seed=0):
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    cfg = json.load(open(BASE / f"models/tm_cfg_s{seed}.json"))
    model = HookedTransformer(HookedTransformerConfig(**cfg)).to(DEVICE)
    model.load_state_dict(t.load(BASE / f"models/tm_model_s{seed}.pt",
                                 map_location=DEVICE))
    model.cfg.use_attn_result = True   # same flags as lsc_acdc_circuit:467-469
    model.cfg.use_attn_in = True
    model.cfg.use_hook_mlp_in = True
    model.eval()
    return model


def load_split(cond, draw, split):
    return json.load(open(DATA_DIR / draw / cond / f"{split}.json"))["examples"]


def make_loader(examples, batch_size, seed, n_samples=None):
    from auto_circuit.data import PromptDataset, PromptDataLoader
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    if n_samples:
        idx = idx[:n_samples]
    cp, xp, ans, wrg = [], [], [], []
    for i in idx:
        e = examples[i]
        cp.append(t.tensor([BOS] + e["token_ids"], device=DEVICE))
        xp.append(t.tensor([BOS] + e["corrupt_token_ids"], device=DEVICE))
        ans.append(t.tensor([e["target_token_id"]], device=DEVICE))
        wrg.append(t.tensor([e["wrong_token_id"]], device=DEVICE))
    ds = PromptDataset(clean_prompts=cp, corrupt_prompts=xp,
                       answers=ans, wrong_answers=wrg)
    n = len(idx)
    bs = min(batch_size, n)
    while bs > 1 and n % bs != 0:
        bs -= 1
    return PromptDataLoader(prompt_dataset=ds, seq_len=SEQ_LEN,
                            diverge_idx=DIVERGE_IDX, batch_size=bs)


def run_circuit(patchable, scores_dev, n_edges, examples):
    from auto_circuit.prune import run_circuits
    from auto_circuit.types import PatchType, AblationType
    set_all_seeds(EVAL_SEED)
    loader1 = make_loader(examples, BATCH, EVAL_SEED)
    with t.no_grad():
        outs = run_circuits(model=patchable, dataloader=loader1,
                            test_edge_counts=[n_edges], prune_scores=scores_dev,
                            patch_type=PatchType.TREE_PATCH,
                            ablation_type=AblationType.RESAMPLE)
    set_all_seeds(EVAL_SEED)
    loader2 = make_loader(examples, BATCH, EVAL_SEED)
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


def base_logits_for(model, examples):
    set_all_seeds(EVAL_SEED)
    loader = make_loader(examples, BATCH, EVAL_SEED)
    L, T, W = [], [], []
    with t.no_grad():
        for b in loader:
            L.append(model(b.clean)[:, -1, :].float())
            T.append(b.answers.squeeze(-1))
            W.append(b.wrong_answers.squeeze(-1))
    return t.cat(L), t.cat(T), t.cat(W)


def run_acdc(patchable, examples, threshold):
    from auto_circuit.prune_algos.ACDC import acdc_prune_scores
    tao_exp, tao_base = threshold_to_tao(threshold)
    set_all_seeds(ACDC_SEED)
    loader = make_loader(examples, BATCH, ACDC_SEED, n_samples=TRAIN_SIZE)
    t0 = time.time()
    scores = acdc_prune_scores(model=patchable, dataloader=loader,
                               official_edges=None, tao_exps=[tao_exp],
                               tao_bases=[tao_base],
                               faithfulness_target="kl_div")
    n_edges = sum(int(t.isinf(v).sum()) for v in scores.values())
    return scores, n_edges, time.time() - t0


def get_patchable(model):
    from auto_circuit.utils.graph_utils import patchable_model
    return patchable_model(model=model, factorized=True,
                           slice_output="last_seq", seq_len=None,
                           separate_qkv=False, device=DEVICE)


def sweep(seed):
    import pandas as pd
    model = load_tm_model(seed)
    patchable = get_patchable(model)
    total = len(patchable.edges)
    train = load_split("mixed", "draw_1", "train")
    val = load_split("mixed", "draw_1", "val")
    bL, bT, bW = base_logits_for(patchable, val)
    base = metrics_from_logits(bL, bT, bW)
    print(f"toy model: {total} edges, base mixed val 2way={base['acc_2way']:.3f}",
          flush=True)
    rows = []
    for thr in SWEEP_THRESHOLDS:
        scores, n_edges, dt = run_acdc(patchable, train, float(thr))
        dev = {k: v.to(DEVICE) for k, v in scores.items()}
        cL, cT, cW = run_circuit(patchable, dev, n_edges, val)
        cm = metrics_from_logits(cL, cT, cW, bL)
        inv, n_inv = invert_prune_scores(scores)
        inv_dev = {k: v.to(DEVICE) for k, v in inv.items()}
        aL, aT, aW = run_circuit(patchable, inv_dev, n_inv, val)
        am = metrics_from_logits(aL, aT, aW)
        rows.append({"threshold": float(thr), "n_edges": n_edges,
                     "edge_frac": n_edges / total,
                     "base_2way": base["acc_2way"],
                     "circuit_2way": cm["acc_2way"], "kl": cm["kl_div"],
                     "ablated_2way": am["acc_2way"], "acdc_seconds": dt})
        print(f"  thr={thr:.2e} edges={n_edges} ({n_edges/total:.1%}) "
              f"circ2way={cm['acc_2way']:.3f} kl={cm['kl_div']:.3f} "
              f"abl2way={am['acc_2way']:.3f} [{dt:.0f}s]", flush=True)
        cleanup_gpu()
    df = pd.DataFrame(rows)
    df["retention"] = df.circuit_2way / df.base_2way
    ok = df[(df.retention >= 0.80) & (df.kl < 0.5)]
    sel = ok.sort_values("edge_frac").iloc[0] if len(ok) else None
    df["selected"] = df.threshold == (sel.threshold if sel is not None else -1)
    (BASE / "sweep").mkdir(exist_ok=True)
    df.to_csv(BASE / f"sweep/sweep_toy_s{seed}.csv", index=False)
    print(f"SELECTED tau*={sel.threshold if sel is not None else 'NONE'}",
          flush=True)


def discover(threshold, seed):
    model = load_tm_model(seed)
    patchable = get_patchable(model)
    total = len(patchable.edges)
    # complete grid: every circuit evaluated on every (condition, draw) test
    test_cache = {(c, d): load_split(c, d, "test")
                  for c in CONDITIONS for d in DRAWS}
    base_cache = {k: base_logits_for(patchable, v) for k, v in test_cache.items()}
    for draw in DRAWS:
        for cond in CONDITIONS:
            d = BASE / f"circuits/s{seed}" / cond / draw
            if (d / "metrics.json").exists():
                print(f"skip s{seed}/{cond}/{draw}", flush=True)
                continue
            d.mkdir(parents=True, exist_ok=True)
            train = load_split(cond, draw, "train")
            scores, n_edges, dt = run_acdc(patchable, train, threshold)
            with open(d / "prune_scores.pkl", "wb") as f:
                pickle.dump({k: v.cpu() for k, v in scores.items()}, f)
            dev = {k: v.to(DEVICE) for k, v in scores.items()}
            transfer = {}
            for (tc, td), ex in test_cache.items():
                L, T, W = run_circuit(patchable, dev, n_edges, ex)
                transfer[f"{tc}|{td}"] = metrics_from_logits(
                    L, T, W, base_cache[(tc, td)][0])
            inv, n_inv = invert_prune_scores(scores)
            inv_dev = {k: v.to(DEVICE) for k, v in inv.items()}
            aL, aT, aW = run_circuit(patchable, inv_dev, n_inv,
                                     test_cache[(cond, draw)])
            json.dump({"condition": cond, "draw": draw, "threshold": threshold,
                       "seed": seed, "n_edges": n_edges, "total_edges": total,
                       "base_metrics": metrics_from_logits(*base_cache[(cond, draw)]),
                       "circuit_metrics": transfer[f"{cond}|{draw}"],
                       "ablation_metrics": metrics_from_logits(aL, aT, aW),
                       "transfer": transfer,
                       "acdc_seed": ACDC_SEED, "eval_seed": EVAL_SEED},
                      open(d / "metrics.json", "w"), indent=1)
            print(f"s{seed}/{cond}/{draw}: edges={n_edges} ({n_edges/total:.1%}) "
                  f"own2way={transfer[f'{cond}|{draw}']['acc_2way']:.3f} "
                  f"[{dt:.0f}s]", flush=True)
            cleanup_gpu()


def aggregate(seed):
    """Compute and persist every reported quantity.

    - complete functional grid incl. WITHIN-condition cross-draw transfer
    - universal-core boost TE, two documented variants:
        univ_AB    = AND(cue_A, cue_B) masks per draw
        univ_ABmix = AND(cue_A, cue_B, mixed) masks per draw (SVA/NB12 form)
    - structure: Jaccard + DIRECTED containment + size-only baseline (assumes edge
      exchangeability; E[|X&Y|] = |X||Y|/N)
    """
    from scipy import stats
    CIRC = BASE / f"circuits/s{seed}"
    scores, metrics = {}, {}
    for c in CONDITIONS:
        for d in DRAWS:
            with open(CIRC / c / d / "prune_scores.pkl", "rb") as f:
                scores[(c, d)] = pickle.load(f)
            metrics[(c, d)] = json.load(open(CIRC / c / d / "metrics.json"))
    total = metrics[(CONDITIONS[0], DRAWS[0])]["total_edges"]

    def edge_set(s):
        return {(m, *i.tolist()) for m, v in s.items()
                for i in t.isinf(v).nonzero()}

    sets = {k: edge_set(v) for k, v in scores.items()}

    def jac(a, b):
        return len(a & b) / len(a | b) if a | b else 1.0

    def acc(sc, sd, tc, td):
        return metrics[(sc, sd)]["transfer"][f"{tc}|{td}"]["acc_2way"]

    # functional grid summaries
    own = [acc(c, d, c, d) for c in ["cue_A", "cue_B"] for d in DRAWS]
    within_cross_draw = [acc(c, d1, c, d2) for c in ["cue_A", "cue_B"]
                         for d1 in DRAWS for d2 in DRAWS if d1 != d2]
    cross_cond = [acc(sc, d1, tc, d2)
                  for sc, tc in [("cue_A", "cue_B"), ("cue_B", "cue_A")]
                  for d1 in DRAWS for d2 in DRAWS]
    print(f"own acc: {np.mean(own):.3f} | within-cond cross-draw: "
          f"{np.mean(within_cross_draw):.3f} "
          f"({min(within_cross_draw):.3f}-{max(within_cross_draw):.3f}) | "
          f"cross-cond: {np.mean(cross_cond):.3f}")

    # universal-core boosts (same-draw, as in NB12 cell 16)
    model = load_tm_model(seed)
    patchable = get_patchable(model)
    univ_te = {}
    for variant, conds in [("univ_AB", ["cue_A", "cue_B"]),
                           ("univ_ABmix", CONDITIONS)]:
        same_b, cross_b = [], []
        for d in DRAWS:
            univ = {}
            for mod in scores[(conds[0], d)]:
                mask = t.isinf(scores[(conds[0], d)][mod]).clone()
                for c in conds[1:]:
                    mask &= t.isinf(scores[(c, d)][mod])
                v = t.zeros_like(scores[(conds[0], d)][mod])
                v[mask] = float("inf")
                univ[mod] = v
            nu = sum(int(t.isinf(v).sum()) for v in univ.values())
            ud = {k: v.to(DEVICE) for k, v in univ.items()}
            for tc in ["cue_A", "cue_B"]:
                test = load_split(tc, d, "test")
                L, T, W = run_circuit(patchable, ud, nu, test)
                u = metrics_from_logits(L, T, W)["acc_2way"]
                oc = "cue_B" if tc == "cue_A" else "cue_A"
                same_b.append(acc(tc, d, tc, d) - u)
                cross_b.append(acc(oc, d, tc, d) - u)
            cleanup_gpu()
        sa, ca = np.array(same_b), np.array(cross_b)
        te = float(ca.mean() / sa.mean()) if sa.mean() > 0 else float("nan")
        univ_te[variant] = {"te": te, "same_boost": float(sa.mean()),
                            "cross_boost": float(ca.mean())}
        print(f"{variant}: TE={te:.3f} (same={sa.mean():.3f}, "
              f"cross={ca.mean():.3f})")

    # structure: jaccard, containment, size-matched null
    j_within = [jac(sets[(c, d1)], sets[(c, d2)])
                for c in ["cue_A", "cue_B"]
                for a, d1 in enumerate(DRAWS) for d2 in DRAWS[a + 1:]]
    j_cross, contain_small, j_null = [], [], []
    for d1 in DRAWS:
        for d2 in DRAWS:
            A, B = sets[("cue_A", d1)], sets[("cue_B", d2)]
            j_cross.append(jac(A, B))
            small, big = (A, B) if len(A) <= len(B) else (B, A)
            contain_small.append(len(A & B) / len(small))
            e_inter = len(A) * len(B) / total
            j_null.append(e_inter / (len(A) + len(B) - e_inter))
    print(f"Jaccard within={np.mean(j_within):.3f} cross={np.mean(j_cross):.3f} "
          f"size-matched null={np.mean(j_null):.3f} | "
          f"containment(smaller in larger)={np.mean(contain_small):.3f}")

    diff = np.array(own) - np.array([acc(("cue_B" if c == "cue_A" else "cue_A"),
                                          d, c, d)
                                     for c in ["cue_A", "cue_B"] for d in DRAWS])
    p = stats.wilcoxon(diff, alternative="greater")[1] if np.any(diff != 0) else 1.0
    json.dump({"seed": seed, "own_acc": float(np.mean(own)),
               "within_cond_cross_draw_acc": float(np.mean(within_cross_draw)),
               "within_cond_cross_draw_min": float(min(within_cross_draw)),
               "cross_cond_acc": float(np.mean(cross_cond)),
               "universal_te": univ_te,
               "jaccard_within": float(np.mean(j_within)),
               "jaccard_cross": float(np.mean(j_cross)),
               "jaccard_size_matched_null": float(np.mean(j_null)),
               "containment_smaller_in_larger": float(np.mean(contain_small)),
               "wilcoxon_own_gt_cross_p": float(p),
               "n_edges": {f"{c}/{d}": metrics[(c, d)]["n_edges"]
                           for c in CONDITIONS for d in DRAWS}},
              open(BASE / f"results/pipeline_verdict_s{seed}.json", "w"),
              indent=1)
    print(f"Wilcoxon own>cross p={p:.5f}; saved pipeline_verdict_s{seed}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "discover", "aggregate"],
                    required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.mode == "sweep":
        sweep(args.seed)
    elif args.mode == "discover":
        import pandas as pd
        thr = args.threshold
        if thr is None:
            df = pd.read_csv(BASE / f"sweep/sweep_toy_s{args.seed}.csv")
            sel = df[df.selected]
            assert len(sel) == 1
            thr = float(sel.threshold.iloc[0])
        print("tau* =", thr, flush=True)
        discover(thr, args.seed)
    else:
        aggregate(args.seed)


if __name__ == "__main__":
    main()
