"""Few-shot/ICL variant: feasibility check.

Task: demonstration-driven copy. Prompt (token ids, BOS added):
    a1..a4 a1..a4 SEP b1..b4 b1..b4 SEP c1..c4   -> predict c1
The query's copy has not started, so the answer is determined only by the
few-shot pattern (two demos), not by within-query induction. Content tokens
(12 distinct per prompt) come from the canonical band pools; corruption =
full content resample; 2-way metric = logit(c1) vs logit(c2).

Gates:
  1. base competence per band x model (top1 + 2-way), 225 examples/cell
  2. one timed ACDC run (160m, tau=1.58e-3, 256 train) -> cost model
"""

import json
import random
import sys
import time
from pathlib import Path

import torch as t

BASE = Path(__file__).resolve().parent.parent
ISC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ISC / "LSC_circuits"))
sys.path.insert(0, str(ISC / "circuit_discovery" / "auto-circuit"))
sys.path.insert(0, str(ISC / "supplementary_experiments" / "code"))

from lsc_acdc_circuit import (  # noqa: E402
    load_model, load_pool, safe_delete_model, cleanup_gpu, set_all_seeds,
    threshold_to_tao,
)
import mean_ablation_cross_band as mab  # noqa: E402

BANDS = ["very_low", "low", "medium", "high", "very_high", "control"]
SEP = 187  # 'Ċ' newline, constant across prompts
N_SEG = 4
SEQ_LEN = 1 + 8 + 1 + 8 + 1 + 4          # BOS + demos/SEPs + query = 23
DIVERGE_IDX = 1
DEVICE = "cuda:0"
MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]


def band_tokens(band):
    pool_dir = (mab.DATA_DIR / "lsc_token_pools"
                / ("unmatched" if band == "very_low" else "matched"))
    pool = load_pool(band if band != "control" else "control", pool_dir)
    toks = [tk["token_id"] for tk in pool["tokens"]]
    weights = None
    if band == "control" and pool.get("frequency_weights"):
        weights = pool["frequency_weights"]
    return toks, weights


def make_examples(band, n, seed):
    toks, weights = band_tokens(band)
    rng = random.Random(seed)
    out, seen = [], set()
    while len(out) < n:
        if weights:
            pick = []
            while len(set(pick)) < 12:
                pick = rng.choices(toks, weights=weights, k=12)
            pick = list(dict.fromkeys(pick))[:12]
        else:
            pick = rng.sample(toks, 12)
        a, b, c = pick[0:4], pick[4:8], pick[8:12]
        key = tuple(pick)
        if key in seen:
            continue
        seen.add(key)
        clean = a + a + [SEP] + b + b + [SEP] + c
        pick2 = rng.sample(toks, 12)
        a2, b2, c2 = pick2[0:4], pick2[4:8], pick2[8:12]
        corrupt = a2 + a2 + [SEP] + b2 + b2 + [SEP] + c2
        out.append({"token_ids": clean, "corrupt_token_ids": corrupt,
                    "target_token_id": c[0], "wrong_token_id": c[1]})
    return out


def base_eval(model, examples, bs=225):
    bos = model.tokenizer.bos_token_id
    n2 = n1 = 0
    with t.no_grad():
        for i in range(0, len(examples), bs):
            ch = examples[i:i + bs]
            ids = t.tensor([[bos] + e["token_ids"] for e in ch], device=DEVICE)
            last = model(ids)[:, -1, :]
            tgt = t.tensor([e["target_token_id"] for e in ch], device=DEVICE)
            wrg = t.tensor([e["wrong_token_id"] for e in ch], device=DEVICE)
            n2 += int((last.gather(1, tgt[:, None])
                       > last.gather(1, wrg[:, None])).sum())
            n1 += int((last.argmax(-1) == tgt).sum())
    return n1 / len(examples), n2 / len(examples)


def main():
    (BASE / "results").mkdir(parents=True, exist_ok=True)
    data = {b: make_examples(b, 225, 42_000 + i)
            for i, b in enumerate(BANDS)}
    rows = []
    for mname in MODELS:
        model = load_model(mname, DEVICE)
        # plain forwards only: per-head result hooks would materialize a
        # [batch, seq, heads, d_head, d_model] tensor (~80 GB at 1.4b/seq 24)
        model.cfg.use_attn_result = False
        model.cfg.use_attn_in = False
        model.cfg.use_hook_mlp_in = False
        for band in BANDS:
            a1, a2 = base_eval(model, data[band])
            rows.append({"model": mname, "band": band,
                         "acc_top1": a1, "acc_2way": a2})
        sub = [r for r in rows if r["model"] == mname]
        print(f"{mname:12s} top1: "
              + " ".join(f"{r['band']}={r['acc_top1']:.3f}" for r in sub)
              + " | 2way: "
              + " ".join(f"{r['band']}={r['acc_2way']:.3f}" for r in sub),
              flush=True)
        safe_delete_model(model)
        cleanup_gpu()
    json.dump(rows, open(BASE / "results/base_gate.json", "w"), indent=1)

    # --- timed ACDC probe: 160m, tau = 1.58e-3, 256 train examples ---
    print("\nACDC cost probe (pythia-160m)...", flush=True)
    from auto_circuit.data import PromptDataset, PromptDataLoader
    from auto_circuit.prune_algos.ACDC import acdc_prune_scores
    from auto_circuit.utils.graph_utils import patchable_model
    model = load_model("pythia-160m", DEVICE)
    bos = model.tokenizer.bos_token_id
    patchable = patchable_model(model=model, factorized=True,
                                slice_output="last_seq", seq_len=None,
                                separate_qkv=False, device=DEVICE)
    train = make_examples("control", 256, 99_001)
    cp = [t.tensor([bos] + e["token_ids"], device=DEVICE) for e in train]
    xp = [t.tensor([bos] + e["corrupt_token_ids"], device=DEVICE) for e in train]
    ans = [t.tensor([e["target_token_id"]], device=DEVICE) for e in train]
    wrg = [t.tensor([e["wrong_token_id"]], device=DEVICE) for e in train]
    ds = PromptDataset(clean_prompts=cp, corrupt_prompts=xp, answers=ans,
                       wrong_answers=wrg)
    loader = PromptDataLoader(prompt_dataset=ds, seq_len=SEQ_LEN,
                              diverge_idx=DIVERGE_IDX, batch_size=256)
    tao_exp, tao_base = threshold_to_tao(0.00158)
    set_all_seeds(42)
    t0 = time.time()
    scores = acdc_prune_scores(model=patchable, dataloader=loader,
                               official_edges=None, tao_exps=[tao_exp],
                               tao_bases=[tao_base],
                               faithfulness_target="kl_div")
    dt = time.time() - t0
    n_edges = sum(int(t.isinf(v).sum()) for v in scores.values())
    print(f"160m ACDC @1.58e-3: {n_edges} edges kept, {dt:.0f}s "
          f"(SVA-160m was 262s; LSC-160m ~1968s)", flush=True)
    json.dump({"acdc_seconds_160m": dt, "n_edges": n_edges},
              open(BASE / "results/cost_probe.json", "w"), indent=1)


if __name__ == "__main__":
    main()
