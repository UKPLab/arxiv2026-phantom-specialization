"""Double-dissociation certificate: pre-locked component groups, held-out data.

Component groups are fixed in advance (not selected from observed drops) and
evaluated on the VAL split, held out from the pipeline's test evaluations:
  SET_A = ALL attention heads in layers 1..3
  SET_B = MLP layers 1..3
PASS iff ablate SET_A -> drop_A >= 0.5, drop_B <= 0.1
     and ablate SET_B -> drop_B >= 0.5, drop_A <= 0.1.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig

BASE = Path(__file__).resolve().parent.parent
DEVICE = "cuda:0" if t.cuda.is_available() else "cpu"
BOS = 0


def load_examples(cond, split="val", draw="draw_1"):  # draw override: --draw
    d = json.load(open(BASE / f"data/{draw}/{cond}/{split}.json"))["examples"]
    ids = t.tensor([[BOS] + e["token_ids"] for e in d], device=DEVICE)
    tgt = t.tensor([e["target_token_id"] for e in d], device=DEVICE)
    return ids, tgt


def top1(model, ids, tgt, hooks=None):
    with t.no_grad():
        logits = (model.run_with_hooks(ids, fwd_hooks=hooks)
                  if hooks else model(ids))[:, -1, :]
    return float((logits.argmax(-1) == tgt).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--draw", default="draw_1",
                    help="draw_2 = separate validation split (its val is "
                         "unused elsewhere; ablation means still come from "
                         "mixed/train, which discovery also uses)")
    args = ap.parse_args()
    cfg = json.load(open(BASE / f"models/tm_cfg_s{args.seed}.json"))
    model = HookedTransformer(HookedTransformerConfig(**cfg)).to(DEVICE)
    model.load_state_dict(t.load(BASE / f"models/tm_model_s{args.seed}.pt",
                                 map_location=DEVICE))
    model.eval()
    L, H = cfg["n_layers"], cfg["n_heads"]

    data = {c: load_examples(c, draw=args.draw) for c in ["cue_A", "cue_B"]}
    base = {c: top1(model, *data[c]) for c in data}

    ids_m, _ = load_examples("mixed", "train", draw=args.draw)
    _, cache = model.run_with_cache(
        ids_m, names_filter=lambda n: n.endswith(("hook_z", "hook_mlp_out")))
    mean_z = {l: cache[f"blocks.{l}.attn.hook_z"].mean(0, keepdim=True)
              for l in range(L)}
    mean_mlp = {l: cache[f"blocks.{l}.hook_mlp_out"].mean(0, keepdim=True)
                for l in range(L)}
    del cache

    def ablate_z(z, hook, layer):
        return mean_z[layer].expand_as(z)

    def ablate_mlp(x, hook, layer):
        return mean_mlp[layer].expand_as(x)

    hooks_a = [(f"blocks.{l}.attn.hook_z", partial(ablate_z, layer=l))
               for l in range(1, L)]
    hooks_b = [(f"blocks.{l}.hook_mlp_out", partial(ablate_mlp, layer=l))
               for l in range(1, L)]
    accA = {c: top1(model, *data[c], hooks_a) for c in data}
    accB = {c: top1(model, *data[c], hooks_b) for c in data}
    dis_a = (base["cue_A"] - accA["cue_A"] >= 0.5
             and base["cue_B"] - accA["cue_B"] <= 0.1)
    dis_b = (base["cue_B"] - accB["cue_B"] >= 0.5
             and base["cue_A"] - accB["cue_A"] <= 0.1)
    verdict = "PASS" if (dis_a and dis_b) else "FAIL"
    out = {"seed": args.seed, "split": "val", "draw": args.draw, "base": base,
           "ablate_attn_L1_3": accA, "ablate_mlp_L1_3": accB,
           "criteria": {"dissoc_A": dis_a, "dissoc_B": dis_b},
           "verdict": verdict}
    suffix = "" if args.draw == "draw_1" else f"_{args.draw}"
    json.dump(out, open(BASE / f"results/certificate_s{args.seed}{suffix}.json",
                        "w"), indent=1)
    print(f"s{args.seed} base A={base['cue_A']:.3f} B={base['cue_B']:.3f} | "
          f"ablate attn: A={accA['cue_A']:.3f} B={accA['cue_B']:.3f} | "
          f"ablate mlp: A={accB['cue_A']:.3f} B={accB['cue_B']:.3f} | "
          f"{verdict}", flush=True)


if __name__ == "__main__":
    main()
