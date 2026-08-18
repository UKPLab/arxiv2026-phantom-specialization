"""Train the two-mechanism toy (DESIGN.md). Gate: >=0.99 on BOTH conditions."""

import json
import random
from pathlib import Path

import torch as t
from transformer_lens import HookedTransformer, HookedTransformerConfig

from tm_data import D_VOCAB, batch_tensors

BASE = Path(__file__).resolve().parent.parent
DEVICE = "cuda:0" if t.cuda.is_available() else "cpu"
CFG = dict(n_layers=4, d_model=128, n_ctx=9, d_head=32, n_heads=4, d_mlp=512,
           act_fn="gelu", d_vocab=D_VOCAB, normalization_type="LN",
           positional_embedding_type="standard", seed=0)
STEPS, BATCH, LR = 8000, 256, 1e-3


def accuracy(model, rng, cue, n=2000):
    ids, tgt, wrg = batch_tensors(rng, n, cue, DEVICE)
    with t.no_grad():
        logits = model(ids)[:, -1, :]
    top1 = float((logits.argmax(-1) == tgt).float().mean())
    two = float((logits.gather(1, tgt[:, None]) > logits.gather(1, wrg[:, None]))
                .float().mean())
    return top1, two


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t.manual_seed(args.seed)
    cfg = dict(CFG, seed=args.seed)
    model = HookedTransformer(HookedTransformerConfig(**cfg)).to(DEVICE)
    opt = t.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = t.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 200) * max(0.1, 1 - s / STEPS))
    rng = random.Random(42 + 100 * args.seed)
    for step in range(1, STEPS + 1):
        ids, tgt, _ = batch_tensors(rng, BATCH, None, DEVICE)
        logits = model(ids)[:, -1, :]
        loss = t.nn.functional.cross_entropy(logits, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == 1:
            ra, rb = random.Random(9991), random.Random(9992)
            a1, a2 = accuracy(model, ra, "cue_A")
            b1, b2 = accuracy(model, rb, "cue_B")
            print(f"step {step:5d} loss {loss.item():.4f} | "
                  f"A top1={a1:.3f} 2way={a2:.3f} | B top1={b1:.3f} 2way={b2:.3f}",
                  flush=True)
            if a1 >= 0.995 and b1 >= 0.995 and step >= 2000:
                print("early stop: both conditions >= 0.995", flush=True)
                break

    (BASE / "models").mkdir(exist_ok=True)
    t.save(model.state_dict(), BASE / f"models/tm_model_s{args.seed}.pt")
    json.dump(cfg, open(BASE / f"models/tm_cfg_s{args.seed}.json", "w"),
              indent=1)
    ra, rb = random.Random(777_1), random.Random(777_2)
    a1, a2 = accuracy(model, ra, "cue_A", 4000)
    b1, b2 = accuracy(model, rb, "cue_B", 4000)
    gate = {"cue_A_top1": a1, "cue_A_2way": a2, "cue_B_top1": b1,
            "cue_B_2way": b2, "pass": a1 >= 0.99 and b1 >= 0.99}
    json.dump(gate, open(BASE / f"models/gate_s{args.seed}.json", "w"),
              indent=1)
    print("GATE:", gate, flush=True)


if __name__ == "__main__":
    main()
