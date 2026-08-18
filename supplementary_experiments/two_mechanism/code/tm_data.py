"""Two-mechanism task data generator.

Sequence (ids, BOS added by loaders): [CUE] x1..x6 [q], length 8.
q = x_i, i in {2..5}.
CUE_A -> answer x_{i+1}          (contextual retrieval; attention)
CUE_B -> answer PERM[q]           (fixed memorized mapping; MLP lookup)
Wrong answer (2-way metric) = the OTHER rule's answer.
Corrupt prompt = same cue, same i, fresh content tokens (retrieval target
AND mapping input destroyed, condition preserved).
"""

import json
import random
from pathlib import Path

BOS, CUE_A, CUE_B = 0, 1, 2
CONTENT = list(range(3, 515))          # 512 content tokens
D_VOCAB = 515
SEQ_IDS = 8                            # without BOS
PREFIX = {"cue_A": [CUE_A], "cue_B": [CUE_B]}
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"
CONDITIONS = ["cue_A", "cue_B", "mixed"]
DRAWS = ["draw_1", "draw_2", "draw_3"]
SPLITS = {"train": 1050, "val": 225, "test": 225}

# fixed vocab derangement for rule B: single n-cycle over a seeded shuffle
# (an n-cycle guarantees a bijection with no fixed points; pairing two
# independent shuffles can collide)
_r = random.Random(7)
_shuf = CONTENT[:]
_r.shuffle(_shuf)
PERM = {_shuf[j]: _shuf[(j + 1) % len(_shuf)] for j in range(len(_shuf))}


def sample_base(rng):
    """Cue-free base example; both cue variants are derived from it."""
    while True:
        xs = rng.sample(CONTENT, 6)
        i = rng.randint(2, 5)          # 1-indexed within x1..x6
        q = xs[i - 1]
        if xs[i] != PERM[q]:           # keep the 2-way metric unambiguous
            break
    xs2 = rng.sample(CONTENT, 6)
    return {"xs": xs, "i": i, "q": q, "xs2": xs2}


def realize(base, cue):
    """Apply a cue to a base example -> paired prompts differ ONLY in cue."""
    xs, i, q, xs2 = base["xs"], base["i"], base["q"], base["xs2"]
    ans_a, ans_b = xs[i], PERM[q]
    tgt, wrg = (ans_a, ans_b) if cue == "cue_A" else (ans_b, ans_a)
    return {"token_ids": PREFIX[cue] + xs + [q],
            "corrupt_token_ids": PREFIX[cue] + xs2 + [xs2[i - 1]],
            "cue": cue, "i": i, "target_token_id": tgt, "wrong_token_id": wrg}


def sample_example(rng, cue=None):
    if cue is None:
        cue = rng.choice(["cue_A", "cue_B"])
    return realize(sample_base(rng), cue)


def batch_tensors(rng, n, cue=None, device="cpu"):
    import torch as t
    ex = [sample_example(rng, cue) for _ in range(n)]
    ids = t.tensor([[BOS] + e["token_ids"] for e in ex], device=device)
    tgt = t.tensor([e["target_token_id"] for e in ex], device=device)
    wrg = t.tensor([e["wrong_token_id"] for e in ex], device=device)
    return ids, tgt, wrg


def dump_datasets():
    """Frozen condition x draw x split datasets from ONE base pool per draw:
    cue_A and cue_B share identical base examples (paired, differ only in the
    cue token); mixed applies a random cue to the same bases."""
    for d_idx, draw in enumerate(DRAWS, 1):
        rng = random.Random(500_000 + 1000 * d_idx)
        seen, bases = set(), {s: [] for s in SPLITS}
        for split, n in SPLITS.items():
            while len(bases[split]) < n:
                b = sample_base(rng)
                key = tuple(b["xs"] + [b["i"]])
                if key in seen:
                    continue
                seen.add(key)
                bases[split].append(b)
        for cond in CONDITIONS:
            crng = random.Random(900_000 + d_idx)
            for split, bb in bases.items():
                ex = []
                for b in bb:
                    cue = (crng.choice(["cue_A", "cue_B"])
                           if cond == "mixed" else cond)
                    e = realize(b, cue)
                    e["example_id"] = len(ex)
                    ex.append(e)
                od = DATA_DIR / draw / cond
                od.mkdir(parents=True, exist_ok=True)
                json.dump({"condition": cond, "draw": draw, "split": split,
                           "paired": True, "n_examples": len(ex),
                           "examples": ex}, open(od / f"{split}.json", "w"))
        print(f"{draw}: dumped {CONDITIONS} (paired bases)")


if __name__ == "__main__":
    dump_datasets()
