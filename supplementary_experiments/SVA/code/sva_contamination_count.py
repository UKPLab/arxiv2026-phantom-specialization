"""Recount the brow/disc contamination from the frozen datasets (CPU).

Counts, from data_final/ and the discovery loader's own batching logic:
1. contaminated examples over all 22,500 (5 conditions x 3 draws x 1500);
2. contaminated TEST examples over 3,375 (5 x 3 x 225);
3. contaminated examples inside ACDC's effective first batches: for each
   of the 15 matched+unmatched+control (condition, draw) circuits,
   reproduce make_loader(seed=ACDC_SEED, n_samples=256) index order and
   take the first batch (batch size reduced from 225 to the largest
   divisor of 256, i.e. 128; PromptDataLoader preserves order,
   shuffle=False), giving 15 x 128 = 1,920 effective training examples.

A prompt is contaminated if its subject (token_ids[1]) or attractor
(token_ids[4]) is one of Ġbrow/Ġbrows/Ġdisc/Ġdiscs.

Output: results/sva_contamination_counts.json
"""

import json
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "code"))

BANDS = ["low", "medium", "high", "very_high", "control"]
DRAWS = ["draw_1", "draw_2", "draw_3"]
SPLITS = ["train", "val", "test"]
BAD_IDS = {6479, 22931, 1262, 28217}  # Ġbrow, Ġbrows, Ġdisc, Ġdiscs
ACDC_SEED, TRAIN_SIZE, BATCH = 42, 256, 225


def is_bad(e):
    return e["token_ids"][1] in BAD_IDS or e["token_ids"][4] in BAD_IDS


def first_batch_indices(n_examples):
    """Replicate sva_discovery.make_loader's order and batch-size rule."""
    rng = random.Random(ACDC_SEED)
    idx = list(range(n_examples))
    rng.shuffle(idx)
    idx = idx[:TRAIN_SIZE]
    n = len(idx)
    bs = min(BATCH, n)
    while bs > 1 and n % bs != 0:
        bs -= 1
    return idx[:bs], bs


def main():
    total = bad_total = test_total = bad_test = 0
    fb_total = fb_bad = 0
    per_cell = {}
    for band in BANDS:
        for draw in DRAWS:
            splits = {s: json.load(open(
                BASE / "data_final" / draw / band / f"{s}.json"))["examples"]
                for s in SPLITS}
            for s, ex in splits.items():
                total += len(ex)
                nbad = sum(map(is_bad, ex))
                bad_total += nbad
                if s == "test":
                    test_total += len(ex)
                    bad_test += nbad
            train = splits["train"]
            fb_idx, bs = first_batch_indices(len(train))
            nb = sum(is_bad(train[i]) for i in fb_idx)
            fb_total += len(fb_idx)
            fb_bad += nb
            per_cell[f"{band}/{draw}"] = {"first_batch_size": bs,
                                          "first_batch_bad": nb}
    out = {
        "bad_token_ids": sorted(BAD_IDS),
        "all_examples": {"total": total, "contaminated": bad_total},
        "test_examples": {"total": test_total, "contaminated": bad_test},
        "acdc_first_batches": {"total": fb_total, "contaminated": fb_bad},
        "per_cell_first_batch": per_cell,
    }
    json.dump(out, open(BASE / "results/sva_contamination_counts.json", "w"),
              indent=1)
    print(f"all: {bad_total}/{total}  test: {bad_test}/{test_total}  "
          f"acdc first batches: {fb_bad}/{fb_total}", flush=True)
    sizes = {c["first_batch_size"] for c in per_cell.values()}
    print(f"first-batch sizes: {sorted(sizes)}", flush=True)
    print("saved results/sva_contamination_counts.json", flush=True)


if __name__ == "__main__":
    main()
