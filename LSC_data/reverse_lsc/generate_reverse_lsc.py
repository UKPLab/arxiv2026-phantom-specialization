#!/usr/bin/env python3
"""
Reverse-Copy LSC Data Generator
================================
Generates a variant of Literal Sequence Copying where the target token T
is placed BEFORE the source prefix instead of after it.

Standard LSC:
    Pos:  0  1  2  3  4   5   6  7 ... 15   16 17 18 19 20
          S1 S2 S3 S4 S5  T   R1 R2... R10  S1 S2 S3 S4 S5

Reverse-copy LSC:
    Pos:  0   1  2  3  4  5   6  7 ... 15   16 17 18 19 20
          T   S1 S2 S3 S4 S5  R1 R2... R10  S1 S2 S3 S4 S5

In both cases:
  - Prediction is at position 20 (second S5): predict T
  - Total sequence length: 21 tokens (no BOS)
  - Corruption: replace positions 16-20 (repeated prefix) with random tokens

The  difference: standard LSC uses induction (match S5 -> copy offset +1),
while reverse-copy requires retrieving T from BEFORE the source prefix
(match S5 -> retrieve offset -5 from match point), which demands a
mechanistically different attention pattern.

PURPOSE: Positive control for the circuit-comparison pipeline.
If the model can solve reverse-copy, ACDC should extract a different circuit,
and cross-task transfer should be low; validating that the pipeline can
detect genuine mechanistic differences.

Uses the SAME token pool and seed structure as standard LSC.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
# Use the same control-band pool as standard LSC
POOL_PATH = SCRIPT_DIR.parent / "lsc_token_pools" / "matched" / "lsc_pool_control.json"
OUTPUT_DIR = SCRIPT_DIR

# ============================================================================
# PARAMETERS
# ============================================================================

N_SOURCE = 5
N_DISTRACT = 10
N_EXAMPLES = 1500
TRAIN_RATIO = 0.70
TEST_RATIO = 0.15
VAL_RATIO = 0.15

# Use a different seed from standard LSC to avoid identical token selections,
# but keep it deterministic
MASTER_SEED = 42
GENERATION_SEED = MASTER_SEED + 2000  # offset from standard LSC's +1000
SPLIT_SEED = MASTER_SEED + 4000  # offset from standard LSC's +3000


# ============================================================================
# SEQUENCE CONSTRUCTION
# ============================================================================


def build_reverse_lsc_sequence(
    source_ids: list,
    source_strings: list,
    target_id: int,
    target_string: str,
    distract_ids: list,
    distract_strings: list,
) -> dict:
    """
    Build a single reverse-copy LSC sequence (NO BOS).

    Structure: [T] [S1..Sn] [R1..Rm] [S1..Sn]

    The target T is placed BEFORE the source prefix, requiring the model
    to retrieve T from before the matched position rather than after it.
    """
    n_source = len(source_ids)
    n_distract = len(distract_ids)

    # Assemble: T first, then source, distractors, repeated source
    full_ids = [target_id] + source_ids + distract_ids + source_ids
    full_strings = [target_string] + source_strings + distract_strings + source_strings

    total_len = len(full_ids)  # Should be 21

    # Position map (0-indexed, no BOS)
    pos_target = 0  # T
    pos_source = [1, n_source]  # S1..Sn
    pos_distract = [n_source + 1, n_source + n_distract]  # R1..Rm
    pos_repeat = [n_source + 1 + n_distract, total_len - 1]  # S1..Sn (copy)

    return {
        "token_ids": full_ids,
        "token_strings": full_strings,
        "total_len": total_len,
        "positions": {
            "target": pos_target,
            "source": pos_source,
            "distraction": pos_distract,
            "repetition": pos_repeat,
        },
        "source_token_ids": source_ids,
        "source_token_strings": source_strings,
        "target_token_id": target_id,
        "target_token_string": target_string,
        "distractor_token_ids": distract_ids,
        "distractor_token_strings": distract_strings,
        "prediction": {
            "position": total_len - 1,  # position 20 (second S5)
            "target_id": target_id,
            "target_string": target_string,
        },
    }


# ============================================================================
# DATASET GENERATION
# ============================================================================


def load_pool(pool_path: Path) -> dict:
    """Load a token pool JSON file."""
    with open(pool_path) as f:
        pool = json.load(f)
    print(f"Loaded pool: {pool_path.name} ({len(pool['tokens'])} tokens)")
    return pool


def generate_examples(pool: dict, n_examples: int, generation_seed: int) -> list:
    """Generate reverse-copy LSC examples."""
    tokens = pool["tokens"]
    n_pool = len(tokens)
    n_needed = N_SOURCE + 1 + N_DISTRACT  # 16 unique tokens per sequence
    band_type = pool.get("band_type", "baseline")

    if n_pool < n_needed:
        raise ValueError(f"Pool ({n_pool}) < tokens needed ({n_needed})")

    pool_ids = [t["token_id"] for t in tokens]
    pool_strings = [t["token_string"] for t in tokens]

    # Frequency weights for control band
    weights = None
    if band_type == "baseline":
        if "frequency_weights" in pool and pool["frequency_weights"]:
            weights = np.array(pool["frequency_weights"])
        else:
            log_freqs = np.array([t["log_frequency"] for t in tokens])
            weights = np.power(10.0, log_freqs)
        weights = np.clip(weights, 1e-10, None)
        print(
            f"  Control: frequency-weighted sampling "
            f"(weight range: {weights.min():.4f}..{weights.max():.4f})"
        )

    examples = []
    for i in range(n_examples):
        example_seed = generation_seed + i
        rng = np.random.RandomState(example_seed)

        # Sample all unique tokens at once
        if weights is not None:
            p = weights / weights.sum()
            indices = rng.choice(n_pool, size=n_needed, replace=False, p=p)
        else:
            indices = rng.choice(n_pool, size=n_needed, replace=False)

        # Assign roles (same assignment as standard LSC)
        source_idx = indices[:N_SOURCE]
        target_idx = indices[N_SOURCE]
        distract_idx = indices[N_SOURCE + 1 :]

        source_ids = [pool_ids[j] for j in source_idx]
        source_strings = [pool_strings[j] for j in source_idx]
        target_id = pool_ids[target_idx]
        target_string = pool_strings[target_idx]
        distract_ids = [pool_ids[j] for j in distract_idx]
        distract_strings = [pool_strings[j] for j in distract_idx]

        seq = build_reverse_lsc_sequence(
            source_ids,
            source_strings,
            target_id,
            target_string,
            distract_ids,
            distract_strings,
        )
        seq["example_id"] = i
        seq["seed"] = example_seed

        # Per-token frequencies
        seq["token_log_frequencies"] = {
            "source": [round(tokens[j]["log_frequency"], 6) for j in source_idx],
            "target": round(tokens[target_idx]["log_frequency"], 6),
            "distraction": [round(tokens[j]["log_frequency"], 6) for j in distract_idx],
        }

        examples.append(seq)

    return examples


def split_examples(examples: list, split_seed: int) -> dict:
    """Split examples into train/test/val."""
    n = len(examples)
    indices = np.arange(n)
    rng = np.random.RandomState(split_seed)
    rng.shuffle(indices)

    n_train = int(n * TRAIN_RATIO)
    n_test = int(n * TEST_RATIO)
    n_val = n - n_train - n_test

    return {
        "train": [examples[i] for i in indices[:n_train]],
        "test": [examples[i] for i in indices[n_train : n_train + n_test]],
        "val": [examples[i] for i in indices[n_train + n_test :]],
    }


def save_split(examples: list, split_name: str, output_dir: Path) -> Path:
    """Save a split to JSON with metadata header."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{split_name}.json"

    data = {
        "task": "reverse_LSC",
        "variant": "positive_control",
        "draw": 1,
        "master_seed": MASTER_SEED,
        "band": "control",
        "band_type": "baseline",
        "split": split_name,
        "sequence_structure": {
            "format": "[T][S1..S5][R1..R10][S1..S5]",
            "n_source": N_SOURCE,
            "n_distract": N_DISTRACT,
            "total_len": N_SOURCE + 1 + N_DISTRACT + N_SOURCE,
            "bos_included": False,
            "prediction_position": N_SOURCE + 1 + N_DISTRACT + N_SOURCE - 1,
            "prediction_description": (
                "At position 20 (second S5), predict T (position 0). "
                "Unlike standard LSC where T follows S5, here T precedes "
                "the source prefix, requiring a different retrieval mechanism."
            ),
        },
        "n_examples": len(examples),
        "token_construction": (
            "Sequences are token ID lists, NOT text. All tokens are word_en "
            "BPE tokens with \u0120 (U+0120) space prefix baked into the "
            "embedding. Feed token_ids directly to the model. Do NOT "
            "re-tokenize. BOS is NOT included; prepend it at inference time "
            "if needed (e.g. TransformerLens prepend_bos=True shifts all "
            "positions by +1)."
        ),
        "created_at": datetime.now().isoformat(),
        "examples": examples,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Saved {split_name}: {len(examples)} examples -> {path}")
    return path


def main():
    print("=" * 60)
    print("Reverse-Copy LSC Generator")
    print("=" * 60)

    # Load pool
    if not POOL_PATH.exists():
        print(f"ERROR: Pool not found at {POOL_PATH}")
        sys.exit(1)
    pool = load_pool(POOL_PATH)

    # Generate
    print(f"\nGenerating {N_EXAMPLES} reverse-copy sequences...")
    print(f"  Structure: [T][S1..S5][R1..R10][S1..S5]")
    print(f"  Prediction: position 20 -> T (position 0)")
    print(f"  Generation seed: {GENERATION_SEED}")
    examples = generate_examples(pool, N_EXAMPLES, GENERATION_SEED)

    # Verify structure
    ex = examples[0]
    assert ex["total_len"] == 21, f"Expected 21, got {ex['total_len']}"
    assert ex["prediction"]["position"] == 20
    assert ex["token_ids"][0] == ex["target_token_id"], "T should be at position 0"
    assert ex["token_ids"][1:6] == ex["source_token_ids"], "S1-S5 at positions 1-5"
    assert ex["token_ids"][16:21] == ex["source_token_ids"], "Repeated S1-S5 at 16-20"
    print(f"  Structure verified")

    # Split
    print(f"\nSplitting ({TRAIN_RATIO:.0%}/{TEST_RATIO:.0%}/{VAL_RATIO:.0%})...")
    splits = split_examples(examples, SPLIT_SEED)

    # Save
    print(f"\nSaving to {OUTPUT_DIR}/")
    for split_name, split_examples_list in splits.items():
        save_split(split_examples_list, split_name, OUTPUT_DIR)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Done. Generated {N_EXAMPLES} reverse-copy LSC sequences.")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Test:  {len(splits['test'])}")
    print(f"  Val:   {len(splits['val'])}")
    print(f"\nExample sequence (first):")
    ex = examples[0]
    print(f"  Tokens: {ex['token_strings'][:6]} ... {ex['token_strings'][16:]}")
    print(f"  T={ex['target_token_string']} at pos 0")
    print(f"  Predict T at pos {ex['prediction']['position']}")


if __name__ == "__main__":
    main()
