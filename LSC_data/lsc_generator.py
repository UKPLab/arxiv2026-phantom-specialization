#!/usr/bin/env python3
"""
LSC Dataset Generator
======================
Generates Literal Sequence Copying datasets per frequency band, with
multiple independent draws for measuring within-band sampling noise.

TASK DEFINITION
---------------
LSC (Literal Sequence Copying) tests language models' ability to perform
induction: recognizing repeated patterns and copying what followed them.

Sequence structure (n_source=5, n_distract=10):

    Position:  0  1  2  3  4     5     6  7 ... 15    16 17 18 19 20
    Token:    S1 S2 S3 S4 S5     T    R1 R2 ... R10   S1 S2 S3 S4 S5
              ─── source ────   tgt   ── distraction ── ─── repetition ─

    Total length: n_source + 1 + n_distract + n_source = 21

Prediction target:
    At position 20 (second S5), the model should predict T.
    The induction mechanism: attend back to the first S5 (pos 4), find
    that T followed it, and copy T as the prediction.

The distractor segment separates source from repetition, forcing the
model to use long-range attention (induction heads) rather than local
bigram statistics.

BOS HANDLING
------------
This script stores RAW task sequences (21 tokens).  BOS is NOT included.

BOS is a model/framework convention, not part of the task definition.
TransformerLens adds BOS via prepend_bos=True (making 22 tokens at
inference time).  HuggingFace does not add BOS by default.

By keeping BOS out of the raw data, we decouple data generation from
inference framework.  Downstream code is responsible for prepending BOS
if needed.  With BOS, position indices shift by +1:

    Without BOS (raw data):  S1=0, S5=4, T=5, R1=6, ..., S5*=20
    With BOS (inference):    BOS=0, S1=1, S5=5, T=6, R1=7, ..., S5*=21

TOKEN ID CONSTRUCTION (Ġ prefix handling)
------------------------------------------
All tokens in our pool are word_en tokens with the Ġ (U+0120) space
prefix; e.g. token ID 1234 = "Ġcat", token ID 5678 = "Ġdog".

The Ġ prefix is PART OF THE TOKEN EMBEDDING.  It is not a separate
character to be added at runtime.  This means:

  WRONG: Construct text " cat dog house ... cat dog house" and
           re-tokenize.  The first token may lose its Ġ prefix, and
           multi-token words could split differently.

  RIGHT: Construct the input as a list of token IDs directly:
           [1234, 5678, 9012, ..., 1234, 5678, 9012, ...]
           Feed these IDs to the model without any tokenizer step.

All downstream code MUST use the token_ids field directly.  The
token_strings field is for human readability and debugging only.

RANDOMNESS CONTROL
------------------
Reproducibility is controlled through a seed hierarchy derived from
a single MASTER_SEED per draw:

    Draw 1: MASTER_SEED = 42
    Draw 2: MASTER_SEED = 100042
    Draw 3: MASTER_SEED = 200042

    GENERATION_SEED  = MASTER + 1000   (sequence construction)
    SPLIT_SEED       = MASTER + 3000   (train/test/val splitting)

Within generation, each (band, example_index) pair gets a deterministic
seed: GENERATION_SEED + band_offset + example_index.  This means:

  - Regenerating a single band does not affect other bands
  - Adding more examples extends the sequence, does not change existing ones
  - The same MASTER_SEED always produces the same dataset
  - Different draws sample different tokens from the SAME pool

Multiple draws quantify within-band token sampling noise.  If results
are consistent across draws, the findings are robust to which specific
tokens were sampled from the pool.

EXPERIMENTAL DESIGN
--------------------
All tokens within a single sequence come from the SAME frequency band.
Source tokens (S1-S5), target token (T), and distractor tokens (R1-R10)
are ALL drawn from the same band pool.  All 16 tokens in a sequence
are unique (sampled without replacement).

For core bands: tokens are sampled uniformly from the pool.
For control: tokens are sampled with frequency weights (probability
proportional to pretraining frequency), reflecting the natural
distribution the model was trained on.

INPUT
-----
LSC-specific token pools (from lsc_token_pools.py):
  LSC_data/lsc_token_pools/matched/lsc_pool_{band}.json
  LSC_data/lsc_token_pools/unmatched/lsc_pool_{band}.json

OUTPUTS
-------
LSC_data/
├── datasets/
│   ├── matched/                       <- --variant auto-detected from pool-dir
│   │   ├── draw_1/
│   │   │   ├── low/
│   │   │   │   ├── train.json
│   │   │   │   ├── test.json
│   │   │   │   └── val.json
│   │   │   ├── medium/ ...
│   │   │   └── control/ ...
│   │   ├── draw_2/ ...
│   │   └── draw_3/ ...
│   └── unmatched/
│       ├── draw_1/
│       │   ├── very_low/ ...
│       │   ├── low/ ...
│       │   └── control/ ...
│       ├── draw_2/ ...
│       └── draw_3/ ...
├── metadata/
│   ├── matched/draw_1/ ...
│   └── unmatched/draw_1/ ...
└── logs/
    └── lsc_generation_<timestamp>.log

Usage:
    # Primary analysis (matched, length-controlled)
    python lsc_generator.py --pool-dir ./lsc_token_pools/matched
    # Robustness check (unmatched, original LC WORD pools)
    python lsc_generator.py --pool-dir ./lsc_token_pools/unmatched
    # Explicit variant override
    python lsc_generator.py --pool-dir ./lsc_token_pools/matched --variant matched
    python lsc_generator.py --bands low medium high very_high control
"""

import json
import argparse
import logging
import sys
from pathlib import Path
from collections import OrderedDict
from datetime import datetime

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
POOL_DIR = SCRIPT_DIR / "lsc_token_pools" / "matched"
OUTPUT_DIR = SCRIPT_DIR

# Band names in canonical order (must match band design / pool export)
BAND_NAMES = [
    "bottom_tail",
    "very_low",
    "low",
    "medium",
    "high",
    "very_high",
    "top_tail",
    "control",
]
DEFAULT_BASE_SEED = 42
DEFAULT_N_DRAWS = 3
# Offset between draws must exceed max per-draw seed range.
# Per draw: generation_seed + 7*10000 + 1500 = generation_seed + 71500
# Using 100_000 gives comfortable margin.
DRAW_SEED_SPACING = 100_000


def compute_seeds(master_seed: int) -> dict:
    """
    Derive all seeds from a single master seed.

    Offsets are chosen to avoid overlap even if individual
    seed spaces span thousands of values.
    """
    return {
        "master": master_seed,
        "generation": master_seed + 1000,
        "split": master_seed + 3000,
    }


def band_seed_offset(band_name: str) -> int:
    """
    Deterministic offset for a band name.

    Uses the canonical band index * 10000 so per-example seeds
    within different bands never collide.
    """
    try:
        idx = BAND_NAMES.index(band_name)
    except ValueError:
        # Fallback for unknown bands
        idx = abs(hash(band_name)) % 100
    return idx * 10000


DEFAULT_N_SOURCE = 5
DEFAULT_N_DISTRACT = 10
DEFAULT_N_EXAMPLES = 1500
TRAIN_RATIO = 0.70
TEST_RATIO = 0.15
VAL_RATIO = 0.15


def setup_logging(output_dir: Path, debug: bool = False):
    """Configure logging to both console and file."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"lsc_generation_{timestamp}.log"

    level = logging.DEBUG if debug else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__), log_file


def sample_token_indices(
    n_pool: int,
    n_needed: int,
    rng: np.random.RandomState,
    weights: np.ndarray = None,
) -> np.ndarray:
    """
    Sample n_needed unique token indices from a pool.

    For core/exploratory bands: uniform sampling (weights=None).
    For control: frequency-weighted sampling.
    """
    if weights is not None:
        p = weights / weights.sum()
        return rng.choice(n_pool, size=n_needed, replace=False, p=p)
    return rng.choice(n_pool, size=n_needed, replace=False)


def build_lsc_sequence(
    source_ids: list,
    source_strings: list,
    target_id: int,
    target_string: str,
    distract_ids: list,
    distract_strings: list,
) -> dict:
    """
    Build a single LSC sequence (NO BOS).

    Structure: [S1..Sn] [T] [R1..Rm] [S1..Sn]
    """
    n_source = len(source_ids)
    n_distract = len(distract_ids)

    # Assemble
    full_ids = source_ids + [target_id] + distract_ids + source_ids
    full_strings = source_strings + [target_string] + distract_strings + source_strings

    total_len = len(full_ids)

    # Position map (0-indexed, no BOS)
    pos_source = [0, n_source - 1]  # S1..Sn
    pos_target = n_source  # T
    pos_distract = [n_source + 1, n_source + n_distract]  # R1..Rm
    pos_repeat = [n_source + 1 + n_distract, total_len - 1]  # S1..Sn (copy)

    return {
        "token_ids": full_ids,
        "token_strings": full_strings,
        "total_len": total_len,
        "positions": {
            "source": pos_source,
            "target": pos_target,
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
            "position": total_len - 1,
            "target_id": target_id,
            "target_string": target_string,
        },
    }


def generate_band_examples(
    pool: dict,
    band_name: str,
    n_source: int,
    n_distract: int,
    n_examples: int,
    generation_seed: int,
) -> list:
    """
    Generate LSC examples for a single frequency band.

    Each example gets a deterministic seed:
        generation_seed + band_offset + example_index

    This means regenerating one band doesn't affect others,
    and extending with more examples doesn't change existing ones.
    """
    logger = logging.getLogger(__name__)

    tokens = pool["tokens"]
    n_pool = len(tokens)
    band_type = pool["band_type"]
    n_needed = n_source + 1 + n_distract

    if n_pool < n_needed:
        logger.error(f"  {band_name}: pool ({n_pool}) < tokens needed ({n_needed})")
        return None

    # Precompute arrays for fast lookup
    pool_ids = [t["token_id"] for t in tokens]
    pool_strings = [t["token_string"] for t in tokens]

    # Frequency weights for control band
    weights = None
    weight_source = None
    if band_type == "baseline":
        # Prefer pre-computed frequency_weights from pool (script 10)
        # Fall back to recomputation from log_frequency for backwards compatibility
        if "frequency_weights" in pool and pool["frequency_weights"]:
            weights = np.array(pool["frequency_weights"])
            weight_source = "pool"
        else:
            log_freqs = np.array([t["log_frequency"] for t in tokens])
            weights = np.power(10.0, log_freqs)
            weight_source = "recomputed"
        weights = np.clip(weights, 1e-10, None)
        logger.info(
            f"  Control: frequency-weighted sampling ({weight_source}) "
            f"(weight range: {weights.min():.4f}..{weights.max():.4f})"
        )

    # Per-band seed offset
    offset = band_seed_offset(band_name)

    examples = []
    for i in range(n_examples):
        # Deterministic per-example seed
        example_seed = generation_seed + offset + i
        rng = np.random.RandomState(example_seed)

        # Sample all unique tokens at once
        indices = sample_token_indices(n_pool, n_needed, rng, weights)

        # Assign roles
        source_idx = indices[:n_source]
        target_idx = indices[n_source]
        distract_idx = indices[n_source + 1 :]

        source_ids = [pool_ids[j] for j in source_idx]
        source_strings = [pool_strings[j] for j in source_idx]
        target_id = pool_ids[target_idx]
        target_string = pool_strings[target_idx]
        distract_ids = [pool_ids[j] for j in distract_idx]
        distract_strings = [pool_strings[j] for j in distract_idx]

        seq = build_lsc_sequence(
            source_ids,
            source_strings,
            target_id,
            target_string,
            distract_ids,
            distract_strings,
        )
        seq["example_id"] = i
        seq["seed"] = example_seed

        # Per-token frequencies for analysis
        seq["token_log_frequencies"] = {
            "source": [round(tokens[j]["log_frequency"], 6) for j in source_idx],
            "target": round(tokens[target_idx]["log_frequency"], 6),
            "distraction": [round(tokens[j]["log_frequency"], 6) for j in distract_idx],
        }

        examples.append(seq)

    return examples


def split_examples(
    examples: list,
    split_seed: int,
    train_ratio: float = TRAIN_RATIO,
    test_ratio: float = TEST_RATIO,
    val_ratio: float = VAL_RATIO,
) -> dict:
    """
    Split examples into train/test/val using a dedicated seed.

    The split seed is independent of generation seeds, so changing
    the split ratio doesn't require regenerating examples.
    """
    assert abs(train_ratio + test_ratio + val_ratio - 1.0) < 1e-6

    n = len(examples)
    indices = np.arange(n)

    rng = np.random.RandomState(split_seed)
    rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_test = int(n * test_ratio)
    # val gets the remainder to avoid rounding issues
    n_val = n - n_train - n_test

    train_idx = indices[:n_train]
    test_idx = indices[n_train : n_train + n_test]
    val_idx = indices[n_train + n_test :]

    splits = {
        "train": [examples[i] for i in sorted(train_idx)],
        "test": [examples[i] for i in sorted(test_idx)],
        "val": [examples[i] for i in sorted(val_idx)],
    }

    return splits


def validate_examples(
    examples: list,
    n_source: int,
    n_distract: int,
) -> list:
    """
    Run sanity checks on generated examples.
    Returns list of warning strings (empty = all good).
    """
    warnings = []
    expected_len = n_source + 1 + n_distract + n_source

    for ex in examples:
        i = ex["example_id"]

        # Length
        if len(ex["token_ids"]) != expected_len:
            warnings.append(f"ex{i}: len={len(ex['token_ids'])} != {expected_len}")

        if ex["total_len"] != expected_len:
            warnings.append(f"ex{i}: total_len field mismatch")

        # Source == repetition
        pos = ex["positions"]
        src = ex["token_ids"][pos["source"][0] : pos["source"][1] + 1]
        rep = ex["token_ids"][pos["repetition"][0] : pos["repetition"][1] + 1]
        if src != rep:
            warnings.append(f"ex{i}: repetition != source")

        # Target at correct position
        if ex["token_ids"][pos["target"]] != ex["target_token_id"]:
            warnings.append(f"ex{i}: target position mismatch")

        # Prediction target matches
        if ex["prediction"]["target_id"] != ex["target_token_id"]:
            warnings.append(f"ex{i}: prediction target mismatch")

        # All unique (source + target + distractors)
        unique = (
            ex["source_token_ids"]
            + [ex["target_token_id"]]
            + ex["distractor_token_ids"]
        )
        if len(set(unique)) != len(unique):
            warnings.append(f"ex{i}: duplicate tokens")

        # Distraction length
        dist = ex["token_ids"][pos["distraction"][0] : pos["distraction"][1] + 1]
        if len(dist) != n_distract:
            warnings.append(f"ex{i}: distraction len={len(dist)} != {n_distract}")

    return warnings


def compute_token_overlap_stats(
    examples: list,
    pool_size: int,
) -> dict:
    """
    Compute statistics about token reuse across examples.

    For small pools, the same tokens will appear in many examples,
    creating statistical dependency. This function quantifies that overlap.

    Returns dict with:
        - tokens_per_example: number of unique tokens per example
        - total_token_slots: total token appearances across all examples
        - unique_tokens_used: number of distinct tokens actually used
        - pool_coverage: fraction of pool that was sampled at least once
        - mean_token_appearances: average times each used token appears
        - max_token_appearances: maximum times any token appears
        - appearance_distribution: {appearances: count} for analysis
    """
    from collections import Counter

    # Count appearances of each token across all examples
    token_counts = Counter()
    tokens_per_example = 0

    for ex in examples:
        # Unique tokens in this example (source + target + distractors)
        unique_tokens = set(
            ex["source_token_ids"]
            + [ex["target_token_id"]]
            + ex["distractor_token_ids"]
        )
        tokens_per_example = len(unique_tokens)
        token_counts.update(unique_tokens)

    total_slots = len(examples) * tokens_per_example
    unique_used = len(token_counts)
    appearances = list(token_counts.values())

    # Distribution of appearances (how many tokens appear 1x, 2x, etc.)
    appearance_dist = Counter(appearances)
    # Summarize into buckets for readability
    buckets = {
        "1-10": sum(c for a, c in appearance_dist.items() if 1 <= a <= 10),
        "11-50": sum(c for a, c in appearance_dist.items() if 11 <= a <= 50),
        "51-100": sum(c for a, c in appearance_dist.items() if 51 <= a <= 100),
        "101+": sum(c for a, c in appearance_dist.items() if a > 100),
    }

    return {
        "tokens_per_example": tokens_per_example,
        "total_token_slots": total_slots,
        "unique_tokens_used": unique_used,
        "pool_size": pool_size,
        "pool_coverage": round(unique_used / pool_size, 4) if pool_size > 0 else 0,
        "mean_token_appearances": round(np.mean(appearances), 2) if appearances else 0,
        "median_token_appearances": int(np.median(appearances)) if appearances else 0,
        "max_token_appearances": max(appearances) if appearances else 0,
        "min_token_appearances": min(appearances) if appearances else 0,
        "appearance_buckets": buckets,
        "note": (
            "High token reuse creates statistical dependency between examples. "
            "If pool_coverage < 1.0, some tokens were never sampled. "
            "If mean_token_appearances >> 1, most tokens appear in multiple examples."
        ),
    }


def save_split(
    examples: list,
    split_name: str,
    band_name: str,
    band_type: str,
    log_freq_range: list,
    sequence_structure: dict,
    output_path: Path,
    draw_index: int = 1,
    master_seed: int = 42,
    variant: str = "matched",
):
    """Save a single split (train/test/val) as JSON."""
    dataset = OrderedDict()
    dataset["task"] = "LSC"
    dataset["variant"] = variant
    dataset["draw"] = draw_index
    dataset["master_seed"] = master_seed
    dataset["band"] = band_name
    dataset["band_type"] = band_type
    dataset["split"] = split_name
    dataset["log_freq_range"] = log_freq_range
    dataset["sequence_structure"] = sequence_structure
    dataset["n_examples"] = len(examples)
    dataset["token_construction"] = (
        "Sequences are token ID lists, NOT text. All tokens are word_en "
        "BPE tokens with Ġ (U+0120) space prefix baked into the embedding. "
        "Feed token_ids directly to the model. Do NOT re-tokenize. "
        "BOS is NOT included; prepend it at inference time if needed "
        "(e.g. TransformerLens prepend_bos=True shifts all positions by +1)."
    )
    dataset["created_at"] = datetime.now().isoformat()
    dataset["examples"] = examples

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)


def generate_single_draw(
    draw_index: int,
    master_seed: int,
    pools: dict,
    bands_to_generate: list,
    n_source: int,
    n_distract: int,
    n_examples: int,
    datasets_dir: Path,
    metadata_dir: Path,
    sequence_structure: dict,
    pool_dir: Path,
    variant: str = "matched",
) -> dict:
    """
    Generate one complete draw (all bands) for a given master seed.

    Returns draw summary dict or None on failure.
    """
    logger = logging.getLogger(__name__)
    seeds = compute_seeds(master_seed)
    n_needed = n_source + 1 + n_distract
    seq_len = n_source + 1 + n_distract + n_source

    logger.info(
        f"  Seeds: master={seeds['master']}, "
        f"generation={seeds['generation']}, split={seeds['split']}"
    )

    summary_bands = OrderedDict()
    all_warnings = []
    skipped = []
    seed_records = []

    for band in bands_to_generate:
        pool = pools[band]
        logger.info(f"  Band: {band} ({pool['n_tokens']:,} tokens)")

        if pool["n_tokens"] < n_needed:
            logger.warning(f"    SKIPPING: pool too small")
            skipped.append(band)
            continue

        # Generate
        examples = generate_band_examples(
            pool=pool,
            band_name=band,
            n_source=n_source,
            n_distract=n_distract,
            n_examples=n_examples,
            generation_seed=seeds["generation"],
        )

        if examples is None:
            skipped.append(band)
            continue

        # Validate
        warnings = validate_examples(examples, n_source, n_distract)
        all_warnings.extend(warnings)
        if warnings:
            for w in warnings:
                logger.warning(f"    VALIDATION: {w}")
        else:
            logger.info(f"    {len(examples)} examples validated")

        # Token overlap analysis
        overlap_stats = compute_token_overlap_stats(examples, pool["n_tokens"])
        logger.info(
            f"    Overlap: coverage={overlap_stats['pool_coverage']:.1%}, "
            f"mean_app={overlap_stats['mean_token_appearances']:.1f}, "
            f"max={overlap_stats['max_token_appearances']}"
        )

        # Split
        splits = split_examples(examples, seeds["split"])
        logger.info(
            f"    Split: train={len(splits['train'])}, "
            f"test={len(splits['test'])}, val={len(splits['val'])}"
        )

        # Save
        band_dir = datasets_dir / band
        band_dir.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in splits.items():
            split_path = band_dir / f"{split_name}.json"
            save_split(
                examples=split_data,
                split_name=split_name,
                band_name=band,
                band_type=pool["band_type"],
                log_freq_range=pool["log_freq_range"],
                sequence_structure=sequence_structure,
                output_path=split_path,
                draw_index=draw_index,
                master_seed=master_seed,
                variant=variant,
            )

        # Record seeds
        offset = band_seed_offset(band)
        seed_records.append(
            {
                "band": band,
                "generation_seed_base": seeds["generation"],
                "band_offset": offset,
                "example_seed_range": [
                    seeds["generation"] + offset,
                    seeds["generation"] + offset + n_examples - 1,
                ],
                "split_seed": seeds["split"],
            }
        )

        # Summary entry
        sampling = (
            "frequency_weighted" if pool["band_type"] == "baseline" else "uniform"
        )
        summary_bands[band] = OrderedDict(
            [
                ("band_type", pool["band_type"]),
                ("pool_size", pool["n_tokens"]),
                ("n_examples", len(examples)),
                ("train", len(splits["train"])),
                ("test", len(splits["test"])),
                ("val", len(splits["val"])),
                ("sampling", sampling),
                ("log_freq_range", pool["log_freq_range"]),
                ("token_overlap", overlap_stats),
            ]
        )

    # ---- Save seed manifest for this draw ----
    seed_manifest = OrderedDict()
    seed_manifest["description"] = (
        "Complete record of all seeds used in this draw. "
        "Reproduce exactly by using the same master_seed."
    )
    seed_manifest["draw_index"] = draw_index
    seed_manifest["master_seed"] = seeds["master"]
    seed_manifest["derived_seeds"] = {
        "generation": seeds["generation"],
        "split": seeds["split"],
    }
    seed_manifest["seed_formula"] = (
        "example_seed = generation_seed + band_offset + example_index"
    )
    seed_manifest["per_band"] = seed_records

    metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata_dir / "seed_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(seed_manifest, f, indent=2)

    # ---- Save generation summary for this draw ----
    summary = OrderedDict()
    summary["task"] = "LSC"
    summary["variant"] = variant
    summary["draw_index"] = draw_index
    summary["master_seed"] = master_seed
    summary["created_at"] = datetime.now().isoformat()
    summary["sequence_structure"] = sequence_structure
    summary["parameters"] = {
        "n_examples_per_band": n_examples,
        "master_seed": seeds["master"],
        "train_ratio": TRAIN_RATIO,
        "test_ratio": TEST_RATIO,
        "val_ratio": VAL_RATIO,
    }
    summary["pool_dir"] = str(pool_dir)
    summary["n_bands_generated"] = len(summary_bands)
    summary["n_bands_skipped"] = len(skipped)
    summary["skipped_bands"] = skipped
    summary["total_examples"] = sum(b["n_examples"] for b in summary_bands.values())
    summary["validation_warnings"] = len(all_warnings)
    summary["bands"] = summary_bands

    summary_path = metadata_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "summary_bands": summary_bands,
        "all_warnings": all_warnings,
        "skipped": skipped,
        "seeds": seeds,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate LSC datasets for each frequency band",
    )
    parser.add_argument(
        "--pool-dir",
        type=Path,
        default=POOL_DIR,
        help="Directory containing lsc_pool_*.json files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Output root directory"
    )
    parser.add_argument(
        "--n-source",
        type=int,
        default=DEFAULT_N_SOURCE,
        help="Source tokens S1..Sn (default: 5)",
    )
    parser.add_argument(
        "--n-distract",
        type=int,
        default=DEFAULT_N_DISTRACT,
        help="Distractor tokens R1..Rm (default: 10)",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=DEFAULT_N_EXAMPLES,
        help="Examples per band before splitting (default: 1500)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=DEFAULT_BASE_SEED,
        help="Base seed; draw i uses base_seed + (i-1) (default: 42)",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=DEFAULT_N_DRAWS,
        help="Number of independent draws (default: 3)",
    )
    parser.add_argument(
        "--bands",
        nargs="*",
        default=None,
        help="Specific bands to generate (default: auto-detect from pool dir)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Pool variant name for output subdir (default: auto-detect "
        "from pool-dir basename, e.g. 'matched' or 'unmatched')",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # ---- Auto-detect variant from pool-dir basename ----
    if args.variant is None:
        dirname = args.pool_dir.name
        if dirname in ("matched", "unmatched"):
            args.variant = dirname
        else:
            args.variant = dirname
            # Will still work, just uses the dir name as variant

    # ---- Setup ----
    out = args.output_dir
    logger, log_file = setup_logging(out, args.debug)

    n_needed = args.n_source + 1 + args.n_distract
    seq_len = args.n_source + 1 + args.n_distract + args.n_source

    # ---- Auto-detect available bands from pool directory ----
    if args.bands:
        bands_to_generate = args.bands
    else:
        available = []
        for band in BAND_NAMES:
            if (args.pool_dir / f"lsc_pool_{band}.json").exists():
                available.append(band)
        bands_to_generate = available

    if not bands_to_generate:
        logger.error(f"No pool files found in {args.pool_dir}")
        return 1

    # ---- Validate pools exist ----
    missing = []
    for band in bands_to_generate:
        if not (args.pool_dir / f"lsc_pool_{band}.json").exists():
            missing.append(band)
    if missing:
        logger.error(f"Missing pool files: {missing}")
        return 1

    # ---- Load pools once (shared across draws) ----
    pools = {}
    for band in bands_to_generate:
        pool_path = args.pool_dir / f"lsc_pool_{band}.json"
        with open(pool_path) as f:
            pools[band] = json.load(f)

    # ---- Compute draw seeds ----
    # Large spacing prevents per-example seed collisions across draws
    draw_seeds = [args.base_seed + i * DRAW_SEED_SPACING for i in range(args.n_draws)]

    # ---- Sequence structure metadata (shared across all bands/draws) ----
    sequence_structure = {
        "format": f"[S1..S{args.n_source}][T][R1..R{args.n_distract}]"
        f"[S1..S{args.n_source}]",
        "n_source": args.n_source,
        "n_distract": args.n_distract,
        "total_len": seq_len,
        "bos_included": False,
        "prediction_position": seq_len - 1,
        "prediction_description": (
            f"At position {seq_len - 1} (second S{args.n_source}), predict T"
        ),
    }

    logger.info("=" * 60)
    logger.info("LSC DATASET GENERATOR")
    logger.info("=" * 60)
    logger.info(f"  Variant:          {args.variant}")
    logger.info(f"  Pool directory:   {args.pool_dir}")
    logger.info(f"  Output directory: {out}")
    logger.info(
        f"  Structure: [S1..S{args.n_source}][T]"
        f"[R1..R{args.n_distract}][S1..S{args.n_source}]"
    )
    logger.info(f"  Sequence length:  {seq_len} tokens (no BOS)")
    logger.info(f"  Unique tokens/seq: {n_needed}")
    logger.info(f"  Examples/band:    {args.n_examples}")
    logger.info(
        f"  Splits:           train={TRAIN_RATIO} / test={TEST_RATIO} / val={VAL_RATIO}"
    )
    logger.info(f"  Draws:            {args.n_draws} (seeds: {draw_seeds})")
    logger.info(f"  Bands:            {bands_to_generate}")
    pool_sizes = {b: pools[b]["n_tokens"] for b in bands_to_generate}
    logger.info(f"  Pool sizes:       {pool_sizes}")
    logger.info(f"  Log file:         {log_file}")
    logger.info("")

    # ---- Generate each draw ----
    draw_results = []
    total_warnings = 0

    for draw_idx, master_seed in enumerate(draw_seeds, start=1):
        draw_label = f"draw_{draw_idx}"
        logger.info("=" * 60)
        logger.info(f"DRAW {draw_idx}/{args.n_draws}  (master_seed={master_seed})")
        logger.info("=" * 60)

        datasets_dir = out / "datasets" / args.variant / draw_label
        metadata_dir = out / "metadata" / args.variant / draw_label

        result = generate_single_draw(
            draw_index=draw_idx,
            master_seed=master_seed,
            pools=pools,
            bands_to_generate=bands_to_generate,
            n_source=args.n_source,
            n_distract=args.n_distract,
            n_examples=args.n_examples,
            datasets_dir=datasets_dir,
            metadata_dir=metadata_dir,
            sequence_structure=sequence_structure,
            pool_dir=args.pool_dir,
            variant=args.variant,
        )

        draw_results.append(result)
        total_warnings += len(result["all_warnings"])
        logger.info("")

    # ---- Final log ----
    logger.info("=" * 60)
    logger.info("LSC GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(
        f"  Structure: [S1..S{args.n_source}][T]"
        f"[R1..R{args.n_distract}][S1..S{args.n_source}]  "
        f"({seq_len} tokens, no BOS)"
    )
    logger.info(f"  Predict T at position {seq_len - 1}")
    logger.info(f"  Draws: {args.n_draws} (seeds: {draw_seeds})")
    logger.info("")

    # Print summary table from first draw (all draws have same structure)
    result0 = draw_results[0]
    logger.info(
        f"  {'band':<15s}  {'type':<12s}  {'pool':>6s}  "
        f"{'total':>6s}  {'train':>6s}  {'test':>5s}  {'val':>5s}  "
        f"{'coverage':>8s}  {'mean_app':>8s}"
    )
    logger.info(
        f"  {'-' * 15}  {'-' * 12}  {'-' * 6}  "
        f"{'-' * 6}  {'-' * 6}  {'-' * 5}  {'-' * 5}  "
        f"{'-' * 8}  {'-' * 8}"
    )
    for name, info in result0["summary_bands"].items():
        overlap = info.get("token_overlap", {})
        coverage_pct = f"{overlap.get('pool_coverage', 0) * 100:.0f}%"
        mean_app = f"{overlap.get('mean_token_appearances', 0):.1f}"
        logger.info(
            f"  {name:<15s}  {info['band_type']:<12s}  "
            f"{info['pool_size']:>6,d}  "
            f"{info['n_examples']:>6,d}  "
            f"{info['train']:>6,d}  "
            f"{info['test']:>5,d}  "
            f"{info['val']:>5,d}  "
            f"{coverage_pct:>8s}  "
            f"{mean_app:>8s}"
        )

    all_skipped = set()
    for r in draw_results:
        all_skipped.update(r["skipped"])
    if all_skipped:
        logger.warning(f"\n  Skipped (pool too small): {sorted(all_skipped)}")

    total_examples = sum(
        sum(b["n_examples"] for b in r["summary_bands"].values()) for r in draw_results
    )
    logger.info(f"\n  Total examples across all draws: {total_examples:,}")
    if total_warnings:
        logger.warning(f"  {total_warnings} validation warnings")
    else:
        logger.info(f"  All validations passed")
    logger.info(f"\n  Output: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
