#!/usr/bin/env python3
"""
Pile-Based Context Extraction
==============================
Pipeline step 14.  Extracts real usage contexts for every token in the
Pythia vocabulary by scanning the Pile training data binary shards.

PURPOSE
-------
POS classification (Script 15) needs example sentences showing how each
token is actually used.  Previous versions extracted contexts from FineWeb
using string matching, which contaminates S-type token contexts (e.g.
bare ``bank`` gets matched to text where the tokenizer would produce
``Ġbank``).  This script fixes that by using **token ID matching** against
the actual training data.

METHOD
------
The Pile is stored as 21 binary shards of uint16 token IDs.  For each
shard we:

  1. Read data in 1 GB chunks (with +/-WINDOW overlap for edge safety).
  2. Build a boolean mask of "needed" token IDs (those still needing
     more context examples).
  3. Use fast NumPy indexing to find positions of needed tokens.
  4. For each hit, extract a window of +/-WINDOW surrounding token IDs.
  5. Store windows in per-token reservoirs (capped at per_shard_limit).

After all shards are processed, the main process:
  - Merges reservoirs from all workers.
  - Samples up to max_contexts windows per token.
  - Decodes each window to human-readable text using the vocabulary.
  - Saves the same pickle format consumed by Script 15.

INPUT
-----
Pile binary shards:  ../Pile/pile_shards/document-*-of-00020.bin
Token vocabulary:    pile_frequencies/merged_token_freq.pkl.gz

OUTPUT
------
context_windows/token_contexts.pkl          Dict[token_string, List[str]]
context_windows/token_contexts_sample.json  Human-readable sample
context_windows/extraction_stats.json       Processing statistics

Usage:
    python 14_extract_pile_contexts.py
    python 14_extract_pile_contexts.py --workers 21 --max-contexts 100
    python 14_extract_pile_contexts.py --per-shard-limit 20
"""

import argparse
import gzip
import json
import logging
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PILE_DIR = SCRIPT_DIR.parent / "Pile" / "pile_shards"
FREQ_PKL = SCRIPT_DIR / "pile_frequencies" / "merged_token_freq.pkl.gz"
OUTPUT_DIR = SCRIPT_DIR / "context_windows"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

WINDOW = 10  # +/-10 tokens around target
CHUNK_SIZE = 500_000_000  # 500M uint16 = 1 GB per read
MAX_CONTEXTS = 100  # Final contexts per token
PER_SHARD_LIMIT = 10  # Contexts collected per token per shard
SEED = 42


def load_vocabulary(freq_pkl_path: Path) -> dict:
    """
    Load token_id -> token_string mapping from the frequency pickle.

    Returns dict mapping int token_id to str token_string.
    """
    logger.info(f"Loading vocabulary from {freq_pkl_path.name} ...")
    with gzip.open(freq_pkl_path, "rb") as f:
        data = pickle.load(f)

    vocab = data["token_to_string"]  # {int: str}
    meta = data.get("metadata", {})
    logger.info(f"  Vocabulary size: {len(vocab):,}")
    logger.info(f"  Pile total tokens: {meta.get('total_tokens', '?'):,}")
    return vocab


# ============================================================================
# BPE WINDOW DECODER
# ============================================================================

# BPE whitespace markers used by GPT-NeoX tokenizer
BPE_SPACE = "\u0120"  # Ġ -> space
BPE_NEWLINE = "\u010a"  # Ċ -> newline
BPE_TAB = "\u0109"  # ĉ -> tab
BPE_CR = "\u010d"  # č -> carriage return


def decode_token_text(token_string: str) -> str:
    """Decode a single token string to readable text."""
    return (
        token_string.replace(BPE_SPACE, " ")
        .replace(BPE_NEWLINE, "\n")
        .replace(BPE_TAB, "\t")
        .replace(BPE_CR, "")
    )


def decode_window(token_ids, vocab: dict, target_pos: int) -> str:
    """
    Decode a window of token IDs to a human-readable context string.

    The target token is wrapped in [...] brackets.
    """
    parts = []
    for i, tid in enumerate(token_ids):
        raw = vocab.get(int(tid), f"<unk_{tid}>")
        text = decode_token_text(raw)
        if i == target_pos:
            target_clean = text.lstrip()
            if not target_clean:
                target_clean = text  # preserve if it was only whitespace
            parts.append(f"[{target_clean}]")
        else:
            parts.append(text)

    result = "".join(parts).strip()
    # Collapse multiple spaces/newlines for cleaner output
    while "  " in result:
        result = result.replace("  ", " ")
    return result


def process_shard(args):
    """
    Process one Pile shard: find token occurrences and extract context windows.

    Args:
        args: tuple of (shard_idx, shard_path, vocab_size, needed_ids,
                         window, per_shard_limit, seed)

    Returns:
        tuple of (shard_idx, reservoirs_dict, stats_dict, error_or_None)
        reservoirs_dict: {token_id: list of np.array windows}
    """
    (shard_idx, shard_path, vocab_size, needed_ids, window, per_shard_limit, seed) = (
        args
    )

    shard_name = os.path.basename(shard_path)
    rng = random.Random(seed + shard_idx)

    try:
        file_size = os.path.getsize(shard_path)
        total_tokens = file_size // 2

        sys.stdout.write(
            f"[Shard {shard_idx:02d}] {shard_name}: {total_tokens:,} tokens\n"
        )
        sys.stdout.flush()

        # Boolean mask: True for tokens that still need contexts
        needed = np.zeros(vocab_size, dtype=bool)
        needed[needed_ids] = True

        # Reservoirs: token_id -> list of uint16 window arrays
        reservoirs = defaultdict(list)
        counts = defaultdict(int)  # total hits per token (for stats)

        offset = 0  # current position in the shard (in tokens)
        chunk_idx = 0

        with open(shard_path, "rb") as f:
            while offset < total_tokens:
                # Compute read range with overlap for window extraction
                read_start = max(0, offset - window)
                read_end = min(offset + CHUNK_SIZE + window, total_tokens)
                read_count = read_end - read_start

                f.seek(read_start * 2)
                raw = f.read(read_count * 2)
                chunk = np.frombuffer(raw, dtype=np.uint16)

                # Effective range (excludes overlap regions)
                eff_start = offset - read_start
                eff_end = min(eff_start + CHUNK_SIZE, len(chunk) - window)
                if eff_end <= eff_start:
                    offset += CHUNK_SIZE
                    continue

                eff_chunk = chunk[eff_start:eff_end]

                # Fast boolean mask to find positions of needed tokens
                hit_mask = needed[eff_chunk]
                positions = np.where(hit_mask)[0]

                if len(positions) > 0:
                    # Map positions back to chunk coordinates
                    chunk_positions = positions + eff_start

                    # Group by token_id for efficient per-token sampling
                    # Use numpy for the grouping
                    hit_ids = chunk[chunk_positions]
                    unique_ids = np.unique(hit_ids)

                    for uid in unique_ids:
                        uid_int = int(uid)
                        if len(reservoirs[uid_int]) >= per_shard_limit:
                            continue

                        # Positions for this token
                        uid_positions = chunk_positions[hit_ids == uid]
                        counts[uid_int] += len(uid_positions)

                        # How many more do we need?
                        remaining = per_shard_limit - len(reservoirs[uid_int])

                        # Sample if too many hits
                        if len(uid_positions) > remaining:
                            indices = rng.sample(range(len(uid_positions)), remaining)
                            uid_positions = uid_positions[indices]

                        # Extract windows
                        for pos in uid_positions:
                            w_start = max(0, pos - window)
                            w_end = min(len(chunk), pos + window + 1)
                            w = chunk[w_start:w_end].copy()
                            # Record the target position within the window
                            target_in_window = pos - w_start
                            reservoirs[uid_int].append((w, target_in_window))

                        # Check if this token is full
                        if len(reservoirs[uid_int]) >= per_shard_limit:
                            needed[uid_int] = False

                # Check if all tokens are satisfied
                if not needed.any():
                    sys.stdout.write(
                        f"[Shard {shard_idx:02d}] All tokens satisfied "
                        f"at offset {offset:,}\n"
                    )
                    sys.stdout.flush()
                    break

                offset += CHUNK_SIZE
                chunk_idx += 1

                if chunk_idx % 5 == 0:
                    n_filled = sum(
                        1 for v in reservoirs.values() if len(v) >= per_shard_limit
                    )
                    pct = 100.0 * offset / total_tokens
                    sys.stdout.write(
                        f"[Shard {shard_idx:02d}] {pct:.0f}%  "
                        f"tokens_filled={n_filled:,}/{len(needed_ids):,}\n"
                    )
                    sys.stdout.flush()

        # Convert reservoirs to serializable format
        result_reservoirs = {}
        for tid, windows in reservoirs.items():
            result_reservoirs[tid] = [(w.tolist(), tp) for w, tp in windows]

        stats = {
            "shard_idx": shard_idx,
            "shard_name": shard_name,
            "total_tokens": total_tokens,
            "unique_tokens_found": len(reservoirs),
            "total_hits": sum(counts.values()),
            "tokens_at_limit": sum(
                1 for v in reservoirs.values() if len(v) >= per_shard_limit
            ),
        }

        sys.stdout.write(
            f"[Shard {shard_idx:02d}] Done: "
            f"{len(reservoirs):,} tokens found, "
            f"{sum(len(v) for v in reservoirs.values()):,} windows\n"
        )
        sys.stdout.flush()

        return (shard_idx, result_reservoirs, stats, None)

    except Exception as e:
        import traceback

        return (shard_idx, {}, {}, f"{e}\n{traceback.format_exc()}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract token contexts from Pile binary shards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pile-dir",
        type=Path,
        default=PILE_DIR,
        help="Directory with Pile binary shards",
    )
    parser.add_argument(
        "--freq-pkl",
        type=Path,
        default=FREQ_PKL,
        help="Path to merged_token_freq.pkl.gz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=21,
        help="Number of parallel workers (default: 21, one per shard)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=WINDOW,
        help=f"Context window size +/-N tokens (default: {WINDOW})",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=MAX_CONTEXTS,
        help=f"Max contexts per token in final output (default: {MAX_CONTEXTS})",
    )
    parser.add_argument(
        "--per-shard-limit",
        type=int,
        default=PER_SHARD_LIMIT,
        help=f"Max contexts per token per shard (default: {PER_SHARD_LIMIT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("PILE-BASED CONTEXT EXTRACTION")
    logger.info("=" * 70)

    # ---- Find shards ----
    shard_files = sorted(args.pile_dir.glob("document-*-of-*.bin"))
    if not shard_files:
        logger.error(f"No shard files found in {args.pile_dir}")
        return 1

    logger.info(f"Found {len(shard_files)} Pile shards")
    total_gb = sum(f.stat().st_size for f in shard_files) / (1024**3)
    logger.info(f"Total size: {total_gb:.1f} GB")

    # ---- Load vocabulary ----
    vocab = load_vocabulary(args.freq_pkl)
    vocab_size = max(vocab.keys()) + 1
    all_token_ids = sorted(vocab.keys())
    logger.info(f"Will extract contexts for {len(all_token_ids):,} tokens")

    # ---- Parameters ----
    logger.info("")
    logger.info(f"Window:          +/-{args.window} tokens")
    logger.info(f"Per-shard limit: {args.per_shard_limit}")
    logger.info(f"Max contexts:    {args.max_contexts}")
    logger.info(f"Workers:         {args.workers}")
    logger.info(f"Seed:            {args.seed}")
    pool_size = len(shard_files) * args.per_shard_limit
    logger.info(
        f"Max pool size:   {len(shard_files)} shards x "
        f"{args.per_shard_limit} = {pool_size} per token"
    )
    logger.info(f"Final sampling:  {pool_size} pool -> {args.max_contexts}")

    # ---- Process shards in parallel ----
    logger.info("")
    logger.info("=" * 70)
    logger.info("PROCESSING SHARDS")
    logger.info("=" * 70)

    worker_args = [
        (
            idx,
            str(sf),
            vocab_size,
            all_token_ids,
            args.window,
            args.per_shard_limit,
            args.seed,
        )
        for idx, sf in enumerate(shard_files)
    ]

    start_time = time.time()
    num_workers = min(args.workers, len(shard_files))

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_shard, worker_args)

    elapsed_shards = time.time() - start_time
    logger.info(f"\nShard processing: {elapsed_shards / 60:.1f} minutes")

    # ---- Check for errors ----
    errors = [(idx, err) for idx, _, _, err in results if err]
    if errors:
        for idx, err in errors:
            logger.error(f"Shard {idx} failed: {err}")

    # ---- Merge reservoirs ----
    logger.info("")
    logger.info("Merging reservoirs from all shards...")

    merged_pools = defaultdict(list)  # token_id -> list of (window, target_pos)
    shard_stats = []

    for shard_idx, reservoirs, stats, err in results:
        if err:
            continue
        shard_stats.append(stats)
        for tid_str, windows in reservoirs.items():
            tid = int(tid_str)
            merged_pools[tid].extend(windows)

    logger.info(f"  Tokens with contexts: {len(merged_pools):,}/{len(all_token_ids):,}")
    total_windows = sum(len(v) for v in merged_pools.values())
    logger.info(f"  Total windows in pool: {total_windows:,}")

    # ---- Sample and decode ----
    logger.info("")
    logger.info(f"Sampling {args.max_contexts} contexts per token and decoding...")

    rng = random.Random(args.seed)
    all_contexts = {}  # token_string -> list of context strings
    n_complete = 0

    for tid in all_token_ids:
        token_string = vocab[tid]
        pool = merged_pools.get(tid, [])

        if not pool:
            continue

        # Sample down to max_contexts
        if len(pool) > args.max_contexts:
            pool = rng.sample(pool, args.max_contexts)

        # Decode each window
        contexts = []
        for window_list, target_pos in pool:
            window_arr = np.array(window_list, dtype=np.uint16)
            ctx = decode_window(window_arr, vocab, target_pos)
            contexts.append(ctx)

        all_contexts[token_string] = contexts
        if len(contexts) >= args.max_contexts:
            n_complete += 1

    logger.info(f"  Tokens with contexts: {len(all_contexts):,}")
    logger.info(f"  Tokens complete ({args.max_contexts}): {n_complete:,}")
    total_ctx = sum(len(v) for v in all_contexts.values())
    logger.info(f"  Total contexts: {total_ctx:,}")

    # ---- Save outputs ----
    logger.info("")
    logger.info("Saving outputs...")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Main pickle
    pkl_path = args.output_dir / "token_contexts.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(all_contexts, f, protocol=pickle.HIGHEST_PROTOCOL)
    pkl_mb = pkl_path.stat().st_size / (1024**2)
    logger.info(f"  {pkl_path} ({pkl_mb:.1f} MB)")

    # Human-readable sample (top tokens by frequency)
    sample_path = args.output_dir / "token_contexts_sample.json"
    sample = {}
    # Use first 20 tokens that have contexts
    sample_count = 0
    for tid in all_token_ids:
        ts = vocab[tid]
        if ts in all_contexts and all_contexts[ts]:
            sample[ts] = {
                "count": len(all_contexts[ts]),
                "examples": all_contexts[ts][:5],
            }
            sample_count += 1
            if sample_count >= 20:
                break

    with open(sample_path, "w") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    logger.info(f"  {sample_path}")

    # Stats
    elapsed_total = time.time() - start_time
    stats = {
        "params": {
            "window": args.window,
            "max_contexts": args.max_contexts,
            "per_shard_limit": args.per_shard_limit,
            "num_shards": len(shard_files),
            "workers": num_workers,
            "seed": args.seed,
            "source": "pile_binary_shards",
        },
        "summary": {
            "total_tokens_in_vocab": len(all_token_ids),
            "with_contexts": len(all_contexts),
            "complete": n_complete,
            "total_contexts": total_ctx,
            "processing_time_minutes": round(elapsed_total / 60, 1),
        },
        "shard_stats": shard_stats,
        "created_at": datetime.now().isoformat(),
    }

    stats_path = args.output_dir / "extraction_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  {stats_path}")

    # Log
    log_path = args.output_dir / f"extraction_{datetime.now():%Y%m%d_%H%M%S}.log"

    # ---- Final report ----
    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(f"Source:                Pile binary shards (token ID matching)")
    logger.info(
        f"Tokens with contexts:  {len(all_contexts):,}/{len(all_token_ids):,} "
        f"({100 * len(all_contexts) / len(all_token_ids):.1f}%)"
    )
    logger.info(
        f"Tokens complete ({args.max_contexts:,}):   {n_complete:,}/{len(all_token_ids):,} "
        f"({100 * n_complete / len(all_token_ids):.1f}%)"
    )
    logger.info(f"Total contexts:        {total_ctx:,}")
    logger.info(f"Processing time:       {elapsed_total / 60:.1f} minutes")
    if errors:
        logger.warning(f"Shard errors:          {len(errors)}")
    logger.info("")
    logger.info("=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
