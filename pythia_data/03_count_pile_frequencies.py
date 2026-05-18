#!/usr/bin/env python3
"""
Count Pile Token Frequencies - Parallel Shard Processing
=========================================================
Counts token frequencies across all Pile shards using multiprocessing.

Since we're just counting token IDs (not analyzing documents), we don't need
the merged file. Processing shards in parallel is faster and simpler.

Key insight: Token counting is parallel.
    total_counts = sum(bincount(shard_i) for all shards)

Usage:
    python 03_count_pile_frequencies.py
    python 03_count_pile_frequencies.py --workers 8
    python 03_count_pile_frequencies.py --workers 21  # one per shard
"""

import numpy as np
import json
import gzip
import pickle
import argparse
from pathlib import Path
from datetime import datetime
import time
from transformers import AutoTokenizer
from multiprocessing import Pool
import os


def log(msg):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def count_shard(args):
    """
    Count token frequencies in a single shard.

    Reads data in chunks to avoid memmap page fault issues on network
    filesystems (BeegFS). Each chunk is loaded into RAM before counting.

    Args:
        args: tuple of (shard_path, vocab_size, shard_idx)

    Returns:
        tuple of (shard_idx, frequencies_array, token_count, max_token_found)
    """
    shard_path, vocab_size, shard_idx = args

    # Process in 1GB chunks (500M uint16 values = 1GB)
    CHUNK_SIZE = 500_000_000

    try:
        import os
        import sys

        shard_name = os.path.basename(shard_path)
        file_size = os.path.getsize(shard_path)
        total_tokens = file_size // 2  # uint16 = 2 bytes

        sys.stdout.write(
            f"[Shard {shard_idx:02d}] {shard_name}: {total_tokens:,} tokens\n"
        )
        sys.stdout.flush()

        # Initialize frequency counter
        frequencies = np.zeros(vocab_size, dtype=np.int64)
        max_token = 0

        # Open file and read in chunks
        with open(shard_path, "rb") as f:
            chunk_idx = 0
            tokens_read = 0

            while True:
                # Read chunk as raw bytes, then convert to uint16
                raw = f.read(CHUNK_SIZE * 2)  # 2 bytes per uint16
                if not raw:
                    break

                # Convert to numpy array
                chunk = np.frombuffer(raw, dtype=np.uint16)
                tokens_read += len(chunk)

                # Count frequencies in this chunk
                chunk_counts = np.bincount(chunk, minlength=vocab_size)
                frequencies += chunk_counts

                # Track max token
                chunk_max = int(chunk.max())
                if chunk_max > max_token:
                    max_token = chunk_max

                chunk_idx += 1
                if chunk_idx % 5 == 0:
                    pct = 100.0 * tokens_read / total_tokens
                    sys.stdout.write(
                        f"[Shard {shard_idx:02d}] {pct:.0f}% ({tokens_read:,} / {total_tokens:,})\n"
                    )
                    sys.stdout.flush()

        sys.stdout.write(
            f"[Shard {shard_idx:02d}] Done: {tokens_read:,} tokens, max_id={max_token}\n"
        )
        sys.stdout.flush()

        return (shard_idx, frequencies, tokens_read, max_token, None)

    except Exception as e:
        import traceback

        return (shard_idx, None, 0, 0, f"{str(e)}\n{traceback.format_exc()}")


def main():
    parser = argparse.ArgumentParser(
        description="Count Pile token frequencies using parallel shard processing"
    )
    default_workers = os.cpu_count() or 8
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Number of parallel workers (default: {default_workers})",
    )
    args = parser.parse_args()

    SCRIPT_DIR = Path(__file__).resolve().parent
    PILE_DIR = SCRIPT_DIR.parent / "Pile"
    shard_dir = PILE_DIR / "pile_shards"
    output_dir = SCRIPT_DIR / "pile_frequencies"
    output_dir.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("PILE TOKEN FREQUENCY COUNTER - PARALLEL SHARD PROCESSING")
    log("=" * 70)
    log(f"Shard directory: {shard_dir}")
    log(f"Output directory: {output_dir}")
    log(f"Workers: {args.workers}")
    log("=" * 70)

    # Find all shard files
    shard_files = sorted(shard_dir.glob("document-*-of-*.bin"))

    if not shard_files:
        log(f"ERROR: No shard files found in {shard_dir}")
        log("  Expected files like: document-00000-of-00020.bin")
        return 1

    log(f"\nFound {len(shard_files)} shards:")
    total_size_gb = 0
    for sf in shard_files:
        size_gb = sf.stat().st_size / (1024**3)
        total_size_gb += size_gb
        log(f"  {sf.name}: {size_gb:.2f} GB")
    log(f"  Total: {total_size_gb:.2f} GB")

    # Load tokenizer to get vocab
    log("\n" + "=" * 70)
    log("LOADING TOKENIZER")
    log("=" * 70)
    log("Loading Pythia tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

    # Build vocabulary lookup
    log("Building vocabulary lookup table...")
    vocab = tokenizer.get_vocab()
    vocab_lookup = {token_id: token_string for token_string, token_id in vocab.items()}

    # CRITICAL: Use actual vocab size from get_vocab(), not tokenizer.vocab_size
    # The data may contain tokens beyond the official vocab_size
    max_token_id = max(vocab.values())
    vocab_size = max_token_id + 1000  # Add buffer for safety

    log(f"Tokenizer loaded")
    log(f"  Official vocab_size: {tokenizer.vocab_size:,}")
    log(f"  Actual vocab entries: {len(vocab_lookup):,}")
    log(f"  Max token ID in vocab: {max_token_id:,}")
    log(f"  Using vocab_size: {vocab_size:,} (with safety buffer)")

    # Prepare worker arguments
    worker_args = [
        (str(shard_path), vocab_size, idx) for idx, shard_path in enumerate(shard_files)
    ]

    # Process shards in parallel
    log("\n" + "=" * 70)
    log("COUNTING TOKEN FREQUENCIES")
    log("=" * 70)
    log(f"Processing {len(shard_files)} shards with {args.workers} workers...")
    log("-" * 70)

    start_time = time.time()

    # Initialize global counters
    global_frequencies = np.zeros(vocab_size, dtype=np.int64)
    total_tokens = 0
    max_token_found = 0
    errors = []

    # Use multiprocessing pool
    num_workers = min(args.workers, len(shard_files))
    log(f"Starting {num_workers} worker processes...")

    with Pool(processes=num_workers) as pool:
        results = pool.map(count_shard, worker_args)

    # Aggregate results
    log("\nAggregating results...")
    for shard_idx, frequencies, token_count, max_token, error in results:
        shard_name = shard_files[shard_idx].name

        if error:
            log(f"  {shard_name}: ERROR - {error}")
            errors.append((shard_name, error))
        else:
            global_frequencies += frequencies
            total_tokens += token_count
            max_token_found = max(max_token_found, max_token)
            log(f"  {shard_name}: {token_count:,} tokens (max ID: {max_token})")

    elapsed = time.time() - start_time

    if errors:
        log(f"\n{len(errors)} shards failed:")
        for shard_name, error in errors:
            log(f"    {shard_name}: {error}")

    # Validate max token found
    if max_token_found >= vocab_size:
        log(f"\nWARNING: Max token ID {max_token_found} >= vocab_size {vocab_size}")
        log("  Some tokens may have been truncated!")
    else:
        log(f"\nMax token ID found: {max_token_found} (within vocab_size {vocab_size})")

    # Convert numpy array to dict
    log("\nConverting to dictionary format...")
    token_counter = {}
    tokens_outside_vocab = []

    for token_id in range(vocab_size):
        count = int(global_frequencies[token_id])
        if count > 0:
            token_counter[token_id] = count
            # Check if this token is outside the known vocabulary
            if token_id not in vocab_lookup:
                tokens_outside_vocab.append((token_id, count))

    log(f"Found {len(token_counter):,} unique tokens with non-zero counts")

    if tokens_outside_vocab:
        log(f"Found {len(tokens_outside_vocab)} token IDs outside vocabulary:")
        for tid, count in tokens_outside_vocab[:10]:
            log(f"    Token ID {tid}: {count:,} occurrences")
        if len(tokens_outside_vocab) > 10:
            log(f"    ... and {len(tokens_outside_vocab) - 10} more")
        log(f"  These will be labeled as '<unk_ID>' in output")

    # Report processing results
    log("\n" + "=" * 70)
    log("PROCESSING COMPLETE")
    log("=" * 70)
    log(f"Shards processed:    {len(shard_files) - len(errors)}/{len(shard_files)}")
    log(f"Total tokens:        {total_tokens:,}")
    log(f"Unique tokens:       {len(token_counter):,}")
    log(f"Processing time:     {elapsed / 60:.1f} minutes ({elapsed:.1f} seconds)")
    log(f"Processing speed:    {total_tokens / elapsed / 1e6:.1f} million tokens/sec")

    # Validate expected corpus size (~300 billion tokens for Pile Standard)
    if 290e9 < total_tokens < 310e9:
        log("\nToken count matches expected Pile STANDARD corpus size!")
        log(f"  Expected: ~300 billion tokens")
        log(f"  Got: {total_tokens / 1e9:.1f} billion tokens")
    else:
        log(f"\nToken count differs from expected:")
        log(f"  Expected: ~300 billion tokens")
        log(f"  Got: {total_tokens / 1e9:.1f} billion tokens")

    log("=" * 70)

    # Create token string lookup
    log("\nCreating token string lookup table...")
    token_to_string = {}
    for tid in token_counter.keys():
        token_to_string[tid] = vocab_lookup.get(tid, f"<unk_{tid}>")

    log(f"All token IDs mapped to strings")

    # Save token frequencies
    log("\nSaving token frequencies...")
    token_freq_path = output_dir / "merged_token_freq.pkl.gz"
    token_freq_data = {
        "frequencies": token_counter,
        "token_to_string": token_to_string,
        "metadata": {
            "total_shards": len(shard_files),
            "shards_processed": len(shard_files) - len(errors),
            "total_tokens": int(total_tokens),
            "unique_tokens": len(token_counter),
            "processing_time_seconds": float(elapsed),
            "processing_time_minutes": float(elapsed / 60),
            "workers_used": num_workers,
            "created_at": datetime.now().isoformat(),
            "source": "pile_shards_parallel",
            "method": "Parallel shard processing with multiprocessing",
        },
    }

    with gzip.open(token_freq_path, "wb", compresslevel=6) as f:
        pickle.dump(token_freq_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    freq_size_mb = token_freq_path.stat().st_size / (1024**2)
    log(f"Token frequencies saved: {token_freq_path.name} ({freq_size_mb:.1f} MB)")

    # Save processing statistics
    log("Saving processing statistics...")
    top_10 = sorted(token_counter.items(), key=lambda x: x[1], reverse=True)[:10]

    stats = {
        "method": "parallel_shard_processing",
        "shard_dir": str(shard_dir),
        "num_shards": len(shard_files),
        "shards_processed": len(shard_files) - len(errors),
        "workers_used": num_workers,
        "total_tokens": int(total_tokens),
        "unique_tokens": len(token_counter),
        "max_token_id_found": int(max_token_found),
        "vocab_size_used": int(vocab_size),
        "processing_time_seconds": float(elapsed),
        "processing_time_minutes": float(elapsed / 60),
        "tokens_per_second": float(total_tokens / elapsed),
        "top_10_tokens": [
            (int(tid), token_to_string[tid], int(count)) for tid, count in top_10
        ],
        "errors": errors if errors else None,
        "processed_at": datetime.now().isoformat(),
    }

    stats_path = output_dir / "merged_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    log(f"Statistics saved: {stats_path.name}")

    # Final summary
    log("\n" + "=" * 70)
    log("ANALYSIS COMPLETE")
    log("=" * 70)
    log(f"Performance:")
    log(f"  Time: {elapsed:.1f} seconds ({elapsed / 60:.1f} minutes)")
    log(f"  Speed: {total_tokens / elapsed / 1e6:.1f} million tokens/sec")
    log(f"  Workers: {num_workers}")
    log(f"\nResults:")
    log(f"  Total tokens:   {total_tokens:,}")
    log(f"  Unique tokens:  {len(token_counter):,}")
    log(f"  Shards:         {len(shard_files) - len(errors)}/{len(shard_files)}")
    log(f"\nOutput:")
    log(f"  {token_freq_path.name} ({freq_size_mb:.1f} MB)")
    log(f"  {stats_path.name}")
    log("\nNext step: python 04_profile_special_tokens.py")
    log("           python 05_export_frequencies.py")
    log("=" * 70)

    return 0 if not errors else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
