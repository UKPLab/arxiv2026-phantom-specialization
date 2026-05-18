#!/usr/bin/env python3
"""
Extract Words from Pile.

Optional analysis script (not part of core pipeline).

Extracts word frequencies from tokenized Pile corpus shards.
Uses direct vocabulary lookup instead of tokenizer.decode() for speed.
Converts BPE Ġ markers to spaces for proper word boundary detection.

Outputs:
    pile_words/word_frequencies.tsv
"""

import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer
from tqdm import tqdm
import csv
import traceback
import logging
import sys
from datetime import datetime


SCRIPT_DIR = Path(__file__).resolve().parent
PILE_DIR = SCRIPT_DIR.parent / "Pile"
SHARD_DIR = PILE_DIR / "pile_shards"
OUTPUT_DIR = SCRIPT_DIR / "pile_words"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOC_SIZE = 2049
MAX_EXAMPLES = 5
CONTEXT_WORDS = 5
NUM_SHARDS = 21  # Shards 00-20

# Checkpoint every N documents to avoid losing progress
CHECKPOINT_INTERVAL = 500_000


def setup_logging():
    """Configure logging to both console and file."""
    log_file = OUTPUT_DIR / "extraction.log"

    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


log = logging.getLogger()


def build_vocab_lookup():
    """Build token_id -> token_string lookup once."""
    log.info("Building vocabulary lookup table...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    vocab = tokenizer.get_vocab()
    lookup = {token_id: token_str for token_str, token_id in vocab.items()}
    log.info(f"Loaded {len(lookup):,} tokens")
    return lookup


def get_checkpoint_path(shard_id: int) -> Path:
    """Get path for shard checkpoint file."""
    return OUTPUT_DIR / f"checkpoint_shard_{shard_id:05d}.json"


def get_output_path(shard_id: int) -> Path:
    """Get path for completed shard output file."""
    return OUTPUT_DIR / f"shard_{shard_id:05d}.json"


def load_checkpoint(shard_id: int) -> tuple[dict, dict, int]:
    """Load checkpoint if exists, returns (counts, examples, docs_processed)."""
    checkpoint_path = get_checkpoint_path(shard_id)
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            log.info(
                f"[Shard {shard_id:02d}] Resuming from checkpoint: {data['docs_processed']:,} docs"
            )
            return (
                defaultdict(int, data["counts"]),
                defaultdict(list, {k: v for k, v in data["examples"].items()}),
                data["docs_processed"],
            )
        except Exception as e:
            log.warning(f"[Shard {shard_id:02d}] Failed to load checkpoint: {e}")
    return defaultdict(int), defaultdict(list), 0


def save_checkpoint(
    shard_id: int, word_counts: dict, word_examples: dict, docs_processed: int
):
    """Save checkpoint to disk."""
    checkpoint_path = get_checkpoint_path(shard_id)
    temp_path = checkpoint_path.with_suffix(".tmp")

    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "counts": dict(word_counts),
                    "examples": dict(word_examples),
                    "docs_processed": docs_processed,
                },
                f,
            )
        temp_path.rename(checkpoint_path)
        log.debug(f"[Shard {shard_id:02d}] Checkpoint saved: {docs_processed:,} docs")
    except Exception as e:
        log.error(f"[Shard {shard_id:02d}] Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()


def save_final_output(
    shard_id: int, word_counts: dict, word_examples: dict, docs_processed: int
):
    """Save final output and remove checkpoint."""
    output_path = get_output_path(shard_id)
    checkpoint_path = get_checkpoint_path(shard_id)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "counts": dict(word_counts),
                "examples": dict(word_examples),
                "docs_processed": docs_processed,
            },
            f,
        )

    # Remove checkpoint after successful save
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    log.info(
        f"[Shard {shard_id:02d}] Saved final output: {len(word_counts):,} unique words"
    )


def process_shard(shard_id: int, vocab_lookup: dict) -> bool:
    """
    Process one shard with checkpointing.

    Returns True if successful, False otherwise.
    """
    import os

    # Check if already completed
    output_path = get_output_path(shard_id)
    if output_path.exists():
        log.info(f"[Shard {shard_id:02d}] Already completed, skipping")
        return True

    shard_file = SHARD_DIR / f"document-{shard_id:05d}-of-00020.bin"
    if not shard_file.exists():
        log.error(f"[Shard {shard_id:02d}] File not found: {shard_file}")
        return False

    try:
        file_size = os.path.getsize(shard_file)
        total_tokens = file_size // 2  # uint16 = 2 bytes
        num_docs = total_tokens // DOC_SIZE

        log.info(
            f"[Shard {shard_id:02d}] Starting: {num_docs:,} documents ({total_tokens:,} tokens)"
        )

        # Load checkpoint or start fresh
        word_counts, word_examples, docs_already_processed = load_checkpoint(shard_id)

        # Read in chunks of ~100MB
        CHUNK_TOKENS = 50_000_000  # 50M tokens = 100MB
        CHUNK_BYTES = CHUNK_TOKENS * 2

        docs_processed = 0
        docs_to_skip = docs_already_processed
        leftover = b""
        last_checkpoint_docs = docs_already_processed

        with open(shard_file, "rb") as f:
            chunk_idx = 0
            while True:
                raw = f.read(CHUNK_BYTES)
                if not raw:
                    break

                # Combine leftover from previous chunk
                raw = leftover + raw
                leftover = b""

                # Convert to numpy array
                data = np.frombuffer(raw, dtype=np.uint16)

                # Process complete documents only
                num_complete_docs = len(data) // DOC_SIZE
                tokens_used = num_complete_docs * DOC_SIZE

                # Save leftover tokens for next chunk
                if tokens_used < len(data):
                    leftover_tokens = len(data) - tokens_used
                    leftover = raw[-(leftover_tokens * 2) :]

                # Process each document in this chunk
                for doc_idx in range(num_complete_docs):
                    # Skip already processed docs (from checkpoint)
                    if docs_to_skip > 0:
                        docs_to_skip -= 1
                        docs_processed += 1
                        continue

                    start = doc_idx * DOC_SIZE
                    end = start + DOC_SIZE
                    token_ids = data[start:end]

                    # Direct vocab lookup
                    token_strings = [
                        vocab_lookup.get(int(tid), f"<unk_{tid}>") for tid in token_ids
                    ]

                    # Join tokens, converting Ġ to space
                    text = "".join(s.replace("Ġ", " ") for s in token_strings)
                    words = text.split()

                    # Count words
                    for i, word in enumerate(words):
                        word_counts[word] += 1

                        if len(word_examples[word]) < MAX_EXAMPLES:
                            start_idx = max(0, i - CONTEXT_WORDS)
                            end_idx = min(len(words), i + CONTEXT_WORDS + 1)
                            word_examples[word].append(
                                " ".join(words[start_idx:end_idx])
                            )

                    docs_processed += 1

                chunk_idx += 1

                # Progress logging
                if chunk_idx % 3 == 0:
                    pct = 100.0 * docs_processed / num_docs
                    log.info(
                        f"[Shard {shard_id:02d}] {pct:.0f}% ({docs_processed:,} / {num_docs:,} docs)"
                    )

                # Checkpoint periodically
                if docs_processed - last_checkpoint_docs >= CHECKPOINT_INTERVAL:
                    save_checkpoint(
                        shard_id, word_counts, word_examples, docs_processed
                    )
                    last_checkpoint_docs = docs_processed

        # Save final output
        save_final_output(shard_id, word_counts, word_examples, docs_processed)
        log.info(
            f"[Shard {shard_id:02d}] Completed: {docs_processed:,} docs, {len(word_counts):,} unique words"
        )
        return True

    except MemoryError as e:
        log.error(f"[Shard {shard_id:02d}] OUT OF MEMORY: {e}")
        log.error(f"[Shard {shard_id:02d}] Current unique words: {len(word_counts):,}")
        # Try to save checkpoint before dying
        try:
            save_checkpoint(shard_id, word_counts, word_examples, docs_processed)
            log.info(
                f"[Shard {shard_id:02d}] Emergency checkpoint saved at {docs_processed:,} docs"
            )
        except Exception:
            pass
        return False

    except Exception as e:
        log.error(f"[Shard {shard_id:02d}] ERROR: {type(e).__name__}: {e}")
        log.error(f"[Shard {shard_id:02d}] Traceback:\n{traceback.format_exc()}")
        # Try to save checkpoint
        try:
            if docs_processed > last_checkpoint_docs:
                save_checkpoint(shard_id, word_counts, word_examples, docs_processed)
                log.info(
                    f"[Shard {shard_id:02d}] Emergency checkpoint saved at {docs_processed:,} docs"
                )
        except Exception:
            pass
        return False


def merge_results():
    """Merge all shard results into final TSV."""
    log.info("=" * 70)
    log.info("MERGING RESULTS")
    log.info("=" * 70)

    global_counts = defaultdict(int)
    global_examples = defaultdict(list)

    shard_files = sorted(OUTPUT_DIR.glob("shard_*.json"))
    log.info(f"Found {len(shard_files)} shard result files")

    if len(shard_files) == 0:
        log.warning("No shard files to merge!")
        return

    for shard_file in tqdm(shard_files, desc="Merging shards"):
        try:
            with open(shard_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            for word, count in data["counts"].items():
                global_counts[word] += count

            for word, examples in data["examples"].items():
                current = global_examples[word]
                needed = MAX_EXAMPLES - len(current)
                if needed > 0:
                    global_examples[word].extend(examples[:needed])
        except Exception as e:
            log.error(f"Failed to load {shard_file}: {e}")
            continue

    log.info(f"Merged {len(global_counts):,} unique words")

    output_tsv = OUTPUT_DIR / "word_frequencies.tsv"
    log.info(f"Writing final TSV: {output_tsv}")

    with open(output_tsv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["word", "count", "examples"])

        sorted_words = sorted(global_counts.items(), key=lambda x: x[1], reverse=True)

        for word, count in tqdm(sorted_words, desc="Writing TSV"):
            examples = global_examples.get(word, [])
            examples_str = " | ".join(examples)
            writer.writerow([word, count, examples_str])

    output_size_mb = output_tsv.stat().st_size / (1024**2)
    log.info(f"Saved: {output_tsv} ({output_size_mb:.1f} MB)")

    log.info("\nTop 20 words:")
    for i, (word, count) in enumerate(sorted_words[:20], 1):
        log.info(f"  {i:2d}. {word!r:30s} : {count:,}")

    log.info("\nCleaning up intermediate files...")
    for shard_file in shard_files:
        shard_file.unlink()
    log.info("Cleanup complete")


def main():
    setup_logging()

    log.info("=" * 70)
    log.info("WORD EXTRACTION - SEQUENTIAL WITH CHECKPOINTING")
    log.info("=" * 70)
    log.info(f"Output directory: {OUTPUT_DIR}")
    log.info(f"Checkpoint interval: {CHECKPOINT_INTERVAL:,} documents")

    # Build vocab lookup once
    vocab_lookup = build_vocab_lookup()

    # Process shards SEQUENTIALLY to avoid OOM
    log.info("\n" + "=" * 70)
    log.info("PROCESSING SHARDS (sequential to avoid OOM)")
    log.info("=" * 70)

    start_time = datetime.now()
    successful = 0
    failed = 0

    for shard_id in range(NUM_SHARDS):
        log.info("-" * 50)
        try:
            if process_shard(shard_id, vocab_lookup):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            log.error(
                f"[Shard {shard_id:02d}] Unhandled exception: {type(e).__name__}: {e}"
            )
            log.error(traceback.format_exc())
            failed += 1

    elapsed = (datetime.now() - start_time).total_seconds()

    log.info("\n" + "=" * 70)
    log.info("PROCESSING COMPLETE")
    log.info("=" * 70)
    log.info(f"Successful: {successful}/{NUM_SHARDS} shards")
    log.info(f"Failed: {failed}/{NUM_SHARDS} shards")
    log.info(f"Elapsed time: {elapsed / 60:.1f} minutes ({elapsed / 3600:.2f} hours)")

    if successful > 0:
        merge_results()
    else:
        log.error("No shards completed successfully, nothing to merge")
        return 1

    log.info("\nALL DONE!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
