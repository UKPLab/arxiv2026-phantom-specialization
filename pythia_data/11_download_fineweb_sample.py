#!/usr/bin/env python3
"""
Prepare FineWeb Sample for Token Validation

Paths (relative to parent directory, sibling to pythia_data):
  - Download: ../fineweb_sample/sample
  - Output:   ../fineweb_sample/processed

Target: 500M tokens
Uses: huggingface_hub
"""

import os
import json
import pickle
import logging
from pathlib import Path
from collections import Counter
from typing import Dict, Set, Tuple
import re
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
FINEWEB_DIR = SCRIPT_DIR.parent / "fineweb_sample"
DOWNLOAD_DIR = str(FINEWEB_DIR / "sample")
OUTPUT_DIR = str(FINEWEB_DIR / "processed")
TARGET_TOKENS = 500_000_000  # 500 million tokens
MIN_LANGUAGE_SCORE = 0.7
TARGET_LANGUAGE = "en"

# Word extraction pattern (alphabetic words only)
WORD_PATTERN = re.compile(r"\b[a-zA-Z]+\b")


def setup_huggingface_cache():
    """
    Set HuggingFace environment variables to control download location
    """
    # Set HuggingFace cache directories
    os.environ["HF_HOME"] = DOWNLOAD_DIR
    os.environ["HF_DATASETS_CACHE"] = DOWNLOAD_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = DOWNLOAD_DIR

    logger.info(f"HuggingFace cache set to: {DOWNLOAD_DIR}")


def download_fineweb_sample() -> str:
    """
    Download FineWeb sample using huggingface_hub
    Downloads to: {DOWNLOAD_DIR}

    Returns:
        Path to downloaded data
    """
    try:
        from huggingface_hub import snapshot_download

        logger.info("=" * 80)
        logger.info("DOWNLOADING FINEWEB SAMPLE")
        logger.info("=" * 80)
        logger.info(f"Target tokens: {TARGET_TOKENS:,}")
        logger.info(f"Language: {TARGET_LANGUAGE}")
        logger.info(f"Min language score: {MIN_LANGUAGE_SCORE}")
        logger.info(f"Download directory: {DOWNLOAD_DIR}")

        Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)

        logger.info("Downloading FineWeb sample/10BT...")

        folder = snapshot_download(
            "HuggingFaceFW/fineweb",
            repo_type="dataset",
            local_dir=DOWNLOAD_DIR,
            allow_patterns="sample/10BT/*",
            max_workers=4,
            resume_download=True,
        )

        logger.info(f"Download complete: {folder}")
        return folder

    except ImportError:
        logger.error(
            "huggingface_hub not installed. Install with: pip install huggingface_hub"
        )
        raise


def process_documents() -> Tuple[Set[str], Counter, Dict]:
    """
    Process downloaded documents and build word dictionaries
    Reads from: {DOWNLOAD_DIR}

    Returns:
        (word_set, word_frequency, metadata)
    """
    logger.info("=" * 80)
    logger.info("PROCESSING DOCUMENTS")
    logger.info("=" * 80)
    logger.info(f"Source directory: {DOWNLOAD_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    word_set = set()
    word_frequency = Counter()

    total_docs = 0
    total_tokens = 0
    total_chars = 0

    # Find all Parquet files in download directory
    parquet_files = sorted(list(Path(DOWNLOAD_DIR).rglob("*.parquet")))

    logger.info(f"Found {len(parquet_files)} Parquet files")

    if not parquet_files:
        logger.error(f"No Parquet files found in {DOWNLOAD_DIR}")
        raise FileNotFoundError(f"No .parquet files in {DOWNLOAD_DIR}")

    try:
        import pyarrow.parquet as pq

        logger.info("Processing Parquet files...")
        logger.info(f"Target: {TARGET_TOKENS:,} tokens")

        for parquet_file in tqdm(parquet_files, desc="Parquet files"):
            if total_tokens >= TARGET_TOKENS:
                logger.info(f"Reached target of {TARGET_TOKENS:,} tokens")
                break

            try:
                logger.info(f"Reading: {parquet_file.name}")

                # CRITICAL: Disable memory mapping to avoid BeegFS issues
                table = pq.read_table(
                    str(parquet_file), memory_map=False, use_threads=False
                )
                df = table.to_pandas()

                logger.info(f"  Loaded {len(df):,} rows")

                # Filter by language and score
                if "language" in df.columns:
                    df = df[df["language"] == TARGET_LANGUAGE]
                if "language_score" in df.columns:
                    df = df[df["language_score"] > MIN_LANGUAGE_SCORE]

                logger.info(f"  After filtering: {len(df):,} rows")

                # Process each document
                for idx, row in enumerate(df.itertuples(), 1):
                    text = getattr(row, "text", "")
                    if not text:
                        continue

                    # Extract words
                    words = WORD_PATTERN.findall(text)
                    words_lower = [w.lower() for w in words if w.isalpha()]

                    # Update statistics
                    word_set.update(words_lower)
                    word_frequency.update(words_lower)

                    total_docs += 1
                    total_tokens += len(words_lower)
                    total_chars += len(text)

                    # Log progress every 10,000 documents
                    if total_docs % 10000 == 0:
                        logger.info(
                            f"  Processed {total_docs:,} docs, {total_tokens:,} tokens, {len(word_set):,} unique words"
                        )

                    # Stop if we've reached target
                    if total_tokens >= TARGET_TOKENS:
                        logger.info(f"  Reached target of {TARGET_TOKENS:,} tokens")
                        break

                logger.info(f"  Completed {parquet_file.name}")

            except Exception as e:
                logger.error(f"  Error processing file {parquet_file}: {e}")
                continue

    except ImportError:
        logger.error("pyarrow not installed. Install with: pip install pyarrow")
        raise

    # Create metadata
    metadata = {
        "total_documents": total_docs,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "unique_words": len(word_set),
        "source": "HuggingFaceFW/fineweb",
        "sample": "10BT",
        "target_tokens": TARGET_TOKENS,
        "language": TARGET_LANGUAGE,
        "min_language_score": MIN_LANGUAGE_SCORE,
        "download_dir": DOWNLOAD_DIR,
        "output_dir": OUTPUT_DIR,
    }

    logger.info("=" * 80)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total documents: {total_docs:,}")
    logger.info(f"Total tokens: {total_tokens:,}")
    logger.info(f"Total characters: {total_chars:,}")
    logger.info(f"Unique words: {len(word_set):,}")
    logger.info(f"Top 20 most common words:")
    for word, count in word_frequency.most_common(20):
        logger.info(f"  {word:20s}: {count:,}")

    # Save to output directory
    logger.info(f"\nSaving processed data to: {OUTPUT_DIR}")

    # Save word set
    word_set_file = Path(OUTPUT_DIR) / "word_set.pkl"
    with open(word_set_file, "wb") as f:
        pickle.dump(word_set, f)
    logger.info(f"Saved word set: {word_set_file}")

    # Save word frequency
    word_freq_file = Path(OUTPUT_DIR) / "word_frequency.pkl"
    with open(word_freq_file, "wb") as f:
        pickle.dump(word_frequency, f)
    logger.info(f"Saved word frequency: {word_freq_file}")

    # Save metadata
    metadata_file = Path(OUTPUT_DIR) / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {metadata_file}")

    # Save human-readable summary
    summary_file = Path(OUTPUT_DIR) / "summary.txt"
    with open(summary_file, "w") as f:
        f.write("FineWeb Sample Processing Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Source: {metadata['source']}\n")
        f.write(f"Sample: {metadata['sample']}\n")
        f.write(f"Language: {metadata['language']}\n")
        f.write(f"Min language score: {metadata['min_language_score']}\n")
        f.write(f"Target tokens: {metadata['target_tokens']:,}\n\n")
        f.write(f"Total documents: {metadata['total_documents']:,}\n")
        f.write(f"Total tokens: {metadata['total_tokens']:,}\n")
        f.write(f"Total characters: {metadata['total_chars']:,}\n")
        f.write(f"Unique words: {metadata['unique_words']:,}\n\n")
        f.write(f"Download directory: {metadata['download_dir']}\n")
        f.write(f"Output directory: {metadata['output_dir']}\n\n")
        f.write("Top 100 most common words:\n")
        f.write("-" * 80 + "\n")
        for i, (word, count) in enumerate(word_frequency.most_common(100), 1):
            f.write(f"{i:3d}. {word:20s}: {count:,}\n")
    logger.info(f"Saved summary: {summary_file}")

    return word_set, word_frequency, metadata


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare FineWeb sample")
    parser.add_argument(
        "--skip-download", action="store_true", help="Skip download, use existing data"
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("FINEWEB PREPARATION")
    logger.info("=" * 80)
    logger.info(f"Download directory: {DOWNLOAD_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Target tokens: {TARGET_TOKENS:,}")
    logger.info("")

    # Setup HuggingFace to use our download directory
    setup_huggingface_cache()

    # Step 1: Download
    if not args.skip_download:
        try:
            download_dir = download_fineweb_sample()
            logger.info(f"Downloaded to: {download_dir}")
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return 1
    else:
        logger.info(f"Skipping download, using existing data in: {DOWNLOAD_DIR}")

    # Step 2: Process documents
    try:
        word_set, word_frequency, metadata = process_documents()
        logger.info("Processing complete")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

    logger.info("")
    logger.info("=" * 80)
    logger.info("ALL DONE!")
    logger.info("=" * 80)
    logger.info(f"Processed data saved to: {OUTPUT_DIR}")
    logger.info(f"Total tokens: {metadata['total_tokens']:,}")
    logger.info(f"Unique words: {metadata['unique_words']:,}")
    logger.info("")
    logger.info("Next step: Run prevalidation")
    logger.info(f"  python 01_prevalidator_fineweb.py \\")
    logger.info(f"    --fineweb-dir {OUTPUT_DIR} \\")
    logger.info(f"    --input tokens.csv --output validated.csv")

    return 0


if __name__ == "__main__":
    exit(main())
