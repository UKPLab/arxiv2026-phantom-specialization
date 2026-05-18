#!/usr/bin/env python3
"""
Merge Pile Shards Using Official Pythia Script
==============================================
One-time operation to merge 21 shards into single file.
This eliminates cross-boundary document issues.

Expected time: 2-4 hours
Expected output size: ~558 GB merged file
"""

import subprocess
import shutil
from pathlib import Path
from datetime import datetime


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def main():
    # Paths - Pile data stored separately from pipeline code
    SCRIPT_DIR = Path(__file__).resolve().parent
    PILE_DIR = SCRIPT_DIR.parent / "Pile"
    pythia_repo = SCRIPT_DIR / "pythia"
    shard_dir = PILE_DIR / "pile_shards"
    merged_dir = PILE_DIR / "pile_merged"

    log("=" * 70)
    log("MERGING PILE SHARDS (Official Pythia Method)")
    log("=" * 70)
    log(f"Shard directory: {shard_dir}")
    log(f"Output directory: {merged_dir}")
    log(f"Expected time: 2-4 hours")
    log(f"Expected output: ~558 GB merged file")
    log("=" * 70)

    # Create output directory
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Verify input files exist
    first_shard = shard_dir / "document-00000-of-00020.bin"
    index_file = shard_dir / "document.idx"

    if not first_shard.exists():
        log(f"ERROR: First shard not found: {first_shard}")
        return 1

    if not index_file.exists():
        log(f"ERROR: Index file not found: {index_file}")
        return 1

    log(f"Found first shard: {first_shard}")
    log(f"Found index file: {index_file}")

    # Run unshard script
    log("\nStep 1/2: Merging 21 shards...")
    log("This will take 2-4 hours. Progress will be shown below.")
    log("-" * 70)

    unshard_script = pythia_repo / "utils" / "unshard_memmap.py"

    if not unshard_script.exists():
        log(f"ERROR: Unshard script not found: {unshard_script}")
        log("Make sure you cloned the Pythia repository correctly.")
        return 1

    cmd = [
        "python3",
        str(unshard_script),
        "--input_file",
        str(first_shard),
        "--num_shards",
        "21",
        "--output_dir",
        str(merged_dir),
    ]

    log(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        log("Shards merged successfully!")
    except subprocess.CalledProcessError as e:
        log(f"ERROR during merging: {e}")
        return 1

    # Copy index file
    log("\nStep 2/2: Copying index file...")
    dest_index = merged_dir / "document.idx"
    shutil.copy2(index_file, dest_index)
    log(f"Copied index to: {dest_index}")

    # Verify output
    merged_bin = merged_dir / "document.bin"
    if merged_bin.exists():
        size_gb = merged_bin.stat().st_size / (1024**3)
        log("\n" + "=" * 70)
        log("MERGE COMPLETE!")
        log("=" * 70)
        log(f"Merged file: {merged_bin}")
        log(f"Size: {size_gb:.2f} GB")
        log(f"Index file: {dest_index}")
        log("\nYou can now run 03_count_pile_frequencies.py on the merged file.")
        log("=" * 70)
        return 0
    else:
        log(f"ERROR: Expected output file not found: {merged_bin}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
