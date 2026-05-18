#!/usr/bin/env python3
"""
Save token frequencies in multiple formats.

Pipeline step 05.  Takes the merged frequency file and writes:
- CSV  (pandas/Excel)
- TSV  (legacy)
- Pickle  (compact, Python-only)

Usage:
    python 05_export_frequencies.py

Outputs:
    pile_frequencies/
    ├── merged_token_frequencies.csv
    ├── merged_token_frequencies.tsv
    └── merged_token_frequencies.pkl.gz
"""

import gzip
import pickle
import csv
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


def log(msg):
    """Print with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    input_dir = SCRIPT_DIR / "pile_frequencies"
    output_dir = input_dir

    log("=" * 70)
    log("SAVING FREQUENCIES IN MULTIPLE FORMATS")
    log("=" * 70)
    log(f"Input directory:  {input_dir}")
    log(f"Output directory: {output_dir}")
    log("=" * 70)

    # Load the frequency file from step 02
    freq_file = input_dir / "merged_token_freq.pkl.gz"

    if not freq_file.exists():
        log(f"ERROR: Frequency file not found: {freq_file}")
        log("Make sure you ran 03_count_pile_frequencies.py first!")
        return 1

    log(f"\nLoading: {freq_file.name}")

    with gzip.open(freq_file, "rb") as f:
        data = pickle.load(f)

    frequencies = data["frequencies"]
    token_to_string = data["token_to_string"]

    total_tokens = sum(frequencies.values())
    log(f"Loaded {len(frequencies):,} unique tokens")
    log(f"Total occurrences: {total_tokens:,}")

    # Sort by frequency (descending)
    log("\nSorting tokens by frequency...")
    sorted_tokens = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)
    log(f"Sorted {len(sorted_tokens):,} tokens")

    # Format 1
    log("\n" + "=" * 70)
    log("[1/3] SAVING CSV FORMAT")
    log("=" * 70)
    csv_path = output_dir / "merged_token_frequencies.csv"
    log(f"Writing: {csv_path.name}")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["token_id", "token_string", "count"])

        for token_id, count in tqdm(
            sorted_tokens, desc="Writing CSV", leave=False, ncols=100
        ):
            token_string = token_to_string.get(token_id, "<unknown>")
            writer.writerow([token_id, token_string, count])

    csv_size = csv_path.stat().st_size / (1024**2)
    log(f"CSV saved: {csv_size:.2f} MB")

    # Format 2 (backward compatibility)
    log("\n" + "=" * 70)
    log("[2/3] SAVING TSV FORMAT")
    log("=" * 70)
    tsv_path = output_dir / "merged_token_frequencies.tsv"
    log(f"Writing: {tsv_path.name}")

    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_ALL)
        writer.writerow(["token_id", "token_string", "count"])

        for token_id, count in tqdm(
            sorted_tokens, desc="Writing TSV", leave=False, ncols=100
        ):
            token_string = token_to_string.get(token_id, "<unknown>")
            writer.writerow([token_id, token_string, count])

    tsv_size = tsv_path.stat().st_size / (1024**2)
    log(f"TSV saved: {tsv_size:.2f} MB")

    # Format 3
    log("\n" + "=" * 70)
    log("[3/3] SAVING PICKLE FORMAT")
    log("=" * 70)
    pkl_path = output_dir / "merged_token_frequencies.pkl.gz"
    log(f"Writing: {pkl_path.name}")

    merged_data = {
        "token_id": [tid for tid, _ in sorted_tokens],
        "token_string": [
            token_to_string.get(tid, "<unknown>") for tid, _ in sorted_tokens
        ],
        "count": [count for _, count in sorted_tokens],
        "metadata": {
            "total_unique_tokens": len(frequencies),
            "total_occurrences": total_tokens,
            "created_at": datetime.now().isoformat(),
            "source": "pile_merged_analysis",
            "format_version": "1.0",
            "description": "Complete token frequencies from Pile corpus (merged file analysis)",
        },
    }

    with gzip.open(pkl_path, "wb") as f:
        pickle.dump(merged_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    pkl_size = pkl_path.stat().st_size / (1024**2)
    log(f"Pickle saved: {pkl_size:.2f} MB")

    # Display top 20 tokens
    log("\n" + "=" * 70)
    log("TOP 20 MOST FREQUENT TOKENS")
    log("=" * 70)
    log(f"{'Rank':<6} {'Token ID':<10} {'Count':<15} Token String")
    log("-" * 70)

    for rank, (token_id, count) in enumerate(sorted_tokens[:20], 1):
        token_string = token_to_string.get(token_id, "<unknown>")
        display_str = (
            repr(token_string)
            if len(token_string) <= 20
            else repr(token_string[:17] + "...")
        )
        log(f"{rank:<6} {token_id:<10} {count:<15,} {display_str}")

    # Summary statistics
    log("\n" + "=" * 70)
    log("SUMMARY STATISTICS")
    log("=" * 70)
    log(f"Total unique tokens: {len(frequencies):,}")
    log(f"Total occurrences:   {total_tokens:,}")
    log(f"Average count:       {total_tokens / len(frequencies):.2f}")
    log(
        f"Most frequent:       {sorted_tokens[0][1]:,} occurrences (token {sorted_tokens[0][0]})"
    )
    log(
        f"Least frequent:      {sorted_tokens[-1][1]:,} occurrences (token {sorted_tokens[-1][0]})"
    )

    # Calculate coverage (what % of corpus do top N tokens cover?)
    log("\nCorpus Coverage:")

    top10_sum = sum(count for _, count in sorted_tokens[:10])
    log(
        f"  Top 10 tokens:    {top10_sum:,} occurrences = {top10_sum / total_tokens * 100:.2f}% of corpus"
    )

    top100_sum = sum(count for _, count in sorted_tokens[:100])
    log(
        f"  Top 100 tokens:   {top100_sum:,} occurrences = {top100_sum / total_tokens * 100:.2f}% of corpus"
    )

    top1000_sum = sum(count for _, count in sorted_tokens[:1000])
    log(
        f"  Top 1000 tokens:  {top1000_sum:,} occurrences = {top1000_sum / total_tokens * 100:.2f}% of corpus"
    )

    # Final summary
    log("\n" + "=" * 70)
    log("ALL FORMATS SAVED SUCCESSFULLY")
    log("=" * 70)
    log(f"\nOutput files in: {output_dir}")
    log(f"  1. CSV (recommended):  {csv_path.name} ({csv_size:.2f} MB)")
    log(f"  2. Pickle (compact):   {pkl_path.name} ({pkl_size:.2f} MB)")
    log(f"  3. TSV (compatible):   {tsv_path.name} ({tsv_size:.2f} MB)")

    log("\nUsage examples:")
    log(f"  CSV:")
    log(f"    import pandas as pd")
    log(f"    import csv")
    log(f"    df = pd.read_csv('{csv_path}', quoting=csv.QUOTE_ALL)")
    log(f"")
    log(f"  Pickle:")
    log(f"    import pickle, gzip")
    log(f"    with gzip.open('{pkl_path}', 'rb') as f:")
    log(f"        data = pickle.load(f)")
    log(f"")
    log(f"  TSV:")
    log(f"    df = pd.read_csv('{tsv_path}', sep='\\t', quoting=csv.QUOTE_ALL)")

    log("\n" + "=" * 70)
    log("NEXT STEP")
    log("=" * 70)
    log("Run: python 06_build_token_dataset.py")
    log("This will compute log-frequencies, percentiles, and deciles.")
    log("=" * 70)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
