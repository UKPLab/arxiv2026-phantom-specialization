#!/usr/bin/env python3
"""
Profile special tokens.

Reads merged frequency data and reports the counts and identities of
Pythia's special tokens.

Reads:
    pile_frequencies/merged_token_freq.pkl.gz

Writes:
    pile_frequencies/special_tokens_analysis.json
    pile_frequencies/special_tokens_report.txt
"""

import pickle
import gzip
import json
from pathlib import Path
from transformers import AutoTokenizer


def log(msg):
    print(msg, flush=True)


def main():
    SCRIPT_DIR = Path(__file__).resolve().parent
    input_file = SCRIPT_DIR / "pile_frequencies" / "merged_token_freq.pkl.gz"
    output_json = SCRIPT_DIR / "pile_frequencies" / "special_tokens_analysis.json"
    output_txt = SCRIPT_DIR / "pile_frequencies" / "special_tokens_report.txt"

    log("=" * 70)
    log("SPECIAL TOKENS ANALYSIS")
    log("=" * 70)
    log(f"Input: {input_file}")

    # Load frequency data
    log("\nLoading frequency data...")
    with gzip.open(input_file, "rb") as f:
        data = pickle.load(f)

    frequencies = data["frequencies"]
    token_to_string = data["token_to_string"]
    total_tokens = data["metadata"]["total_tokens"]

    log(f"Loaded {len(frequencies):,} tokens")
    log(f"  Total tokens in corpus: {total_tokens:,}")

    # Load tokenizer to identify special tokens
    log("\nLoading Pythia tokenizer to identify special tokens...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

    # Identify special tokens
    special_tokens = {}
    special_token_ids = set()

    # Get all special tokens
    if hasattr(tokenizer, "all_special_tokens"):
        for special_token in tokenizer.all_special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(special_token)
            special_tokens[token_id] = special_token
            special_token_ids.add(token_id)

    # Add by attribute
    special_attrs = ["eos_token_id", "bos_token_id", "pad_token_id", "unk_token_id"]
    for attr in special_attrs:
        if hasattr(tokenizer, attr):
            token_id = getattr(tokenizer, attr)
            if token_id is not None:
                token_string = tokenizer.convert_ids_to_tokens([token_id])[0]
                special_tokens[token_id] = token_string
                special_token_ids.add(token_id)

    log(f"Identified {len(special_tokens)} special token types")

    # Analyze special token frequencies
    log("\n" + "=" * 70)
    log("SPECIAL TOKEN FREQUENCIES")
    log("=" * 70)

    special_token_data = []
    special_total = 0

    for token_id in sorted(special_token_ids):
        if token_id in frequencies:
            count = frequencies[token_id]
            token_str = special_tokens[token_id]
            pct = count / total_tokens * 100

            special_token_data.append(
                {
                    "token_id": token_id,
                    "token_string": token_str,
                    "count": count,
                    "percentage": pct,
                }
            )

            special_total += count

            log(f"  {repr(token_str):30s} (ID {token_id:5d}): {count:15,} ({pct:.4f}%)")

    if special_token_data:
        special_pct = special_total / total_tokens * 100
        log(
            f"\n  TOTAL SPECIAL TOKENS: {special_total:,} ({special_pct:.4f}% of corpus)"
        )
    else:
        log("  No special tokens found with non-zero counts")

    # Check for EOD token specifically
    log("\n" + "=" * 70)
    log("EOD TOKEN ANALYSIS")
    log("=" * 70)

    eod_id = 0  # EOD is typically token 0
    if eod_id in frequencies:
        eod_count = frequencies[eod_id]
        eod_pct = eod_count / total_tokens * 100
        eod_token_str = token_to_string.get(eod_id, "<unknown>")

        log(f"  Token ID: {eod_id}")
        log(f"  Token string: {repr(eod_token_str)}")
        log(f"  Count: {eod_count:,}")
        log(f"  Percentage: {eod_pct:.4f}%")
        log(f"  Expected: ~7-8 million (one per logical document)")

        if 6_000_000 < eod_count < 9_000_000:
            log(f"  Count looks reasonable!")
        else:
            log(f"  Unexpected count (expected 6-9 million)")
    else:
        log("  EOD token (ID 0) not found in frequencies!")

    # Top tokens analysis
    log("\n" + "=" * 70)
    log("TOP 20 TOKENS (Including Special Tokens)")
    log("=" * 70)

    sorted_tokens = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:20]

    log(f"{'Rank':<6} {'Token ID':<10} {'Special?':<10} {'Count':<16} {'%':<8} Token")
    log("-" * 70)

    for rank, (token_id, count) in enumerate(sorted_tokens, 1):
        is_special = "YES" if token_id in special_token_ids else "no"
        pct = count / total_tokens * 100
        token_str = token_to_string.get(token_id, f"<unk_{token_id}>")

        # Truncate long strings
        if len(token_str) > 30:
            display_str = repr(token_str[:27] + "...")
        else:
            display_str = repr(token_str)

        log(
            f"{rank:<6} {token_id:<10} {is_special:<10} {count:<16,} {pct:<8.4f} {display_str}"
        )

    # Save JSON report
    log("\n" + "=" * 70)
    log("SAVING REPORTS")
    log("=" * 70)

    report_data = {
        "special_tokens": {
            "total_count": special_total,
            "total_percentage": special_total / total_tokens * 100,
            "num_types": len(special_token_data),
            "tokens": special_token_data,
        },
        "eod_token": {
            "token_id": eod_id,
            "count": frequencies.get(eod_id, 0),
            "percentage": frequencies.get(eod_id, 0) / total_tokens * 100,
            "token_string": token_to_string.get(eod_id, "<unknown>"),
        },
        "top_20_tokens": [
            {
                "rank": rank,
                "token_id": token_id,
                "token_string": token_to_string.get(token_id, f"<unk_{token_id}>"),
                "count": count,
                "percentage": count / total_tokens * 100,
                "is_special": token_id in special_token_ids,
            }
            for rank, (token_id, count) in enumerate(sorted_tokens, 1)
        ],
        "metadata": {
            "total_tokens": total_tokens,
            "unique_tokens": len(frequencies),
            "source_file": str(input_file),
        },
    }

    with open(output_json, "w") as f:
        json.dump(report_data, f, indent=2)

    log(f"Saved JSON report: {output_json}")

    # Save text report
    with open(output_txt, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("SPECIAL TOKENS ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total tokens in corpus: {total_tokens:,}\n")
        f.write(f"Unique tokens: {len(frequencies):,}\n\n")

        f.write("SPECIAL TOKENS:\n")
        f.write("-" * 70 + "\n")
        if special_token_data:
            for st in special_token_data:
                f.write(
                    f"  {st['token_string']:30s} (ID {st['token_id']:5d}): "
                    f"{st['count']:15,} ({st['percentage']:.4f}%)\n"
                )
            f.write(
                f"\nTotal special tokens: {special_total:,} ({special_total / total_tokens * 100:.4f}%)\n"
            )
        else:
            f.write("  No special tokens found\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("EOD TOKEN:\n")
        f.write("-" * 70 + "\n")
        if eod_id in frequencies:
            eod_count = frequencies[eod_id]
            f.write(f"  Token ID: {eod_id}\n")
            f.write(f"  Token: {repr(token_to_string.get(eod_id, '<unknown>'))}\n")
            f.write(f"  Count: {eod_count:,}\n")
            f.write(f"  Percentage: {eod_count / total_tokens * 100:.4f}%\n")
        else:
            f.write("  EOD token not found!\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("TOP 20 TOKENS:\n")
        f.write("-" * 70 + "\n")
        for item in report_data["top_20_tokens"]:
            special_mark = " [SPECIAL]" if item["is_special"] else ""
            f.write(
                f"  {item['rank']:2d}. {repr(item['token_string']):30s} "
                f"({item['count']:,}, {item['percentage']:.4f}%){special_mark}\n"
            )

    log(f"Saved text report: {output_txt}")

    log("\n" + "=" * 70)
    log("ANALYSIS COMPLETE")
    log("=" * 70)
    log(f"\nGenerated files:")
    log(f"  1. {output_json}")
    log(f"  2. {output_txt}")
    log(f"\nYou can now review these files for special token information.")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
