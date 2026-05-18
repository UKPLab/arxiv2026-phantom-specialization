#!/usr/bin/env python3
"""
Token Pool Exporter
====================
Pipeline step 10.  Exports the actual token pools for each frequency band.

PURPOSE
-------
Takes the categorized vocabulary (08) and the canonical band definitions (09),
filters word_en tokens into their respective bands, and exports per-band
JSON files ready for downstream dataset generation (LSC, IOI, etc.).

Each pool file contains the tokens with all their metadata so that
dataset generators can filter further (e.g. by capitalization for IOI names)
without re-running the categorizer.

INPUT
-----
token_categories.csv        Categorized vocabulary  (from 08_token_categorizer.py)
final_bands.json            Canonical band definitions  (from 09_band_designer.py)

OUTPUTS
-------
token_pools/
├── pool_bottom_tail.json
├── pool_very_low.json
├── pool_low.json
├── pool_medium.json
├── pool_high.json
├── pool_very_high.json
├── pool_top_tail.json
├── pool_control.json
└── pool_summary.json     Summary counts and validation

Usage:
    python 10_token_pool_exporter.py
    python 10_token_pool_exporter.py --categories-csv ... --bands-json ...
"""

import json
import argparse
import logging
import sys
from pathlib import Path
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
CATEGORIES_CSV = SCRIPT_DIR / "token_categories" / "token_categories.csv"
BANDS_JSON = SCRIPT_DIR / "band_design" / "final_bands.json"
OUTPUT_DIR = SCRIPT_DIR / "token_pools"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# Columns from 08 CSV to include in each pool entry
EXPORT_FIELDS = [
    "token_id",
    "token_string",
    "content_text",
    "log_frequency",
    "has_space_prefix",
    "token_label",
    "capitalization",
    "n_content_chars",
    "is_ascii",
    "primary_script",
]

# Optional fields
OPTIONAL_FIELDS = [
    "freq_per_million",
    "raw_count",
    "percentile",
    "mixed_subtype",
    "is_single_token_no_space",  # For IOI: verifies token encodes as single token
    "is_single_token_with_space",  # For IOI: verifies token encodes as single token with space prefix
]


def row_to_entry(row: pd.Series, fields: list, optional: list) -> OrderedDict:
    """Convert a DataFrame row to a clean ordered dict."""
    entry = OrderedDict()
    for f in fields:
        val = row[f]
        # Clean up numpy types for JSON serialization
        if isinstance(val, (np.integer,)):
            val = int(val)
        elif isinstance(val, (np.floating,)):
            val = round(float(val), 6)
        elif isinstance(val, (np.bool_,)):
            val = bool(val)
        entry[f] = val
    for f in optional:
        if f in row.index and pd.notna(row[f]):
            val = row[f]
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = round(float(val), 6)
            entry[f] = val
    return entry


def build_pool(
    wdf: pd.DataFrame,
    band_name: str,
    band_def: dict,
    fields: list,
    optional: list,
    chosen_k: int = 5,
) -> dict:
    """
    Build a token pool for a single band.

    For core/exploratory bands: filter by log_freq_range.
    For control: include all word_en tokens with frequency weights.

    Boundary convention (matches 09_band_designer):
      - Core band i < k-1:  [lo, hi)   (exclusive upper to prevent overlap)
      - Core band k-1:      [lo, hi]   (inclusive upper for last core band)
      - bottom_tail:         [lo, hi)   (exclusive upper to prevent overlap with first core band)
      - top_tail:            (lo, hi]   (exclusive lower to prevent overlap with last core band)
    """
    band_type = band_def["type"]
    lo, hi = band_def["log_freq_range"]

    if band_type == "core":
        band_index = band_def.get("band_index", 0)
        if band_index < chosen_k - 1:
            # Non-last core band: [lo, hi)
            mask = (wdf["log_frequency"] >= lo) & (wdf["log_frequency"] < hi)
        else:
            # Last core band: [lo, hi]
            mask = (wdf["log_frequency"] >= lo) & (wdf["log_frequency"] <= hi)
        pool_df = wdf[mask].copy()
    elif band_type == "exploratory":
        if band_name == "bottom_tail":
            # [lo, hi); exclusive upper to avoid overlap with first core band
            mask = (wdf["log_frequency"] >= lo) & (wdf["log_frequency"] < hi)
        else:
            # top_tail (lo, hi]; exclusive lower to avoid overlap with last core band
            mask = (wdf["log_frequency"] > lo) & (wdf["log_frequency"] <= hi)
        pool_df = wdf[mask].copy()
    elif band_type == "baseline":
        # Control: all word_en tokens
        pool_df = wdf.copy()
    else:
        raise ValueError(f"Unknown band type: {band_type}")

    # Sort by log_frequency
    pool_df = pool_df.sort_values("log_frequency")

    # Build token list
    tokens = []
    for _, row in pool_df.iterrows():
        tokens.append(row_to_entry(row, fields, optional))

    # Compute frequency weights for control band
    weights = None
    if band_type == "baseline" and "freq_per_million" in pool_df.columns:
        freqs = pool_df["freq_per_million"].values.astype(float)
        weights = (freqs / freqs.sum()).tolist()
        weights = [round(w, 10) for w in weights]

    # Capitalization counts
    cap_counts = pool_df["capitalization"].value_counts().to_dict()
    cap_counts = {str(k): int(v) for k, v in cap_counts.items()}

    # Build pool object
    pool = OrderedDict()
    pool["band_name"] = band_name
    pool["band_type"] = band_type
    pool["log_freq_range"] = [round(lo, 4), round(hi, 4)]
    pool["n_tokens"] = len(tokens)
    pool["capitalization"] = cap_counts

    if band_type == "core":
        pool["band_index"] = band_def.get("band_index")
        pool["center"] = band_def.get("center")
        pool["freq_ratio"] = band_def.get("freq_ratio")

    if band_type == "exploratory":
        pool["notes"] = band_def.get("notes", "")

    if band_type == "baseline":
        pool["sampling"] = band_def.get("sampling", "frequency_weighted")
        pool["sampling_description"] = band_def.get("sampling_description", "")
        if weights is not None:
            pool["frequency_weights_included"] = True

    # Log-freq stats for this pool
    if len(tokens) > 0:
        lf = pool_df["log_frequency"].values
        pool["log_freq_stats"] = OrderedDict(
            [
                ("min", round(float(lf.min()), 4)),
                ("max", round(float(lf.max()), 4)),
                ("mean", round(float(lf.mean()), 4)),
                ("median", round(float(np.median(lf)), 4)),
                ("std", round(float(lf.std()), 4)),
            ]
        )

    pool["tokens"] = tokens

    # Add weights array for control (separate from tokens for cleanliness)
    if weights is not None:
        pool["frequency_weights"] = weights

    return pool


def main():
    parser = argparse.ArgumentParser(
        description="Export token pools for each frequency band",
    )
    parser.add_argument(
        "--categories-csv",
        type=Path,
        default=CATEGORIES_CSV,
        help="Path to token_categories.csv",
    )
    parser.add_argument(
        "--bands-json", type=Path, default=BANDS_JSON, help="Path to final_bands.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory"
    )
    args = parser.parse_args()

    # ---- Validate inputs ----
    if not args.categories_csv.exists():
        logger.error(f"Categories CSV not found: {args.categories_csv}")
        return 1
    if not args.bands_json.exists():
        logger.error(f"Bands JSON not found: {args.bands_json}")
        return 1

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load categories ----
    logger.info(f"Loading categories: {args.categories_csv}")
    df = pd.read_csv(args.categories_csv, keep_default_na=False)
    logger.info(f"  {len(df):,} tokens loaded")

    # Filter to word_en
    wdf = df[df["is_word_en"].astype(str) == "True"].copy()
    logger.info(f"  word_en tokens: {len(wdf):,}")

    # Determine available fields
    fields = [f for f in EXPORT_FIELDS if f in wdf.columns]
    optional = [f for f in OPTIONAL_FIELDS if f in wdf.columns]
    missing = [f for f in EXPORT_FIELDS if f not in wdf.columns]
    if missing:
        logger.warning(f"  Missing expected fields (skipped): {missing}")
    logger.info(f"  Export fields: {fields}")
    logger.info(f"  Optional fields: {optional}")

    # ---- Load band definitions ----
    logger.info(f"Loading bands: {args.bands_json}")
    with open(args.bands_json) as f:
        bands_data = json.load(f)

    bands = bands_data["bands"]
    logger.info(f"  {len(bands)} bands defined: {list(bands.keys())}")

    # ---- Build and export pools ----
    logger.info("")
    summary_bands = OrderedDict()
    total_core_tokens = 0
    total_exported = 0

    chosen_k = bands_data.get("chosen_k", 5)

    for band_name, band_def in bands.items():
        logger.info(f"Building pool: {band_name}")
        pool = build_pool(wdf, band_name, band_def, fields, optional, chosen_k=chosen_k)

        # Save
        pool_path = out / f"pool_{band_name}.json"
        with open(pool_path, "w") as f:
            json.dump(pool, f, indent=2)

        n = pool["n_tokens"]
        cap = pool["capitalization"]
        n_cap = cap.get("capitalized", 0)
        n_low = cap.get("lowercase", 0)
        btype = pool["band_type"]

        logger.info(
            f"  -> {pool_path.name}: {n:,} tokens "
            f"(lower={n_low:,}, cap={n_cap:,}) [{btype}]"
        )

        # Summary entry
        summary_bands[band_name] = OrderedDict(
            [
                ("type", btype),
                ("n_tokens", n),
                ("capitalization", cap),
                ("log_freq_range", pool["log_freq_range"]),
                ("file", pool_path.name),
            ]
        )
        if "log_freq_stats" in pool:
            summary_bands[band_name]["log_freq_stats"] = pool["log_freq_stats"]

        if btype == "core":
            total_core_tokens += n
        if btype != "baseline":
            total_exported += n

    # ---- Validation ----
    logger.info("")
    logger.info("Validating...")

    # Check: core bands should cover all word_en tokens in core range
    core_range = bands_data.get("core_range", [None, None])
    if core_range[0] is not None:
        core_mask = (wdf["log_frequency"] >= core_range[0]) & (
            wdf["log_frequency"] <= core_range[1]
        )
        expected_core = int(core_mask.sum())
        if total_core_tokens != expected_core:
            logger.warning(
                f"  Core token count mismatch: "
                f"exported={total_core_tokens}, "
                f"expected={expected_core}"
            )
        else:
            logger.info(
                f"  Core bands cover all {expected_core:,} "
                f"tokens in range [{core_range[0]:.3f}, {core_range[1]:.3f}]"
            )

    # Check: no token should appear in multiple core/exploratory bands
    # (control intentionally overlaps all bands)
    logger.info(f"  Total non-control tokens exported: {total_exported:,}")

    # ---- Save summary ----
    summary = OrderedDict()
    summary["created_at"] = datetime.now().isoformat()
    summary["source_categories"] = str(args.categories_csv)
    summary["source_bands"] = str(args.bands_json)
    summary["n_word_en_total"] = len(wdf)
    summary["chosen_k"] = bands_data.get("chosen_k")
    summary["bands"] = summary_bands

    summary_path = out / "pool_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved: {summary_path}")

    # ---- Final log ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("TOKEN POOL EXPORT COMPLETE")
    logger.info("=" * 60)
    logger.info(
        f"  {'band':<15s}  {'type':<12s}  {'tokens':>7s}  {'cap':>5s}  {'lower':>7s}"
    )
    logger.info(f"  {'-' * 15}  {'-' * 12}  {'-' * 7}  {'-' * 5}  {'-' * 7}")
    for name, info in summary_bands.items():
        cap = info["capitalization"]
        logger.info(
            f"  {name:<15s}  {info['type']:<12s}  "
            f"{info['n_tokens']:>7,d}  "
            f"{cap.get('capitalized', 0):>5,d}  "
            f"{cap.get('lowercase', 0):>7,d}"
        )
    logger.info(f"\n  Output directory: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
