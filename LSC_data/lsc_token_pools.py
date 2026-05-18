#!/usr/bin/env python3
"""
LSC Token Pool Generator
=========================
Creates task-specific token pools for Literal Sequence Copying from
generic validated pools (script 11).

PURPOSE
-------
Applies LSC-specific filters to validated (WORD-only) pools:
  1. Lowercase only              (eliminates capitalization confound)
  2. Exact character length matching across bands  (eliminates length confound)

Exports pools into two subdirectories:
  matched/   - Length-controlled pools (matchable bands + control)
  unmatched/ - Original LC WORD pools (all 8 bands, no length matching)

The report includes recommendations on which bands to drop during
circuit discovery (based on pool size and matching status).

Pipeline position:
  Script 11 (validated pools, all caps) -> THIS -> lsc_generator.py

INPUT
-----
pythia_data/token_pools_validated/pool_validated_*.json
pythia_data/band_design/final_bands.json

OUTPUTS
-------
lsc_token_pools/
├── matched/
│   ├── lsc_pool_low.json         Length-matched LC WORD
│   ├── lsc_pool_medium.json
│   ├── lsc_pool_high.json
│   ├── lsc_pool_very_high.json
│   └── lsc_pool_control.json     Union of matched cores, freq-weighted
├── unmatched/
│   ├── lsc_pool_{band}.json      One per band (all 8), LC WORD only
│   └── lsc_pool_control.json     All LC WORD control tokens, freq-weighted
├── lsc_pool_report.txt
└── lsc_pool_summary.json

Usage:
    python lsc_token_pools.py
"""

import json
import argparse
import logging
import sys
from pathlib import Path
from collections import OrderedDict, Counter
from datetime import datetime

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATED_DIR = SCRIPT_DIR.parent / "pythia_data" / "token_pools_validated"
BANDS_JSON = SCRIPT_DIR.parent / "pythia_data" / "band_design" / "final_bands.json"
OUTPUT_DIR = SCRIPT_DIR / "lsc_token_pools"

DEFAULT_MATCH_BANDS = ["low", "medium", "high", "very_high"]
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_validated_pool(validated_dir: Path, band_name: str) -> dict:
    """Load a validated pool JSON file."""
    path = validated_dir / f"pool_validated_{band_name}.json"
    with open(path) as f:
        return json.load(f)


def strip_unused_fields(tokens: list) -> list:
    """Remove upstream metadata fields not used by LSC pipeline."""
    drop = {"percentile"}
    return [{k: v for k, v in t.items() if k not in drop} for t in tokens]


def filter_lowercase(tokens: list) -> list:
    """Filter tokens to lowercase only."""
    return [t for t in tokens if t["capitalization"] == "lowercase"]


def compute_length_overlap(lc_by_band: dict, match_bands: list) -> tuple:
    """
    Compute matchable overlap across bands by character length.

    Returns
    -------
    overlap : dict  {length: min_count_across_bands}
    by_length : dict  {band: {length: [tokens]}}
    excluded : dict  {length: [bands_missing]}
    """
    by_length = {}
    for band in match_bands:
        bl = {}
        for t in lc_by_band[band]:
            L = t["n_content_chars"]
            bl.setdefault(L, []).append(t)
        by_length[band] = bl

    all_lengths = set()
    for bl in by_length.values():
        all_lengths.update(bl.keys())

    overlap = {}
    excluded = {}
    for L in sorted(all_lengths):
        counts = {b: len(by_length[b].get(L, [])) for b in match_bands}
        min_count = min(counts.values())
        if min_count > 0:
            overlap[L] = min_count
        else:
            missing = [b for b, c in counts.items() if c == 0]
            excluded[L] = missing

    return overlap, by_length, excluded


def create_matched_pool(
    overlap: dict,
    by_length: dict,
    band_name: str,
    band_def: dict,
    rng: np.random.Generator,
) -> dict:
    """Sample tokens to create a length-matched pool for one band."""
    selected = []
    for L in sorted(overlap.keys()):
        candidates = by_length[band_name].get(L, [])
        n = overlap[L]
        if len(candidates) > n:
            indices = rng.choice(len(candidates), size=n, replace=False)
            indices.sort()
            selected.extend([candidates[int(i)] for i in indices])
        else:
            selected.extend(candidates)

    selected.sort(key=lambda t: t["log_frequency"])
    matched_n = sum(overlap.values())

    pool = OrderedDict()
    pool["band_name"] = band_name
    pool["band_type"] = band_def["type"]
    pool["log_freq_range"] = [
        round(band_def["log_freq_range"][0], 4),
        round(band_def["log_freq_range"][1], 4),
    ]
    pool["n_tokens"] = len(selected)
    pool["matching"] = OrderedDict(
        [
            ("method", "lowercase_word_length_matched"),
            (
                "filters",
                [
                    "FineWeb WORD validated (from script 11)",
                    "lowercase only",
                    "exact character length matching across bands",
                ],
            ),
            ("n_per_band", matched_n),
            ("lengths_matched", sorted(overlap.keys())),
            ("tokens_per_length", {str(k): v for k, v in sorted(overlap.items())}),
        ]
    )

    cap_counts = Counter(t["capitalization"] for t in selected)
    pool["capitalization"] = {str(k): int(v) for k, v in sorted(cap_counts.items())}

    if band_def["type"] == "core":
        pool["band_index"] = band_def.get("band_index")
        pool["center"] = band_def.get("center")
        pool["freq_ratio"] = band_def.get("freq_ratio")

    if selected:
        lf = np.array([t["log_frequency"] for t in selected])
        pool["log_freq_stats"] = OrderedDict(
            [
                ("min", round(float(lf.min()), 4)),
                ("max", round(float(lf.max()), 4)),
                ("mean", round(float(lf.mean()), 4)),
                ("median", round(float(np.median(lf)), 4)),
                ("std", round(float(lf.std()), 4)),
            ]
        )

    pool["tokens"] = selected
    return pool


def create_matched_control(
    matched_pools: dict, match_bands: list, band_def: dict
) -> dict:
    """Create matched control from union of matched core bands."""
    all_tokens = []
    for band in match_bands:
        all_tokens.extend(matched_pools[band]["tokens"])

    all_tokens.sort(key=lambda t: t["log_frequency"])

    # Frequency weights
    freqs = []
    for t in all_tokens:
        if "freq_per_million" in t:
            freqs.append(t["freq_per_million"])
        else:
            freqs.append(10 ** t["log_frequency"])
    total = sum(freqs)
    weights = [round(f / total, 10) for f in freqs]

    log_freqs = [t["log_frequency"] for t in all_tokens]

    pool = OrderedDict()
    pool["band_name"] = "control"
    pool["band_type"] = "baseline"
    pool["log_freq_range"] = [round(min(log_freqs), 4), round(max(log_freqs), 4)]
    pool["n_tokens"] = len(all_tokens)
    pool["matching"] = OrderedDict(
        [
            ("method", "union_of_matched_core_bands"),
            (
                "description",
                "Union of length-matched lowercase WORD tokens from core bands. "
                "Frequency-weighted sampling.",
            ),
            ("core_bands", match_bands),
        ]
    )
    pool["capitalization"] = {"lowercase": len(all_tokens)}
    pool["sampling"] = "frequency_weighted"
    pool["sampling_description"] = band_def.get("sampling_description", "")
    pool["frequency_weights_included"] = True

    if log_freqs:
        lf = np.array(log_freqs)
        pool["log_freq_stats"] = OrderedDict(
            [
                ("min", round(float(lf.min()), 4)),
                ("max", round(float(lf.max()), 4)),
                ("mean", round(float(lf.mean()), 4)),
                ("median", round(float(np.median(lf)), 4)),
                ("std", round(float(lf.std()), 4)),
            ]
        )

    pool["tokens"] = all_tokens
    pool["frequency_weights"] = weights
    return pool


def create_unmatched_pool(lc_tokens: list, band_name: str, band_def: dict) -> dict:
    """Create an unmatched LSC pool (lowercase WORD, no length matching)."""
    tokens = sorted(lc_tokens, key=lambda t: t["log_frequency"])

    pool = OrderedDict()
    pool["band_name"] = band_name
    pool["band_type"] = band_def["type"]
    pool["log_freq_range"] = [
        round(band_def["log_freq_range"][0], 4),
        round(band_def["log_freq_range"][1], 4),
    ]
    pool["n_tokens"] = len(tokens)
    pool["matching"] = OrderedDict(
        [
            ("method", "lowercase_word_unmatched"),
            (
                "filters",
                [
                    "FineWeb WORD validated (from script 11)",
                    "lowercase only",
                ],
            ),
        ]
    )
    pool["capitalization"] = {"lowercase": len(tokens)}

    if band_def["type"] == "core":
        pool["band_index"] = band_def.get("band_index")
        pool["center"] = band_def.get("center")
        pool["freq_ratio"] = band_def.get("freq_ratio")

    if band_def["type"] == "exploratory":
        pool["notes"] = band_def.get("notes", "")

    if band_def["type"] == "baseline":
        pool["sampling"] = band_def.get("sampling", "frequency_weighted")
        pool["sampling_description"] = band_def.get("sampling_description", "")
        if tokens:
            freqs = []
            for t in tokens:
                if "freq_per_million" in t:
                    freqs.append(t["freq_per_million"])
                else:
                    freqs.append(10 ** t["log_frequency"])
            total = sum(freqs)
            pool["frequency_weights"] = [round(f / total, 10) for f in freqs]
            pool["frequency_weights_included"] = True

    if tokens:
        lf = np.array([t["log_frequency"] for t in tokens])
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
    return pool


def write_report(
    all_bands,
    band_types,
    lc_counts,
    match_bands,
    overlap,
    excluded,
    matched_pools,
    unmatched_pools,
    output_pools,
    output_path,
):
    """Write human-readable LSC pool report."""
    MIN_TOKENS_PER_SEQ = 16
    lines = []
    w = 80
    lines.append("=" * w)
    lines.append("LSC TOKEN POOL REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Task: Literal Sequence Copying (lowercase, length-matched)")
    lines.append("=" * w)
    lines.append("")

    # --- Section 1: LC WORD availability ---
    lines.append("1. LOWERCASE WORD TOKENS PER BAND")
    lines.append("-" * w)
    lines.append(
        f"{'Band':<14s} {'Type':<12s}  {'Validated':>9s}  {'LC_WORD':>8s}  {'%LC':>6s}"
    )
    lines.append(f"{'-' * 14} {'-' * 12}  {'-' * 9}  {'-' * 8}  {'-' * 6}")
    for band in all_bands:
        btype = band_types[band]
        n_lc = unmatched_pools[band]["n_tokens"]
        pct = lc_counts[band]["pct"] if band in lc_counts else 0
        lines.append(
            f"{band:<14s} {btype:<12s}  {lc_counts[band]['n_validated']:>9d}  "
            f"{n_lc:>8d}  {pct:>5.1f}%"
        )
    lines.append("")

    # --- Section 2: Length matching ---
    lines.append("2. LENGTH MATCHING")
    lines.append("-" * w)
    matched_n = sum(overlap.values())
    lines.append(f"Matched bands: {match_bands}")
    lines.append(f"Matched lengths: {sorted(overlap.keys())}")
    lines.append(f"Tokens per band: {matched_n}")
    lines.append("")

    col_w = max(8, max(len(b) for b in match_bands) + 1)
    hdr = f"{'Len':>4s}  {'Match':>6s}  "
    hdr += "  ".join(f"{b:>{col_w}s}" for b in match_bands)
    lines.append(hdr)
    lines.append(
        f"{'-' * 4}  {'-' * 6}  " + "  ".join(f"{'-' * col_w}" for _ in match_bands)
    )
    for L in sorted(overlap.keys()):
        avail = []
        for band in match_bands:
            n_lc = len(
                [
                    t
                    for t in unmatched_pools[band]["tokens"]
                    if t["n_content_chars"] == L
                ]
            )
            avail.append(f"{n_lc:>{col_w}d}")
        lines.append(f"{L:>4d}  {overlap[L]:>6d}  " + "  ".join(avail))

    if excluded:
        lines.append("")
        lines.append("Excluded lengths (missing in at least one band):")
        for L in sorted(excluded.keys()):
            lines.append(f"  Length {L}: missing in {excluded[L]}")
    lines.append("")

    # --- Section 3: Output pool summary (all bands) ---
    lines.append("3. OUTPUT POOL SUMMARY (all bands)")
    lines.append("-" * w)
    lines.append(
        f"{'Band':<14s}  {'N':>6s}  {'Method':<28s}  {'LogF_u':>8s}  {'LogF_s':>8s}"
    )
    lines.append(f"{'-' * 14}  {'-' * 6}  {'-' * 28}  {'-' * 8}  {'-' * 8}")
    for band in all_bands:
        p = output_pools[band]
        lfs = p.get("log_freq_stats", {})
        method = p["matching"]["method"]
        lines.append(
            f"{band:<14s}  {p['n_tokens']:>6d}  {method:<28s}  "
            f"{lfs.get('mean', 0):>8.4f}  {lfs.get('std', 0):>8.4f}"
        )
    lines.append("")

    # --- Section 4: Matched pool statistics ---
    lines.append("4. MATCHED POOL STATISTICS")
    lines.append("-" * w)
    lines.append(
        f"{'Band':<14s}  {'N':>6s}  {'LogF_u':>8s}  {'LogF_s':>8s}  "
        f"{'Len_u':>6s}  {'Len_s':>6s}"
    )
    lines.append(f"{'-' * 14}  {'-' * 6}  {'-' * 8}  {'-' * 8}  {'-' * 6}  {'-' * 6}")
    for band in match_bands:
        mp = matched_pools[band]
        lfs = mp.get("log_freq_stats", {})
        lengths = [t["n_content_chars"] for t in mp["tokens"]]
        lines.append(
            f"{band:<14s}  {mp['n_tokens']:>6d}  "
            f"{lfs.get('mean', 0):>8.4f}  {lfs.get('std', 0):>8.4f}  "
            f"{np.mean(lengths):>6.2f}  {np.std(lengths):>6.2f}"
        )
    if "control" in matched_pools:
        mp = matched_pools["control"]
        lfs = mp.get("log_freq_stats", {})
        lengths = [t["n_content_chars"] for t in mp["tokens"]]
        lines.append(
            f"{'control':<14s}  {mp['n_tokens']:>6d}  "
            f"{lfs.get('mean', 0):>8.4f}  {lfs.get('std', 0):>8.4f}  "
            f"{np.mean(lengths):>6.2f}  {np.std(lengths):>6.2f}"
        )
    lines.append("")

    # --- Section 5: Frequency shift check ---
    lines.append("5. FREQUENCY SHIFT (unmatched vs matched)")
    lines.append("-" * w)
    lines.append(f"{'Band':<14s}  {'Unm_u':>8s}  {'Mat_u':>8s}  {'Delta':>8s}")
    lines.append(f"{'-' * 14}  {'-' * 8}  {'-' * 8}  {'-' * 8}")
    for band in match_bands:
        um = unmatched_pools[band].get("log_freq_stats", {})
        mm = matched_pools[band].get("log_freq_stats", {})
        um_mean = um.get("mean", 0)
        mm_mean = mm.get("mean", 0)
        delta = mm_mean - um_mean
        lines.append(f"{band:<14s}  {um_mean:>8.4f}  {mm_mean:>8.4f}  {delta:>+8.4f}")
    lines.append("")

    # --- Section 6: Assessment ---
    lines.append("6. ASSESSMENT")
    lines.append("-" * w)
    if matched_n >= 500:
        lines.append(
            f"PASS: Matched pool ({matched_n}/band) adequate for "
            "LSC dataset generation."
        )
    elif matched_n >= 300:
        lines.append(
            f"WARNING: Matched pool ({matched_n}/band) usable but "
            "may cause token reuse."
        )
    else:
        lines.append(f"CRITICAL: Matched pool ({matched_n}/band) too small.")

    lines.append("")
    lines.append(f"With {matched_n} tokens and 16 unique tokens/example:")
    n_examples = 1500
    reuse = n_examples * 16 / matched_n
    lines.append(f"  1500 examples -> ~{reuse:.1f}x avg token reuse")
    lines.append("")

    # --- Section 7: Recommendations ---
    lines.append("7. RECOMMENDATIONS")
    lines.append("-" * w)
    lines.append("Based on pool sizes, matching status, and minimum requirements")
    lines.append(f"(16 unique tokens per LSC sequence):")
    lines.append("")
    lines.append(
        f"  {'Band':<14s}  {'Pool':>6s}  {'Matched':>8s}  {'Recommendation':<40s}"
    )
    lines.append(f"  {'-' * 14}  {'-' * 6}  {'-' * 8}  {'-' * 40}")
    for band in all_bands:
        p = output_pools[band]
        n = p["n_tokens"]
        is_matched = band in matched_pools
        matched_str = "yes" if is_matched else "no"

        if n < MIN_TOKENS_PER_SEQ:
            rec = "DROP -- cannot fill one sequence"
        elif not is_matched and n < 100:
            rec = "CAUTION -- unmatched, very small pool"
        elif not is_matched:
            rec = "CAUTION -- unmatched, no length control"
        else:
            rec = "KEEP"

        lines.append(f"  {band:<14s}  {n:>6d}  {matched_str:>8s}  {rec:<40s}")

    lines.append("")
    keep = [b for b in all_bands if b in matched_pools]
    caution = [
        b
        for b in all_bands
        if b not in matched_pools and output_pools[b]["n_tokens"] >= MIN_TOKENS_PER_SEQ
    ]
    drop = [b for b in all_bands if output_pools[b]["n_tokens"] < MIN_TOKENS_PER_SEQ]

    lines.append(f"  Recommended primary analysis: {keep}")
    if caution:
        lines.append(f"  Include with caveats:         {caution}")
    if drop:
        lines.append(f"  Recommend dropping:           {drop}")

    lines.append("")
    lines.append("=" * w)
    lines.append("END OF REPORT")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate LSC-specific token pools from validated pools",
    )
    parser.add_argument(
        "--validated-dir",
        type=Path,
        default=VALIDATED_DIR,
        help="Directory with pool_validated_*.json from script 11",
    )
    parser.add_argument(
        "--bands-json", type=Path, default=BANDS_JSON, help="Path to final_bands.json"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for LSC token pools",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED, help="Random seed for matched sampling"
    )
    parser.add_argument(
        "--match-bands",
        type=str,
        nargs="+",
        default=DEFAULT_MATCH_BANDS,
        help="Bands to include in length matching (default: low medium high very_high)",
    )
    args = parser.parse_args()

    # ---- Validate inputs ----
    if not args.validated_dir.exists():
        logger.error(f"Validated pool dir not found: {args.validated_dir}")
        return 1
    if not args.bands_json.exists():
        logger.error(f"Bands JSON not found: {args.bands_json}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matched_dir = args.output_dir / "matched"
    unmatched_dir = args.output_dir / "unmatched"
    matched_dir.mkdir(parents=True, exist_ok=True)
    unmatched_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ---- Load band definitions ----
    logger.info(f"Loading bands: {args.bands_json}")
    with open(args.bands_json) as f:
        bands_data = json.load(f)

    all_bands = bands_data["summary"]["condition_names_ordered"]
    band_types = {name: bands_data["bands"][name]["type"] for name in all_bands}
    match_bands = args.match_bands

    # Validate match bands exist
    for b in match_bands:
        if b not in bands_data["bands"]:
            logger.error(f"Match band '{b}' not found in bands JSON")
            return 1

    # ---- Load validated pools & filter to lowercase ----
    logger.info(f"Loading validated pools from: {args.validated_dir}")
    lc_by_band = {}  # band -> [lowercase tokens]
    lc_counts = {}  # band -> stats
    validated_sizes = {}

    for band_name in all_bands:
        pool = load_validated_pool(args.validated_dir, band_name)
        n_validated = pool["n_tokens"]
        validated_sizes[band_name] = n_validated
        lc_tokens = filter_lowercase(pool["tokens"])
        lc_tokens = strip_unused_fields(lc_tokens)
        lc_by_band[band_name] = lc_tokens

        pct = round(100 * len(lc_tokens) / n_validated, 1) if n_validated > 0 else 0
        lc_counts[band_name] = {
            "n_validated": n_validated,
            "n_lc": len(lc_tokens),
            "pct": pct,
        }
        logger.info(
            f"  {band_name}: {n_validated:,} validated -> "
            f"{len(lc_tokens):,} lowercase ({pct}%)"
        )

    # ---- Create unmatched pools (all bands) ----
    logger.info("")
    logger.info("Creating unmatched LSC pools (lowercase WORD only)...")
    unmatched_pools = {}
    for band_name in all_bands:
        band_def = bands_data["bands"][band_name]
        unmatched_pools[band_name] = create_unmatched_pool(
            lc_by_band[band_name], band_name, band_def
        )
        logger.info(f"  {band_name}: {unmatched_pools[band_name]['n_tokens']:,} tokens")

    # ---- Compute length overlap for matched bands ----
    logger.info("")
    logger.info(f"Computing length overlap for: {match_bands}")
    overlap, by_length, excluded = compute_length_overlap(lc_by_band, match_bands)
    matched_n = sum(overlap.values())
    logger.info(f"  Matched lengths: {sorted(overlap.keys())}")
    logger.info(f"  Tokens per band: {matched_n}")

    # ---- Create matched pools ----
    logger.info("")
    logger.info("Creating matched LSC pools...")
    matched_pools = {}
    for band_name in match_bands:
        band_def = bands_data["bands"][band_name]
        matched_pools[band_name] = create_matched_pool(
            overlap, by_length, band_name, band_def, rng
        )
        mp = matched_pools[band_name]
        lfs = mp.get("log_freq_stats", {})
        logger.info(
            f"  {band_name}: {mp['n_tokens']:,} tokens, "
            f"log_freq {lfs.get('mean', 0):.4f} +/- {lfs.get('std', 0):.4f}"
        )

    # Matched control
    control_def = bands_data["bands"]["control"]
    matched_pools["control"] = create_matched_control(
        matched_pools, match_bands, control_def
    )
    logger.info(
        f"  control: {matched_pools['control']['n_tokens']:,} tokens "
        f"(union of {len(match_bands)} matched bands)"
    )

    # ---- Validate matched pools ----
    logger.info("")
    logger.info("Validating matched pools...")
    sizes = {b: matched_pools[b]["n_tokens"] for b in match_bands}
    if len(set(sizes.values())) == 1:
        logger.info(f"  PASS: All matched bands have n={list(sizes.values())[0]}")
    else:
        logger.error(f"  FAIL: Unequal sizes: {sizes}")
        return 1

    ref_dist = Counter(
        t["n_content_chars"] for t in matched_pools[match_bands[0]]["tokens"]
    )
    for b in match_bands[1:]:
        dist = Counter(t["n_content_chars"] for t in matched_pools[b]["tokens"])
        if dist != ref_dist:
            logger.error(f"  FAIL: {b} length distribution differs")
            return 1
    logger.info("  PASS: Identical length distributions across matched bands")

    # ---- Combine output_pools view for report (matched where possible) ----
    MIN_TOKENS_PER_SEQ = 16  # source + target + distractors
    output_pools = OrderedDict()
    for band_name in all_bands:
        if band_name in matched_pools:
            output_pools[band_name] = matched_pools[band_name]
        else:
            output_pools[band_name] = unmatched_pools[band_name]

    # ---- Export matched pools ----
    logger.info("")
    logger.info("Exporting matched pools...")
    for band_name in matched_pools:
        path = matched_dir / f"lsc_pool_{band_name}.json"
        with open(path, "w") as f:
            json.dump(matched_pools[band_name], f, indent=2)
        n = matched_pools[band_name]["n_tokens"]
        logger.info(f"  matched/{path.name}  ({n:,} tokens)")

    # ---- Export unmatched pools (all bands) ----
    logger.info("Exporting unmatched pools...")
    for band_name in all_bands:
        path = unmatched_dir / f"lsc_pool_{band_name}.json"
        with open(path, "w") as f:
            json.dump(unmatched_pools[band_name], f, indent=2)
        n = unmatched_pools[band_name]["n_tokens"]
        flag = ""
        if n < MIN_TOKENS_PER_SEQ:
            flag = "  [TOO SMALL]"
        logger.info(f"  unmatched/{path.name}  ({n:,} tokens){flag}")

    # ---- Summary JSON ----
    summary = OrderedDict()
    summary["created_at"] = datetime.now().isoformat()
    summary["task"] = "LSC"
    summary["seed"] = args.seed
    summary["source"] = str(args.validated_dir)
    summary["match_bands"] = match_bands
    summary["matched_n_per_band"] = matched_n
    summary["matched_lengths"] = sorted(overlap.keys())
    summary["tokens_per_length"] = {str(k): v for k, v in sorted(overlap.items())}

    summary["bands"] = OrderedDict()
    for b in all_bands:
        p = output_pools[b]
        n = p["n_tokens"]
        method = p["matching"]["method"]
        viable = n >= MIN_TOKENS_PER_SEQ
        recommendation = "keep"
        if not viable:
            recommendation = "drop"
        elif method == "lowercase_word_unmatched":
            recommendation = "caution"

        summary["bands"][b] = OrderedDict(
            [
                ("type", band_types[b]),
                ("n_tokens", n),
                ("matching_method", method),
                ("viable", viable),
                ("recommendation", recommendation),
                ("log_freq_stats", p.get("log_freq_stats")),
            ]
        )

    summary["recommendations"] = OrderedDict(
        [
            (
                "primary_analysis",
                [
                    b
                    for b in all_bands
                    if summary["bands"][b]["recommendation"] == "keep"
                ],
            ),
            (
                "include_with_caveats",
                [
                    b
                    for b in all_bands
                    if summary["bands"][b]["recommendation"] == "caution"
                ],
            ),
            (
                "drop",
                [
                    b
                    for b in all_bands
                    if summary["bands"][b]["recommendation"] == "drop"
                ],
            ),
        ]
    )

    summary_path = args.output_dir / "lsc_pool_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  {summary_path.name}")

    # ---- Report ----
    report_path = args.output_dir / "lsc_pool_report.txt"
    write_report(
        all_bands,
        band_types,
        lc_counts,
        match_bands,
        overlap,
        excluded,
        matched_pools,
        unmatched_pools,
        output_pools,
        report_path,
    )

    # ---- Final summary ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("LSC TOKEN POOL GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Matched pools ({matched_dir}):")
    for b in matched_pools:
        mp = matched_pools[b]
        logger.info(f"    {b:<14s}  {mp['n_tokens']:>6,d} tokens")
    logger.info(f"  Unmatched pools ({unmatched_dir}):")
    for b in all_bands:
        up = unmatched_pools[b]
        logger.info(f"    {b:<14s}  {up['n_tokens']:>6,d} tokens")
    logger.info("")
    logger.info(f"  Matched bands: {match_bands} ({matched_n}/band)")
    rec = summary["recommendations"]
    logger.info(f"  Recommendations:")
    logger.info(f"    Primary analysis:      {rec['primary_analysis']}")
    if rec["include_with_caveats"]:
        logger.info(f"    Include with caveats:  {rec['include_with_caveats']}")
    if rec["drop"]:
        logger.info(f"    Drop:                  {rec['drop']}")
    logger.info(f"  Output: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
