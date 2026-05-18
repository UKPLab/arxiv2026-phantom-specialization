#!/usr/bin/env python3
"""
Generic Confound Profiler & Validated Pool Exporter
=====================================================
Pipeline step 13.  Profiles token-level confounds across frequency bands
and exports validated pools for downstream task generators.

PURPOSE
-------
Task-INDEPENDENT confound analysis and pool validation.  This script:
  1. Applies word_en filtering via Script 08's token_label (removes
     subword fragments, non-English tokens, etc.)
  2. Profiles the 3 critical confounds across bands:
       - Character length distribution
       - Capitalization distribution
       - Word vs subword composition
     Plus auxiliary profiling: token ID, space prefix, ASCII status
  3. Exports VALIDATED pools (word_en only, all capitalizations preserved)
     ready for task-specific generators to further filter and match

DESIGN PRINCIPLE
----------------
This script does NOT do task-specific matching.  It exports the largest
possible validated pool per band.  Task generators (LSC, IOI, etc.)
handle their own matching:
  - LSC: filters to lowercase, matches on exact character length
  - IOI: filters for capitalized names, lowercase verbs/nouns, etc.

Usage:
    python 13_confound_profiler.py
"""

import json
import argparse
import logging
import sys
from pathlib import Path
from collections import OrderedDict, Counter
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
POOL_DIR = SCRIPT_DIR / "token_pools"
BANDS_JSON = SCRIPT_DIR / "band_design" / "final_bands.json"
CATEGORIES_CSV = SCRIPT_DIR / "token_categories" / "token_categories.csv"
OUTPUT_DIR = SCRIPT_DIR / "confound_analysis"
VALIDATED_DIR = SCRIPT_DIR / "token_pools_validated"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_pool(pool_dir: Path, band_name: str) -> dict:
    """Load a token pool JSON file."""
    path = pool_dir / f"pool_{band_name}.json"
    with open(path) as f:
        return json.load(f)


def load_category_lookup(csv_path: Path) -> dict:
    """
    Load token categories from Script 08 as a token_id -> info lookup.

    Returns dict: {token_id: {"token_label": str, "has_space_prefix": bool}}
    """
    logger.info(f"Loading token categories: {csv_path}")
    df = pd.read_csv(
        csv_path,
        usecols=[
            "token_id",
            "token_label",
            "has_space_prefix",
        ],
    )
    lookup = {}
    for _, row in df.iterrows():
        hsp = row["has_space_prefix"]
        if isinstance(hsp, str):
            hsp = hsp.lower() == "true"
        elif pd.isna(hsp):
            hsp = False

        lookup[int(row["token_id"])] = {
            "token_label": str(row["token_label"]),
            "has_space_prefix": bool(hsp),
        }
    logger.info(f"  {len(lookup):,} tokens loaded")
    return lookup


def is_verified_word(token_id: int, cat_lookup: dict) -> bool:
    """
    Check if a token is a verified English word token.

    Uses Script 08's token_label == "word_en" which already implies:
      - has_space_prefix = True (Ġ prefix)
      - All content characters are ASCII Latin letters
    """
    info = cat_lookup.get(token_id)
    if info is None:
        return False
    return info["token_label"] == "word_en"


def profile_band(pool: dict, cat_lookup: dict) -> OrderedDict:
    """
    Compute confound profiles for a single band.

    Profiles the 3 critical confounds:
      1. Character length distribution (+ stats)
      2. Capitalization distribution
      3. Token label composition (word_en vs other)
    Plus auxiliary: token ID stats, space prefix, ASCII status.
    """
    tokens = pool["tokens"]
    if not tokens:
        return OrderedDict([("n_total", 0)])

    # -- Extract arrays --
    lengths = [t["n_content_chars"] for t in tokens]
    caps = [t["capitalization"] for t in tokens]
    log_freqs = [t["log_frequency"] for t in tokens]
    token_ids = [t["token_id"] for t in tokens]

    # -- Token label distribution --
    labels = Counter()
    for t in tokens:
        info = cat_lookup.get(t["token_id"])
        if info:
            labels[info["token_label"]] += 1
        else:
            labels["MISSING"] += 1

    # -- word_en subset stats --
    word_tokens = [t for t in tokens if is_verified_word(t["token_id"], cat_lookup)]
    word_lengths = [t["n_content_chars"] for t in word_tokens]
    word_log_freqs = [t["log_frequency"] for t in word_tokens]
    word_caps = [t["capitalization"] for t in word_tokens]
    word_ids = [t["token_id"] for t in word_tokens]

    # -- Build profile --
    profile = OrderedDict()
    profile["n_total"] = len(tokens)
    profile["n_word"] = len(word_tokens)
    profile["pct_word"] = round(100 * len(word_tokens) / len(tokens), 1)

    # Confound 1: token label breakdown
    profile["token_labels"] = dict(sorted(labels.items()))

    # Confound 2: capitalization (all tokens)
    profile["capitalization_all"] = dict(sorted(Counter(caps).items()))
    # Capitalization (WORD-only)
    profile["capitalization_word"] = dict(sorted(Counter(word_caps).items()))

    # Confound 3: character length (all tokens)
    profile["length_distribution_all"] = dict(sorted(Counter(lengths).items()))
    profile["length_stats_all"] = _length_stats(lengths)
    # Length (WORD-only)
    profile["length_distribution_word"] = dict(sorted(Counter(word_lengths).items()))
    profile["length_stats_word"] = _length_stats(word_lengths) if word_lengths else None

    # Auxiliary: capitalization breakdown within WORD pool
    for cap_type in ["lowercase", "capitalized", "uppercase"]:
        subset = [t for t in word_tokens if t["capitalization"] == cap_type]
        key = f"n_word_{cap_type}"
        profile[key] = len(subset)

    # Auxiliary: token ID stats (documents the inherent confound)
    profile["token_id_stats_all"] = _id_stats(token_ids, log_freqs)
    profile["token_id_stats_word"] = (
        _id_stats(word_ids, word_log_freqs) if word_tokens else None
    )

    # Auxiliary: space prefix
    n_space = sum(1 for t in tokens if t.get("has_space_prefix"))
    n_space_word = sum(1 for t in word_tokens if t.get("has_space_prefix"))
    profile["pct_space_prefix_all"] = round(100 * n_space / len(tokens), 1)
    profile["pct_space_prefix_word"] = (
        round(100 * n_space_word / len(word_tokens), 1) if word_tokens else None
    )

    # Auxiliary: ASCII
    n_ascii = sum(1 for t in tokens if t.get("is_ascii"))
    n_ascii_word = sum(1 for t in word_tokens if t.get("is_ascii"))
    profile["pct_ascii_all"] = round(100 * n_ascii / len(tokens), 1)
    profile["pct_ascii_word"] = (
        round(100 * n_ascii_word / len(word_tokens), 1) if word_tokens else None
    )

    # Log-frequency stats
    profile["log_freq_stats_all"] = OrderedDict(
        [
            ("mean", round(float(np.mean(log_freqs)), 4)),
            ("std", round(float(np.std(log_freqs)), 4)),
        ]
    )
    profile["log_freq_stats_word"] = (
        OrderedDict(
            [
                ("mean", round(float(np.mean(word_log_freqs)), 4)),
                ("std", round(float(np.std(word_log_freqs)), 4)),
            ]
        )
        if word_log_freqs
        else None
    )

    return profile


def _length_stats(lengths: list) -> OrderedDict:
    """Compute summary stats for a list of character lengths."""
    return OrderedDict(
        [
            ("min", int(min(lengths))),
            ("max", int(max(lengths))),
            ("mean", round(float(np.mean(lengths)), 2)),
            ("median", round(float(np.median(lengths)), 2)),
            ("std", round(float(np.std(lengths)), 2)),
        ]
    )


def _id_stats(token_ids: list, log_freqs: list) -> OrderedDict:
    """Compute token ID stats and correlation with log-frequency."""
    ids = np.array(token_ids)
    freqs = np.array(log_freqs)
    corr = float(np.corrcoef(ids, freqs)[0, 1]) if len(ids) > 1 else None
    return OrderedDict(
        [
            ("mean", round(float(ids.mean()), 1)),
            ("std", round(float(ids.std()), 1)),
            ("min", int(ids.min())),
            ("max", int(ids.max())),
            ("corr_with_log_freq", round(corr, 4) if corr is not None else None),
        ]
    )


def build_validated_pool(
    pool: dict,
    band_name: str,
    band_def: dict,
    cat_lookup: dict,
) -> dict:
    """
    Build a validated pool: word_en only, all capitalizations preserved.

    Enriches each token with category metadata from Script 08.
    """
    validated_tokens = []
    for t in pool["tokens"]:
        if not is_verified_word(t["token_id"], cat_lookup):
            continue
        info = cat_lookup.get(t["token_id"], {})
        enriched = OrderedDict(t)
        enriched["token_label"] = info.get("token_label", "MISSING")
        validated_tokens.append(enriched)

    # Sort by log_frequency
    validated_tokens.sort(key=lambda t: t["log_frequency"])

    # Capitalization counts
    cap_counts = Counter(t["capitalization"] for t in validated_tokens)
    cap_counts = {str(k): int(v) for k, v in sorted(cap_counts.items())}

    # Length distribution
    len_counts = Counter(t["n_content_chars"] for t in validated_tokens)
    len_counts = {int(k): int(v) for k, v in sorted(len_counts.items())}

    # Build pool object
    vpool = OrderedDict()
    vpool["band_name"] = band_name
    vpool["band_type"] = band_def["type"]
    vpool["log_freq_range"] = [
        round(band_def["log_freq_range"][0], 4),
        round(band_def["log_freq_range"][1], 4),
    ]
    vpool["n_tokens"] = len(validated_tokens)
    vpool["validation"] = OrderedDict(
        [
            ("method", "word_en_filtered"),
            (
                "description",
                "word_en tokens only (Ġ-prefixed ASCII Latin from Script 08). "
                "All capitalizations preserved. "
                "No length matching applied; task generators handle matching.",
            ),
            ("filter", "token_label == word_en"),
            ("n_original", len(pool["tokens"])),
            ("n_validated", len(validated_tokens)),
            (
                "retention_pct",
                round(100 * len(validated_tokens) / len(pool["tokens"]), 1)
                if pool["tokens"]
                else 0,
            ),
        ]
    )
    vpool["capitalization"] = cap_counts
    vpool["length_distribution"] = len_counts

    if band_def["type"] == "core":
        vpool["band_index"] = band_def.get("band_index")
        vpool["center"] = band_def.get("center")
        vpool["freq_ratio"] = band_def.get("freq_ratio")

    if band_def["type"] == "exploratory":
        vpool["notes"] = band_def.get("notes", "")

    if band_def["type"] == "baseline":
        vpool["sampling"] = band_def.get("sampling", "frequency_weighted")
        vpool["sampling_description"] = band_def.get("sampling_description", "")
        # Compute frequency weights for control band
        if validated_tokens:
            freqs = []
            for t in validated_tokens:
                if "freq_per_million" in t:
                    freqs.append(t["freq_per_million"])
                else:
                    freqs.append(10 ** t["log_frequency"])
            total_freq = sum(freqs)
            vpool["frequency_weights"] = [round(f / total_freq, 10) for f in freqs]
            vpool["frequency_weights_included"] = True

    if validated_tokens:
        lf = np.array([t["log_frequency"] for t in validated_tokens])
        vpool["log_freq_stats"] = OrderedDict(
            [
                ("min", round(float(lf.min()), 4)),
                ("max", round(float(lf.max()), 4)),
                ("mean", round(float(lf.mean()), 4)),
                ("median", round(float(np.median(lf)), 4)),
                ("std", round(float(lf.std()), 4)),
            ]
        )

    vpool["tokens"] = validated_tokens
    return vpool


def make_figure(profiles, all_bands, output_path):
    """Generate confound profile figure."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Generic Confound Profile (word_en validated pools)\n"
        f"Bands: {', '.join(all_bands)}",
        fontsize=13,
        fontweight="bold",
    )

    palette = [
        "#d62728",
        "#ff7f0e",
        "#2ca02c",
        "#1f77b4",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
    ]
    colors = {b: palette[i % len(palette)] for i, b in enumerate(all_bands)}

    max_len = 15
    lengths_range = range(1, max_len + 1)

    # ---- Panel 1: WORD validation funnel ----
    ax = axes[0, 0]
    x = np.arange(len(all_bands))
    w = 0.35
    orig = [profiles[b]["n_total"] for b in all_bands]
    word = [profiles[b]["n_word"] for b in all_bands]
    ax.bar(x - w / 2, orig, w, label="All tokens", color="#BBDEFB", edgecolor="white")
    ax.bar(
        x + w / 2, word, w, label="WORD validated", color="#4CAF50", edgecolor="white"
    )
    for i, b in enumerate(all_bands):
        pct = profiles[b]["pct_word"]
        ax.text(
            i + w / 2,
            word[i] + max(orig) * 0.01,
            f"{pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(all_bands, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Token count")
    ax.set_title("word_en validation funnel")
    ax.legend(fontsize=7)

    # ---- Panel 2: Token label distribution (stacked %) ----
    ax = axes[0, 1]
    label_types = [
        "word_en",
        "subword_en",
        "word_other",
        "subword_other",
        "mixed",
        "numeric",
        "punctuation",
        "whitespace",
    ]
    x = np.arange(len(all_bands))
    bottoms = np.zeros(len(all_bands))
    label_colors = {
        "word_en": "#4CAF50",
        "subword_en": "#F44336",
        "word_other": "#9E9E9E",
        "subword_other": "#BDBDBD",
        "mixed": "#9C27B0",
        "numeric": "#FF9800",
        "punctuation": "#795548",
        "whitespace": "#607D8B",
    }
    for lab in label_types:
        vals = []
        for band in all_bands:
            total = profiles[band]["n_total"]
            count = profiles[band]["token_labels"].get(lab, 0)
            vals.append(100 * count / total if total > 0 else 0)
        if any(v > 0 for v in vals):
            ax.bar(
                x,
                vals,
                bottom=bottoms,
                label=lab,
                color=label_colors.get(lab, "#888"),
                alpha=0.85,
            )
            bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(all_bands, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("% of tokens")
    ax.set_title("Token label composition (confound)")
    ax.legend(fontsize=6)

    # ---- Panel 3: Capitalization distribution (WORD-only, stacked %) ----
    ax = axes[1, 0]
    cap_types = ["lowercase", "capitalized", "uppercase", "mixed_case"]
    cap_colors = {
        "lowercase": "#42A5F5",
        "capitalized": "#FFA726",
        "uppercase": "#EF5350",
        "mixed_case": "#AB47BC",
    }
    x = np.arange(len(all_bands))
    bottoms = np.zeros(len(all_bands))
    for cap in cap_types:
        vals = []
        for band in all_bands:
            total = profiles[band]["n_word"]
            count = profiles[band]["capitalization_word"].get(cap, 0)
            vals.append(100 * count / total if total > 0 else 0)
        if any(v > 0 for v in vals):
            ax.bar(
                x,
                vals,
                bottom=bottoms,
                label=cap,
                color=cap_colors.get(cap, "#888"),
                alpha=0.85,
            )
            bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(all_bands, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("% of WORD tokens")
    ax.set_title("Capitalization distribution (confound)")
    ax.legend(fontsize=7)

    # ---- Panel 4: Length distribution (WORD-only, grouped bars) ----
    ax = axes[1, 1]
    n_bands = len(all_bands)
    width = 0.8 / n_bands
    offsets = np.arange(n_bands) - (n_bands - 1) / 2
    for i, band in enumerate(all_bands):
        dist = profiles[band]["length_distribution_word"]
        total = profiles[band]["n_word"]
        if total == 0:
            continue
        pcts = [100 * dist.get(L, 0) / total for L in lengths_range]
        positions = np.array(list(lengths_range)) + offsets[i] * width
        ax.bar(positions, pcts, width=width, label=band, color=colors[band], alpha=0.8)
    ax.set_xlabel("Character length")
    ax.set_ylabel("% of WORD tokens")
    ax.set_title("Character length distribution (confound)")
    ax.legend(fontsize=6, ncol=2)
    ax.set_xlim(0.5, max_len + 0.5)
    ax.set_xticks(range(1, max_len + 1))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


def write_text_report(profiles, all_bands, band_types, validated_pools, output_path):
    """Write human-readable confound report."""
    lines = []
    w = 90
    lines.append("=" * w)
    lines.append("GENERIC CONFOUND PROFILE & VALIDATED POOL REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Bands profiled: {all_bands}")
    lines.append(
        f"Validation: word_en filter from Script 08 (all capitalizations preserved)"
    )
    lines.append("=" * w)
    lines.append("")

    # --- Section 1: Word/subword composition ---
    lines.append("1. TOKEN LABEL COMPOSITION (Script 08 categories)")
    lines.append("-" * w)
    label_types = ["word_en", "subword_en", "mixed", "numeric"]
    hdr = f"{'Band':<14s} {'Type':<8s}  {'Total':>6s}  {'word_en':>8s}  {'%word':>6s}  "
    hdr += "  ".join(f"{s:>10s}" for s in label_types[1:])
    lines.append(hdr)
    lines.append(
        f"{'-' * 14} {'-' * 8}  {'-' * 6}  {'-' * 8}  {'-' * 6}  "
        + "  ".join(f"{'-' * 10}" for _ in label_types[1:])
    )
    for band in all_bands:
        p = profiles[band]
        btype = band_types[band]
        labs = p["token_labels"]
        rest = "  ".join(f"{labs.get(s, 0):>10d}" for s in label_types[1:])
        lines.append(
            f"{band:<14s} {btype:<8s}  {p['n_total']:>6d}  {labs.get('word_en', 0):>8d}  "
            f"{p['pct_word']:>5.1f}%  {rest}"
        )
    lines.append("")

    # --- Section 2: Capitalization distribution (WORD-only) ---
    lines.append("2. CAPITALIZATION DISTRIBUTION (WORD-only)")
    lines.append("-" * w)
    cap_types = ["lowercase", "capitalized", "uppercase", "mixed_case"]
    hdr = f"{'Band':<14s}  {'N_WORD':>6s}  "
    hdr += "  ".join(f"{c[:7]:>8s}" for c in cap_types)
    lines.append(hdr)
    lines.append(
        f"{'-' * 14}  {'-' * 6}  " + "  ".join(f"{'-' * 8}" for _ in cap_types)
    )
    for band in all_bands:
        p = profiles[band]
        cap = p["capitalization_word"]
        vals = "  ".join(f"{cap.get(c, 0):>8d}" for c in cap_types)
        lines.append(f"{band:<14s}  {p['n_word']:>6d}  {vals}")
    lines.append("")

    # Capitalization percentages
    lines.append("   Capitalization % within WORD pool:")
    hdr2 = f"   {'Band':<14s}  "
    hdr2 += "  ".join(f"{'%' + c[:6]:>8s}" for c in cap_types)
    lines.append(hdr2)
    lines.append(f"   {'-' * 14}  " + "  ".join(f"{'-' * 8}" for _ in cap_types))
    for band in all_bands:
        p = profiles[band]
        cap = p["capitalization_word"]
        n = p["n_word"]
        vals = "  ".join(
            f"{100 * cap.get(c, 0) / n:>7.1f}%" if n > 0 else f"{'N/A':>8s}"
            for c in cap_types
        )
        lines.append(f"   {band:<14s}  {vals}")
    lines.append("")

    # --- Section 3: Character length distribution (WORD-only) ---
    lines.append("3. CHARACTER LENGTH DISTRIBUTION (WORD-only)")
    lines.append("-" * w)
    for band in all_bands:
        st = profiles[band]["length_stats_word"]
        if st:
            lines.append(
                f"  {band:<14s}  mean={st['mean']:.2f}  std={st['std']:.2f}  "
                f"range=[{st['min']}, {st['max']}]"
            )
    lines.append("")
    lines.append("   Per-length counts (WORD-only):")
    all_lens = set()
    for band in all_bands:
        all_lens.update(profiles[band]["length_distribution_word"].keys())
    all_lens = sorted(all_lens)
    display_lens = [L for L in all_lens if L <= 15]

    col_w = max(8, max(len(b) for b in all_bands) + 1)
    hdr = f"   {'Len':>4s}  " + "  ".join(f"{b:>{col_w}s}" for b in all_bands)
    lines.append(hdr)
    lines.append(f"   {'-' * 4}  " + "  ".join(f"{'-' * col_w}" for _ in all_bands))
    for L in display_lens:
        vals = []
        for band in all_bands:
            c = profiles[band]["length_distribution_word"].get(L, 0)
            vals.append(f"{c:>{col_w}d}")
        lines.append(f"   {L:>4d}  " + "  ".join(vals))
    lines.append("")

    # --- Section 4: Token ID confound (acknowledged limitation) ---
    lines.append("4. TOKEN ID CONFOUND (acknowledged limitation)")
    lines.append("-" * w)
    lines.append("   Token IDs correlate with frequency (GPT-NeoX tokenizer artifact).")
    lines.append("   Cannot be controlled; reported for transparency.")
    lines.append("")
    hdr = (
        f"   {'Band':<14s}  {'ID_mean':>8s}  {'ID_std':>8s}  "
        f"{'ID_min':>8s}  {'ID_max':>8s}  {'r(ID,freq)':>10s}"
    )
    lines.append(hdr)
    lines.append(
        f"   {'-' * 14}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 10}"
    )
    for band in all_bands:
        ids = profiles[band]["token_id_stats_word"]
        if ids:
            lines.append(
                f"   {band:<14s}  {ids['mean']:>8.0f}  {ids['std']:>8.0f}  "
                f"{ids['min']:>8d}  {ids['max']:>8d}  "
                f"{ids['corr_with_log_freq']:>+10.4f}"
            )
    lines.append("")

    # --- Section 5: Validated pool summary ---
    lines.append("5. VALIDATED POOL SUMMARY")
    lines.append("-" * w)
    hdr = (
        f"{'Band':<14s} {'Type':<8s}  {'Original':>8s}  {'WORD':>8s}  "
        f"{'Kept%':>6s}  {'LogF_u':>8s}  {'LogF_s':>8s}"
    )
    lines.append(hdr)
    lines.append(
        f"{'-' * 14} {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 6}  {'-' * 8}  {'-' * 8}"
    )
    for band in all_bands:
        p = profiles[band]
        vp = validated_pools[band]
        btype = band_types[band]
        lfs = vp.get("log_freq_stats", {})
        lines.append(
            f"{band:<14s} {btype:<8s}  {p['n_total']:>8d}  {vp['n_tokens']:>8d}  "
            f"{vp['validation']['retention_pct']:>5.1f}%  "
            f"{lfs.get('mean', 0):>8.4f}  {lfs.get('std', 0):>8.4f}"
        )
    lines.append("")

    # --- Section 6: Assessment ---
    lines.append("6. ASSESSMENT")
    lines.append("-" * w)
    for btype_label, btype_key in [
        ("Core", "core"),
        ("Exploratory", "exploratory"),
        ("Baseline", "baseline"),
    ]:
        type_bands = [b for b in all_bands if band_types[b] == btype_key]
        if not type_bands:
            continue
        min_word = min(profiles[b]["n_word"] for b in type_bands)
        min_band = min(type_bands, key=lambda b: profiles[b]["n_word"])
        lines.append(f"{btype_label} bands ({', '.join(type_bands)}):")
        lines.append(f"  Smallest WORD pool: {min_word} tokens (band: {min_band})")
        if btype_key == "core":
            if min_word >= 500:
                lines.append(
                    "  PASS: Sufficient for downstream task-specific matching."
                )
            elif min_word >= 200:
                lines.append("  WARNING: Smallest band may limit matching options.")
            else:
                lines.append("  CRITICAL: Too few WORD tokens for reliable matching.")
        lines.append("")

    lines.append("Confounds to control in task generators:")
    lines.append("  1. Character length: match on exact length across bands")
    lines.append(
        "  2. Capitalization:   filter by task requirements (e.g., lowercase for LSC)"
    )
    lines.append("  3. Word/subword:     ALREADY CONTROLLED (WORD-only pools)")
    lines.append("")
    lines.append("Acknowledged limitation:")
    lines.append(
        "  Token ID correlates with frequency (r ~ -0.8). Inherent to tokenizer."
    )
    lines.append("  Report in paper; cannot be experimentally controlled.")

    lines.append("")
    lines.append("=" * w)
    lines.append("END OF REPORT")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generic confound profiler & validated pool exporter",
    )
    parser.add_argument(
        "--pool-dir",
        type=Path,
        default=POOL_DIR,
        help="Directory containing pool_*.json from step 10",
    )
    parser.add_argument(
        "--bands-json",
        type=Path,
        default=BANDS_JSON,
        help="Path to final_bands.json from step 09",
    )
    parser.add_argument(
        "--categories-csv",
        type=Path,
        default=CATEGORIES_CSV,
        help="Path to token_categories.csv from Script 08",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for confound analysis",
    )
    parser.add_argument(
        "--validated-dir",
        type=Path,
        default=VALIDATED_DIR,
        help="Output directory for validated pools",
    )
    args = parser.parse_args()

    # ---- Validate inputs ----
    if not args.pool_dir.exists():
        logger.error(f"Pool directory not found: {args.pool_dir}")
        return 1
    if not args.bands_json.exists():
        logger.error(f"Bands JSON not found: {args.bands_json}")
        return 1
    if not args.categories_csv.exists():
        logger.error(f"Categories CSV not found: {args.categories_csv}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.validated_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load band definitions ----
    logger.info(f"Loading bands: {args.bands_json}")
    with open(args.bands_json) as f:
        bands_data = json.load(f)

    # Get ordered band names from the JSON
    all_bands = bands_data["summary"]["condition_names_ordered"]
    band_types = {name: bands_data["bands"][name]["type"] for name in all_bands}
    logger.info(f"  {len(all_bands)} bands: {all_bands}")

    cat_lookup = load_category_lookup(args.categories_csv)

    # ---- Load all pools ----
    logger.info(f"Loading pools from: {args.pool_dir}")
    pools = {}
    for band_name in all_bands:
        pools[band_name] = load_pool(args.pool_dir, band_name)
        logger.info(
            f"  {band_name} [{band_types[band_name]}]: "
            f"{pools[band_name]['n_tokens']:,} tokens"
        )

    # ---- Profile confounds (all bands) ----
    logger.info("")
    logger.info("Profiling confounds...")
    profiles = {}
    for band_name in all_bands:
        profiles[band_name] = profile_band(pools[band_name], cat_lookup)
        p = profiles[band_name]
        logger.info(
            f"  {band_name}: {p['n_total']:,} total -> "
            f"{p['n_word']:,} WORD ({p['pct_word']}%)"
        )

    # ---- Build validated pools (all bands) ----
    logger.info("")
    logger.info("Building validated pools (WORD-only, all caps preserved)...")
    validated_pools = {}
    for band_name in all_bands:
        band_def = bands_data["bands"][band_name]
        validated_pools[band_name] = build_validated_pool(
            pools[band_name], band_name, band_def, cat_lookup
        )
        vp = validated_pools[band_name]
        logger.info(
            f"  {band_name}: {vp['n_tokens']:,} validated tokens "
            f"({vp['validation']['retention_pct']:.1f}% retained)"
        )

    # ---- Export validated pools ----
    logger.info("")
    logger.info("Exporting validated pools...")
    for band_name in all_bands:
        path = args.validated_dir / f"pool_validated_{band_name}.json"
        with open(path, "w") as f:
            json.dump(validated_pools[band_name], f, indent=2)
        logger.info(f"  Saved: {path.name}")

    # ---- Summary JSON ----
    summary = OrderedDict()
    summary["created_at"] = datetime.now().isoformat()
    summary["validation_method"] = "word_en_filtered"
    summary["description"] = (
        "word_en tokens (Script 08). All capitalizations preserved. "
        "Task generators handle capitalization filtering and length matching."
    )
    summary["source_pools"] = str(args.pool_dir)
    summary["categories_csv"] = str(args.categories_csv)
    summary["all_bands"] = all_bands
    summary["bands"] = OrderedDict()
    for band_name in all_bands:
        vp = validated_pools[band_name]
        p = profiles[band_name]
        summary["bands"][band_name] = OrderedDict(
            [
                ("type", band_types[band_name]),
                ("n_original", p["n_total"]),
                ("n_validated", vp["n_tokens"]),
                ("retention_pct", vp["validation"]["retention_pct"]),
                ("capitalization", vp["capitalization"]),
                ("log_freq_stats", vp.get("log_freq_stats")),
            ]
        )

    summary_path = args.validated_dir / "validated_pool_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Saved: {summary_path.name}")

    # ---- Reports ----
    report_json = OrderedDict()
    report_json["created_at"] = datetime.now().isoformat()
    report_json["all_bands"] = all_bands
    report_json["band_types"] = band_types
    report_json["profiles"] = profiles
    report_json["confounds_profiled"] = [
        "word_vs_subword (token_label)",
        "capitalization",
        "character_length",
        "token_id (acknowledged limitation)",
    ]

    report_path = args.output_dir / "confound_report.json"
    with open(report_path, "w") as f:
        json.dump(report_json, f, indent=2)
    logger.info(f"  Saved: {report_path}")

    txt_path = args.output_dir / "confound_report.txt"
    write_text_report(profiles, all_bands, band_types, validated_pools, txt_path)

    fig_path = args.output_dir / "fig_confound_profile.png"
    make_figure(profiles, all_bands, fig_path)

    # ---- Final summary ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("CONFOUND PROFILING COMPLETE")
    logger.info("=" * 60)
    logger.info(
        f"  {'Band':<14s} {'Type':<8s}  {'Original':>8s}  {'WORD':>8s}  "
        f"{'lower':>7s}  {'capital':>7s}  {'upper':>7s}"
    )
    logger.info(
        f"  {'-' * 14} {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 7}  {'-' * 7}  {'-' * 7}"
    )
    for band in all_bands:
        p = profiles[band]
        cap = p["capitalization_word"]
        logger.info(
            f"  {band:<14s} {band_types[band]:<8s}  "
            f"{p['n_total']:>8,d}  {p['n_word']:>8,d}  "
            f"{cap.get('lowercase', 0):>7,d}  "
            f"{cap.get('capitalized', 0):>7,d}  "
            f"{cap.get('uppercase', 0):>7,d}"
        )
    logger.info("")
    logger.info(f"  Validated pools: {args.validated_dir}")
    logger.info(f"  Confound report: {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
