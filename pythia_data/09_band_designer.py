#!/usr/bin/env python3
"""
Band Designer
==============
Pipeline step 09.  Defines frequency bands for experiments.

PURPOSE
-------
Takes the categorized vocabulary from 08 and answers:
  - For k = 3..8 equal-width bands, how many word_en tokens per band?
  - What's the capitalization breakdown per band?
  - Where does the token budget become too thin?

The "sweet spot" is the largest k where every band still has enough
tokens for all planned tasks (LSC needs any word_en, IOI needs
capitalized word_en for names).

DESIGN DECISIONS/JUSTIFICATION
--------------------------------------------

Why equal-width bands in log-space?
  The independent variable is frequency on a logarithmic scale; a
  token appearing 10x/million vs 100x/million is the same perceptual
  distance as 100 vs 1000.  Equal-width in log-space means equal
  multiplicative ratios between conditions.  This is the standard
  approach in any experiment where the independent variable spans
  orders of magnitude.  The alternative which is equal-width in linear
  space, would put 99% of tokens in the first band and leave the rest empty.

Why k=5 specifically?
  Not arbitrary.  The profiler (07) established that the distribution
  is unimodal with no natural clusters, so the number of bands cannot
  be derived from the data's intrinsic structure.  Instead, k is
  determined by crossing two constraints:

    1. Within-band frequency ratio must be small enough that each
       condition genuinely controls for frequency (below ~4x).
    2. The minimum token pool must be large enough for diverse
       dataset generation (above ~500 word_en tokens).

  k=5 is the largest k satisfying both.  k=4 has 4.7x ratio: nearly
  half an order of magnitude of uncontrolled variance per band.  k=6
  drops to 512 minimum tokens, reducing sampling diversity for no
  additional statistical power.  5 gradient points can show a
  monotonic trend; 3 can only show "low != high."

Why p1-p99 as the core range?
  The tails are qualitatively different populations.  The profiler
  showed an 82x density difference between bulk and tails.  The
  bottom 1% scatters 271 tokens across 4.6 log-freq units. that's
  not a controllable experimental condition, it's noise.  Including
  tails in the core range would either widen every band (worse control)
  or create bands with too few tokens.

How do adjacent bands differ meaningfully?
  Adjacent band centers are separated by 0.54 log-freq units: a 3.5x
  frequency ratio.  The token at the center of very_low appears
  roughly once per 0.7 million tokens; at the center of very_high,
  roughly 100 times per million.  That's a 140x difference across the
  full gradient.  If circuit structure doesn't vary across a 140x
  frequency manipulation, frequency genuinely doesn't matter.

Why a control band?
  The control is frequency-weighted random sampling from the full
  word_en pool and it reflects the natural frequency mix the model sees
  during pretraining.  It represents what prior MI work has implicitly
  measured when selecting words "normally" without controlling for
  frequency.  If band-specific circuits differ from control, prior
  results were frequency-confounded.  The control is the null
  hypothesis, not a redundant condition.

FINAL SCHEME (8 conditions)
----------------------------
  bottom_tail   exploratory   Ultra-rare, sparse > report with caveats
  very_low      core          Gradient point 1
  low           core          Gradient point 2
  medium        core          Gradient point 3
  high          core          Gradient point 4
  very_high     core          Gradient point 5
  top_tail      exploratory   Ultra-common, sparse > report with caveats
  control       baseline      Frequency-weighted random (null hypothesis)

PAPER-CITABLE SUMMARY
-----------------------
  The selection procedure follows a three-stage empirical pipeline:
  distribution profiling (07) established unimodality and absence of
  natural clusters (ruling out data-driven discretization), token
  categorization (08) identified the usable experimental pool (27,037
  English word-onset tokens), and band evaluation across k=3-8 (this
  script) identified k=5 as maximizing gradient resolution while
  maintaining within-band frequency control (3.5x ratio) and
  sufficient token diversity (>=827 per band).  This is reproducible and
  the profiler, categorizer, and designer scripts output deterministic
  results from the same input data.

INPUT
-----
token_categories.csv  (from 08_token_categorizer.py)

OUTPUTS
-------
band_schemes.json          All candidate schemes (k=3..8) with counts
band_scheme_report.txt     Human-readable comparison table
fig_band_schemes.png       Visual comparison of schemes
final_bands.json           CANONICAL band definitions for all downstream tasks

Usage:
    python 09_band_designer.py
    python 09_band_designer.py --csv-file path/to/token_categories.csv
    python 09_band_designer.py --csv-file ... --chosen-k 5
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


SCRIPT_DIR = Path(__file__).resolve().parent
CATEGORIES_CSV = SCRIPT_DIR / "token_categories" / "token_categories.csv"
OUTPUT_DIR = SCRIPT_DIR / "band_design"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# BAND COMPUTATION
# ============================================================================

# Canonical band names for the final scheme.
# Index 0..k-1 maps to core band names (low-freq -> high-freq).
CORE_BAND_NAMES = {
    3: ["very_low", "medium", "very_high"],
    4: ["very_low", "low", "high", "very_high"],
    5: ["very_low", "low", "medium", "high", "very_high"],
    6: ["very_low", "low", "low_medium", "high_medium", "high", "very_high"],
    7: ["very_low", "low", "low_medium", "medium", "high_medium", "high", "very_high"],
    8: [
        "very_low",
        "low",
        "low_medium",
        "medium",
        "high_medium",
        "high",
        "high_very_high",
        "very_high",
    ],
}


def build_final_scheme(
    scheme: dict,
    log_freqs: np.ndarray,
    capitalizations: np.ndarray,
    k: int,
) -> OrderedDict:
    """
    Build the canonical final band definition from a computed scheme.

    Adds:
      - Human-readable names to each band
      - Control band definition (frequency-weighted random sampling)
      - Band type labels (exploratory / core / baseline)
    """
    names = CORE_BAND_NAMES.get(k)
    if names is None:
        names = [f"band_{i + 1}" for i in range(k)]

    final = OrderedDict()
    final["description"] = (
        "Canonical frequency band definitions for all downstream tasks. "
        "Import this file : do not hardcode boundaries."
    )
    final["chosen_k"] = k
    final["band_width"] = scheme["band_width"]
    final["freq_ratio_per_band"] = scheme["freq_ratio_per_band"]
    final["core_range"] = scheme["range"]
    final["created_at"] = datetime.now().isoformat()

    bands = OrderedDict()

    for b in scheme["bands"]:
        bid = b["band_id"]

        if bid == "bottom_tail":
            entry = OrderedDict(
                [
                    ("type", "exploratory"),
                    ("log_freq_range", b["range"]),
                    ("n_tokens", b["n_tokens"]),
                    ("capitalization", b["capitalization"]),
                    (
                        "notes",
                        "Ultra-rare tokens. Sparse sampling: report with caveats.",
                    ),
                ]
            )
            bands["bottom_tail"] = entry

        elif bid == "top_tail":
            entry = OrderedDict(
                [
                    ("type", "exploratory"),
                    ("log_freq_range", b["range"]),
                    ("n_tokens", b["n_tokens"]),
                    ("capitalization", b["capitalization"]),
                    (
                        "notes",
                        "Ultra-common tokens. Sparse sampling: report with caveats.",
                    ),
                ]
            )
            bands["top_tail"] = entry

        else:
            idx = b["band_index"]
            name = names[idx]
            entry = OrderedDict(
                [
                    ("type", "core"),
                    ("band_index", idx),
                    ("log_freq_range", b["range"]),
                    ("center", b["center"]),
                    ("freq_ratio", b["freq_ratio"]),
                    ("n_tokens", b["n_tokens"]),
                    ("capitalization", b["capitalization"]),
                ]
            )
            bands[name] = entry

    # Control band: frequency-weighted random sample from full word_en
    cap_counts = _cap_counts(capitalizations)
    bands["control"] = OrderedDict(
        [
            ("type", "baseline"),
            (
                "log_freq_range",
                [round(float(log_freqs.min()), 4), round(float(log_freqs.max()), 4)],
            ),
            ("sampling", "frequency_weighted"),
            (
                "sampling_description",
                "Random sample from full word_en pool, weighted by token "
                "pretraining frequency. Reflects the natural frequency mix "
                "the model sees during training. This is the null hypothesis: "
                "what circuit emerges when frequency is not controlled.",
            ),
            ("n_tokens_available", int(len(log_freqs))),
            ("capitalization", cap_counts),
        ]
    )

    final["bands"] = bands

    # Quick summary for logging
    final["summary"] = OrderedDict(
        [
            ("n_conditions", len(bands)),
            ("core_bands", names),
            ("exploratory_bands", ["bottom_tail", "top_tail"]),
            ("baseline_bands", ["control"]),
            (
                "condition_names_ordered",
                ["bottom_tail"] + names + ["top_tail", "control"],
            ),
        ]
    )

    return final


def compute_band_scheme(
    log_freqs: np.ndarray,
    capitalizations: np.ndarray,
    k: int,
    range_low: float,
    range_high: float,
) -> dict:
    """
    Compute a k-band equal-width scheme over [range_low, range_high].

    Tokens below range_low -> bottom_tail
    Tokens above range_high -> top_tail
    Core divided into k equal-width bands.

    Returns dict with band definitions, token counts, and
    capitalization breakdowns.
    """
    width = (range_high - range_low) / k
    boundaries = [range_low + i * width for i in range(k + 1)]

    bands = []

    # Bottom tail
    tail_low_mask = log_freqs < range_low
    tail_low_caps = capitalizations[tail_low_mask]
    bands.append(
        {
            "band_id": "bottom_tail",
            "band_index": -1,
            "range": [float(log_freqs.min()), float(range_low)],
            "range_width": float(range_low - log_freqs.min()),
            "n_tokens": int(tail_low_mask.sum()),
            "capitalization": _cap_counts(tail_low_caps),
        }
    )

    # Core bands
    for i in range(k):
        lo = boundaries[i]
        hi = boundaries[i + 1]
        # Include right edge for last band
        if i < k - 1:
            mask = (log_freqs >= lo) & (log_freqs < hi)
        else:
            mask = (log_freqs >= lo) & (log_freqs <= hi)

        band_caps = capitalizations[mask]
        band_center = (lo + hi) / 2
        freq_ratio = 10**width  # multiplicative ratio within band

        bands.append(
            {
                "band_id": f"band_{i + 1}",
                "band_index": i,
                "range": [round(lo, 4), round(hi, 4)],
                "center": round(band_center, 4),
                "range_width": round(width, 4),
                "freq_ratio": round(freq_ratio, 2),
                "n_tokens": int(mask.sum()),
                "capitalization": _cap_counts(band_caps),
            }
        )

    # Top tail
    tail_high_mask = log_freqs > range_high
    tail_high_caps = capitalizations[tail_high_mask]
    bands.append(
        {
            "band_id": "top_tail",
            "band_index": k,
            "range": [float(range_high), float(log_freqs.max())],
            "range_width": float(log_freqs.max() - range_high),
            "n_tokens": int(tail_high_mask.sum()),
            "capitalization": _cap_counts(tail_high_caps),
        }
    )

    # Summary stats
    core_bands = [b for b in bands if b["band_index"] >= 0 and b["band_index"] < k]
    core_counts = [b["n_tokens"] for b in core_bands]
    cap_counts = [b["capitalization"].get("capitalized", 0) for b in core_bands]

    scheme = {
        "k": k,
        "range": [round(range_low, 4), round(range_high, 4)],
        "total_range_width": round(range_high - range_low, 4),
        "band_width": round(width, 4),
        "freq_ratio_per_band": round(10**width, 2),
        "boundaries": [round(b, 4) for b in boundaries],
        "bands": bands,
        "core_summary": {
            "total_tokens": int(sum(core_counts)),
            "min_tokens": int(min(core_counts)) if core_counts else 0,
            "max_tokens": int(max(core_counts)) if core_counts else 0,
            "mean_tokens": int(np.mean(core_counts)) if core_counts else 0,
            "std_tokens": int(np.std(core_counts)) if core_counts else 0,
            "min_capitalized": int(min(cap_counts)) if cap_counts else 0,
            "counts_per_band": core_counts,
            "capitalized_per_band": cap_counts,
        },
    }

    return scheme


def _cap_counts(caps: np.ndarray) -> dict:
    """Count capitalization types."""
    if len(caps) == 0:
        return {}
    unique, counts = np.unique(caps, return_counts=True)
    return {str(u): int(c) for u, c in zip(unique, counts)}


def plot_band_schemes(
    log_freqs: np.ndarray,
    schemes: dict,
    path: Path,
) -> None:
    """
    Compare all k schemes visually.

    Top panel: histogram with band boundaries overlaid for each k.
    Bottom panel: token counts per band for each k (bar chart).
    """
    ks = sorted(schemes.keys())
    n_schemes = len(ks)

    fig = plt.figure(figsize=(18, 5 + 4 * n_schemes))
    gs = gridspec.GridSpec(
        n_schemes + 1, 2, hspace=0.4, wspace=0.3, height_ratios=[2] + [1] * n_schemes
    )

    # Color palette for bands
    band_colors = [
        "#4285f4",
        "#ea4335",
        "#fbbc04",
        "#34a853",
        "#ff6d01",
        "#46bdc6",
        "#7b1fa2",
        "#c2185b",
    ]

    # --- Top left: histogram with all boundary sets ---
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_hist.hist(
        log_freqs, bins=120, color="#ddd", edgecolor="white", linewidth=0.3, alpha=0.8
    )
    line_styles = ["-", "--", "-.", ":", "-", "--"]
    for i, k in enumerate(ks):
        s = schemes[k]
        for b in s["boundaries"]:
            ax_hist.axvline(
                b,
                color=band_colors[i % len(band_colors)],
                linewidth=1.0,
                linestyle=line_styles[i % len(line_styles)],
                alpha=0.7,
            )
        # Label just the first boundary for legend
        ax_hist.axvline(
            s["boundaries"][0],
            color=band_colors[i % len(band_colors)],
            linewidth=1.5,
            linestyle=line_styles[i % len(line_styles)],
            alpha=0.7,
            label=f"k={k}",
        )
    ax_hist.set_xlabel("log₁₀(freq per million)", fontsize=10)
    ax_hist.set_ylabel("Count", fontsize=10)
    ax_hist.set_title(
        "word_en Distribution with Band Boundaries", fontsize=12, fontweight="bold"
    )
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.2)

    # --- Top right: min tokens per band vs k ---
    ax_min = fig.add_subplot(gs[0, 1])
    ks_list = list(ks)
    min_tokens = [schemes[k]["core_summary"]["min_tokens"] for k in ks_list]
    min_caps = [schemes[k]["core_summary"]["min_capitalized"] for k in ks_list]

    x = np.arange(len(ks_list))
    w = 0.35
    ax_min.bar(
        x - w / 2,
        min_tokens,
        w,
        label="Min word_en",
        color="#4285f4",
        edgecolor="black",
        linewidth=0.5,
    )
    ax_min.bar(
        x + w / 2,
        min_caps,
        w,
        label="Min capitalized",
        color="#fbbc04",
        edgecolor="black",
        linewidth=0.5,
    )
    ax_min.set_xticks(x)
    ax_min.set_xticklabels([f"k={k}" for k in ks_list], fontsize=10)
    ax_min.set_ylabel("Minimum tokens in any band", fontsize=10)
    ax_min.set_title("Feasibility: Minimum Band Size", fontsize=12, fontweight="bold")
    ax_min.legend(fontsize=9)
    ax_min.grid(axis="y", alpha=0.3)

    # Threshold lines
    for threshold, label in [(50, "50"), (100, "100"), (500, "500")]:
        ax_min.axhline(threshold, color="#999", linewidth=0.8, linestyle=":", alpha=0.5)
        ax_min.text(
            len(ks_list) - 0.5,
            threshold + 20,
            f"n={label}",
            fontsize=7,
            color="#666",
            ha="right",
        )

    # --- Per-scheme breakdown rows ---
    for row_idx, k in enumerate(ks):
        s = schemes[k]
        core = [b for b in s["bands"] if b["band_index"] >= 0 and b["band_index"] < k]

        # Left: token counts per band
        ax = fig.add_subplot(gs[row_idx + 1, 0])
        band_names = [b["band_id"] for b in core]
        band_totals = [b["n_tokens"] for b in core]
        band_cap = [b["capitalization"].get("capitalized", 0) for b in core]
        band_lower = [b["capitalization"].get("lowercase", 0) for b in core]

        x = np.arange(len(core))
        ax.bar(
            x,
            band_lower,
            label="lowercase",
            color="#4285f4",
            edgecolor="white",
            linewidth=0.3,
        )
        ax.bar(
            x,
            band_cap,
            bottom=band_lower,
            label="capitalized",
            color="#fbbc04",
            edgecolor="white",
            linewidth=0.3,
        )
        other = [t - l - c for t, l, c in zip(band_totals, band_lower, band_cap)]
        ax.bar(
            x,
            other,
            bottom=[l + c for l, c in zip(band_lower, band_cap)],
            label="other",
            color="#cccccc",
            edgecolor="white",
            linewidth=0.3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(band_names, fontsize=8)
        ax.set_ylabel("Tokens", fontsize=9)
        ax.set_title(
            f"k={k}: Token counts per band (width={s['band_width']:.3f}, "
            f"ratio={s['freq_ratio_per_band']:.1f}x)",
            fontsize=10,
            fontweight="bold",
        )
        if row_idx == 0:
            ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.2)

        # Add count labels
        for i, (total, cap) in enumerate(zip(band_totals, band_cap)):
            ax.text(
                i,
                total + max(band_totals) * 0.02,
                f"{total:,}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

        # Right: band ranges visualization
        ax2 = fig.add_subplot(gs[row_idx + 1, 1])
        for i, b in enumerate(core):
            lo, hi = b["range"]
            ax2.barh(
                i,
                hi - lo,
                left=lo,
                height=0.7,
                color=band_colors[i % len(band_colors)],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.7,
            )
            ax2.text(
                (lo + hi) / 2,
                i,
                f"n={b['n_tokens']:,}",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
            )
        ax2.set_yticks(range(len(core)))
        ax2.set_yticklabels([b["band_id"] for b in core], fontsize=8)
        ax2.set_xlabel("log₁₀(freq per million)", fontsize=9)
        ax2.set_title(f"k={k}: Band Ranges", fontsize=10, fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)
        ax2.invert_yaxis()

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def write_report(
    schemes: dict,
    range_low: float,
    range_high: float,
    n_total: int,
    n_tails: dict,
    path: Path,
) -> None:
    """Human-readable comparison of all schemes."""
    lines = []

    def section(title):
        lines.append("")
        lines.append("=" * 80)
        lines.append(title)
        lines.append("=" * 80)

    section("BAND DESIGN REPORT")
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Total word_en tokens: {n_total:,}")
    lines.append(f"Core range (p1-p99): [{range_low:.4f}, {range_high:.4f}]")
    lines.append(f"Core range width: {range_high - range_low:.4f} log-freq units")
    lines.append(f"Core range freq ratio: {10 ** (range_high - range_low):.1f}x")

    section("TAIL BANDS")
    lines.append(
        f"  bottom_tail (< p1): {n_tails['bottom']:>5,d} tokens  "
        f"[range: {n_tails['bottom_range'][0]:.3f} to {range_low:.3f}]"
    )
    lines.append(
        f"  top_tail    (> p99): {n_tails['top']:>5,d} tokens  "
        f"[range: {range_high:.3f} to {n_tails['top_range'][1]:.3f}]"
    )

    section("SCHEME COMPARISON")
    # Header
    header = f"{'k':>3s}  {'width':>6s}  {'ratio':>6s}  {'min_tok':>8s}  {'max_tok':>8s}  {'min_cap':>8s}  {'balance':>8s}  band counts"
    lines.append(header)
    lines.append("-" * len(header) + "-" * 40)

    for k in sorted(schemes.keys()):
        s = schemes[k]
        cs = s["core_summary"]
        balance = cs["min_tokens"] / cs["max_tokens"] if cs["max_tokens"] > 0 else 0
        counts_str = "  ".join(f"{c:>5,d}" for c in cs["counts_per_band"])
        lines.append(
            f"{k:>3d}  {s['band_width']:>6.3f}  {s['freq_ratio_per_band']:>5.1f}x  "
            f"{cs['min_tokens']:>8,d}  {cs['max_tokens']:>8,d}  "
            f"{cs['min_capitalized']:>8,d}  {balance:>8.3f}  {counts_str}"
        )

    section("DETAILED BREAKDOWN PER SCHEME")
    for k in sorted(schemes.keys()):
        s = schemes[k]
        lines.append("")
        lines.append(
            f"--- k={k} (band_width={s['band_width']:.4f}, "
            f"freq_ratio={s['freq_ratio_per_band']:.1f}x) ---"
        )
        lines.append("")
        lines.append(
            f"  {'band':<12s}  {'range':>20s}  {'total':>7s}  "
            f"{'lower':>7s}  {'cap':>7s}  {'upper':>7s}  {'other':>7s}"
        )
        lines.append("  " + "-" * 80)

        for b in s["bands"]:
            if b["band_id"] in ("bottom_tail", "top_tail"):
                continue
            cap = b["capitalization"]
            lo_s = cap.get("lowercase", 0)
            ca_s = cap.get("capitalized", 0)
            up_s = cap.get("uppercase", 0)
            ot_s = b["n_tokens"] - lo_s - ca_s - up_s
            rng = f"[{b['range'][0]:>7.3f}, {b['range'][1]:>6.3f}]"
            lines.append(
                f"  {b['band_id']:<12s}  {rng:>20s}  {b['n_tokens']:>7,d}  "
                f"{lo_s:>7,d}  {ca_s:>7,d}  {up_s:>7,d}  {ot_s:>7,d}"
            )

    section("INTERPRETATION GUIDE")
    lines.append("  'width'    = log-freq span of each band")
    lines.append("  'ratio'    = multiplicative frequency ratio within band")
    lines.append("               (e.g. 4.3x means tokens in band span a")
    lines.append("                4.3-fold frequency range)")
    lines.append("  'min_tok'  = fewest word_en tokens in any core band")
    lines.append("  'min_cap'  = fewest capitalized tokens in any core band")
    lines.append("               (proper noun candidates for IOI)")
    lines.append("  'balance'  = min/max ratio (1.0 = perfectly balanced)")
    lines.append("")
    lines.append("  Pick the largest k where min_tok and min_cap are")
    lines.append("  sufficient for your dataset generation needs.")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Design frequency bands from categorized tokens",
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=CATEGORIES_CSV,
        help="Path to token_categories.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--k-min", type=int, default=3, help="Minimum number of bands to evaluate"
    )
    parser.add_argument(
        "--k-max", type=int, default=8, help="Maximum number of bands to evaluate"
    )
    parser.add_argument(
        "--chosen-k",
        type=int,
        default=5,
        help="Chosen k for final scheme export (default: 5)",
    )
    parser.add_argument(
        "--range-pct-low",
        type=float,
        default=1.0,
        help="Lower percentile for core range (default: 1)",
    )
    parser.add_argument(
        "--range-pct-high",
        type=float,
        default=99.0,
        help="Upper percentile for core range (default: 99)",
    )
    args = parser.parse_args()

    if not args.csv_file.exists():
        logger.error(f"CSV not found: {args.csv_file}")
        return 1

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load ----
    logger.info(f"Loading: {args.csv_file}")
    df = pd.read_csv(args.csv_file, keep_default_na=False)
    logger.info(f"  {len(df):,} tokens loaded")

    # ---- Filter to word_en ----
    mask = df["is_word_en"].astype(str) == "True"
    wdf = df[mask].copy()
    logger.info(f"  word_en tokens: {len(wdf):,}")

    if len(wdf) == 0:
        logger.error("No word_en tokens found!")
        return 1

    log_freqs = wdf["log_frequency"].values
    capitalizations = wdf["capitalization"].values

    # ---- Define range ----
    range_low = float(np.percentile(log_freqs, args.range_pct_low))
    range_high = float(np.percentile(log_freqs, args.range_pct_high))

    logger.info(
        f"  Core range (p{args.range_pct_low:.0f}-p{args.range_pct_high:.0f}): "
        f"[{range_low:.4f}, {range_high:.4f}]"
    )
    logger.info(
        f"  Range width: {range_high - range_low:.4f} log-freq units "
        f"({10 ** (range_high - range_low):.1f}x frequency ratio)"
    )

    # Tail counts
    n_bottom_tail = int((log_freqs < range_low).sum())
    n_top_tail = int((log_freqs > range_high).sum())
    logger.info(f"  Bottom tail: {n_bottom_tail:,} tokens")
    logger.info(f"  Top tail: {n_top_tail:,} tokens")

    # ---- Compute all schemes ----
    schemes = {}
    for k in range(args.k_min, args.k_max + 1):
        scheme = compute_band_scheme(
            log_freqs,
            capitalizations,
            k,
            range_low,
            range_high,
        )
        schemes[k] = scheme
        cs = scheme["core_summary"]
        logger.info(
            f"  k={k}: width={scheme['band_width']:.3f}  "
            f"ratio={scheme['freq_ratio_per_band']:.1f}x  "
            f"min_tok={cs['min_tokens']:,}  "
            f"min_cap={cs['min_capitalized']:,}  "
            f"counts={cs['counts_per_band']}"
        )

    # ---- Save ----
    logger.info("\nSaving outputs...")

    # All schemes JSON
    output_data = OrderedDict()
    output_data["created_at"] = datetime.now().isoformat()
    output_data["source_csv"] = str(args.csv_file)
    output_data["n_word_en_total"] = len(wdf)
    output_data["core_range"] = {
        "percentiles": [args.range_pct_low, args.range_pct_high],
        "log_freq": [round(range_low, 4), round(range_high, 4)],
        "width": round(range_high - range_low, 4),
        "freq_ratio": round(10 ** (range_high - range_low), 1),
    }
    output_data["tails"] = {
        "bottom_tail": {
            "n_tokens": n_bottom_tail,
            "range": [round(float(log_freqs.min()), 4), round(range_low, 4)],
        },
        "top_tail": {
            "n_tokens": n_top_tail,
            "range": [round(range_high, 4), round(float(log_freqs.max()), 4)],
        },
    }
    output_data["schemes"] = {str(k): v for k, v in schemes.items()}

    json_path = out / "band_schemes.json"
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Saved: {json_path}")

    # Report
    n_tails = {
        "bottom": n_bottom_tail,
        "top": n_top_tail,
        "bottom_range": [float(log_freqs.min()), range_low],
        "top_range": [range_high, float(log_freqs.max())],
    }
    write_report(
        schemes,
        range_low,
        range_high,
        len(wdf),
        n_tails,
        out / "band_scheme_report.txt",
    )

    # Plot
    plot_band_schemes(log_freqs, schemes, out / "fig_band_schemes.png")

    # ---- Export final scheme ----
    chosen_k = args.chosen_k
    if chosen_k not in schemes:
        logger.error(
            f"Chosen k={chosen_k} not in evaluated range [{args.k_min}, {args.k_max}]"
        )
        return 1

    final = build_final_scheme(
        schemes[chosen_k],
        log_freqs,
        capitalizations,
        chosen_k,
    )

    final_path = out / "final_bands.json"
    with open(final_path, "w") as f:
        json.dump(final, f, indent=2)
    logger.info(f"Saved: {final_path}")

    # ---- Summary ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("BAND DESIGN COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  word_en pool: {len(wdf):,} tokens")
    logger.info(f"  Core range:   [{range_low:.3f}, {range_high:.3f}]")
    logger.info(f"  Schemes evaluated: k={args.k_min}..{args.k_max}")
    logger.info(f"  Chosen k: {chosen_k}")
    logger.info("")
    logger.info(
        f"  {'k':>3s}  {'band_width':>10s}  {'min_tokens':>10s}  {'min_cap':>8s}"
    )
    for k in sorted(schemes.keys()):
        cs = schemes[k]["core_summary"]
        marker = " ◄" if k == chosen_k else ""
        logger.info(
            f"  {k:>3d}  {schemes[k]['band_width']:>10.3f}  "
            f"{cs['min_tokens']:>10,d}  {cs['min_capitalized']:>8,d}{marker}"
        )

    logger.info("")
    logger.info(f"  FINAL SCHEME (k={chosen_k}):")
    for name, band in final["bands"].items():
        btype = band["type"]
        n = band.get("n_tokens", band.get("n_tokens_available", "?"))
        rng = band["log_freq_range"]
        logger.info(
            f"    {name:<15s}  {btype:<12s}  [{rng[0]:>7.3f}, {rng[1]:>6.3f}]  n={n:>6}"
        )

    logger.info(f"\nOutputs in: {out}")
    logger.info(f"CANONICAL BANDS: {final_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
