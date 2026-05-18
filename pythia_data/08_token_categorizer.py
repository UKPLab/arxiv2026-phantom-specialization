#!/usr/bin/env python3
"""
Token Categorizer
==================
Pipeline step 08.  Enriches the token vocabulary with character-level
Unicode analysis and assigns a clean category label to each token.

PURPOSE
-------
Every downstream task (LSC, IOI, etc.) needs to filter the vocabulary
for usable tokens.  Rather than each task implementing its own ad-hoc
filter, this script categorizes every token ONCE using proper Unicode
introspection, producing labels that any task can filter on.

CATEGORIES
----------
    word_en             Ġ-prefixed + all Latin letters.  These tokens
                        begin a new word in text and are the primary
                        experimental pool for English tasks (LSC, IOI).
    subword_en          Bare (no Ġ) + all Latin letters.  Continuation
                        fragments like 'ing', 'tion', 'ment'.  Not usable
                        as standalone words.
    word_other          Ġ-prefixed + all letters but non-Latin script
                        (Greek, Cyrillic, CJK, etc.)
    subword_other       Bare + non-Latin script letters
    numeric             All content chars are digits
    punctuation         All content chars are punctuation or symbols
    whitespace          Token is pure whitespace / control characters
    mixed               Token contains characters from multiple groups
                        (e.g. letters+digits, letters+punctuation)

BPE HANDLING
------------
GPT-NeoX BPE uses Ġ (U+0120, LATIN CAPITAL LETTER G WITH DOT ABOVE)
as the space-prefix marker in the vocabulary representation.  This script:
  1. Detects the Ġ prefix
  2. Records has_space_prefix = True
  3. Strips it before character analysis
  4. The classification is based on CONTENT characters only

INPUT
-----
all_tokens_complete.csv  (from 06_build_token_dataset.py)
Required columns:  token_id, token_string, log_frequency

OUTPUTS
-------
token_categories.csv               Original CSV + new columns
token_categories_summary.json      Category counts and cross-tabs
token_categories_report.txt        Human-readable summary

Usage:
    python 08_token_categorizer.py
    python 08_token_categorizer.py --csv-file path/to/tokens.csv
"""

import json
import argparse
import logging
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Tuple, List
from collections import OrderedDict, Counter
from datetime import datetime

import re

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE0_CSV = SCRIPT_DIR / "token_dataset" / "all_tokens_complete.csv"
OUTPUT_DIR = SCRIPT_DIR / "token_categories"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# UNICODE CONSTANTS
# ============================================================================

# BPE space marker used by GPT-NeoX tokenizer
BPE_SPACE_CHAR = "Ġ"  # U+0120, LATIN CAPITAL LETTER G WITH DOT ABOVE

# BPE whitespace markers: GPT-NeoX bytes_to_unicode() maps whitespace bytes
# to Latin-Extended characters. These look like Latin letters to Unicode
# but actually represent whitespace/control characters:
#   Ġ (U+0120) = space (0x20)
#   Ċ (U+010A) = newline (0x0A)
#   ĉ (U+0109) = tab (0x09)
#   č (U+010D) = carriage return (0x0D)
BPE_WHITESPACE_MARKERS = frozenset("ĠĊĉč")

# Unicode category groups
LETTER_CATEGORIES = {"Lu", "Ll", "Lt", "Lm", "Lo"}
NUMBER_CATEGORIES = {"Nd", "Nl", "No"}
SEPARATOR_CATEGORIES = {"Zs", "Zl", "Zp"}
MARK_CATEGORIES = {"Mn", "Mc", "Me"}
PUNCTUATION_CATEGORIES = {"Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"}
SYMBOL_CATEGORIES = {"Sm", "Sc", "Sk", "So"}
CONTROL_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn"}

# Roman numeral regex (validates 1-4000).
ROMAN_NUMERAL_PATTERN = re.compile(
    r"^(?=.)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", re.IGNORECASE
)

CATEGORY_GROUP_MAP = {}
for cat in LETTER_CATEGORIES:
    CATEGORY_GROUP_MAP[cat] = "letter"
for cat in NUMBER_CATEGORIES:
    CATEGORY_GROUP_MAP[cat] = "number"
for cat in SEPARATOR_CATEGORIES:
    CATEGORY_GROUP_MAP[cat] = "separator"
for cat in MARK_CATEGORIES:
    CATEGORY_GROUP_MAP[cat] = "mark"
for cat in PUNCTUATION_CATEGORIES:
    CATEGORY_GROUP_MAP[cat] = "punctuation"
for cat in SYMBOL_CATEGORIES:
    CATEGORY_GROUP_MAP[cat] = "symbol"
for cat in CONTROL_CATEGORIES:
    CATEGORY_GROUP_MAP[cat] = "control"


# Latin script detection: check if Unicode name starts with "LATIN"
# This catches all Latin letters including accented (é, ñ, ü, etc.)
def _get_script(ch: str) -> str:
    """Get the script of a character from its Unicode name."""
    try:
        name = unicodedata.name(ch, "")
    except ValueError:
        return "unknown"
    if not name:
        return "unknown"

    # Common scripts by name prefix
    for script in [
        "LATIN",
        "GREEK",
        "CYRILLIC",
        "ARABIC",
        "HEBREW",
        "CJK",
        "HANGUL",
        "KATAKANA",
        "HIRAGANA",
        "THAI",
        "DEVANAGARI",
        "BENGALI",
        "TAMIL",
        "GEORGIAN",
        "ARMENIAN",
        "ETHIOPIC",
        "TIBETAN",
        "MYANMAR",
    ]:
        if script in name:
            return script.lower()

    # Digits
    if "DIGIT" in name:
        return "digit"

    return "other"


# ============================================================================
# CHARACTER ANALYSIS
# ============================================================================


def analyze_characters(text: str) -> Dict:
    """
    Full Unicode analysis of a string.

    Returns character-level breakdown:
      - Per-character: Unicode category, group, script
      - Aggregated: category counts, group counts, script counts
      - Derived: is_ascii, is_all_latin, is_all_letter, etc.
    """
    if not text:
        return {
            "n_chars": 0,
            "categories": {},
            "groups": {},
            "scripts": {},
            "is_ascii": True,
            "is_all_letter": False,
            "is_all_latin": False,
            "is_all_digit": False,
            "is_all_punct": False,
            "has_letter": False,
            "has_digit": False,
            "has_punct": False,
            "has_symbol": False,
        }

    categories = Counter()
    groups = Counter()
    scripts = Counter()
    is_ascii = True

    for ch in text:
        cat = unicodedata.category(ch)
        group = CATEGORY_GROUP_MAP.get(cat, "other")
        categories[cat] += 1
        groups[group] += 1

        if ord(ch) > 127:
            is_ascii = False

        # Only track script for letters
        if group == "letter":
            scripts[_get_script(ch)] += 1

    n = len(text)
    n_letter = groups.get("letter", 0)
    n_digit = groups.get("number", 0)
    n_punct = groups.get("punctuation", 0)
    n_symbol = groups.get("symbol", 0)

    return {
        "n_chars": n,
        "categories": dict(categories),
        "groups": dict(groups),
        "scripts": dict(scripts),
        "is_ascii": is_ascii,
        "is_all_letter": n_letter == n,
        "is_all_latin": scripts.get("latin", 0) == n_letter == n and n > 0,
        "is_all_digit": n_digit == n and n > 0,
        "is_all_punct": (n_punct + n_symbol) == n and n > 0,
        "has_letter": n_letter > 0,
        "has_digit": n_digit > 0,
        "has_punct": n_punct > 0,
        "has_symbol": n_symbol > 0,
    }


def categorize_token(token_string: str) -> Dict:
    """
    Categorize a single token.

    Steps:
      1. Detect and strip BPE space prefix (Ġ)
      2. Handle pure whitespace / empty tokens
      3. Analyze content characters
      4. Assign final label incorporating word vs subword distinction

    Label logic for alphabetic tokens:
      - Ġ prefix means this token begins a new word in text.
        For all-Latin tokens: Ġ-prefixed -> word_en, bare -> subword_en
      - word_en tokens are the experimental pool (can function as
        standalone words or word-starts in sentences)
      - subword_en tokens are continuation fragments (ing, tion, ment)
        and are not usable as standalone words
    """
    result = {
        "original_string": token_string,
        "has_space_prefix": False,
        "content_text": "",
        "content_repr": "",
        "n_content_chars": 0,
        "token_label": "unknown",
        "capitalization": "",
        "is_word_en": False,
        "is_roman_numeral": False,
        "is_abbreviation": False,
        "is_single_letter_word": False,
    }

    if not token_string:
        result["token_label"] = "whitespace"
        return result

    # Step 1: Detect BPE space prefix
    content = token_string
    if content.startswith(BPE_SPACE_CHAR):
        result["has_space_prefix"] = True
        content = content[1:]  # strip the Ġ

    # Step 2: Handle whitespace / empty after stripping
    if not content or content.isspace():
        result["token_label"] = "whitespace"
        result["content_text"] = content
        result["content_repr"] = repr(content)
        return result

    # Step 2b: Handle BPE whitespace markers
    # After stripping the first Ġ prefix, check if remaining content is purely
    # BPE whitespace markers (Ġ=space, Ċ=newline, ĉ=tab, č=CR).
    # These look like Latin letters to Unicode but represent whitespace.
    # Example: "ĠĠĠĠ" = 4 spaces, "ĠĊ" = space+newline (indent)
    if content and all(ch in BPE_WHITESPACE_MARKERS for ch in content):
        result["token_label"] = "whitespace"
        result["content_text"] = content
        result["content_repr"] = repr(content)
        result["n_content_chars"] = len(content)
        return result

    # Also handle other whitespace-like content
    # (newlines, tabs, etc. that survived as tokens)
    all_control = all(
        unicodedata.category(ch) in CONTROL_CATEGORIES | SEPARATOR_CATEGORIES
        for ch in content
    )
    if all_control:
        result["token_label"] = "whitespace"
        result["content_text"] = content
        result["content_repr"] = repr(content)
        result["n_content_chars"] = len(content)
        return result

    result["content_text"] = content
    result["content_repr"] = repr(content)
    result["n_content_chars"] = len(content)

    # Step 3: Character analysis
    analysis = analyze_characters(content)
    result["char_analysis"] = analysis

    # Step 4: Assign label
    groups = analysis["groups"]
    n_groups_present = sum(
        1 for g in ["letter", "number", "punctuation", "symbol"] if groups.get(g, 0) > 0
    )

    has_space = result["has_space_prefix"]

    if n_groups_present == 0:
        # Only marks, separators, control chars, etc.
        result["token_label"] = "whitespace"
    elif n_groups_present == 1:
        # Pure single category
        if analysis["is_all_letter"]:
            if analysis["is_all_latin"] and analysis["is_ascii"]:
                # word_en / subword_en: Latin script AND ASCII only (a-z/A-Z)
                # This excludes non-ASCII Latin (é, ñ, ü, ß, þ, ð, æ, etc.)
                # which are valid Latin letters but not English-compatible
                if has_space:
                    result["token_label"] = "word_en"
                    result["is_word_en"] = True
                else:
                    result["token_label"] = "subword_en"
            else:
                # word_other / subword_other: non-Latin OR non-ASCII Latin
                # Includes: Greek, Cyrillic, CJK, etc. AND accented Latin
                if has_space:
                    result["token_label"] = "word_other"
                else:
                    result["token_label"] = "subword_other"
        elif analysis["is_all_digit"]:
            result["token_label"] = "numeric"
        elif analysis["is_all_punct"]:
            result["token_label"] = "punctuation"
        else:
            result["token_label"] = "mixed"
    else:
        # Multiple character groups
        result["token_label"] = "mixed"

    # Subcategorize mixed tokens for information
    if result["token_label"] == "mixed":
        has_l = analysis["has_letter"]
        has_d = analysis["has_digit"]
        has_p = analysis["has_punct"] or analysis["has_symbol"]
        if has_l and has_d and not has_p:
            result["mixed_subtype"] = "alpha_num"
        elif has_l and has_p and not has_d:
            result["mixed_subtype"] = "alpha_punct"
        elif has_d and has_p and not has_l:
            result["mixed_subtype"] = "num_punct"
        else:
            result["mixed_subtype"] = "other"

    # Capitalization analysis (for any token that contains letters)
    # Applied to content_text (after Ġ stripping)
    # Separate from label > orthogonal axis for filtering
    #
    #   lowercase    : all letters are lowercase       (Ġthe -> the)
    #   capitalized  : first letter upper, rest lower  (ĠJohn -> John)
    #   uppercase    : all letters are uppercase        (ĠUSA -> USA)
    #   mixed_case   : other patterns                  (ĠiPhone -> iPhone)
    #   no_letters   : token has no letter characters
    if analysis.get("has_letter", False) if "char_analysis" in result else False:
        letters_only = [
            ch for ch in content if unicodedata.category(ch) in LETTER_CATEGORIES
        ]
        if not letters_only:
            result["capitalization"] = "no_letters"
        elif all(ch.islower() for ch in letters_only):
            result["capitalization"] = "lowercase"
        elif all(ch.isupper() for ch in letters_only):
            result["capitalization"] = "uppercase"
        elif letters_only[0].isupper() and all(ch.islower() for ch in letters_only[1:]):
            result["capitalization"] = "capitalized"
        else:
            result["capitalization"] = "mixed_case"
    else:
        result["capitalization"] = "no_letters"

    # Roman numeral: length > 1, all-Latin content, matches regex
    if (
        content
        and len(content) > 1
        and result["token_label"] in ("word_en", "subword_en")
        and ROMAN_NUMERAL_PATTERN.match(content)
    ):
        result["is_roman_numeral"] = True

    # Abbreviation: uppercase, 2-5 chars, all alpha, ASCII
    if (
        content
        and content.isupper()
        and content.isalpha()
        and content.isascii()
        and 2 <= len(content) <= 5
    ):
        result["is_abbreviation"] = True

    # Special single-letter words (I, a, A with space prefix)
    if content in ("I", "a", "A") and result["has_space_prefix"]:
        result["is_single_letter_word"] = True

    return result


def process_vocabulary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process all tokens and add categorization columns.
    """
    logger.info(f"Categorizing {len(df):,} tokens...")

    has_space_prefix = []
    content_text = []
    n_content_chars = []
    token_label = []
    capitalization = []
    is_word_en = []
    mixed_subtype = []
    is_ascii = []
    scripts_primary = []
    char_groups = []
    is_roman_numeral = []
    is_abbreviation = []
    is_single_letter_word = []

    for i, row in df.iterrows():
        ts = str(row["token_string"]) if pd.notna(row["token_string"]) else ""
        cat = categorize_token(ts)

        has_space_prefix.append(cat["has_space_prefix"])
        content_text.append(cat["content_text"])
        n_content_chars.append(cat["n_content_chars"])
        token_label.append(cat["token_label"])
        capitalization.append(cat["capitalization"])
        is_word_en.append(cat["is_word_en"])
        is_roman_numeral.append(cat["is_roman_numeral"])
        is_abbreviation.append(cat["is_abbreviation"])
        is_single_letter_word.append(cat["is_single_letter_word"])
        mixed_subtype.append(cat.get("mixed_subtype", ""))

        analysis = cat.get("char_analysis", {})
        is_ascii.append(analysis.get("is_ascii", True))

        # Primary script (most common script among letter chars)
        scripts = analysis.get("scripts", {})
        if scripts:
            primary = max(scripts, key=scripts.get)
            scripts_primary.append(primary)
        else:
            scripts_primary.append("")

        # Compact groups representation
        groups = analysis.get("groups", {})
        char_groups.append(
            "+".join(f"{g}:{n}" for g, n in sorted(groups.items())) if groups else ""
        )

    df = df.copy()
    df["has_space_prefix"] = has_space_prefix
    df["content_text"] = content_text
    df["n_content_chars"] = n_content_chars
    df["token_label"] = token_label
    df["capitalization"] = capitalization
    df["is_word_en"] = is_word_en
    df["is_roman_numeral"] = is_roman_numeral
    df["is_abbreviation"] = is_abbreviation
    df["is_single_letter_word"] = is_single_letter_word
    df["mixed_subtype"] = mixed_subtype
    df["is_ascii"] = is_ascii
    df["primary_script"] = scripts_primary
    df["char_groups"] = char_groups

    logger.info(f"  Done. Label distribution:")
    for label, count in df["token_label"].value_counts().items():
        pct = count / len(df) * 100
        logger.info(f"    {label:<20s}: {count:>6,d} ({pct:>5.1f}%)")

    return df


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================


def compute_summary(df: pd.DataFrame) -> Dict:
    """
    Compute summary statistics for the categorized vocabulary.
    """
    summary = OrderedDict()
    summary["created_at"] = datetime.now().isoformat()
    summary["n_tokens_total"] = int(len(df))

    # Category counts
    label_counts = df["token_label"].value_counts().to_dict()
    summary["category_counts"] = {k: int(v) for k, v in sorted(label_counts.items())}
    summary["category_pct"] = {
        k: round(v / len(df) * 100, 2) for k, v in sorted(label_counts.items())
    }

    # Mixed subtypes
    mixed_mask = df["token_label"] == "mixed"
    if mixed_mask.any():
        mixed_sub = df.loc[mixed_mask, "mixed_subtype"].value_counts().to_dict()
        summary["mixed_subtypes"] = {k: int(v) for k, v in mixed_sub.items()}

    # Script distribution (for letter-containing tokens)
    script_counts = (
        df.loc[df["primary_script"] != "", "primary_script"].value_counts().to_dict()
    )
    summary["script_counts"] = {k: int(v) for k, v in script_counts.items()}

    # Capitalization distribution
    cap_counts = df["capitalization"].value_counts().to_dict()
    summary["capitalization_counts"] = {
        k: int(v) for k, v in sorted(cap_counts.items())
    }

    # Cross-tab: token_label x capitalization (the useful filter combinations)
    if "capitalization" in df.columns:
        cross = {}
        for label in sorted(df["token_label"].unique()):
            mask = df["token_label"] == label
            cap_dist = df.loc[mask, "capitalization"].value_counts().to_dict()
            cross[label] = {k: int(v) for k, v in sorted(cap_dist.items())}
        summary["label_x_capitalization"] = cross

    # Space prefix stats
    summary["has_space_prefix"] = {
        "true": int(df["has_space_prefix"].sum()),
        "false": int((~df["has_space_prefix"]).sum()),
    }

    # ASCII stats
    summary["is_ascii"] = {
        "true": int(df["is_ascii"].sum()),
        "false": int((~df["is_ascii"]).sum()),
    }

    # Content length distribution by category
    length_by_cat = {}
    for label in sorted(df["token_label"].unique()):
        mask = df["token_label"] == label
        lengths = df.loc[mask, "n_content_chars"]
        length_by_cat[label] = {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": round(float(lengths.mean()), 2),
            "median": int(lengths.median()),
        }
    summary["content_length_by_category"] = length_by_cat

    # Frequency distribution by category
    if "log_frequency" in df.columns:
        freq_by_cat = {}
        for label in sorted(df["token_label"].unique()):
            mask = df["token_label"] == label
            lf = df.loc[mask, "log_frequency"]
            freq_by_cat[label] = {
                "n": int(mask.sum()),
                "log_freq_min": round(float(lf.min()), 4),
                "log_freq_max": round(float(lf.max()), 4),
                "log_freq_mean": round(float(lf.mean()), 4),
                "log_freq_median": round(float(lf.median()), 4),
                "log_freq_std": round(float(lf.std()), 4),
                "log_freq_p5": round(float(lf.quantile(0.05)), 4),
                "log_freq_p25": round(float(lf.quantile(0.25)), 4),
                "log_freq_p75": round(float(lf.quantile(0.75)), 4),
                "log_freq_p95": round(float(lf.quantile(0.95)), 4),
            }
        summary["frequency_by_category"] = freq_by_cat

        # The key number: word_en frequency profile
        en_mask = df["is_word_en"]
        if en_mask.any():
            en_lf = df.loc[en_mask, "log_frequency"]
            pctiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
            en_profile = {
                "n_tokens": int(en_mask.sum()),
                "pct_of_vocab": round(en_mask.sum() / len(df) * 100, 2),
                "log_freq_range": [
                    round(float(en_lf.min()), 4),
                    round(float(en_lf.max()), 4),
                ],
                "percentiles": {
                    f"p{p}": round(float(en_lf.quantile(p / 100)), 4) for p in pctiles
                },
            }
            summary["word_en_profile"] = en_profile

    return summary


def plot_category_overview(df: pd.DataFrame, path: Path) -> None:
    """Category distribution and frequency profiles by category."""
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    labels_ordered = df["token_label"].value_counts().index.tolist()
    colors_map = {
        "word_en": "#4285f4",
        "subword_en": "#a0c4ff",
        "word_other": "#9e9e9e",
        "subword_other": "#d0d0d0",
        "numeric": "#fbbc04",
        "punctuation": "#ea4335",
        "whitespace": "#34a853",
        "mixed": "#ff6d01",
        "unknown": "#cccccc",
    }

    # --- 1. Category bar chart ---
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["token_label"].value_counts()
    bars = ax1.barh(
        range(len(counts)),
        counts.values,
        color=[colors_map.get(l, "#999") for l in counts.index],
        edgecolor="black",
        linewidth=0.5,
    )
    ax1.set_yticks(range(len(counts)))
    ax1.set_yticklabels(counts.index, fontsize=10)
    for i, (cnt, label) in enumerate(zip(counts.values, counts.index)):
        pct = cnt / len(df) * 100
        ax1.text(
            cnt + len(df) * 0.01, i, f"{cnt:,} ({pct:.1f}%)", va="center", fontsize=9
        )
    ax1.set_xlabel("Token count", fontsize=11)
    ax1.set_title("Token Category Distribution", fontsize=12, fontweight="bold")
    ax1.invert_yaxis()
    ax1.grid(axis="x", alpha=0.3)

    # --- 2. Frequency distributions by category ---
    ax2 = fig.add_subplot(gs[0, 1])
    if "log_frequency" in df.columns:
        for label in labels_ordered:
            mask = df["token_label"] == label
            if mask.sum() < 10:
                continue
            lf = df.loc[mask, "log_frequency"]
            ax2.hist(
                lf,
                bins=80,
                alpha=0.4,
                density=True,
                color=colors_map.get(label, "#999"),
                label=label,
                edgecolor="none",
            )
        ax2.set_xlabel("log₁₀(freq per million)", fontsize=11)
        ax2.set_ylabel("Density", fontsize=11)
        ax2.set_title(
            "Frequency Distribution by Category", fontsize=12, fontweight="bold"
        )
        ax2.legend(fontsize=8, loc="upper right")
        ax2.grid(alpha=0.3)

    # --- 3. word_en detail ---
    ax3 = fig.add_subplot(gs[1, 0])
    en_mask = df["is_word_en"]
    if en_mask.any() and "log_frequency" in df.columns:
        en_lf = df.loc[en_mask, "log_frequency"]
        ax3.hist(
            en_lf,
            bins=100,
            color="#4285f4",
            alpha=0.6,
            edgecolor="white",
            linewidth=0.3,
        )
        # Mark percentiles
        for p, ls in [(5, ":"), (25, "--"), (50, "-"), (75, "--"), (95, ":")]:
            val = en_lf.quantile(p / 100)
            ax3.axvline(val, color="#ea4335", linewidth=1.2, linestyle=ls, alpha=0.7)
            ax3.annotate(
                f"p{p}",
                xy=(val, ax3.get_ylim()[1] * 0.9),
                fontsize=7,
                ha="center",
                color="#ea4335",
            )
        ax3.set_xlabel("log₁₀(freq per million)", fontsize=11)
        ax3.set_ylabel("Count", fontsize=11)
        ax3.set_title(
            f"word_en Frequency Distribution (n={en_mask.sum():,})",
            fontsize=12,
            fontweight="bold",
        )
        ax3.grid(alpha=0.3)

    # --- 4. Content length by category ---
    ax4 = fig.add_subplot(gs[1, 1])
    box_data = []
    box_labels = []
    for label in labels_ordered:
        mask = df["token_label"] == label
        if mask.sum() < 10:
            continue
        box_data.append(df.loc[mask, "n_content_chars"].values)
        box_labels.append(label)
    if box_data:
        bp = ax4.boxplot(box_data, vert=True, patch_artist=True, showfliers=False)
        for patch, label in zip(bp["boxes"], box_labels):
            patch.set_facecolor(colors_map.get(label, "#999"))
            patch.set_alpha(0.6)
        ax4.set_xticklabels(box_labels, fontsize=9, rotation=30, ha="right")
        ax4.set_ylabel("Content length (chars)", fontsize=11)
        ax4.set_title("Token Length by Category", fontsize=12, fontweight="bold")
        ax4.grid(axis="y", alpha=0.3)

    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_en_by_frequency(df: pd.DataFrame, path: Path) -> None:
    """
    For word_en tokens: show how many are available in
    sliding windows across the frequency axis.  This is the key
    plot for band feasibility.
    """
    en_mask = df["is_word_en"]
    if not en_mask.any() or "log_frequency" not in df.columns:
        logger.warning(
            "No word_en tokens or no log_frequency column, "
            "skipping en_by_frequency plot"
        )
        return

    en_lf = df.loc[en_mask, "log_frequency"].values

    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    # Panel 1: histogram of word_en tokens
    ax = axes[0]
    n_bins = 100
    counts, edges, _ = ax.hist(
        en_lf, bins=n_bins, color="#4285f4", alpha=0.6, edgecolor="white", linewidth=0.3
    )
    ax.set_ylabel("Token count per bin", fontsize=11)
    ax.set_title(
        f"word_en tokens across frequency axis (n={len(en_lf):,})",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(alpha=0.3)

    # Panel 2: cumulative count: how many en tokens above/below each threshold
    ax2 = axes[1]
    sorted_lf = np.sort(en_lf)
    cum = np.arange(1, len(sorted_lf) + 1)

    ax2.plot(
        sorted_lf, cum, color="#4285f4", linewidth=2, label="Cumulative count (<= x)"
    )
    ax2.plot(
        sorted_lf,
        len(sorted_lf) - cum,
        color="#ea4335",
        linewidth=2,
        linestyle="--",
        label="Remaining count (> x)",
    )

    # Mark key thresholds
    for threshold in [50, 100, 200, 500]:
        ax2.axhline(threshold, color="#999", linewidth=0.5, linestyle=":")
        ax2.text(
            sorted_lf[-1],
            threshold + 10,
            f"n={threshold}",
            fontsize=7,
            color="#666",
            ha="right",
        )

    ax2.set_xlabel("log₁₀(freq per million)", fontsize=11)
    ax2.set_ylabel("Token count", fontsize=11)
    ax2.set_title("Cumulative word_en Availability", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def write_report(summary: Dict, path: Path) -> None:
    """Human-readable summary."""
    lines = []

    def section(title):
        lines.append("")
        lines.append("=" * 80)
        lines.append(title)
        lines.append("=" * 80)

    section("TOKEN CATEGORIZATION REPORT")
    lines.append(f"Generated: {summary['created_at']}")
    lines.append(f"Total tokens: {summary['n_tokens_total']:,}")

    section("CATEGORY DISTRIBUTION")
    for cat, count in sorted(summary["category_counts"].items(), key=lambda x: -x[1]):
        pct = summary["category_pct"][cat]
        lines.append(f"  {cat:<20s}: {count:>6,d}  ({pct:>5.1f}%)")

    if "mixed_subtypes" in summary:
        lines.append("")
        lines.append("  Mixed token subtypes:")
        for sub, count in sorted(
            summary["mixed_subtypes"].items(), key=lambda x: -x[1]
        ):
            lines.append(f"    {sub:<20s}: {count:>6,d}")

    section("SCRIPT DISTRIBUTION (letter-containing tokens)")
    for script, count in sorted(summary["script_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"  {script:<20s}: {count:>6,d}")

    section("CAPITALIZATION DISTRIBUTION")
    for cap, count in sorted(
        summary.get("capitalization_counts", {}).items(), key=lambda x: -x[1]
    ):
        lines.append(f"  {cap:<20s}: {count:>6,d}")

    if "label_x_capitalization" in summary:
        lines.append("")
        lines.append("  Cross-tab (label x capitalization):")
        for label, cap_dist in summary["label_x_capitalization"].items():
            parts = ", ".join(
                f"{k}={v}" for k, v in sorted(cap_dist.items(), key=lambda x: -x[1])
            )
            lines.append(f"    {label:<18s}: {parts}")

    section("SPACE PREFIX")
    sp = summary["has_space_prefix"]
    lines.append(f"  With space prefix:    {sp['true']:>6,d}")
    lines.append(f"  Without space prefix: {sp['false']:>6,d}")

    section("CONTENT LENGTH BY CATEGORY")
    for cat, stats in summary.get("content_length_by_category", {}).items():
        lines.append(
            f"  {cat:<20s}: min={stats['min']}  max={stats['max']}  "
            f"mean={stats['mean']:.1f}  median={stats['median']}"
        )

    if "frequency_by_category" in summary:
        section("FREQUENCY PROFILE BY CATEGORY")
        for cat, stats in summary["frequency_by_category"].items():
            lines.append(
                f"  {cat:<20s}: n={stats['n']:>6,d}  "
                f"log_freq=[{stats['log_freq_min']:>7.3f}, "
                f"{stats['log_freq_max']:>6.3f}]  "
                f"mean={stats['log_freq_mean']:>6.3f}  "
                f"std={stats['log_freq_std']:.3f}"
            )

    if "word_en_profile" in summary:
        section("ALPHABETIC_EN PROFILE (key for experiments)")
        ep = summary["word_en_profile"]
        lines.append(
            f"  Total word_en tokens: {ep['n_tokens']:,} "
            f"({ep['pct_of_vocab']}% of vocabulary)"
        )
        lines.append(
            f"  Log-freq range: [{ep['log_freq_range'][0]:.4f}, "
            f"{ep['log_freq_range'][1]:.4f}]"
        )
        lines.append(f"  Percentiles:")
        for k, v in ep["percentiles"].items():
            lines.append(f"    {k:>5s}: {v:.4f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Categorize tokens by Unicode character analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=STAGE0_CSV,
        help="Path to all_tokens_complete.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory"
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
    logger.info(f"  Columns: {list(df.columns)}")

    # Verify required columns
    required = {"token_id", "token_string"}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return 1

    # ---- Categorize ----
    df = process_vocabulary(df)

    # ---- Summary ----
    summary = compute_summary(df)

    # ---- Save ----
    logger.info("\nSaving outputs...")

    csv_path = out / "token_categories.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

    json_path = out / "token_categories_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved: {json_path}")

    write_report(summary, out / "token_categories_report.txt")

    # ---- Visualizations ----
    plot_category_overview(df, out / "fig_categories.png")
    plot_en_by_frequency(df, out / "fig_en_by_frequency.png")

    # ---- Final log ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("CATEGORIZATION COMPLETE")
    logger.info("=" * 60)
    for cat, count in sorted(summary["category_counts"].items(), key=lambda x: -x[1]):
        pct = summary["category_pct"][cat]
        logger.info(f"  {cat:<20s}: {count:>6,d} ({pct:>5.1f}%)")
    if "word_en_profile" in summary:
        ep = summary["word_en_profile"]
        logger.info(f"")
        logger.info(f"  word_en pool: {ep['n_tokens']:,} tokens")
        logger.info(
            f"  Log-freq range:     [{ep['log_freq_range'][0]:.3f}, "
            f"{ep['log_freq_range'][1]:.3f}]"
        )
    logger.info(f"\nOutputs in: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
