#!/usr/bin/env python3
"""
Prevalidate tokens against a FineWeb word corpus (pipeline step 12).

For each token: clean BPE markers, run rule-based Unicode/script analysis,
and (when needed) consult the FineWeb word corpus to decide whether the
token primarily appears as a standalone word or as a subword fragment.
Classification is driven by the substring/word ratio rather than absolute
counts.
"""

import pandas as pd
import numpy as np
import logging
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import Counter
import unicodedata
import string
import re
from tqdm import tqdm
import time
import pickle
import json

logger = logging.getLogger(__name__)

# Unicode category groups
LETTER_CATEGORIES = {"Lu", "Ll", "Lt", "Lm", "Lo"}
NUMBER_CATEGORIES = {"Nd", "Nl", "No"}
SEPARATOR_CATEGORIES = {"Zs", "Zl", "Zp"}
MARK_CATEGORIES = {"Mn", "Mc", "Me"}
PUNCTUATION_CATEGORIES = {"Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"}
SYMBOL_CATEGORIES = {"Sm", "Sc", "Sk", "So"}
CONTROL_CATEGORIES = {"Cc", "Cf"}

# Roman numeral regex (validates 1-4000). Source:
# https://stackoverflow.com/questions/267399/how-do-you-match-only-valid-roman-numerals-with-a-regular-expression
ROMAN_NUMERAL_PATTERN = re.compile(
    r"^(?=.)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$", re.IGNORECASE
)


class PrevalidationWorker:
    """Worker for prevalidation using FineWeb corpus"""

    def __init__(self, worker_id: int, fineweb_dir: str):
        self.worker_id = worker_id
        self.fineweb_dir = fineweb_dir
        self.word_set = None
        self.word_frequency = None
        self.metadata = None
        self.substring_threshold = 10  # Higher threshold due to larger corpus
        self._load_fineweb_data()
        logger.info(f"[Worker {worker_id}] Initialized with FineWeb corpus")

    def _load_fineweb_data(self):
        """Load preprocessed FineWeb word data"""
        try:
            logger.info(
                f"[Worker {self.worker_id}] Loading FineWeb data from: {self.fineweb_dir}"
            )

            # Load word set
            word_set_file = Path(self.fineweb_dir) / "word_set.pkl"
            with open(word_set_file, "rb") as f:
                self.word_set = pickle.load(f)
            logger.info(
                f"[Worker {self.worker_id}] Loaded word set: {len(self.word_set):,} unique words"
            )

            # Load word frequency
            word_freq_file = Path(self.fineweb_dir) / "word_frequency.pkl"
            with open(word_freq_file, "rb") as f:
                self.word_frequency = pickle.load(f)
            logger.info(
                f"[Worker {self.worker_id}] Loaded word frequency: {len(self.word_frequency):,} words"
            )

            # Load metadata
            metadata_file = Path(self.fineweb_dir) / "metadata.json"
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)
            logger.info(f"[Worker {self.worker_id}] FineWeb metadata:")
            logger.info(f"  - Total tokens: {self.metadata['total_tokens']:,}")
            logger.info(f"  - Total documents: {self.metadata['total_documents']:,}")
            logger.info(f"  - Unique words: {self.metadata['unique_words']:,}")

        except Exception as e:
            logger.error(f"[Worker {self.worker_id}] Failed to load FineWeb data: {e}")
            logger.error(
                f"Please run 11_download_fineweb_sample.py first to download and process FineWeb data"
            )
            raise

    def clean_token(self, token: str) -> Tuple[str, Dict]:
        """
        Clean token and extract metadata
        Returns: (cleaned_token, metadata)
        """
        if not isinstance(token, str):
            if pd.isna(token):
                token = ""
            else:
                token = str(token)

        metadata = {
            "original_token": token,
            "has_leading_space": token.startswith("Ġ") or token.startswith(" "),
            "has_trailing_space": token.endswith(" "),
            "length_original": len(token),
        }

        # Check if only BPE markers before removing them (for whitespace detection)
        is_only_bpe_whitespace = bool(re.match(r"^[Ġ Ċ\s]+$", token))

        # Remove BPE markers
        cleaned = token.replace("Ġ", "").replace("Ċ", "").strip()

        metadata.update(
            {
                "cleaned_token": cleaned,
                "length_cleaned": len(cleaned),
                "is_empty": len(cleaned) == 0,
                "is_only_bpe_whitespace": is_only_bpe_whitespace,
                "has_internal_space": " " in cleaned,
                "is_capitalized": cleaned[0].isupper() if cleaned else False,
                "is_all_caps": cleaned.isupper() and cleaned.isalpha()
                if cleaned
                else False,
                "is_title_case": cleaned.istitle() if cleaned else False,
            }
        )

        return cleaned, metadata

    def analyze_unicode(self, text: str) -> Dict:
        """Analyze Unicode categories of characters"""
        if not text:
            return {
                "total_chars": 0,
                "letter_count": 0,
                "letter_pct": 0.0,
                "number_count": 0,
                "number_pct": 0.0,
                "separator_count": 0,
                "separator_pct": 0.0,
                "mark_count": 0,
                "mark_pct": 0.0,
                "symbol_count": 0,
                "symbol_pct": 0.0,  # Merged punctuation into symbol
                "control_count": 0,
                "control_pct": 0.0,
                "other_count": 0,
                "other_pct": 0.0,
                "dominant_type": "empty",
            }

        # Count by Unicode categories
        category_counts = Counter()
        for char in text:
            cat = unicodedata.category(char)
            category_counts[cat] += 1

        total = len(text)

        # Group counts (merge punctuation and symbol)
        letter_count = sum(
            cnt for cat, cnt in category_counts.items() if cat in LETTER_CATEGORIES
        )
        number_count = sum(
            cnt for cat, cnt in category_counts.items() if cat in NUMBER_CATEGORIES
        )
        separator_count = sum(
            cnt for cat, cnt in category_counts.items() if cat in SEPARATOR_CATEGORIES
        )
        mark_count = sum(
            cnt for cat, cnt in category_counts.items() if cat in MARK_CATEGORIES
        )
        symbol_count = sum(
            cnt
            for cat, cnt in category_counts.items()
            if cat in PUNCTUATION_CATEGORIES or cat in SYMBOL_CATEGORIES
        )
        control_count = sum(
            cnt for cat, cnt in category_counts.items() if cat in CONTROL_CATEGORIES
        )

        accounted = (
            letter_count
            + number_count
            + separator_count
            + mark_count
            + symbol_count
            + control_count
        )
        other_count = total - accounted

        # Determine dominant type
        type_counts = {
            "letter": letter_count,
            "number": number_count,
            "separator": separator_count,
            "mark": mark_count,
            "symbol": symbol_count,
            "control": control_count,
            "other": other_count,
        }
        dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
        if type_counts[dominant_type] / total < 0.5:
            dominant_type = "mixed"

        return {
            "total_chars": total,
            "letter_count": letter_count,
            "letter_pct": (letter_count / total * 100) if total > 0 else 0.0,
            "number_count": number_count,
            "number_pct": (number_count / total * 100) if total > 0 else 0.0,
            "separator_count": separator_count,
            "separator_pct": (separator_count / total * 100) if total > 0 else 0.0,
            "mark_count": mark_count,
            "mark_pct": (mark_count / total * 100) if total > 0 else 0.0,
            "symbol_count": symbol_count,
            "symbol_pct": (symbol_count / total * 100) if total > 0 else 0.0,
            "control_count": control_count,
            "control_pct": (control_count / total * 100) if total > 0 else 0.0,
            "other_count": other_count,
            "other_pct": (other_count / total * 100) if total > 0 else 0.0,
            "dominant_type": dominant_type,
        }

    def rule_based_analysis(
        self, cleaned: str, unicode_analysis: Dict, metadata: Dict
    ) -> Dict:
        """
        Apply rule-based analysis using Unicode categories and patterns
        Returns: dict with objective features
        """
        if not cleaned:
            return {
                "is_empty": True,
                "is_only_whitespace": metadata.get("is_only_bpe_whitespace", False),
                "is_only_alphabet": False,
                "is_only_number": False,
                "is_only_symbol": False,
                "is_only_mark": False,
                "is_only_separator": False,
                "is_single_char": False,
                "is_special_single_I": False,
                "is_special_single_a": False,
                "is_special_single_A": False,
                "is_roman_numeral": False,
                "is_likely_abbreviation": False,
            }

        total_chars = len(cleaned)

        # Check if only one type of Unicode category
        is_only_whitespace = all(c.isspace() for c in cleaned)
        is_only_alphabet = all(c.isalpha() for c in cleaned)
        is_only_number = all(c.isdigit() for c in cleaned)
        is_only_symbol = all(
            unicodedata.category(c) in PUNCTUATION_CATEGORIES
            or unicodedata.category(c) in SYMBOL_CATEGORIES
            for c in cleaned
        )
        is_only_mark = all(unicodedata.category(c) in MARK_CATEGORIES for c in cleaned)
        is_only_separator = all(
            unicodedata.category(c) in SEPARATOR_CATEGORIES for c in cleaned
        )

        # Single character checks
        is_single_char = total_chars == 1
        is_special_single_I = cleaned == "I"
        is_special_single_a = cleaned == "a"
        is_special_single_A = cleaned == "A"

        # Roman numeral check (length > 1 only)
        is_roman_numeral = self._is_roman_numeral(cleaned) and len(cleaned) > 1

        # Abbreviation detection: ONLY capital alphabetic letters, 2-5 length
        is_likely_abbreviation = (
            cleaned.isupper() and 2 <= total_chars <= 5 and cleaned.isalpha()
        )

        return {
            "is_empty": False,
            "is_only_whitespace": is_only_whitespace,
            "is_only_alphabet": is_only_alphabet,
            "is_only_number": is_only_number,
            "is_only_symbol": is_only_symbol,
            "is_only_mark": is_only_mark,
            "is_only_separator": is_only_separator,
            "is_single_char": is_single_char,
            "is_special_single_I": is_special_single_I,
            "is_special_single_a": is_special_single_a,
            "is_special_single_A": is_special_single_A,
            "is_roman_numeral": is_roman_numeral,
            "is_likely_abbreviation": is_likely_abbreviation,
        }

    def _is_roman_numeral(self, s: str) -> bool:
        """
        Check if string is a valid Roman numeral using regex
        Only matches strings with length > 1 (single chars are excluded)
        """
        if not s or len(s) <= 1:
            return False
        return bool(ROMAN_NUMERAL_PATTERN.match(s))

    def get_initial_classification(
        self, cleaned: str, rule_analysis: Dict
    ) -> Tuple[str, str]:
        """
        Get initial classification based on rule-based features
        Returns: (primary_signal, confidence_level)
        """
        # Check whitespace BEFORE empty (using original token info)
        if rule_analysis["is_only_whitespace"]:
            return "WHITESPACE", "high"

        if not cleaned:
            return "EMPTY", "high"

        if rule_analysis["is_only_number"]:
            return "NUMBER", "high"

        if rule_analysis["is_only_symbol"]:
            return "SYMBOL", "high"

        if rule_analysis["is_only_mark"]:
            return "MARK", "high"

        if rule_analysis["is_only_separator"]:
            return "SEPARATOR", "high"

        if rule_analysis["is_roman_numeral"]:
            return "ROMAN_NUMERAL", "high"

        # Special single-letter words
        if (
            rule_analysis["is_special_single_I"]
            or rule_analysis["is_special_single_a"]
            or rule_analysis["is_special_single_A"]
        ):
            return "WORD", "high"

        # Other single alphabetic characters
        if rule_analysis["is_single_char"] and cleaned.isalpha():
            return "SINGLE_CHAR_FRAGMENT", "high"

        # Mixed content
        if not rule_analysis["is_only_alphabet"]:
            if rule_analysis["is_likely_abbreviation"]:
                return "ABBREVIATION", "medium"
            else:
                return "MIXED", "medium"

        # Pure alphabetic needs corpus analysis
        return "NEEDS_CORPUS_ANALYSIS", "none"

    def validate_with_fineweb(self, word: str) -> Dict:
        """
        Check word against FineWeb corpus

        Returns validation results with frequency counts and ratio analysis
        """
        results = {
            # Full match validation
            "fineweb_valid": False,
            "fineweb_exact_freq": 0,
            "fineweb_substring_count": 0,
        }

        if not word or not word.strip():
            return results

        word_lower = word.lower()

        # Full-word match
        if word_lower in self.word_set:
            results["fineweb_valid"] = True
            results["fineweb_exact_freq"] = self.word_frequency.get(word_lower, 0)

        # Substring match: how many other words contain this as substring
        if len(word_lower) >= 1:
            results["fineweb_substring_count"] = sum(
                1 for w in self.word_set if word_lower in w and w != word_lower
            )

        # Aggregate. exact_freq / substring_count are stored for stats only;
        # classification is based on the ratio, not absolute counts.
        word_vote_count = 1 if results["fineweb_valid"] else 0
        results["word_vote_count"] = word_vote_count
        results["word_vote_sources"] = "fineweb" if results["fineweb_valid"] else ""

        # Total exact matches (for statistics only)
        results["total_exact_matches"] = results["fineweb_exact_freq"]

        # Substring votes
        subword_votes = (
            1 if results["fineweb_substring_count"] >= self.substring_threshold else 0
        )
        results["subword_vote_count"] = subword_votes

        # Total substring occurrences (for statistics only)
        results["total_substring_occurrences"] = results["fineweb_substring_count"]

        # Ratio analysis (NO exact frequency thresholds)
        denominator = max(results["total_exact_matches"], 1)
        results["substring_to_word_ratio"] = (
            results["total_substring_occurrences"] / denominator
        )

        # Word confidence based ONLY on ratio (not counts)
        ratio = results["substring_to_word_ratio"]
        if ratio < 0.1:
            word_confidence = "very_high"  # Word appears 10x more than subtoken
        elif ratio < 0.2:
            word_confidence = "high"  # Word appears 5-10x more
        elif ratio < 0.33:
            word_confidence = "medium"  # Word appears 3-5x more
        elif ratio < 0.5:
            word_confidence = "low"  # Word appears 2-3x more
        elif ratio < 1.0:
            word_confidence = "very_low"  # Word appears 1-2x more
        else:
            word_confidence = "none"  # Subtoken appears more

        results["word_confidence"] = word_confidence

        # Subword confidence based on inverse ratio
        if ratio > 10:
            subword_confidence = "very_high"  # Subtoken appears 10x more
        elif ratio > 5:
            subword_confidence = "high"  # Subtoken appears 5-10x more
        elif ratio > 3:
            subword_confidence = "medium"  # Subtoken appears 3-5x more
        elif ratio > 2:
            subword_confidence = "low"  # Subtoken appears 2-3x more
        elif ratio > 1:
            subword_confidence = "very_low"  # Subtoken appears 1-2x more
        else:
            subword_confidence = "none"  # Word appears more

        results["subword_confidence"] = subword_confidence

        # Try lowercase if capitalized
        results["validated_as_lowercase"] = False
        if word_vote_count == 0 and word and word[0].isupper() and word != word.lower():
            lowercase_results = self.validate_with_fineweb(word.lower())
            if lowercase_results["word_vote_count"] > results["word_vote_count"]:
                for key in [
                    "fineweb_valid",
                    "fineweb_exact_freq",
                    "word_vote_count",
                    "word_confidence",
                    "word_vote_sources",
                    "total_exact_matches",
                ]:
                    results[key] = lowercase_results[key]
                results["validated_as_lowercase"] = True

        return results

    def get_final_classification(
        self,
        cleaned: str,
        initial_signal: str,
        initial_confidence: str,
        corpus_results: Dict,
        has_leading_space: bool = None,
    ) -> Tuple[str, str]:
        """
        Get final classification combining rule-based and corpus-based results.

        IMPORTANT: The Ġ prefix (has_leading_space) is the PRIMARY authority.
        - has_leading_space=True  -> Token IS a standalone word (tokenizer truth)
        - has_leading_space=False -> Token is a continuation/substring

        The corpus analysis provides supporting evidence but does NOT override
        the tokenizer's encoding decision.

        Ratio = substring_count / exact_freq
        - ratio < 1: token appears more as standalone word in corpus
        - ratio > 1: token appears more as substring in corpus
        - ratio ≈ 1: ambiguous usage in corpus
        """
        if initial_signal != "NEEDS_CORPUS_ANALYSIS":
            return initial_signal, initial_confidence

        # Ġ PREFIX IS THE PRIMARY AUTHORITY
        # If token has leading space, it's definitionally a standalone word
        if has_leading_space is True:
            # Use corpus ratio to set confidence level
            substring_ratio = corpus_results.get("substring_to_word_ratio", 0)
            if substring_ratio < 0.5:
                return "WORD", "very_high"  # Corpus agrees strongly
            elif substring_ratio < 1.0:
                return "WORD", "high"  # Corpus agrees
            elif substring_ratio < 2.0:
                return "WORD", "medium"  # Corpus somewhat disagrees but Ġ wins
            else:
                return "WORD", "low"  # Corpus disagrees but Ġ still wins

        # NO Ġ PREFIX: Use corpus analysis to determine classification
        # These are typically subtokens, but could be start-of-sentence words
        # (e.g., "The", "I" without Ġ at sentence/document start)

        exact_freq = corpus_results.get("total_exact_matches", 0)
        substring_count = corpus_results.get("total_substring_occurrences", 0)
        substring_ratio = corpus_results.get("substring_to_word_ratio", 0)

        # Special case: never appears as standalone word (only as substring)
        if exact_freq == 0:
            if substring_count > 0:
                # Pure subtoken
                return "SUBTOKEN", "medium"
            else:
                # Never appears in corpus at all
                return "UNKNOWN", "none"

        # RATIO-BASED CLASSIFICATION with explicit range checks
        # Ranges are symmetric around ratio = 1.0 (equal usage)

        # WORD classifications (ratio < 1: word more common than subtoken)
        if 0.0 <= substring_ratio < 0.2:
            # Word appears 5x+ more than subtoken (inverse of >= 5.0)
            return "WORD", "very_high"

        elif 0.2 <= substring_ratio < 0.33:
            # Word appears 3-5x more than subtoken (inverse of 3.0-5.0)
            return "WORD", "high"

        elif 0.33 <= substring_ratio < 0.5:
            # Word appears 2-3x more than subtoken (inverse of 2.0-3.0)
            return "WORD", "medium"

        elif 0.5 <= substring_ratio < 0.67:
            # Word appears 1.5-2x more than subtoken (inverse of 1.5-2.0)
            return "WORD", "low"

        elif 0.67 <= substring_ratio < 0.75:
            # Word appears 1.33-1.5x more than subtoken (inverse of 1.33-1.5)
            return "WORD", "very_low"

        # AMBIGUOUS zone (very close to equal usage)
        elif 0.75 <= substring_ratio < 0.9:
            # Word slightly more common (within 10-25% of equal)
            return "AMBIGUOUS", "word_leaning"

        elif 0.9 <= substring_ratio < 1.1:
            # Nearly equal usage (within 10% of equal)
            return "AMBIGUOUS", "balanced"

        elif 1.1 <= substring_ratio < 1.33:
            # Subtoken slightly more common (within 10-33% of equal)
            return "AMBIGUOUS", "subtoken_leaning"

        # SUBTOKEN classifications (ratio > 1: subtoken more common than word)
        elif 1.33 <= substring_ratio < 1.5:
            # Subtoken appears 1.33-1.5x more than word
            return "SUBTOKEN", "very_low"

        elif 1.5 <= substring_ratio < 2.0:
            # Subtoken appears 1.5-2x more than word
            return "SUBTOKEN", "low"

        elif 2.0 <= substring_ratio < 3.0:
            # Subtoken appears 2-3x more than word
            return "SUBTOKEN", "medium"

        elif 3.0 <= substring_ratio < 5.0:
            # Subtoken appears 3-5x more than word
            return "SUBTOKEN", "high"

        elif substring_ratio >= 5.0:
            # Subtoken appears 5x+ more than word
            return "SUBTOKEN", "very_high"

        else:
            # Should never reach here, but safety fallback
            return "UNKNOWN", "none"

    def process_token(self, token: str) -> Dict:
        """Process single token through prevalidation"""
        cleaned, metadata = self.clean_token(token)
        unicode_analysis = self.analyze_unicode(cleaned)
        rule_analysis = self.rule_based_analysis(cleaned, unicode_analysis, metadata)

        # Get initial classification
        initial_signal, initial_confidence = self.get_initial_classification(
            cleaned, rule_analysis
        )

        # Corpus analysis if needed
        if initial_signal == "NEEDS_CORPUS_ANALYSIS":
            corpus_validation = self.validate_with_fineweb(cleaned)
        else:
            corpus_validation = {
                "fineweb_valid": False,
                "fineweb_exact_freq": 0,
                "fineweb_substring_count": 0,
                "word_vote_count": 0,
                "word_vote_sources": "",
                "total_exact_matches": 0,
                "subword_vote_count": 0,
                "total_substring_occurrences": 0,
                "substring_to_word_ratio": 0.0,
                "word_confidence": "none",
                "subword_confidence": "none",
                "validated_as_lowercase": False,
            }

        # Final classification (pass has_leading_space for Ġ-aware classification)
        primary_signal, confidence_level = self.get_final_classification(
            cleaned,
            initial_signal,
            initial_confidence,
            corpus_validation,
            has_leading_space=metadata.get("has_leading_space", False),
        )

        # Combine results
        result = {
            **metadata,
            **unicode_analysis,
            **rule_analysis,
            **corpus_validation,
            "primary_signal": primary_signal,
            "confidence_level": confidence_level,
        }

        return result

    def process_chunk(self, tokens: List[Dict]) -> List[Dict]:
        """Process chunk of tokens with progress logging"""
        results = []
        chunk_size = len(tokens)

        logger.info(f"[Worker {self.worker_id}] Processing {chunk_size:,} tokens...")

        last_log_time = time.time()
        for idx, token_data in enumerate(tokens):
            token_string = token_data["token_string"]
            validation_result = self.process_token(token_string)

            result = {**token_data, **validation_result}
            results.append(result)

            if time.time() - last_log_time > 30:
                progress_pct = (idx + 1) / chunk_size * 100
                logger.info(
                    f"[Worker {self.worker_id}] Progress: {idx + 1:,}/{chunk_size:,} ({progress_pct:.1f}%)"
                )
                last_log_time = time.time()

        logger.info(f"[Worker {self.worker_id}] Completed {chunk_size:,} tokens")
        return results


def worker_function(args):
    """Worker function for multiprocessing"""
    worker_id, chunk, fineweb_dir = args
    worker = PrevalidationWorker(worker_id, fineweb_dir)
    return worker.process_chunk(chunk)


def run_prevalidation(
    input_csv: str, output_csv: str, fineweb_dir: str, num_workers: int = 8
) -> pd.DataFrame:
    """
    Run prevalidation with FineWeb corpus
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: PREVALIDATION WITH FINEWEB CORPUS")
    logger.info("=" * 80)

    logger.info(f"Loading tokens from: {input_csv}")
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded {len(df):,} tokens")

    logger.info(f"FineWeb corpus directory: {fineweb_dir}")

    tokens = df.to_dict("records")

    # Split into chunks
    chunk_size = max(1, len(tokens) // num_workers)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunks.append(tokens[i : i + chunk_size])

    actual_workers = len(chunks)
    logger.info(f"Split into {actual_workers} chunks for {num_workers} workers")

    # Process in parallel
    logger.info(f"Starting {actual_workers} parallel workers...")

    start_time = time.time()
    with mp.Pool(processes=num_workers) as pool:
        worker_args = [(i, chunk, fineweb_dir) for i, chunk in enumerate(chunks)]
        results = pool.map(worker_function, worker_args)
    elapsed = time.time() - start_time

    # Flatten results
    all_results = [item for sublist in results for item in sublist]

    logger.info(f"Processed {len(all_results):,} tokens in {elapsed / 60:.1f} minutes")
    logger.info(f"Average speed: {len(all_results) / elapsed:.1f} tokens/sec")

    # Create DataFrame
    result_df = pd.DataFrame(all_results)

    # Ensure output directory exists
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    # Save
    logger.info(f"Saving to: {output_csv}")
    result_df.to_csv(output_csv, index=False)

    # Print statistics
    logger.info("=" * 80)
    logger.info("PREVALIDATION STATISTICS (FINEWEB - PURE RATIO-BASED)")
    logger.info("=" * 80)
    logger.info(f"Total tokens: {len(result_df):,}")

    logger.info(f"\n=== PRIMARY SIGNAL CLASSIFICATION ===")
    signal_counts = result_df["primary_signal"].value_counts()
    for signal, count in signal_counts.items():
        pct = count / len(result_df) * 100
        logger.info(f"  {signal:25s}: {count:7,} ({pct:5.2f}%)")

    logger.info(f"\n=== CONFIDENCE LEVEL DISTRIBUTION ===")
    conf_counts = result_df["confidence_level"].value_counts()
    for conf, count in conf_counts.items():
        pct = count / len(result_df) * 100
        logger.info(f"  {conf:10s}: {count:7,} ({pct:5.2f}%)")

    # Word statistics
    logger.info(f"\n=== WORD STATISTICS (from FineWeb) ===")
    word_tokens = result_df[result_df["primary_signal"] == "WORD"]
    if len(word_tokens) > 0:
        logger.info(f"Total WORD tokens: {len(word_tokens):,}")
        logger.info(
            f"  Mean exact frequency: {word_tokens['total_exact_matches'].mean():.1f}"
        )
        logger.info(
            f"  Median exact frequency: {word_tokens['total_exact_matches'].median():.1f}"
        )
        logger.info(
            f"  Mean substring occurrences: {word_tokens['total_substring_occurrences'].mean():.1f}"
        )
        logger.info(
            f"  Mean substring/word ratio: {word_tokens['substring_to_word_ratio'].mean():.2f}"
        )

    # Subtoken statistics
    logger.info(f"\n=== SUBTOKEN STATISTICS ===")
    subtoken_tokens = result_df[result_df["primary_signal"] == "SUBTOKEN"]
    if len(subtoken_tokens) > 0:
        logger.info(f"Total SUBTOKEN tokens: {len(subtoken_tokens):,}")
        logger.info(
            f"  Mean exact frequency: {subtoken_tokens['total_exact_matches'].mean():.1f}"
        )
        logger.info(
            f"  Mean substring occurrences: {subtoken_tokens['total_substring_occurrences'].mean():.1f}"
        )
        logger.info(
            f"  Mean substring/word ratio: {subtoken_tokens['substring_to_word_ratio'].mean():.2f}"
        )

    logger.info(f"\nStage 1 complete")

    return result_df


def main():
    """Test Stage 1 with FineWeb"""
    import argparse

    SCRIPT_DIR = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Stage 1: Prevalidation with FineWeb")
    parser.add_argument(
        "--input",
        default=str(SCRIPT_DIR / "token_dataset" / "all_tokens_complete.csv"),
        help="Input CSV",
    )
    parser.add_argument(
        "--output",
        default=str(
            SCRIPT_DIR / "fineweb_validated" / "all_tokens_fineweb_prevalidated.csv"
        ),
        help="Output CSV",
    )
    parser.add_argument(
        "--fineweb-dir",
        default=str(SCRIPT_DIR.parent / "fineweb_sample" / "processed"),
        help="FineWeb processed data directory",
    )
    parser.add_argument("--workers", type=int, default=16, help="Number of workers")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    df = run_prevalidation(args.input, args.output, args.fineweb_dir, args.workers)

    # Print samples
    logger.info("\n" + "=" * 80)
    logger.info("SAMPLE ANALYSIS")
    logger.info("=" * 80)

    logger.info("\n=== Sample WORDS (low ratio) ===")
    words = df[df["primary_signal"] == "WORD"].nsmallest(15, "substring_to_word_ratio")
    for _, row in words.iterrows():
        logger.info(
            f"  '{row['cleaned_token']}': freq={row['total_exact_matches']}, "
            f"substr={row['total_substring_occurrences']}, ratio={row['substring_to_word_ratio']:.2f}"
        )

    logger.info("\n=== Sample SUBTOKENS (high ratio) ===")
    subtokens = df[df["primary_signal"] == "SUBTOKEN"].nlargest(
        15, "substring_to_word_ratio"
    )
    for _, row in subtokens.iterrows():
        logger.info(
            f"  '{row['cleaned_token']}': freq={row['total_exact_matches']}, "
            f"substr={row['total_substring_occurrences']}, ratio={row['substring_to_word_ratio']:.2f}"
        )

    logger.info("\n=== Checking specific examples ===")
    test_tokens = ["ortunatley", "tradem", "remn", "supernat", "occas"]
    for token in test_tokens:
        row = df[df["cleaned_token"] == token]
        if len(row) > 0:
            row = row.iloc[0]
            logger.info(
                f"  '{token}': signal={row['primary_signal']}, "
                f"freq={row['total_exact_matches']}, "
                f"substr={row['total_substring_occurrences']}, "
                f"ratio={row['substring_to_word_ratio']:.2f}"
            )


if __name__ == "__main__":
    main()
