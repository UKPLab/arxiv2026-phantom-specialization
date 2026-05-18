#!/usr/bin/env python3
"""
Load and query Pile token frequency data.

Supports CSV and TSV formats (auto-detected via pandas).
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FrequencyDataLoader:
    """
    Load and query Pile corpus token frequency data

    Attributes:
        file_path: Path to merged token frequencies file (CSV or TSV)
        data: DataFrame with columns [token_id, token_string, count]
        token_to_freq: Dict mapping token_id -> count
        token_to_string: Dict mapping token_id -> string representation
        string_to_token: Dict mapping string representation -> token_id
        total_tokens: Total number of tokens in Pile corpus
    """

    def __init__(self, file_path: str = None):
        """
        Initialize loader with path to frequency file

        Args:
            file_path: Path to merged token frequencies (CSV or TSV)
        """
        if file_path is None:
            file_path = str(
                Path(__file__).resolve().parent.parent
                / "pile_frequencies"
                / "merged_token_frequencies.tsv"
            )
        self.file_path = Path(file_path)
        self.data: Optional[pd.DataFrame] = None
        self.token_to_freq: Dict[int, int] = {}
        self.token_to_string: Dict[int, str] = {}
        self.string_to_token: Dict[str, int] = {}
        self.total_tokens: int = 0
        self._loaded: bool = False

        # Validate path exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"Frequency file not found: {self.file_path}")

        logger.info(f"Initialized FrequencyDataLoader with path: {self.file_path}")

    def load(self) -> None:
        """
        Load frequency file into memory and build lookup structures

        Raises:
            ValueError: If file format is invalid
            FileNotFoundError: If file doesn't exist
        """
        logger.info("Loading frequency data from file...")

        try:
            # Detect file format from extension
            if self.file_path.suffix == ".csv":
                separator = ","
                logger.info("Detected CSV format")
            elif self.file_path.suffix == ".tsv":
                separator = "\t"
                logger.info("Detected TSV format")
            else:
                # Default to CSV
                separator = ","
                logger.warning(
                    f"Unknown extension {self.file_path.suffix}, assuming CSV"
                )

            # token_string forced to str so values like "nan" are not coerced to float NaN
            self.data = pd.read_csv(
                self.file_path,
                sep=separator,
                dtype={"token_id": int, "count": int, "token_string": str},
                quoting=csv.QUOTE_ALL,
                encoding="utf-8",
                keep_default_na=False,
                na_values=[],
            )

            # Check for required columns
            required_cols = {"token_id", "token_string", "count"}
            if not required_cols.issubset(self.data.columns):
                raise ValueError(
                    f"File missing required columns.\n"
                    f"Expected: {required_cols}\n"
                    f"Got: {set(self.data.columns)}"
                )

            logger.info(f"Loaded {len(self.data)} tokens from file")

            # Build lookup dictionaries
            self._build_lookups()

            # Calculate total tokens in corpus
            self.total_tokens = int(self.data["count"].sum())
            logger.info(f"Total tokens in Pile corpus: {self.total_tokens:,}")

            self._loaded = True
            logger.info("Frequency data loaded successfully")

        except Exception as e:
            logger.error(f"Error loading frequency data: {e}")
            raise

    def _build_lookups(self) -> None:
        """
        Build efficient lookup dictionaries
        """
        logger.info("Building lookup dictionaries...")

        for _, row in self.data.iterrows():
            token_id = int(row["token_id"])
            count = int(row["count"])
            token_string = row["token_string"]

            self.token_to_freq[token_id] = count
            self.token_to_string[token_id] = token_string

            if token_string not in self.string_to_token:
                self.string_to_token[token_string] = token_id

        logger.info(
            f"Built lookup dicts: {len(self.token_to_freq)} token_ids, {len(self.string_to_token)} unique strings"
        )

    def get_frequency(self, token_id: int) -> int:
        """
        Get raw count for token_id

        Args:
            token_id: Token ID from Pythia vocabulary

        Returns:
            Raw count of occurrences in Pile corpus

        Raises:
            ValueError: If token_id not found or data not loaded
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")

        if token_id not in self.token_to_freq:
            raise ValueError(f"Token ID {token_id} not found in frequency data")

        return self.token_to_freq[token_id]

    def get_frequency_by_string(self, token_string: str) -> int:
        """
        Get raw count by token string representation

        Args:
            token_string: String representation of token (e.g., " the", "\\n")

        Returns:
            Raw count of occurrences in Pile corpus

        Raises:
            ValueError: If token_string not found or data not loaded
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")

        if token_string not in self.string_to_token:
            raise ValueError(
                f"Token string '{token_string}' not found in frequency data"
            )

        token_id = self.string_to_token[token_string]
        return self.token_to_freq[token_id]

    def get_token_info(self, token_id: int) -> Dict:
        """
        Get complete information for a token

        Args:
            token_id: Token ID from Pythia vocabulary

        Returns:
            Dict with keys: token_id, string, count, freq_per_million
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")

        if token_id not in self.token_to_freq:
            raise ValueError(f"Token ID {token_id} not found in frequency data")

        count = self.token_to_freq[token_id]
        string = self.token_to_string[token_id]
        freq_per_million = (count / self.total_tokens) * 1e6

        return {
            "token_id": token_id,
            "string": string,
            "count": count,
            "freq_per_million": freq_per_million,
        }

    def get_top_n(self, n: int = 1000) -> List[Tuple[int, str, int]]:
        """
        Get top N most frequent tokens

        Args:
            n: Number of top tokens to return

        Returns:
            List of tuples: (token_id, string, count), sorted by count descending
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")

        # Sort by count descending
        top_data = self.data.nlargest(n, "count")

        result = [
            (int(row["token_id"]), row["token_string"], int(row["count"]))
            for _, row in top_data.iterrows()
        ]

        return result

    def get_bottom_n(self, n: int = 1000) -> List[Tuple[int, str, int]]:
        """
        Get bottom N least frequent tokens

        Args:
            n: Number of bottom tokens to return

        Returns:
            List of tuples: (token_id, string, count), sorted by count ascending
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")

        # Sort by count ascending
        bottom_data = self.data.nsmallest(n, "count")

        result = [
            (int(row["token_id"]), row["token_string"], int(row["count"]))
            for _, row in bottom_data.iterrows()
        ]

        return result

    def get_all_tokens_above_threshold(self, min_count: int = 5) -> List[int]:
        """
        Get all token IDs with count >= threshold

        Args:
            min_count: Minimum count threshold (default: 5 for statistical significance)

        Returns:
            List of token_ids meeting threshold
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")

        filtered = self.data[self.data["count"] >= min_count]
        token_ids = filtered["token_id"].tolist()

        logger.info(f"Found {len(token_ids)} tokens with count >= {min_count}")
        return token_ids

    def compute_frequency_stats(self) -> Dict:
        """
        Compute statistics on frequency distribution

        Returns:
            Dict with distribution statistics
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")

        counts = self.data["count"].values

        stats = {
            "total_unique_tokens": len(self.data),
            "total_token_occurrences": int(self.total_tokens),
            "mean_count": float(np.mean(counts)),
            "median_count": float(np.median(counts)),
            "std_count": float(np.std(counts)),
            "min_count": int(np.min(counts)),
            "max_count": int(np.max(counts)),
            "percentiles": {
                "p1": float(np.percentile(counts, 1)),
                "p5": float(np.percentile(counts, 5)),
                "p10": float(np.percentile(counts, 10)),
                "p25": float(np.percentile(counts, 25)),
                "p50": float(np.percentile(counts, 50)),
                "p75": float(np.percentile(counts, 75)),
                "p90": float(np.percentile(counts, 90)),
                "p95": float(np.percentile(counts, 95)),
                "p99": float(np.percentile(counts, 99)),
            },
            "top10_coverage": float(
                np.sort(counts)[-10:].sum() / self.total_tokens * 100
            ),
            "top100_coverage": float(
                np.sort(counts)[-100:].sum() / self.total_tokens * 100
            ),
            "top1000_coverage": float(
                np.sort(counts)[-1000:].sum() / self.total_tokens * 100
            ),
        }

        return stats

    def export_stats_to_json(self, output_path: Path) -> None:
        """
        Export frequency statistics to JSON file

        Args:
            output_path: Path to save JSON file
        """
        import json

        stats = self.compute_frequency_stats()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Exported frequency statistics to {output_path}")

    def get_tokens_by_ids(self, token_ids: List[int]) -> pd.DataFrame:
        """
        Get DataFrame of tokens for specified token IDs

        Args:
            token_ids: List of token IDs to retrieve

        Returns:
            DataFrame with token information
        """
        if not self._loaded:
            raise ValueError("Data not loaded. Call load() first.")

        result = self.data[self.data["token_id"].isin(token_ids)].copy()
        return result

    def __repr__(self) -> str:
        if self._loaded:
            return f"FrequencyDataLoader(tokens={len(self.data)}, total_occurrences={self.total_tokens:,}, loaded=True)"
        else:
            return f"FrequencyDataLoader(path={self.file_path}, loaded=False)"
