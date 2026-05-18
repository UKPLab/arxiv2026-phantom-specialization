#!/usr/bin/env python3
"""
Log-frequency and percentile transforms for Pile token frequencies.

    log_freq = log10((count / total_tokens) * 1e6)

(frequency per million in log space).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from .frequency_loader import FrequencyDataLoader


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FrequencyTransformer:
    """
    Transform raw frequencies to log-frequencies and compute percentiles

    Formula: log_freq = log10((count / total_tokens) * 1e6)
    This gives frequency per million tokens in log-space.

    Attributes:
        freq_loader: FrequencyDataLoader instance
        log_frequencies: Dict mapping token_id -> log_frequency
        normalized_frequencies: Dict mapping token_id -> freq_per_million
        percentiles: Dict mapping token_id -> percentile (0-100)
    """

    def __init__(self, frequency_loader: FrequencyDataLoader):
        """
        Initialize transformer with frequency loader

        Args:
            frequency_loader: Loaded FrequencyDataLoader instance

        Raises:
            ValueError: If frequency_loader is not loaded
        """
        if not frequency_loader._loaded:
            raise ValueError(
                "FrequencyDataLoader must be loaded before creating transformer"
            )

        self.freq_loader = frequency_loader
        self.log_frequencies: Dict[int, float] = {}
        self.normalized_frequencies: Dict[int, float] = {}
        self.percentiles: Dict[int, float] = {}
        self._computed: bool = False

        logger.info("Initialized FrequencyTransformer")

    def compute_log_frequencies(self) -> None:
        """
        Apply log10 transformation to all frequencies

        Formula: log10((count / total_tokens) * 1e6)
        - Normalizes to per-million tokens
        - Applies log10 transformation (no epsilon needed; minimum freq_per_million > 0)
        """
        logger.info("Computing log-frequencies for all tokens...")

        total_tokens = self.freq_loader.total_tokens

        for token_id, count in self.freq_loader.token_to_freq.items():
            # Normalize to frequency per million
            freq_per_million = (count / total_tokens) * 1e6

            # Log transform (no epsilon needed: min count >= 1, so freq_per_million > 0)
            log_freq = np.log10(freq_per_million)

            self.normalized_frequencies[token_id] = freq_per_million
            self.log_frequencies[token_id] = log_freq

        self._computed = True
        logger.info(f"Computed log-frequencies for {len(self.log_frequencies)} tokens")

    def get_log_frequency(self, token_id: int) -> float:
        """
        Get log-frequency for a specific token

        Args:
            token_id: Token ID from vocabulary

        Returns:
            Log-frequency value

        Raises:
            ValueError: If log-frequencies not computed or token_id not found
        """
        if not self._computed:
            raise ValueError(
                "Log-frequencies not computed. Call compute_log_frequencies() first."
            )

        if token_id not in self.log_frequencies:
            raise ValueError(f"Token ID {token_id} not found in log-frequencies")

        return self.log_frequencies[token_id]

    def get_normalized_frequency(self, token_id: int) -> float:
        """
        Get normalized frequency (per million) for a specific token

        Args:
            token_id: Token ID from vocabulary

        Returns:
            Frequency per million tokens

        Raises:
            ValueError: If frequencies not computed or token_id not found
        """
        if not self._computed:
            raise ValueError(
                "Frequencies not computed. Call compute_log_frequencies() first."
            )

        if token_id not in self.normalized_frequencies:
            raise ValueError(f"Token ID {token_id} not found in normalized frequencies")

        return self.normalized_frequencies[token_id]

    def compute_percentiles(
        self, token_ids: Optional[List[int]] = None
    ) -> Dict[int, float]:
        """
        Compute frequency percentiles for tokens

        Args:
            token_ids: Optional list of token IDs to compute percentiles for.
                      If None, computes for all tokens.

        Returns:
            Dict mapping token_id -> percentile (0-100)
        """
        if not self._computed:
            raise ValueError(
                "Log-frequencies not computed. Call compute_log_frequencies() first."
            )

        logger.info("Computing percentiles...")

        # Get token IDs to process
        if token_ids is None:
            token_ids = list(self.log_frequencies.keys())

        # Get log-frequencies for these tokens
        log_freqs = [self.log_frequencies[tid] for tid in token_ids]

        # Compute percentile rank for each token
        self.percentiles = {}
        for token_id in token_ids:
            token_log_freq = self.log_frequencies[token_id]
            # Percentile: percentage of tokens with frequency <= this token's frequency
            percentile = (
                np.sum(np.array(log_freqs) <= token_log_freq) / len(log_freqs)
            ) * 100
            self.percentiles[token_id] = percentile

        logger.info(f"Computed percentiles for {len(self.percentiles)} tokens")
        return self.percentiles

    def get_percentile(self, token_id: int) -> float:
        """
        Get percentile for a specific token

        Args:
            token_id: Token ID from vocabulary

        Returns:
            Percentile value (0-100)

        Raises:
            ValueError: If percentiles not computed or token_id not found
        """
        if not self.percentiles:
            raise ValueError(
                "Percentiles not computed. Call compute_percentiles() first."
            )

        if token_id not in self.percentiles:
            raise ValueError(f"Token ID {token_id} not found in percentiles")

        return self.percentiles[token_id]

    def get_frequency_statistics(self) -> Dict:
        """
        Compute comprehensive frequency statistics

        Returns:
            Dict with statistics in both raw and log-space
        """
        if not self._computed:
            raise ValueError(
                "Log-frequencies not computed. Call compute_log_frequencies() first."
            )

        log_freqs = np.array(list(self.log_frequencies.values()))
        norm_freqs = np.array(list(self.normalized_frequencies.values()))
        raw_freqs = np.array(list(self.freq_loader.token_to_freq.values()))

        stats = {
            "raw_frequencies": {
                "mean": float(np.mean(raw_freqs)),
                "median": float(np.median(raw_freqs)),
                "std": float(np.std(raw_freqs)),
                "min": float(np.min(raw_freqs)),
                "max": float(np.max(raw_freqs)),
                "total": float(np.sum(raw_freqs)),
            },
            "normalized_frequencies": {
                "mean": float(np.mean(norm_freqs)),
                "median": float(np.median(norm_freqs)),
                "std": float(np.std(norm_freqs)),
                "min": float(np.min(norm_freqs)),
                "max": float(np.max(norm_freqs)),
            },
            "log_frequencies": {
                "mean": float(np.mean(log_freqs)),
                "median": float(np.median(log_freqs)),
                "std": float(np.std(log_freqs)),
                "min": float(np.min(log_freqs)),
                "max": float(np.max(log_freqs)),
                "range": float(np.max(log_freqs) - np.min(log_freqs)),
            },
            "percentiles": {
                f"p{p}": float(np.percentile(log_freqs, p))
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            },
        }

        return stats

    def get_tokens_in_log_range(
        self, min_log_freq: float, max_log_freq: float, include_upper: bool = False
    ) -> List[int]:
        """
        Get all token IDs with log-frequency in specified range

        Uses half-open interval [min, max) by default to prevent double-counting
        when calling with adjacent ranges.

        Args:
            min_log_freq: Minimum log-frequency (inclusive)
            max_log_freq: Maximum log-frequency (exclusive, unless include_upper=True)
            include_upper: If True, use closed interval [min, max] (for final bin)

        Returns:
            List of token IDs in range
        """
        if not self._computed:
            raise ValueError(
                "Log-frequencies not computed. Call compute_log_frequencies() first."
            )

        if include_upper:
            matching = [
                token_id
                for token_id, log_freq in self.log_frequencies.items()
                if min_log_freq <= log_freq <= max_log_freq
            ]
        else:
            matching = [
                token_id
                for token_id, log_freq in self.log_frequencies.items()
                if min_log_freq <= log_freq < max_log_freq
            ]

        logger.info(
            f"Found {len(matching)} tokens in log-frequency range [{min_log_freq:.2f}, {max_log_freq:.2f}{']' if include_upper else ')'}"
        )
        return matching

    def get_tokens_in_percentile_range(
        self, min_percentile: float, max_percentile: float, include_upper: bool = False
    ) -> List[int]:
        """
        Get all token IDs with percentile in specified range

        Uses half-open interval [min, max) by default to prevent double-counting
        when calling with adjacent ranges.

        Args:
            min_percentile: Minimum percentile (0-100, inclusive)
            max_percentile: Maximum percentile (0-100, exclusive, unless include_upper=True)
            include_upper: If True, use closed interval [min, max] (for final bin)

        Returns:
            List of token IDs in range
        """
        if not self.percentiles:
            raise ValueError(
                "Percentiles not computed. Call compute_percentiles() first."
            )

        if include_upper:
            matching = [
                token_id
                for token_id, percentile in self.percentiles.items()
                if min_percentile <= percentile <= max_percentile
            ]
        else:
            matching = [
                token_id
                for token_id, percentile in self.percentiles.items()
                if min_percentile <= percentile < max_percentile
            ]

        logger.info(
            f"Found {len(matching)} tokens in percentile range [{min_percentile}, {max_percentile}{']' if include_upper else ')'}"
        )
        return matching

    def export_statistics(self, output_path: Path) -> None:
        """
        Export frequency statistics to JSON

        Args:
            output_path: Path to save statistics JSON
        """
        if not self._computed:
            raise ValueError(
                "Log-frequencies not computed. Call compute_log_frequencies() first."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = self.get_frequency_statistics()

        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Exported frequency statistics to {output_path}")

    def export_transformed_frequencies(
        self, output_path: Path, include_percentiles: bool = True
    ) -> None:
        """
        Export all transformed frequencies to JSON

        Args:
            output_path: Path to save transformed frequencies JSON
            include_percentiles: Whether to include percentile information
        """
        if not self._computed:
            raise ValueError(
                "Log-frequencies not computed. Call compute_log_frequencies() first."
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {}
        for token_id in self.log_frequencies.keys():
            token_string = self.freq_loader.token_to_string[token_id]
            raw_count = self.freq_loader.token_to_freq[token_id]

            token_data = {
                "token_string": token_string,
                "raw_count": int(raw_count),
                "freq_per_million": float(self.normalized_frequencies[token_id]),
                "log_frequency": float(self.log_frequencies[token_id]),
            }

            if include_percentiles and token_id in self.percentiles:
                token_data["percentile"] = float(self.percentiles[token_id])

            export_data[str(token_id)] = token_data

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(
            f"Exported transformed frequencies for {len(export_data)} tokens to {output_path}"
        )

    def get_token_info_comprehensive(self, token_id: int) -> Dict:
        """
        Get comprehensive frequency information for a token

        Args:
            token_id: Token ID from vocabulary

        Returns:
            Dict with all frequency information
        """
        if not self._computed:
            raise ValueError(
                "Log-frequencies not computed. Call compute_log_frequencies() first."
            )

        if token_id not in self.log_frequencies:
            raise ValueError(f"Token ID {token_id} not found")

        info = {
            "token_id": token_id,
            "token_string": self.freq_loader.token_to_string[token_id],
            "raw_count": int(self.freq_loader.token_to_freq[token_id]),
            "freq_per_million": float(self.normalized_frequencies[token_id]),
            "log_frequency": float(self.log_frequencies[token_id]),
        }

        if token_id in self.percentiles:
            info["percentile"] = float(self.percentiles[token_id])

        return info

    def __repr__(self) -> str:
        if self._computed:
            return f"FrequencyTransformer(tokens={len(self.log_frequencies)}, computed=True)"
        else:
            return "FrequencyTransformer(computed=False)"
