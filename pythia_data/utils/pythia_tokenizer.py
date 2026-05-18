#!/usr/bin/env python3
"""
Wrapper around the Pythia (GPT-NeoX) tokenizer with single-token utilities.

Single-token status uses a round-trip decode->encode check rather than
inspecting raw vocab keys, since Ġ-prefixed tokens require the round-trip
to behave correctly.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from transformers import AutoTokenizer


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PythiaTokenizer:
    """
    Wrapper around HuggingFace Pythia tokenizer with verification utilities

    Attributes:
        model_name: HuggingFace model name (e.g., "EleutherAI/pythia-70m")
        tokenizer: Loaded HuggingFace tokenizer
        vocab: Dict mapping token_string -> token_id (e.g., vocab['Ġthe'] -> 253)
        vocab_size: Size of vocabulary
        verification_cache: Cache of single-token verification results
    """

    def __init__(self, model_name: str = "EleutherAI/pythia-70m"):
        """
        Initialize tokenizer wrapper

        Args:
            model_name: HuggingFace model identifier for Pythia
        """
        self.model_name = model_name
        self.tokenizer = None
        self.vocab: Dict[str, int] = {}
        self.vocab_size: int = 0
        self.verification_cache: Dict[str, Dict] = {}
        self._loaded: bool = False

        logger.info(f"Initialized PythiaTokenizer for model: {model_name}")

    def load(self) -> None:
        """
        Load tokenizer from HuggingFace

        Raises:
            Exception: If tokenizer cannot be loaded
        """
        logger.info(f"Loading tokenizer from HuggingFace: {self.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.vocab = self.tokenizer.get_vocab()
            self.vocab_size = len(self.vocab)

            logger.info(
                f"Loaded tokenizer successfully. Vocabulary size: {self.vocab_size:,}"
            )
            self._loaded = True

        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise

    def is_single_token(
        self, text: str, test_with_space: bool = True
    ) -> Tuple[bool, bool, Dict[str, List[int]]]:
        """
        Verify if text tokenizes to a single token

        Tests both with and without leading space since tokens may differ:
        - "the" might be token_id 253
        - " the" might be token_id 254 (different token!)

        Args:
            text: Text to verify
            test_with_space: Whether to also test with leading space

        Returns:
            Tuple of (is_single_no_space, is_single_with_space, token_info)
            where token_info = {
                'no_space': [token_ids],
                'with_space': [token_ids]
            }
        """
        if not self._loaded:
            raise ValueError("Tokenizer not loaded. Call load() first.")

        # Check cache
        if text in self.verification_cache:
            cached = self.verification_cache[text]
            if test_with_space:
                return (
                    cached["is_single_no_space"],
                    cached["is_single_with_space"],
                    {
                        "no_space": cached["tokens_no_space"],
                        "with_space": cached["tokens_with_space"],
                    },
                )
            else:
                return (
                    cached["is_single_no_space"],
                    False,
                    {"no_space": cached["tokens_no_space"], "with_space": []},
                )

        # Tokenize without space
        tokens_no_space = self.tokenizer.encode(text, add_special_tokens=False)
        is_single_no_space = len(tokens_no_space) == 1

        # Tokenize with space (always compute for cache)
        tokens_with_space = self.tokenizer.encode(" " + text, add_special_tokens=False)
        is_single_with_space = len(tokens_with_space) == 1

        # Cache complete results
        self.verification_cache[text] = {
            "is_single_no_space": is_single_no_space,
            "is_single_with_space": is_single_with_space,
            "tokens_no_space": tokens_no_space,
            "tokens_with_space": tokens_with_space,
        }

        token_info = {"no_space": tokens_no_space, "with_space": tokens_with_space}

        return is_single_no_space, is_single_with_space, token_info

    def batch_verify_single_tokens(
        self, text_list: List[str], test_with_space: bool = True
    ) -> Dict[str, Dict]:
        """
        Verify list of texts for single-token status

        Args:
            text_list: List of texts to verify
            test_with_space: Whether to test with leading space

        Returns:
            Dict mapping text -> verification result with keys:
                - is_single_no_space: bool
                - is_single_with_space: bool
                - tokens_no_space: List[int]
                - tokens_with_space: List[int]
                - recommendation: str (which version to use)
        """
        if not self._loaded:
            raise ValueError("Tokenizer not loaded. Call load() first.")

        results = {}

        for text in text_list:
            is_single_no, is_single_with, token_info = self.is_single_token(
                text, test_with_space
            )

            # Determine recommendation
            if is_single_no and is_single_with:
                recommendation = "both_single"
            elif is_single_no:
                recommendation = "use_no_space"
            elif is_single_with:
                recommendation = "use_with_space"
            else:
                recommendation = "multi_token"

            results[text] = {
                "is_single_no_space": is_single_no,
                "is_single_with_space": is_single_with,
                "tokens_no_space": token_info["no_space"],
                "tokens_with_space": token_info["with_space"],
                "recommendation": recommendation,
            }

        return results

    def get_token_by_id(self, token_id: int) -> str:
        """
        Get string representation of token_id

        Args:
            token_id: Token ID from vocabulary

        Returns:
            String representation of token

        Raises:
            ValueError: If token_id not in vocabulary
        """
        if not self._loaded:
            raise ValueError("Tokenizer not loaded. Call load() first.")

        # Decode single token
        decoded = self.tokenizer.decode([token_id])
        return decoded

    def get_token_ids_by_string(self, token_string: str) -> List[int]:
        """
        Get token IDs for a string (may be multiple)

        Args:
            token_string: String to tokenize

        Returns:
            List of token IDs
        """
        if not self._loaded:
            raise ValueError("Tokenizer not loaded. Call load() first.")

        return self.tokenizer.encode(token_string, add_special_tokens=False)

    def tokenize_and_analyze(self, text: str) -> Dict:
        """
        Tokenize text and return detailed analysis

        Args:
            text: Text to tokenize

        Returns:
            Dict with keys:
                - token_ids: List of token IDs
                - tokens: List of decoded token strings
                - num_tokens: Number of tokens
                - original_text: Original input text
        """
        if not self._loaded:
            raise ValueError("Tokenizer not loaded. Call load() first.")

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        return {
            "token_ids": token_ids,
            "tokens": tokens,
            "num_tokens": len(token_ids),
            "original_text": text,
        }

    def export_vocabulary(self, output_path: Path) -> None:
        """
        Export full vocabulary to JSON file

        Args:
            output_path: Path to save vocabulary JSON
        """
        if not self._loaded:
            raise ValueError("Tokenizer not loaded. Call load() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create exportable vocab (token_id -> string)
        vocab_export = {
            str(token_id): token_str for token_str, token_id in self.vocab.items()
        }

        with open(output_path, "w") as f:
            json.dump(vocab_export, f, indent=2)

        logger.info(
            f"Exported vocabulary ({len(vocab_export)} tokens) to {output_path}"
        )

    def get_vocabulary_statistics(self) -> Dict:
        """
        Compute statistics about vocabulary

        Returns:
            Dict with vocabulary statistics
        """
        if not self._loaded:
            raise ValueError("Tokenizer not loaded. Call load() first.")

        # Analyze token strings
        token_strings = list(self.vocab.keys())

        alphabetic_count = sum(
            1 for t in token_strings if t.strip() and t.strip().isalpha()
        )
        numeric_count = sum(
            1 for t in token_strings if t.strip() and t.strip().isdigit()
        )
        alphanumeric_count = sum(
            1 for t in token_strings if t.strip() and t.strip().isalnum()
        )
        whitespace_count = sum(1 for t in token_strings if t.isspace())
        special_char_count = sum(
            1 for t in token_strings if not t.strip().isalnum() and not t.isspace()
        )

        return {
            "total_vocab_size": self.vocab_size,
            "alphabetic_tokens": alphabetic_count,
            "numeric_tokens": numeric_count,
            "alphanumeric_tokens": alphanumeric_count,
            "whitespace_tokens": whitespace_count,
            "special_char_tokens": special_char_count,
            "model_name": self.model_name,
        }

    def filter_vocabulary_by_pattern(
        self, pattern_func: callable, test_with_space: bool = False
    ) -> List[Tuple[str, int]]:
        """
        Filter vocabulary by custom pattern function

        Args:
            pattern_func: Function that takes token_string and returns bool
            test_with_space: Whether to test tokens with leading space removed

        Returns:
            List of (token_string, token_id) tuples matching pattern
        """
        if not self._loaded:
            raise ValueError("Tokenizer not loaded. Call load() first.")

        matching = []

        for token_str, token_id in self.vocab.items():
            test_str = token_str.lstrip() if test_with_space else token_str
            if pattern_func(test_str):
                matching.append((token_str, token_id))

        logger.info(f"Found {len(matching)} tokens matching pattern")
        return matching

    def clear_cache(self) -> None:
        """Clear verification cache"""
        self.verification_cache.clear()
        logger.info("Cleared verification cache")

    def __repr__(self) -> str:
        if self._loaded:
            return f"PythiaTokenizer(model={self.model_name}, vocab_size={self.vocab_size:,}, loaded=True)"
        else:
            return f"PythiaTokenizer(model={self.model_name}, loaded=False)"
