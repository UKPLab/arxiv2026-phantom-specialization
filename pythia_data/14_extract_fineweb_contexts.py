#!/usr/bin/env python3
"""
Extract FineWeb context windows (Ġ-aware).

Uses the Ġ prefix (has_leading_space) as the PRIMARY authority for matching:
- has_leading_space=True  -> Match as STANDALONE WORD (space-delimited)
- has_leading_space=False -> Match as SUBSTRING (within words)

This preserves the tokenizer's actual encoding decisions rather than
guessing from corpus statistics.

Architecture:
    - All parquet files processed in parallel (1 worker per parquet)
    - Per-parquet context limit prevents source bias
    - Early stopping per parquet when all tokens satisfied
    - Inverted matching: split text ONCE, dict-lookup words, enumerate substrings
    - Final random sampling from pooled per-parquet results

Input:
    - fineweb_validated/all_tokens_fineweb_prevalidated.csv
    - fineweb_sample/sample/sample/10BT/*.parquet

Output:
    - context_windows/token_contexts.pkl
    - context_windows/token_contexts_sample.json
    - context_windows/extraction_stats.json
"""

import json
import pickle
import random
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pyarrow.parquet as pq
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TOKEN_CSV = (
    SCRIPT_DIR / "fineweb_validated" / "all_tokens_fineweb_prevalidated.csv"
)
DEFAULT_FINEWEB_DIR = Path("fineweb_sample/sample/sample/10BT")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "context_windows"

DEFAULT_MAX_CONTEXTS = 100  # Final contexts per token
DEFAULT_PER_PARQUET_LIMIT = 100  # Contexts per token per parquet file
DEFAULT_WINDOW_SIZE = 10
DEFAULT_RANDOM_SEED = 42
DEFAULT_WORKERS = 15

PUNCT_CHARS = ".,;:!?()[]{}\"'-\u2014\u2013\u201c\u201d\u2018\u2019"


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ctx")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(console)

    fh = logging.FileHandler(
        output_dir / f"extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    return logger


def load_tokens(csv_path: Path, logger: logging.Logger) -> Dict[str, dict]:
    """
    Load tokens preserving the ORIGINAL token_string as key.

    Returns dict keyed by token_string (with Ġ if present):
        {
            "Ġthe": {"cleaned": "the", "has_leading_space": True, "match_mode": "word", ...},
            "the": {"cleaned": "the", "has_leading_space": False, "match_mode": "substring", ...},
        }
    """
    import csv

    tokens = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            token_string = row.get("token_string", "").strip()
            cleaned = row.get("cleaned_token", "").strip()

            if not token_string or not cleaned:
                continue

            # Parse has_leading_space (stored as string "True"/"False")
            has_leading_space_str = row.get("has_leading_space", "False")
            has_leading_space = has_leading_space_str.lower() == "true"

            primary_signal = row.get("primary_signal", "")

            # Match mode logic:
            # 1. AMBIGUOUS tokens (ratio ~1.0) -> collect BOTH word and substring
            # 2. has_leading_space=True (Ġ prefix) -> word match
            # 3. has_leading_space=False BUT primary_signal=WORD -> word match (start-of-sentence)
            # 4. has_leading_space=False AND primary_signal!=WORD -> substring match

            if primary_signal == "AMBIGUOUS":
                match_mode = "both"  # Collect word AND substring contexts
            elif has_leading_space:
                match_mode = "word"
            elif primary_signal == "WORD":
                # No Ġ but classified as WORD (e.g., "The", "I" at sentence start)
                match_mode = "word"
            else:
                match_mode = "substring"

            tokens[token_string] = {
                "cleaned": cleaned,
                "cleaned_lower": cleaned.lower(),
                "has_leading_space": has_leading_space,
                "match_mode": match_mode,
                "primary_signal": primary_signal,
                "token_id": row.get("token_id", ""),
            }

    # Count stats
    word_tokens = sum(1 for t in tokens.values() if t["match_mode"] == "word")
    substr_tokens = sum(1 for t in tokens.values() if t["match_mode"] == "substring")
    both_tokens = sum(1 for t in tokens.values() if t["match_mode"] == "both")

    logger.info(f"Loaded {len(tokens):,} tokens")
    logger.info(f"  Word tokens (Ġ or start-of-sentence): {word_tokens:,}")
    logger.info(f"  Substring tokens (no Ġ):              {substr_tokens:,}")
    logger.info(f"  Both (AMBIGUOUS):                     {both_tokens:,}")

    return tokens


def build_word_context(i: int, original_words: List[str], window_size: int) -> str:
    """Build context window string for exact word match at position i."""
    start = max(0, i - window_size)
    end = min(len(original_words), i + window_size + 1)

    parts = []
    if start > 0:
        parts.append("...")
    parts.extend(original_words[start:i])
    parts.append(f"[{original_words[i]}]")
    parts.extend(original_words[i + 1 : end])
    if end < len(original_words):
        parts.append("...")

    return " ".join(parts)


def build_substr_context(
    i: int, start_idx: int, sub_len: int, original_words: List[str], window_size: int
) -> str:
    """Build context window string for substring match at position i.
    start_idx and sub_len give the exact position within original_words[i]."""
    start = max(0, i - window_size)
    end = min(len(original_words), i + window_size + 1)

    word_orig = original_words[i]
    highlighted = (
        word_orig[:start_idx]
        + "["
        + word_orig[start_idx : start_idx + sub_len]
        + "]"
        + word_orig[start_idx + sub_len :]
    )

    parts = []
    if start > 0:
        parts.append("...")
    parts.extend(original_words[start:i])
    parts.append(highlighted)
    parts.extend(original_words[i + 1 : end])
    if end < len(original_words):
        parts.append("...")

    return " ".join(parts)


def process_parquet_streaming(args: Tuple) -> dict:
    """
    Stream documents from parquet, extract contexts using inverted matching.

    Instead of iterating 50K token groups per document (each re-splitting text),
    splits text ONCE and uses:
      - Dict lookup for exact word matches: O(1) per word
      - Substring enumeration for substring matches: O(word_len^2) per word
        with set lookup, instead of O(14K) `in` checks per word

    Args is tuple: (parquet_path, tokens_dict, window_size, per_parquet_limit, seed)
    """
    parquet_path, tokens_info, window_size, per_parquet_limit, seed = args

    random.seed(seed)
    filename = Path(parquet_path).name

    # === Build lookup structures ===
    # Word lookup: case-SENSITIVE exact word match (word + both modes)
    word_lookup = {}
    for token_string, info in tokens_info.items():
        if info["match_mode"] in ("word", "both"):
            word_lookup.setdefault(info["cleaned"], []).append(token_string)

    # Substring lookup: case-SENSITIVE (keyed by original cleaned token, not lowered)
    substr_lookup = {}
    for token_string, info in tokens_info.items():
        if info["match_mode"] in ("substring", "both"):
            substr_lookup.setdefault(info["cleaned"], []).append(token_string)

    substr_set = set(substr_lookup.keys())

    # Pre-compute substring length bounds to skip impossible matches
    if substr_set:
        min_substr_len = min(len(s) for s in substr_set)
        max_substr_len = max(len(s) for s in substr_set)
    else:
        min_substr_len = 1
        max_substr_len = 0

    # Context tracking
    contexts = defaultdict(list)
    contexts_seen = defaultdict(set)
    tokens_complete = set()
    total_tokens = len(tokens_info)

    # Stream parquet
    parquet_file = pq.ParquetFile(parquet_path)
    docs_total = parquet_file.metadata.num_rows
    docs_processed = 0
    early_stopped = False
    log_interval = max(docs_total // 10, 10000)

    print(
        f"  [{filename}] Started: {docs_total:,} docs, {total_tokens:,} tokens "
        f"(word_groups={len(word_lookup):,}, substr_groups={len(substr_lookup):,})",
        flush=True,
    )

    for batch in parquet_file.iter_batches(batch_size=10000, columns=["text"]):
        texts = batch["text"].to_pylist()

        for text in texts:
            if not text:
                continue

            docs_processed += 1

            # Progress logging
            if docs_processed % log_interval == 0:
                pct = 100 * docs_processed / docs_total
                print(
                    f"  [{filename}] {docs_processed:,}/{docs_total:,} ({pct:.0f}%) | "
                    f"{len(tokens_complete):,}/{total_tokens:,} tokens done",
                    flush=True,
                )

            # === SPLIT TEXT ONCE (case-preserved, punctuation-stripped) ===
            original_words = []
            for word in text.split():
                stripped = word.strip(PUNCT_CHARS)
                if stripped:
                    original_words.append(stripped)

            n_words = len(original_words)

            # === Process each word position ===
            for i in range(n_words):
                wo = original_words[i]

                # --- Exact word match (O(1) dict lookup, case-sensitive) ---
                ts_list = word_lookup.get(wo)
                if ts_list is not None:
                    active = [ts for ts in ts_list if ts not in tokens_complete]
                    if not active:
                        # All tokens for this word are done, remove from lookup
                        del word_lookup[wo]
                    else:
                        ctx = build_word_context(i, original_words, window_size)
                        for ts in active:
                            if ctx not in contexts_seen[ts]:
                                contexts_seen[ts].add(ctx)
                                contexts[ts].append(ctx)
                                if len(contexts[ts]) >= per_parquet_limit:
                                    tokens_complete.add(ts)

                # --- Substring match (case-sensitive, enumerate substrings of original word) ---
                # Only proper substrings: length < word_len (excludes exact match)
                # Bounded by actual token lengths in substr_set
                wo = original_words[i]
                wo_len = len(wo)
                if wo_len > min_substr_len:
                    upper = min(wo_len, max_substr_len + 1)
                    for sub_len in range(min_substr_len, upper):
                        for start in range(wo_len - sub_len + 1):
                            sub = wo[start : start + sub_len]
                            ts_list = substr_lookup.get(sub)
                            if ts_list is not None:
                                active = [
                                    ts for ts in ts_list if ts not in tokens_complete
                                ]
                                if not active:
                                    # Clean up completed group
                                    substr_set.discard(sub)
                                    del substr_lookup[sub]
                                else:
                                    ctx = build_substr_context(
                                        i, start, sub_len, original_words, window_size
                                    )
                                    for ts in active:
                                        if ctx not in contexts_seen[ts]:
                                            contexts_seen[ts].add(ctx)
                                            contexts[ts].append(ctx)
                                            if len(contexts[ts]) >= per_parquet_limit:
                                                tokens_complete.add(ts)

            # Early stopping: all tokens hit per-parquet limit
            if len(tokens_complete) >= total_tokens:
                early_stopped = True
                print(
                    f"  [{filename}] EARLY STOP at doc {docs_processed:,}/{docs_total:,} "
                    f"({100 * docs_processed / docs_total:.1f}%)",
                    flush=True,
                )
                break

        if early_stopped:
            break

    total_ctx = sum(len(v) for v in contexts.values())
    status = "early-stop" if early_stopped else "full-scan"
    print(
        f"  [{filename}] Done ({status}): {docs_processed:,}/{docs_total:,} docs | "
        f"{len(tokens_complete):,}/{total_tokens:,} tokens complete | "
        f"{total_ctx:,} contexts",
        flush=True,
    )

    # Free dedup sets before returning
    del contexts_seen

    return {
        "filename": filename,
        "contexts": dict(contexts),
        "docs_processed": docs_processed,
        "docs_total": docs_total,
        "tokens_complete": len(tokens_complete),
        "early_stopped": early_stopped,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=Path, default=DEFAULT_TOKEN_CSV)
    parser.add_argument("--fineweb-dir", type=Path, default=DEFAULT_FINEWEB_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-contexts", type=int, default=DEFAULT_MAX_CONTEXTS)
    parser.add_argument(
        "--per-parquet-limit", type=int, default=DEFAULT_PER_PARQUET_LIMIT
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    logger = setup_logging(args.output_dir)

    logger.info("=" * 70)
    logger.info("FINEWEB CONTEXT EXTRACTION (Ġ-AWARE)")
    logger.info("=" * 70)
    logger.info(f"Final contexts: {args.max_contexts} per token")
    logger.info(f"Per-parquet limit: {args.per_parquet_limit} per token")
    logger.info(
        f"Window: +/-{args.window_size} | Workers: {args.workers} | Seed: {args.seed}"
    )
    logger.info("")
    logger.info("Matching Strategy:")
    logger.info(
        "  1. AMBIGUOUS (ratio ~1.0)           -> Collect BOTH word and substring"
    )
    logger.info("  2. Ġ prefix (has_leading_space)     -> Match as STANDALONE WORD")
    logger.info(
        "  3. No Ġ but primary_signal=WORD    -> Match as WORD (start-of-sentence)"
    )
    logger.info("  4. No Ġ and not WORD               -> Match as SUBSTRING")
    logger.info("")
    logger.info(
        "Matching Engine: inverted (split-once + dict lookup + substring enumeration)"
    )
    logger.info("")

    # Load tokens
    tokens = load_tokens(args.tokens, logger)

    # Get parquet files
    parquet_files = sorted(args.fineweb_dir.glob("*.parquet"))
    random.seed(args.seed)
    random.shuffle(parquet_files)
    num_parquets = len(parquet_files)
    max_pool = num_parquets * args.per_parquet_limit

    logger.info(f"Found {num_parquets} parquet files (shuffled)")
    logger.info(
        f"Max pool per token: {num_parquets} x {args.per_parquet_limit} = {max_pool}"
    )
    logger.info(f"Final sampling: {max_pool} pool -> {args.max_contexts} per token")

    # Prepare worker arguments all parquets get all tokens
    worker_args = [
        (str(pf), tokens, args.window_size, args.per_parquet_limit, args.seed + i)
        for i, pf in enumerate(parquet_files)
    ]

    n_workers = min(args.workers, num_parquets)
    logger.info("")
    logger.info(
        f"Processing {num_parquets} parquets with {n_workers} parallel workers..."
    )
    logger.info("-" * 70)

    # Process all parquets in parallel
    all_contexts: Dict[str, List[str]] = defaultdict(list)
    total_docs = 0
    total_docs_available = 0
    parquets_early_stopped = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_parquet_streaming, arg): arg[0]
            for arg in worker_args
        }

        pbar = tqdm(total=num_parquets, desc="Parquets", unit="file")

        for future in as_completed(futures):
            result = future.result()
            total_docs += result["docs_processed"]
            total_docs_available += result["docs_total"]
            if result["early_stopped"]:
                parquets_early_stopped += 1

            # Merge per-parquet contexts into global pool
            new_contexts = 0
            for token_str, ctx_list in result["contexts"].items():
                all_contexts[token_str].extend(ctx_list)
                new_contexts += len(ctx_list)

            pbar.update(1)
            pbar.set_postfix_str(f"+{new_contexts:,} ctx | {result['filename']}")

        pbar.close()

    logger.info("-" * 70)
    logger.info(f"Total docs processed: {total_docs:,} / {total_docs_available:,}")
    logger.info(f"Parquets early-stopped: {parquets_early_stopped}/{num_parquets}")

    # Pool stats before sampling
    pool_sizes = [len(all_contexts.get(ts, [])) for ts in tokens]
    tokens_with_pool = sum(1 for s in pool_sizes if s > 0)
    avg_pool = sum(pool_sizes) / len(pool_sizes) if pool_sizes else 0
    logger.info(
        f"Context pool: {tokens_with_pool:,}/{len(tokens):,} tokens have contexts, "
        f"avg pool size: {avg_pool:.1f}"
    )

    # Random sample to final count
    logger.info("")
    logger.info(
        f"Sampling {args.max_contexts} contexts per token from pooled results..."
    )

    random.seed(args.seed)
    final_contexts = {}

    for token_str in tokens:
        pool = all_contexts.get(token_str, [])
        if len(pool) > args.max_contexts:
            final_contexts[token_str] = random.sample(pool, args.max_contexts)
        else:
            final_contexts[token_str] = pool

    # Free the large pool
    del all_contexts

    # Compute stats
    complete = sum(
        1 for ts in tokens if len(final_contexts.get(ts, [])) >= args.max_contexts
    )
    with_ctx = sum(1 for ts in tokens if len(final_contexts.get(ts, [])) > 0)
    total_ctx = sum(len(c) for c in final_contexts.values())

    logger.info("")
    logger.info("=" * 70)
    logger.info("RESULTS")
    logger.info("=" * 70)
    logger.info(
        f"Tokens with contexts:  {with_ctx:,}/{len(tokens):,} ({100 * with_ctx / len(tokens):.1f}%)"
    )
    logger.info(
        f"Tokens complete ({args.max_contexts}):   {complete:,}/{len(tokens):,} ({100 * complete / len(tokens):.1f}%)"
    )
    logger.info(f"Total contexts:        {total_ctx:,}")

    # Stats by match mode
    word_with = sum(
        1
        for ts, info in tokens.items()
        if info["match_mode"] == "word" and len(final_contexts.get(ts, [])) > 0
    )
    word_total = sum(1 for _, info in tokens.items() if info["match_mode"] == "word")
    substr_with = sum(
        1
        for ts, info in tokens.items()
        if info["match_mode"] == "substring" and len(final_contexts.get(ts, [])) > 0
    )
    substr_total = sum(
        1 for _, info in tokens.items() if info["match_mode"] == "substring"
    )
    both_with = sum(
        1
        for ts, info in tokens.items()
        if info["match_mode"] == "both" and len(final_contexts.get(ts, [])) > 0
    )
    both_total = sum(1 for _, info in tokens.items() if info["match_mode"] == "both")

    logger.info("")
    logger.info("By Match Mode:")
    if word_total > 0:
        logger.info(
            f"  Word tokens:           {word_with:,}/{word_total:,} ({100 * word_with / word_total:.1f}% with contexts)"
        )
    if substr_total > 0:
        logger.info(
            f"  Substring tokens:      {substr_with:,}/{substr_total:,} ({100 * substr_with / substr_total:.1f}% with contexts)"
        )
    if both_total > 0:
        logger.info(
            f"  Both (AMBIGUOUS):      {both_with:,}/{both_total:,} ({100 * both_with / both_total:.1f}% with contexts)"
        )

    # Save outputs
    logger.info("")
    logger.info("Saving outputs...")

    # Main pickle with contexts
    pkl_path = args.output_dir / "token_contexts.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(final_contexts, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"  {pkl_path} ({pkl_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Sample JSON for inspection
    sample = {}
    for token_str, ctxs in sorted(final_contexts.items(), key=lambda x: -len(x[1]))[
        :100
    ]:
        if ctxs:
            info = tokens[token_str]
            sample[token_str] = {
                "count": len(ctxs),
                "match_mode": info["match_mode"],
                "has_leading_space": info["has_leading_space"],
                "examples": ctxs[:5],
            }
    sample_path = args.output_dir / "token_contexts_sample.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    logger.info(f"  {sample_path}")

    # Stats JSON
    by_signal = defaultdict(lambda: {"total": 0, "with_ctx": 0, "complete": 0})
    by_mode = defaultdict(lambda: {"total": 0, "with_ctx": 0, "complete": 0})

    for token_str, info in tokens.items():
        sig = info.get("primary_signal", "UNKNOWN")
        mode = info["match_mode"]
        cnt = len(final_contexts.get(token_str, []))

        by_signal[sig]["total"] += 1
        by_mode[mode]["total"] += 1

        if cnt > 0:
            by_signal[sig]["with_ctx"] += 1
            by_mode[mode]["with_ctx"] += 1
        if cnt >= args.max_contexts:
            by_signal[sig]["complete"] += 1
            by_mode[mode]["complete"] += 1

    stats = {
        "params": {
            "max_contexts": args.max_contexts,
            "per_parquet_limit": args.per_parquet_limit,
            "window_size": args.window_size,
            "num_parquets": num_parquets,
            "workers": n_workers,
        },
        "summary": {
            "total_tokens": len(tokens),
            "with_contexts": with_ctx,
            "complete": complete,
            "total_contexts": total_ctx,
            "total_docs_processed": total_docs,
            "total_docs_available": total_docs_available,
            "parquets_early_stopped": parquets_early_stopped,
        },
        "by_match_mode": dict(by_mode),
        "by_primary_signal": dict(by_signal),
    }
    stats_path = args.output_dir / "extraction_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  {stats_path}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
