#!/usr/bin/env python3
"""
LSC Representational Activation Extraction
============================================

Extract activations from Pythia models for representational analysis:
- Token embeddings (W_E, no position encoding)
- Residual stream at prediction position (all layers)
- Attention patterns from prediction position (all layers/heads)
- Attention output at prediction position (all layers)
- MLP output and pre-activations at prediction position (all layers)
- Logit lens metrics (P(correct), rank, KL from final)
- OV copy scores per head (per model)

Processes one model at a time to manage GPU memory.
Outputs 60 NPZ files (4 models x 5 bands x 3 draws) + 4 copy score files.

Usage:
    python run_extraction.py
    python run_extraction.py --models pythia-70m pythia-160m
    python run_extraction.py --bands low high --draws draw_1
    python run_extraction.py --batch-size 32 --full-sequence
    python run_extraction.py --dry-run
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add this directory to path for utils import
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils.constants import (
    MODELS,
    BANDS,
    DRAWS,
    MODEL_DIR_NAMES,
    MODEL_INFO,
    DATASETS_BASE,
    ACTIVATIONS_DIR,
    EXTRACTION_DIR,
    EXTRACTION_BATCH_SIZE,
)
from utils.extraction import (
    get_activation_filename,
    load_dataset_for_extraction,
    run_extraction_pipeline,
)


def discover_configurations(models, bands, draws, skip_existing=True):
    """Discover which configurations need extraction.

    Returns:
        (to_extract, existing, missing_data) tuples.
    """
    to_extract = []
    existing = []
    missing_data = []

    for model in models:
        for band in bands:
            for draw in draws:
                # Check if dataset exists
                data_path = DATASETS_BASE / draw / band / "test.json"
                if not data_path.exists():
                    missing_data.append(f"{model}/{band}/{draw}")
                    continue

                # Check if output exists
                filename = get_activation_filename(model, band, draw)
                output_path = ACTIVATIONS_DIR / filename

                if skip_existing and output_path.exists():
                    existing.append(f"{model}/{band}/{draw}")
                else:
                    to_extract.append((model, band, draw))

    return to_extract, existing, missing_data


def estimate_storage(models, bands, draws):
    """Estimate total storage needed for extraction."""
    total_mb = 0
    for model in models:
        info = MODEL_INFO.get(model, {})
        n_layers = info.get("n_layers", 6)
        n_heads = info.get("n_heads", 8)
        d_model = info.get("d_model", 512)
        d_mlp = d_model * 4

        n_examples = 225  # Typical test set size
        n_configs = len(bands) * len(draws)

        # Per-config storage estimate (float32 = 4 bytes)
        bytes_per_config = (
            n_examples * 21 * d_model * 4  # token_embeddings
            + n_examples * n_layers * d_model * 4  # resid_post_predpos
            + n_examples * n_layers * n_heads * 22 * 4  # attn_pattern_predpos
            + n_examples * n_layers * d_model * 4  # attn_out_predpos
            + n_examples * n_layers * d_model * 4  # mlp_out_predpos
            + n_examples * n_layers * d_mlp * 4  # mlp_pre_predpos
            + n_examples * n_layers * 4 * 3  # logit_lens (3 arrays)
            + n_examples * 21 * 4  # input_ids
            + n_examples * 4  # target_ids
        )
        mb_per_config = bytes_per_config / (1024 * 1024)
        total_mb += mb_per_config * n_configs

        # Copy scores
        total_mb += n_layers * n_heads * 4 / (1024 * 1024)

    return total_mb


def main():
    parser = argparse.ArgumentParser(
        description="Extract representational activations from Pythia models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_extraction.py                          # Extract all
  python run_extraction.py --models pythia-70m      # Single model
  python run_extraction.py --dry-run                # Preview only
  python run_extraction.py --batch-size 32          # Smaller batches
  python run_extraction.py --full-sequence          # Include full-seq resid
  python run_extraction.py --no-skip                # Re-extract existing
""",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Models to process (default: {MODELS})",
    )
    parser.add_argument(
        "--bands", nargs="+", default=None, help=f"Bands to process (default: {BANDS})"
    )
    parser.add_argument(
        "--draws", nargs="+", default=None, help=f"Draws to process (default: {DRAWS})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=EXTRACTION_BATCH_SIZE,
        help=f"Batch size for forward passes (default: {EXTRACTION_BATCH_SIZE})",
    )
    parser.add_argument(
        "--full-sequence",
        action="store_true",
        help="Also extract full-sequence residual stream (large!)",
    )
    parser.add_argument(
        "--no-skip", action="store_true", help="Re-extract even if output files exist"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List configurations without extracting"
    )
    args = parser.parse_args()

    models = args.models or MODELS
    bands = args.bands or BANDS
    draws = args.draws or DRAWS
    skip_existing = not args.no_skip

    print("=" * 70)
    print("LSC REPRESENTATIONAL ACTIVATION EXTRACTION")
    print("=" * 70)
    print(f"Models:         {models}")
    print(f"Bands:          {bands}")
    print(f"Draws:          {draws}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Full sequence:  {args.full_sequence}")
    print(f"Skip existing:  {skip_existing}")
    print(f"Output:         {ACTIVATIONS_DIR}")
    print(f"Total configs:  {len(models) * len(bands) * len(draws)}")
    print("=" * 70)

    # Discover what needs to be done
    to_extract, existing, missing_data = discover_configurations(
        models, bands, draws, skip_existing=skip_existing
    )

    print(f"\nTo extract:     {len(to_extract)}")
    print(f"Already exist:  {len(existing)}")
    if missing_data:
        print(f"Missing data:   {len(missing_data)}")
        for m in missing_data[:5]:
            print(f"  {m}")
        if len(missing_data) > 5:
            print(f"  ... and {len(missing_data) - 5} more")

    # Storage estimate
    est_mb = estimate_storage(models, bands, draws)
    print(f"\nEstimated storage: {est_mb:.0f} MB ({est_mb / 1024:.1f} GB)")

    if args.dry_run:
        print("\n[DRY RUN] Would extract:")
        for model, band, draw in to_extract[:20]:
            info = MODEL_INFO.get(model, {})
            print(
                f"  {model}/{band}/{draw} "
                f"(L={info.get('n_layers', '?')}, H={info.get('n_heads', '?')}, "
                f"d={info.get('d_model', '?')})"
            )
        if len(to_extract) > 20:
            print(f"  ... and {len(to_extract) - 20} more")
        print("\nWould also compute copy scores for each model:")
        for model in models:
            print(f"  copy_scores_{MODEL_DIR_NAMES.get(model, model)}.npz")
        return

    if not to_extract:
        print("\nNothing to extract (all configurations already exist).")
        return

    # Run extraction
    print(f"\nStarting extraction at {datetime.now().isoformat()}")
    start_time = time.time()

    summary = run_extraction_pipeline(
        models=models,
        bands=bands,
        draws=draws,
        batch_size=args.batch_size,
        extract_full_sequence=args.full_sequence,
        skip_existing=skip_existing,
        verbose=True,
    )

    total_time = time.time() - start_time

    # Final summary
    n_success = sum(1 for r in summary if r.get("status") == "success")
    n_skip = sum(1 for r in summary if r.get("status") == "skipped")
    n_fail = sum(1 for r in summary if r.get("status") == "error")

    print(f"\n{'=' * 70}")
    print(f"EXTRACTION COMPLETE ({total_time / 60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"Successful: {n_success}")
    print(f"Skipped:    {n_skip}")
    print(f"Failed:     {n_fail}")

    if n_success > 0:
        success_rows = [r for r in summary if r.get("status") == "success"]
        total_size = sum(r.get("file_size_mb", 0) for r in success_rows)
        total_extract_time = sum(r.get("extraction_time_s", 0) for r in success_rows)
        print(f"\nTotal size:   {total_size:.0f} MB ({total_size / 1024:.1f} GB)")
        print(f"Extract time: {total_extract_time / 60:.1f} min")

        print("\nBy model:")
        for model in models:
            model_rows = [r for r in success_rows if r.get("model") == model]
            if model_rows:
                size = sum(r.get("file_size_mb", 0) for r in model_rows)
                t = sum(r.get("extraction_time_s", 0) for r in model_rows)
                print(f"  {model}: {len(model_rows)} configs, {size:.0f} MB, {t:.0f}s")

    if n_fail > 0:
        print("\nFailed configurations:")
        for r in summary:
            if r.get("status") == "error":
                print(
                    f"  {r['model']}/{r['band']}/{r['draw']}: {r.get('error', 'unknown')}"
                )

    print(f"\nSummary CSV: {EXTRACTION_DIR / 'extraction_summary.csv'}")
    print(f"Activations:  {ACTIVATIONS_DIR}")


if __name__ == "__main__":
    main()
