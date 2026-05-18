#!/usr/bin/env python3
"""
LSC Circuit-Mode Activation Extraction
========================================

Extract activations from Pythia models with mean ablation applied:
non-circuit edges have their outputs replaced with dataset-mean activations.
This isolates the computational path through the discovered ACDC circuits.

Key differences from base extraction (run_extraction.py):
- Models loaded with fold_ln=True (matching circuit discovery)
- Mean ablation hooks replace non-circuit component outputs
- No token embeddings or full-sequence extraction
- Circuit metadata (edge counts, head masks) included in NPZ

Processes one model at a time to manage GPU memory.
Outputs 60 NPZ files (4 models x 5 bands x 3 draws).

Usage:
    python run_circuit_extraction.py
    python run_circuit_extraction.py --models pythia-70m pythia-160m
    python run_circuit_extraction.py --bands low high --draws draw_1
    python run_circuit_extraction.py --dry-run
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
    DATASETS_BASE,
    CIRCUITS_DIR,
    CIRCUIT_ACTIVATIONS_DIR,
    EXTRACTION_BATCH_SIZE,
    CIRCUIT_FOLD_LN,
)
from utils.circuit_loading import (
    get_circuit_activation_path,
    load_prune_scores,
    get_circuit_summary,
)
from utils.circuit_extraction import run_circuit_extraction_pipeline


def discover_configurations(models, bands, draws, skip_existing=True):
    """Discover which configurations need extraction."""
    to_extract = []
    existing = []
    missing_data = []
    missing_circuit = []

    for model in models:
        for band in bands:
            for draw in draws:
                # Check dataset
                data_path = DATASETS_BASE / draw / band / "test.json"
                if not data_path.exists():
                    missing_data.append(f"{model}/{band}/{draw}")
                    continue

                # Check prune scores
                model_dir = MODEL_DIR_NAMES.get(model, model.replace("-", "_"))
                circuit_path = (
                    CIRCUITS_DIR / model_dir / band / draw / "prune_scores.pkl"
                )
                if not circuit_path.exists():
                    missing_circuit.append(f"{model}/{band}/{draw}")
                    continue

                # Check output
                output_path = get_circuit_activation_path(model, band, draw)
                if skip_existing and output_path.exists():
                    existing.append(f"{model}/{band}/{draw}")
                else:
                    to_extract.append((model, band, draw))

    return to_extract, existing, missing_data, missing_circuit


def main():
    parser = argparse.ArgumentParser(
        description="Extract circuit-mode activations from Pythia models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_circuit_extraction.py                          # Extract all
  python run_circuit_extraction.py --models pythia-70m      # Single model
  python run_circuit_extraction.py --dry-run                # Preview only
  python run_circuit_extraction.py --batch-size 32          # Smaller batches
  python run_circuit_extraction.py --no-skip                # Re-extract existing
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
    print("LSC CIRCUIT-MODE ACTIVATION EXTRACTION")
    print("=" * 70)
    print(f"Models:         {models}")
    print(f"Bands:          {bands}")
    print(f"Draws:          {draws}")
    print(f"Batch size:     {args.batch_size}")
    print(f"fold_ln:        {CIRCUIT_FOLD_LN}")
    print(f"Skip existing:  {skip_existing}")
    print(f"Circuit dir:    {CIRCUITS_DIR}")
    print(f"Output:         {CIRCUIT_ACTIVATIONS_DIR}")
    print(f"Total configs:  {len(models) * len(bands) * len(draws)}")
    print("=" * 70)

    # Discover what needs to be done
    to_extract, existing, missing_data, missing_circuit = discover_configurations(
        models, bands, draws, skip_existing=skip_existing
    )

    print(f"\nTo extract:       {len(to_extract)}")
    print(f"Already exist:    {len(existing)}")
    if missing_data:
        print(f"Missing data:     {len(missing_data)}")
    if missing_circuit:
        print(f"Missing circuits: {len(missing_circuit)}")
        for m in missing_circuit[:5]:
            print(f"  {m}")
        if len(missing_circuit) > 5:
            print(f"  ... and {len(missing_circuit) - 5} more")

    if args.dry_run:
        print("\n[DRY RUN] Would extract:")
        for model, band, draw in to_extract[:20]:
            try:
                ps = load_prune_scores(model, band, draw)
                summary = get_circuit_summary(ps)
                print(f"  {model}/{band}/{draw} ({summary['total_edges']} edges)")
            except Exception:
                print(f"  {model}/{band}/{draw}")
        if len(to_extract) > 20:
            print(f"  ... and {len(to_extract) - 20} more")
        return

    if not to_extract:
        print("\nNothing to extract.")
        return

    # Run extraction
    print(f"\nStarting circuit extraction at {datetime.now().isoformat()}")
    start_time = time.time()

    summary = run_circuit_extraction_pipeline(
        models=models,
        bands=bands,
        draws=draws,
        batch_size=args.batch_size,
        skip_existing=skip_existing,
        verbose=True,
    )

    total_time = time.time() - start_time

    # Final summary
    n_success = sum(1 for r in summary if r.get("status") == "success")
    n_skip = sum(1 for r in summary if r.get("status") == "skipped")
    n_fail = sum(1 for r in summary if r.get("status") == "error")

    print(f"\n{'=' * 70}")
    print(f"CIRCUIT EXTRACTION COMPLETE ({total_time / 60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"Successful: {n_success}")
    print(f"Skipped:    {n_skip}")
    print(f"Failed:     {n_fail}")

    if n_success > 0:
        success_rows = [r for r in summary if r.get("status") == "success"]
        total_size = sum(r.get("file_size_mb", 0) for r in success_rows)
        print(f"\nTotal size: {total_size:.0f} MB ({total_size / 1024:.1f} GB)")

    if n_fail > 0:
        print("\nFailed configurations:")
        for r in summary:
            if r.get("status") == "error":
                print(
                    f"  {r['model']}/{r['band']}/{r['draw']}: {r.get('error', 'unknown')}"
                )


if __name__ == "__main__":
    main()
