#!/usr/bin/env python3
"""
LSC Circuit Structure Extraction
=================================

Extract structural information from 60 prune_scores.pkl files:
- Detailed edges with full source->destination parsing
- Layer-to-layer flow matrix
- Component-level flow (attn/mlp/resid patterns)
- Head-level connectivity
- Skip connections, input/output edges

Outputs JSON and CSV files for downstream analysis (no GPU needed after).

Usage:
    python run_extraction.py
    python run_extraction.py --models pythia-70m pythia-160m
    python run_extraction.py --dry-run
"""

import os
import sys
import json
import csv
import time
import argparse
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from utils.constants import (
    MODELS,
    BANDS,
    DRAWS,
    MODEL_DIR_NAMES,
    CIRCUITS_BASE,
    EXTRACTION_DIR,
    INDIVIDUAL_CIRCUITS_DIR,
    ALL_CIRCUITS_JSON,
    CIRCUITS_SUMMARY_CSV,
)
from utils.extraction import (
    process_single_circuit,
    flatten_structure_to_row,
    get_circuit_path,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract structural info from LSC circuits"
    )
    parser.add_argument(
        "--models", nargs="+", default=None, help="Models to process (default: all)"
    )
    parser.add_argument(
        "--bands", nargs="+", default=None, help="Bands to process (default: all)"
    )
    parser.add_argument(
        "--draws", nargs="+", default=None, help="Draws to process (default: all)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List files without processing"
    )
    args = parser.parse_args()

    models = args.models or MODELS
    bands = args.bands or BANDS
    draws = args.draws or DRAWS

    print("=" * 70)
    print("LSC CIRCUIT STRUCTURE EXTRACTION")
    print("=" * 70)
    print(f"Models: {models}")
    print(f"Bands: {bands}")
    print(f"Draws: {draws}")
    print(f"Total circuits: {len(models) * len(bands) * len(draws)}")
    print(f"Output: {EXTRACTION_DIR}")
    print("=" * 70)

    # Discover circuit files
    tasks = []
    missing = []
    for model in models:
        for band in bands:
            for draw in draws:
                path = get_circuit_path(model, band, draw)
                if path.exists():
                    tasks.append((model, band, draw))
                else:
                    missing.append(f"{model}/{band}/{draw}")

    print(f"\nFound: {len(tasks)} circuit files")
    if missing:
        print(f"Missing: {len(missing)} files:")
        for m in missing[:10]:
            print(f"  {m}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for model, band, draw in tasks[:20]:
            print(f"  {model}/{band}/{draw}")
        if len(tasks) > 20:
            print(f"  ... and {len(tasks) - 20} more")
        return

    if not tasks:
        print("No circuit files found!")
        return

    EXTRACTION_DIR.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_CIRCUITS_DIR.mkdir(parents=True, exist_ok=True)

    # Process all circuits
    results = []
    start_time = time.time()

    for i, (model, band, draw) in enumerate(tasks):
        t0 = time.time()
        structure = process_single_circuit(model, band, draw)
        elapsed = time.time() - t0

        results.append(structure)

        status = "ok" if structure.get("status") == "success" else "FAIL"
        edges = structure.get("total_edges", "?")
        skip = structure.get("n_skip_connections", "?")
        frac = structure.get("edge_fraction", 0)

        total_elapsed = time.time() - start_time
        remaining = len(tasks) - (i + 1)
        avg = total_elapsed / (i + 1)
        eta = avg * remaining

        print(
            f"[{i + 1}/{len(tasks)}] {status} {model}/{band}/{draw}: "
            f"{edges} edges ({frac:.1%}), {skip} skip "
            f"({elapsed:.1f}s, ETA: {eta / 60:.1f}min)"
        )

    # Save results
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]

    print(f"\nProcessed: {len(successful)} successful, {len(failed)} failed")

    # Save combined JSON
    print("Saving combined structure file...")
    combined = {
        "metadata": {
            "extraction_timestamp": datetime.now().isoformat(),
            "n_circuits": len(successful),
            "n_failed": len(failed),
            "models": models,
            "bands": bands,
            "draws": draws,
        },
        "circuits": {r["circuit_id"]: r for r in successful},
        "failed": {r.get("circuit_id", "unknown"): r for r in failed} if failed else {},
    }
    with open(ALL_CIRCUITS_JSON, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"  {ALL_CIRCUITS_JSON}")

    # Save individual JSONs
    print("Saving individual circuit files...")
    for r in successful:
        out_path = INDIVIDUAL_CIRCUITS_DIR / f"{r['circuit_id']}.json"
        with open(out_path, "w") as f:
            json.dump(r, f, indent=2)

    # Save summary CSV
    print("Saving summary CSV...")
    rows = [flatten_structure_to_row(r) for r in successful]
    rows = [r for r in rows if r is not None]

    if rows:
        with open(CIRCUITS_SUMMARY_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {CIRCUITS_SUMMARY_CSV}")

    # Print summary
    total_time = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"EXTRACTION COMPLETE ({total_time:.1f}s)")
    print(f"{'=' * 70}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Output: {EXTRACTION_DIR}")

    if successful:
        import statistics

        fracs = [r["edge_fraction"] for r in successful]
        edges = [r["total_edges"] for r in successful]
        skips = [r.get("n_skip_connections", 0) for r in successful]

        print(
            f"\nEdge fraction: {statistics.mean(fracs):.1%} +/- {statistics.stdev(fracs):.1%}"
        )
        print(
            f"Total edges: {statistics.mean(edges):.0f} +/- {statistics.stdev(edges):.0f}"
        )
        print(
            f"Skip connections: {statistics.mean(skips):.1f} +/- {statistics.stdev(skips):.1f}"
        )

        print("\nBy model:")
        for model in models:
            model_results = [r for r in successful if r["model"] == model]
            if model_results:
                mf = [r["edge_fraction"] for r in model_results]
                ms = [r.get("n_skip_connections", 0) for r in model_results]
                print(
                    f"  {model}: {statistics.mean(mf):.1%} edge fraction, "
                    f"{statistics.mean(ms):.1f} skip ({len(model_results)} circuits)"
                )


if __name__ == "__main__":
    main()
