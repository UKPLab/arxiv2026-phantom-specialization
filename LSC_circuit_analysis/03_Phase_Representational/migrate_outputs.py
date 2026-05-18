#!/usr/bin/env python3
"""
Migrate Phase 3 outputs from flat to hierarchical directory structure.

Moves files from:
  outputs/analysis/01_*.csv  -> outputs/embedding/base/analysis/
  outputs/viz/viz_01_*.png   -> outputs/embedding/base/viz/
  outputs/analysis/02_*.csv  -> outputs/residual_stream/base/analysis/
  etc.

Usage:
    python migrate_outputs.py --dry-run   # Preview moves
    python migrate_outputs.py             # Execute moves
"""

import os as _os
from pathlib import Path as _Path


def _find_project_root() -> _Path:
    env = _os.environ.get("PROJECT_ROOT")
    if env:
        return _Path(env).resolve()
    for p in _Path(__file__).resolve().parents:
        if (p / "src" / "config.py").exists():
            return p
    return _Path(__file__).resolve().parents[1]


PROJECT_ROOT = _find_project_root()
import shutil
import argparse
from pathlib import Path

OUTPUT_DIR = PROJECT_ROOT / "LSC_circuit_analysis/03_Phase_Representational/outputs"

# Mapping: notebook prefix -> domain name
PREFIX_TO_DOMAIN = {
    "01": "embedding",
    "02": "residual_stream",
    "03": "logit_lens",
    "04": "attention",
    "05": "mlp",
    "06": "info_theoretic",
    "07": "inferential",
}


def discover_moves():
    """Discover all file moves needed."""
    moves = []

    analysis_dir = OUTPUT_DIR / "analysis"
    viz_dir = OUTPUT_DIR / "viz"

    # Analysis CSVs
    if analysis_dir.exists():
        for f in sorted(analysis_dir.iterdir()):
            if not f.is_file():
                continue
            prefix = f.name[:2]
            domain = PREFIX_TO_DOMAIN.get(prefix)
            if domain:
                dest = OUTPUT_DIR / domain / "base" / "analysis" / f.name
                moves.append((f, dest))

    # Viz PNGs
    if viz_dir.exists():
        for f in sorted(viz_dir.iterdir()):
            if not f.is_file():
                continue
            # viz files are named viz_NN_... where NN is the notebook prefix
            name = f.name
            if name.startswith("viz_"):
                prefix = name[4:6]  # Extract NN from viz_NN_...
                domain = PREFIX_TO_DOMAIN.get(prefix)
                if domain:
                    dest = OUTPUT_DIR / domain / "base" / "viz" / f.name
                    moves.append((f, dest))

    return moves


def main():
    parser = argparse.ArgumentParser(
        description="Migrate outputs to hierarchical structure"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview moves without executing"
    )
    args = parser.parse_args()

    moves = discover_moves()

    if not moves:
        print("No files to migrate.")
        return

    # Group by domain for display
    domains = {}
    for src, dst in moves:
        domain = dst.parent.parent.parent.name
        domains.setdefault(domain, []).append((src, dst))

    print(f"{'DRY RUN - ' if args.dry_run else ''}Migration plan:")
    print(f"Total files: {len(moves)}")
    print()

    for domain, domain_moves in sorted(domains.items()):
        analysis_count = sum(1 for _, d in domain_moves if "/analysis/" in str(d))
        viz_count = sum(1 for _, d in domain_moves if "/viz/" in str(d))
        print(
            f"  {domain}: {analysis_count} analysis + {viz_count} viz = {len(domain_moves)} files"
        )

    if args.dry_run:
        print("\nDetailed moves:")
        for src, dst in moves:
            print(f"  {src.name}")
            print(f"    -> {dst.relative_to(OUTPUT_DIR)}")
        print(f"\nRe-run without --dry-run to execute.")
        return

    # Execute moves
    print("\nExecuting moves...")
    moved = 0
    errors = 0
    for src, dst in moves:
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            moved += 1
        except Exception as e:
            print(f"  ERROR: {src.name}: {e}")
            errors += 1

    print(f"\nCopied: {moved}, Errors: {errors}")

    if errors == 0 and moved > 0:
        # Verify all copies succeeded before removing originals
        all_ok = True
        for src, dst in moves:
            if not dst.exists():
                print(f"  VERIFY FAILED: {dst}")
                all_ok = False
            elif dst.stat().st_size != src.stat().st_size:
                print(f"  SIZE MISMATCH: {dst}")
                all_ok = False

        if all_ok:
            print("\nAll copies verified. Removing originals...")
            for src, dst in moves:
                src.unlink()
            print("Done. Original files removed.")

            # Remove empty directories
            for d in [OUTPUT_DIR / "analysis", OUTPUT_DIR / "viz"]:
                if d.exists() and not any(d.iterdir()):
                    d.rmdir()
                    print(f"  Removed empty directory: {d.name}/")
        else:
            print("\nVerification failed! Original files kept.")


if __name__ == "__main__":
    main()
