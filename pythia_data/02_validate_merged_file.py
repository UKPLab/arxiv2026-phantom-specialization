#!/usr/bin/env python3
"""
Compare Both Index Files
=========================
Checks both pile_standard and pile_merged index files
to determine which one is correct for the merged file.

Usage:
    python compare_index_files.py
"""

import numpy as np
import struct
from pathlib import Path


def log(msg):
    print(msg, flush=True)


def read_index_file(idx_path):
    """Read and parse index file"""
    if not idx_path.exists():
        return None

    log(f"\n{'=' * 70}")
    log(f"Reading: {idx_path}")
    log("=" * 70)

    with open(idx_path, "rb") as f:
        magic = f.read(9)
        log(f"  Magic: {magic}")

        version = struct.unpack("<Q", f.read(8))[0]
        log(f"  Version: {version}")

        dtype_code = struct.unpack("<B", f.read(1))[0]
        dtype_map = {
            1: "uint8",
            2: "int8",
            3: "int16",
            4: "int32",
            5: "int64",
            6: "float32",
            7: "float64",
            8: "uint16",
        }
        dtype_name = dtype_map.get(dtype_code, "unknown")
        log(f"  Dtype code: {dtype_code} ({dtype_name})")

        n_docs = struct.unpack("<Q", f.read(8))[0]
        log(f"  Number of documents: {n_docs:,}")

        n_bytes = struct.unpack("<Q", f.read(8))[0]
        log(f"  Total bytes (n_bytes field): {n_bytes:,} ({n_bytes / 1e9:.2f} GB)")

        # Read document metadata (sequentially - sizes then pointers)
        doc_sizes = np.frombuffer(f.read(n_docs * 4), dtype=np.int32)
        doc_pointers = np.frombuffer(f.read(n_docs * 8), dtype=np.int64)

        log(f"\n  Document size statistics:")
        log(f"    All same size: {len(np.unique(doc_sizes)) == 1}")
        log(f"    Most common size: {np.bincount(doc_sizes).argmax()}")
        log(f"    Min size: {doc_sizes.min()}")
        log(f"    Max size: {doc_sizes.max()}")

        log(f"\n  Pointer statistics:")
        log(f"    First doc pointer: {doc_pointers[0]:,}")
        log(f"    Last doc pointer: {doc_pointers[-1]:,}")
        log(f"    Max pointer + max size: {doc_pointers.max() + doc_sizes.max():,}")

        log(f"\n  Sample documents:")
        log(f"    First 3:")
        for i in range(min(3, n_docs)):
            log(f"      Doc {i}: ptr={doc_pointers[i]:,}, size={doc_sizes[i]}")

        log(f"    Last 3:")
        for i in range(max(0, n_docs - 3), n_docs):
            log(f"      Doc {i}: ptr={doc_pointers[i]:,}, size={doc_sizes[i]}")

        # Calculate expected file size
        expected_size = doc_pointers[-1] + doc_sizes[-1]
        log(
            f"\n  Expected data file size: {expected_size:,} bytes ({expected_size / 1e9:.2f} GB)"
        )
        log(f"    (last doc pointer + last doc size)")

        return {
            "path": idx_path,
            "dtype_code": dtype_code,
            "dtype_name": dtype_name,
            "n_docs": n_docs,
            "n_bytes": n_bytes,
            "doc_sizes": doc_sizes,
            "doc_pointers": doc_pointers,
            "expected_size": expected_size,
        }


def test_index_with_merged_file(idx_info, merged_file_path):
    """Test if an index file works with the merged file"""
    if idx_info is None:
        return None

    log(f"\n{'=' * 70}")
    log(f"Testing {idx_info['path'].name} with merged file")
    log("=" * 70)

    file_size = merged_file_path.stat().st_size
    log(f"  Merged file size: {file_size:,} bytes ({file_size / 1e9:.2f} GB)")

    # Check if sizes match
    size_match = abs(idx_info["expected_size"] - file_size) < 1000
    log(f"  Index expected size: {idx_info['expected_size']:,} bytes")
    log(f"  Size match: {'YES' if size_match else 'NO'}")

    if not size_match:
        diff = abs(idx_info["expected_size"] - file_size)
        log(f"  Difference: {diff:,} bytes ({diff / 1e9:.2f} GB)")

    # Try to memory-map
    dtype_map_np = {
        1: np.uint8,
        2: np.int8,
        3: np.int16,
        4: np.int32,
        5: np.int64,
        6: np.float32,
        7: np.float64,
        8: np.uint16,
    }
    dtype = dtype_map_np.get(idx_info["dtype_code"], np.uint16)

    try:
        data = np.memmap(merged_file_path, dtype=dtype, mode="r")
        log(f"  Memmap length: {len(data):,} tokens")

        # Test reading documents
        readable_count = 0
        test_indices = [
            0,
            idx_info["n_docs"] // 4,
            idx_info["n_docs"] // 2,
            3 * idx_info["n_docs"] // 4,
            idx_info["n_docs"] - 1,
        ]

        log(f"\n  Testing document reads:")
        for idx in test_indices:
            if idx >= idx_info["n_docs"]:
                continue

            ptr = int(idx_info["doc_pointers"][idx])
            size = int(idx_info["doc_sizes"][idx])

            if ptr >= 0 and size > 0 and ptr + size <= len(data):
                status = ""
                readable_count += 1
            else:
                status = ""

            log(
                f"    {status} Doc {idx:,}: ptr={ptr:,}, size={size}, "
                f"end={ptr + size:,}, data_len={len(data):,}"
            )

        # Count all readable documents
        all_readable = 0
        for i in range(idx_info["n_docs"]):
            ptr = int(idx_info["doc_pointers"][i])
            size = int(idx_info["doc_sizes"][i])
            if ptr >= 0 and size > 0 and ptr + size <= len(data):
                all_readable += 1

        readable_pct = all_readable / idx_info["n_docs"] * 100
        log(
            f"\n  Total readable: {all_readable:,} / {idx_info['n_docs']:,} ({readable_pct:.2f}%)"
        )

        return {
            "size_match": size_match,
            "readable_count": all_readable,
            "readable_percent": readable_pct,
        }

    except Exception as e:
        log(f"  ERROR: {e}")
        return None


def main():
    log("=" * 70)
    log("COMPARING BOTH INDEX FILES")
    log("=" * 70)

    # Paths - Pile data stored separately from pipeline code
    SCRIPT_DIR = Path(__file__).resolve().parent
    PILE_DIR = SCRIPT_DIR.parent / "Pile"
    standard_idx = PILE_DIR / "pile_shards" / "document.idx"
    merged_idx = PILE_DIR / "pile_merged" / "document.idx"
    merged_bin = PILE_DIR / "pile_merged" / "document.bin"

    # Check what exists
    log(f"\nFile existence check:")
    log(f"  Standard index: {'EXISTS' if standard_idx.exists() else 'MISSING'}")
    log(f"  Merged index:   {'EXISTS' if merged_idx.exists() else 'MISSING'}")
    log(f"  Merged binary:  {'EXISTS' if merged_bin.exists() else 'MISSING'}")

    if not merged_bin.exists():
        log("\nERROR: Merged binary file not found!")
        return 1

    # Read both index files
    standard_info = read_index_file(standard_idx)
    merged_info = read_index_file(merged_idx)

    # Compare if both exist
    if standard_info and merged_info:
        log("\n" + "=" * 70)
        log("COMPARING INDEX FILES")
        log("=" * 70)

        # Check if they're identical
        if standard_info["n_bytes"] == merged_info["n_bytes"]:
            log("  n_bytes field: IDENTICAL")
            if standard_info["n_bytes"] == 1:
                log("    Both show n_bytes=1 (CORRUPTED)")
        else:
            log("  n_bytes field: DIFFERENT")
            log(f"    Standard: {standard_info['n_bytes']:,}")
            log(f"    Merged:   {merged_info['n_bytes']:,}")

        if standard_info["n_docs"] == merged_info["n_docs"]:
            log("  n_docs field: IDENTICAL")
        else:
            log("  n_docs field: DIFFERENT")

        # Check if pointers are identical
        if np.array_equal(standard_info["doc_pointers"], merged_info["doc_pointers"]):
            log("  Document pointers: IDENTICAL")
            log("    -> Merged index was COPIED from standard (not regenerated)")
        else:
            log("  Document pointers: DIFFERENT")
            log("    -> Merged index was REGENERATED by unshard script")

    # Test both with merged file
    log("\n" + "=" * 70)
    log("TESTING WITH MERGED FILE")
    log("=" * 70)

    standard_test = None
    merged_test = None

    if standard_info:
        standard_test = test_index_with_merged_file(standard_info, merged_bin)

    if merged_info:
        merged_test = test_index_with_merged_file(merged_info, merged_bin)

    # Final recommendation
    log("\n" + "=" * 70)
    log("RECOMMENDATION")
    log("=" * 70)

    if not merged_info:
        log("No index file in pile_merged directory!")
        log("  The unshard script did NOT create a new index.")
        log("  You must use the fixed version that doesn't need an index.")
        log("\n  Use: 03_count_pile_frequencies.py")

    elif standard_test and merged_test:
        if merged_test["readable_percent"] > standard_test["readable_percent"]:
            log("Use the MERGED index file")
            log(f"  It can read {merged_test['readable_percent']:.1f}% of documents")
            log(f"  vs {standard_test['readable_percent']:.1f}% with standard index")
        elif merged_test["readable_percent"] == standard_test["readable_percent"]:
            if merged_test["readable_percent"] < 99:
                log("BOTH index files have the same (low) readable rate!")
                log(
                    f"  Both can only read {merged_test['readable_percent']:.1f}% of documents"
                )
                log("\n  RECOMMENDATION: Use the fixed version (no index needed)")
                log("  Run: 03_count_pile_frequencies.py")
            else:
                log("Both index files work equally well")
                log("  Use either one")
        else:
            log("Use the STANDARD index file")
            log(f"  It can read {standard_test['readable_percent']:.1f}% of documents")
            log(f"  vs {merged_test['readable_percent']:.1f}% with merged index")

    elif merged_test:
        if merged_test["readable_percent"] > 95:
            log("Use the MERGED index file")
            log(f"  It can read {merged_test['readable_percent']:.1f}% of documents")
        else:
            log("Merged index has issues")
            log(f"  Can only read {merged_test['readable_percent']:.1f}% of documents")
            log("\n  Use: 03_count_pile_frequencies.py")

    log("\n" + "=" * 70)
    log("BEST SOLUTION")
    log("=" * 70)
    log("Since Pile has fixed document size (2049 tokens),")
    log("you don't actually NEED an index file at all!")
    log("\nThe fixed version calculates positions directly:")
    log("  Document N starts at position: N x 2049")
    log("\nThis is faster, simpler, and 100% reliable.")
    log("\n-> Use: 03_count_pile_frequencies.py")
    log("=" * 70)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
