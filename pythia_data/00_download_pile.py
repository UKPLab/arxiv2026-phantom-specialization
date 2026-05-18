"""
Download the deduplicated, preshuffled Pile dataset from the HuggingFace Hub.

Pipeline step 00. Output goes to ../Pile/pile_shards/ relative to this file.
The download is resumable.
"""

from huggingface_hub import snapshot_download
from pathlib import Path
import time
from datetime import datetime


def log(msg):
    """Print with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


# Where to save - Pile data stored separately from pipeline code
PILE_DIR = Path(__file__).resolve().parent.parent / "Pile"
output_dir = PILE_DIR / "pile_shards"
output_dir.mkdir(parents=True, exist_ok=True)

log("=" * 70)
log("PYTHIA PILE DOWNLOAD")
log("=" * 70)
log(f"Output directory: {output_dir}")
log(f"Expected size: ~600 GB")
log(f"Expected time: 12-48 hours (depends on network)")
log("Starting download...")
log("=" * 70)

start_time = time.time()

try:
    snapshot_download(
        repo_id="EleutherAI/pile-standard-pythia-preshuffled",
        repo_type="dataset",
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    elapsed = time.time() - start_time
    hours = elapsed / 3600

    log("=" * 70)
    log("DOWNLOAD COMPLETE!")
    log(f"Time taken: {hours:.2f} hours")
    log(f"Files saved to: {output_dir}")
    log("=" * 70)

    # List what was downloaded
    log("\nDownloaded files:")
    for f in sorted(output_dir.glob("*")):
        size_gb = f.stat().st_size / (1024**3)
        log(f"  {f.name}: {size_gb:.2f} GB")

except KeyboardInterrupt:
    log("\nDownload interrupted by user (Ctrl+C)")
    log("You can resume by running this script again")
    exit(1)

except Exception as e:
    log(f"\nError during download: {e}")
    log("You can try running this script again to resume")
    exit(1)
