"""
Shared constants for Phase Functional analysis.

This module contains all constants used across the analysis notebooks,
ensuring consistency and reducing redundancy.
"""

from pathlib import Path
from typing import Dict, List, Optional

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


BASE_PATH = PROJECT_ROOT / "LSC_circuit_analysis/01_Phase_Functional"
CIRCUITS_BASE = PROJECT_ROOT / "LSC_circuits/circuit_discovery/circuits"
SUMMARY_PATH = PROJECT_ROOT / "LSC_circuits/circuit_discovery/summary"

DISCOVERY_SUMMARY = SUMMARY_PATH / "discovery_summary.json"
CROSS_BAND_TRANSFER_SUMMARY = SUMMARY_PATH / "cross_band_transfer.json"

# Output paths
OUTPUT_DIR = BASE_PATH / "outputs"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
VIZ_DIR = OUTPUT_DIR / "viz"


def get_output_dirs():
    """Create and return output directories."""
    for d in [OUTPUT_DIR, ANALYSIS_DIR, VIZ_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_DIR, VIZ_DIR


# =============================================================================
# MODELS
# =============================================================================

MODELS: List[str] = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
]

MODEL_DIR_NAMES: Dict[str, str] = {
    "pythia-70m": "pythia_70m",
    "pythia-160m": "pythia_160m",
    "pythia-410m": "pythia_410m",
    "pythia-1b": "pythia_1b",
    "pythia-1.4b": "pythia_1.4b",
}

MODEL_CAPACITY: Dict[str, int] = {
    "pythia-70m": 70,
    "pythia-160m": 160,
    "pythia-410m": 410,
    "pythia-1b": 1000,
    "pythia-1.4b": 1400,
}

MODEL_LAYERS: Dict[str, int] = {
    "pythia-70m": 6,
    "pythia-160m": 12,
    "pythia-410m": 24,
    "pythia-1b": 16,
    "pythia-1.4b": 24,
}

MODEL_HEADS: Dict[str, int] = {
    "pythia-70m": 8,
    "pythia-160m": 12,
    "pythia-410m": 16,
    "pythia-1b": 8,
    "pythia-1.4b": 16,
}

MODEL_TOTAL_EDGES: Dict[str, int] = {
    "pythia-70m": 1324,
    "pythia-160m": 11467,
    "pythia-410m": 80581,
    "pythia-1b": 10009,
    "pythia-1.4b": 80581,
}

# Pareto-optimized thresholds (one per model)
MODEL_THRESHOLDS: Dict[str, float] = {
    "pythia-70m": 0.00158,
    "pythia-160m": 0.000631,
    "pythia-410m": 0.000251,
    "pythia-1b": 0.00158,
    "pythia-1.4b": 0.000631,
}


# =============================================================================
# FREQUENCY BANDS
# =============================================================================

BANDS: List[str] = ["low", "medium", "high", "very_high", "control"]
FREQUENCY_BANDS: List[str] = ["low", "medium", "high", "very_high"]
CONTROL_BAND: str = "control"

# Ordinal frequency rank (for trend tests; control excluded)
FREQUENCY_RANK: Dict[str, Optional[int]] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "very_high": 4,
    "control": None,
}

BAND_NAMES: Dict[str, str] = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "very_high": "Very High",
    "control": "Control",
}

# Groupings for asymmetric transfer analysis
LOW_FREQ_BANDS: List[str] = ["low", "medium"]
HIGH_FREQ_BANDS: List[str] = ["high", "very_high"]

# Adjacent band pairs
ADJACENT_PAIRS: List[tuple] = [
    ("low", "medium"),
    ("medium", "high"),
    ("high", "very_high"),
]

# Cross-spectrum pairs (distance >= 2)
CROSS_SPECTRUM_PAIRS: List[tuple] = [
    ("low", "high"),
    ("low", "very_high"),
    ("medium", "very_high"),
]


# =============================================================================
# DRAWS
# =============================================================================

DRAWS: List[str] = ["draw_1", "draw_2", "draw_3"]
N_DRAWS: int = 3


BASE_METRICS: List[str] = [
    "accuracy",
    "top5_accuracy",
    "top10_accuracy",
    "mean_correct_prob",
]
CIRCUIT_METRICS: List[str] = [
    "accuracy",
    "top5_accuracy",
    "top10_accuracy",
    "mean_correct_prob",
    "kl_div",
]
PRIMARY_METRICS: List[str] = ["accuracy", "top5_accuracy"]

DERIVED_METRIC_NAMES: List[str] = [
    "retention_ratio",
    "completeness",
    "generalization_gap",
    "transfer_ratio",
    "top1_top5_gap",
]


BAND_COLORS: Dict[str, str] = {
    "low": "#d62728",
    "medium": "#ff7f0e",
    "high": "#2ca02c",
    "very_high": "#1f77b4",
    "control": "#7f7f7f",
}

MODEL_COLORS: Dict[str, str] = {
    "pythia-70m": "#1f77b4",
    "pythia-160m": "#ff7f0e",
    "pythia-410m": "#2ca02c",
    "pythia-1b": "#d62728",
    "pythia-1.4b": "#9467bd",
}

FIGURE_DEFAULTS = {
    "figsize": (14, 8),
    "dpi": 150,
    "font_size": 11,
    "title_size": 14,
    "label_size": 12,
}


# =============================================================================
# STATISTICAL PARAMETERS
# =============================================================================

ALPHA: float = 0.05
N_BOOTSTRAP: int = 10000
RANDOM_SEED: int = 42


# =============================================================================
# CROSS-PHASE PATHS
# =============================================================================

RANDOM_BASELINE_DIR = PROJECT_ROOT / "LSC_circuits/random_baseline"
RANDOM_BASELINE_RESULTS = RANDOM_BASELINE_DIR / "random_baseline_results.json"

PER_EXAMPLE_DIR = PROJECT_ROOT / "LSC_circuits/per_example_eval"

PHASE2_ANALYSIS_DIR = (
    PROJECT_ROOT / "LSC_circuit_analysis/02_Phase_Structural/outputs/analysis"
)
