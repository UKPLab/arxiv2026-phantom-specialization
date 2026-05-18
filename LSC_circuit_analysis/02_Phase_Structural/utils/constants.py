"""
Shared constants for Phase Structural analysis.

Imports shared constants from Phase 1 (Functional) and adds
structural-specific paths, component types, and edge categories.
"""

import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List

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


_PHASE1_PATH = PROJECT_ROOT / "LSC_circuit_analysis/01_Phase_Functional"

# Use importlib to avoid circular import when this module is also named utils.constants
_spec = importlib.util.spec_from_file_location(
    "phase1_constants", str(_PHASE1_PATH / "utils" / "constants.py")
)
_phase1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase1)

# Models
MODELS = _phase1.MODELS
MODEL_DIR_NAMES = _phase1.MODEL_DIR_NAMES
MODEL_LAYERS = _phase1.MODEL_LAYERS
MODEL_HEADS = _phase1.MODEL_HEADS
MODEL_CAPACITY = _phase1.MODEL_CAPACITY
MODEL_TOTAL_EDGES = _phase1.MODEL_TOTAL_EDGES
MODEL_THRESHOLDS = _phase1.MODEL_THRESHOLDS
# Bands
BANDS = _phase1.BANDS
FREQUENCY_BANDS = _phase1.FREQUENCY_BANDS
CONTROL_BAND = _phase1.CONTROL_BAND
FREQUENCY_RANK = _phase1.FREQUENCY_RANK
BAND_NAMES = _phase1.BAND_NAMES
LOW_FREQ_BANDS = _phase1.LOW_FREQ_BANDS
HIGH_FREQ_BANDS = _phase1.HIGH_FREQ_BANDS
ADJACENT_PAIRS = _phase1.ADJACENT_PAIRS
CROSS_SPECTRUM_PAIRS = _phase1.CROSS_SPECTRUM_PAIRS
# Draws
DRAWS = _phase1.DRAWS
N_DRAWS = _phase1.N_DRAWS
# Plotting
BAND_COLORS = _phase1.BAND_COLORS
MODEL_COLORS = _phase1.MODEL_COLORS
FIGURE_DEFAULTS = _phase1.FIGURE_DEFAULTS
# Statistical
ALPHA = _phase1.ALPHA
N_BOOTSTRAP = _phase1.N_BOOTSTRAP
RANDOM_SEED = _phase1.RANDOM_SEED


PHASE2_BASE = PROJECT_ROOT / "LSC_circuit_analysis/02_Phase_Structural"
CIRCUITS_BASE = PROJECT_ROOT / "LSC_circuits/circuit_discovery/circuits"

# Output paths
OUTPUT_DIR = PHASE2_BASE / "outputs"
EXTRACTION_DIR = OUTPUT_DIR / "extraction"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
VIZ_DIR = OUTPUT_DIR / "viz"

# Extraction output files
ALL_CIRCUITS_JSON = EXTRACTION_DIR / "all_circuits_structure.json"
CIRCUITS_SUMMARY_CSV = EXTRACTION_DIR / "circuits_summary.csv"
INDIVIDUAL_CIRCUITS_DIR = EXTRACTION_DIR / "circuits"


def get_output_dirs():
    """Create and return output directories."""
    for d in [
        OUTPUT_DIR,
        EXTRACTION_DIR,
        ANALYSIS_DIR,
        VIZ_DIR,
        INDIVIDUAL_CIRCUITS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_DIR, VIZ_DIR


# =============================================================================
# MODEL ARCHITECTURE INFO (for edge parsing)
# =============================================================================

MODEL_INFO: Dict[str, Dict] = {
    "pythia-70m": {"n_layers": 6, "n_heads": 8, "d_model": 512},
    "pythia-160m": {"n_layers": 12, "n_heads": 12, "d_model": 768},
    "pythia-410m": {"n_layers": 24, "n_heads": 16, "d_model": 1024},
    "pythia-1b": {"n_layers": 16, "n_heads": 8, "d_model": 2048},
    "pythia-1.4b": {"n_layers": 24, "n_heads": 16, "d_model": 2048},
}


# =============================================================================
# COMPONENT TYPES
# =============================================================================

# Source component types (where edges come FROM)
SOURCE_TYPES: List[str] = ["embed", "attn", "mlp"]

# Destination component types (where edges go TO)
DESTINATION_TYPES: List[str] = ["attn_in", "mlp_in", "resid_post"]

# Simplified destination types for flow analysis
DEST_SIMPLIFIED: Dict[str, str] = {
    "attn_in": "attn",
    "mlp_in": "mlp",
    "resid_post": "resid",
}

# All component types
COMPONENT_TYPES: List[str] = ["embed", "attn", "mlp", "resid"]


# =============================================================================
# EDGE CATEGORIES
# =============================================================================

EDGE_CATEGORIES: List[str] = ["input", "output", "local", "forward", "skip"]

EDGE_CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "input": "Edges from embedding to first layer components",
    "output": "Edges to final residual stream (model output)",
    "local": "Edges within the same layer",
    "forward": "Edges to the next layer (layer distance = 1)",
    "skip": "Edges skipping layers (layer distance > 1)",
}


# =============================================================================
# PLOTTING (structural-specific)
# =============================================================================

COMPONENT_COLORS: Dict[str, str] = {
    "embed": "#e377c2",
    "attn": "#1f77b4",
    "attn_in": "#1f77b4",
    "mlp": "#ff7f0e",
    "mlp_in": "#ff7f0e",
    "resid": "#2ca02c",
    "resid_post": "#2ca02c",
}

EDGE_CATEGORY_COLORS: Dict[str, str] = {
    "input": "#e377c2",
    "output": "#d62728",
    "local": "#2ca02c",
    "forward": "#1f77b4",
    "skip": "#9467bd",
}
