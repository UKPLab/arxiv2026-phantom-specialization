"""
Shared constants for Phase Representational analysis.

Imports shared constants from Phase 2 (Structural), which chains through
Phase 1 (Functional). Adds representational-specific paths, model configs,
hook points, and LSC sequence structure constants.
"""

import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple

# =============================================================================
# IMPORT SHARED CONSTANTS FROM PHASE 2 (which re-exports Phase 1)
# =============================================================================

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


_PHASE2_PATH = PROJECT_ROOT / "LSC_circuit_analysis/02_Phase_Structural"

_spec = importlib.util.spec_from_file_location(
    "phase2_constants", str(_PHASE2_PATH / "utils" / "constants.py")
)
_phase2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_phase2)

# --- From Phase 1 (via Phase 2) ---
# Models
MODELS = _phase2.MODELS
MODEL_DIR_NAMES = _phase2.MODEL_DIR_NAMES
MODEL_LAYERS = _phase2.MODEL_LAYERS
MODEL_HEADS = _phase2.MODEL_HEADS
MODEL_CAPACITY = _phase2.MODEL_CAPACITY
MODEL_TOTAL_EDGES = _phase2.MODEL_TOTAL_EDGES
MODEL_THRESHOLDS = _phase2.MODEL_THRESHOLDS
# Bands
BANDS = _phase2.BANDS
FREQUENCY_BANDS = _phase2.FREQUENCY_BANDS
CONTROL_BAND = _phase2.CONTROL_BAND
FREQUENCY_RANK = _phase2.FREQUENCY_RANK
BAND_NAMES = _phase2.BAND_NAMES
LOW_FREQ_BANDS = _phase2.LOW_FREQ_BANDS
HIGH_FREQ_BANDS = _phase2.HIGH_FREQ_BANDS
ADJACENT_PAIRS = _phase2.ADJACENT_PAIRS
CROSS_SPECTRUM_PAIRS = _phase2.CROSS_SPECTRUM_PAIRS
# Draws
DRAWS = _phase2.DRAWS
N_DRAWS = _phase2.N_DRAWS
# Plotting
BAND_COLORS = _phase2.BAND_COLORS
MODEL_COLORS = _phase2.MODEL_COLORS
FIGURE_DEFAULTS = _phase2.FIGURE_DEFAULTS
# Statistical
ALPHA = _phase2.ALPHA
N_BOOTSTRAP = _phase2.N_BOOTSTRAP
RANDOM_SEED = _phase2.RANDOM_SEED

# --- From Phase 2 ---
MODEL_INFO = _phase2.MODEL_INFO
CIRCUITS_BASE = _phase2.CIRCUITS_BASE
COMPONENT_TYPES = _phase2.COMPONENT_TYPES


# =============================================================================
# PHASE 3 PATHS
# =============================================================================

PHASE3_BASE = PROJECT_ROOT / "LSC_circuit_analysis/03_Phase_Representational"
DATASETS_BASE = PROJECT_ROOT / "LSC_data/datasets/matched"
TOKEN_POOLS_BASE = PROJECT_ROOT / "LSC_data/lsc_token_pools/matched"

# Output paths
OUTPUT_DIR = PHASE3_BASE / "outputs"
EXTRACTION_DIR = OUTPUT_DIR / "extraction"
ACTIVATIONS_DIR = EXTRACTION_DIR / "activations"
CIRCUIT_ACTIVATIONS_DIR = EXTRACTION_DIR / "circuit_activations"

# Legacy flat output dirs (kept for backward compatibility during migration)
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
VIZ_DIR = OUTPUT_DIR / "viz"

# Hierarchical output directories (domain-first)
DOMAIN_DIRS: Dict[str, Path] = {
    "embedding": OUTPUT_DIR / "embedding",
    "residual_stream": OUTPUT_DIR / "residual_stream",
    "logit_lens": OUTPUT_DIR / "logit_lens",
    "attention": OUTPUT_DIR / "attention",
    "mlp": OUTPUT_DIR / "mlp",
    "info_theoretic": OUTPUT_DIR / "info_theoretic",
    "transfer": OUTPUT_DIR / "transfer",
    "inferential": OUTPUT_DIR / "inferential",
}


def get_domain_dirs(domain: str, variant: str = "base") -> Tuple[Path, Path]:
    """Get (analysis_dir, viz_dir) for a domain and variant.

    Args:
        domain: Analysis domain (e.g. 'residual_stream', 'attention').
        variant: 'base', 'circuit', or 'comparison'.

    Returns:
        (analysis_dir, viz_dir) tuple, directories created if needed.
    """
    base = DOMAIN_DIRS[domain]
    if variant in ("base", "circuit", "comparison"):
        analysis = base / variant / "analysis"
        viz = base / variant / "viz"
    else:
        analysis = base / "analysis"
        viz = base / "viz"
    analysis.mkdir(parents=True, exist_ok=True)
    viz.mkdir(parents=True, exist_ok=True)
    return analysis, viz


def get_output_dirs():
    """Create and return legacy flat output directories."""
    for d in [
        OUTPUT_DIR,
        EXTRACTION_DIR,
        ACTIVATIONS_DIR,
        CIRCUIT_ACTIVATIONS_DIR,
        ANALYSIS_DIR,
        VIZ_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
    return ANALYSIS_DIR, VIZ_DIR


# Phase 1 & 2 output paths (for cross-phase loading)
PHASE1_ANALYSIS_DIR = (
    PROJECT_ROOT / "LSC_circuit_analysis/01_Phase_Functional/outputs/analysis"
)
PHASE2_ANALYSIS_DIR = (
    PROJECT_ROOT / "LSC_circuit_analysis/02_Phase_Structural/outputs/analysis"
)

# Circuit discovery paths
CIRCUITS_DIR = PROJECT_ROOT / "LSC_circuits/circuit_discovery/circuits"
CIRCUIT_FOLD_LN: bool = True  # Match circuit discovery setting (fold_ln=True)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

HF_MODEL_NAMES: Dict[str, str] = {
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
}

MODEL_D_MODEL: Dict[str, int] = {
    "pythia-70m": 512,
    "pythia-160m": 768,
    "pythia-410m": 1024,
    "pythia-1b": 2048,
    "pythia-1.4b": 2048,
}

MODEL_D_MLP: Dict[str, int] = {
    "pythia-70m": 2048,
    "pythia-160m": 3072,
    "pythia-410m": 4096,
    "pythia-1b": 8192,
    "pythia-1.4b": 8192,
}


# =============================================================================
# TRANSFORMERLENS HOOK POINTS
# =============================================================================

HOOK_POINTS: Dict[str, str] = {
    "embed": "hook_embed",
    "pos_embed": "hook_pos_embed",
    "resid_pre": "blocks.{layer}.hook_resid_pre",
    "resid_mid": "blocks.{layer}.hook_resid_mid",
    "resid_post": "blocks.{layer}.hook_resid_post",
    "attn_out": "blocks.{layer}.attn.hook_result",
    "attn_pattern": "blocks.{layer}.attn.hook_pattern",
    "mlp_out": "blocks.{layer}.mlp.hook_post",
    "mlp_pre": "blocks.{layer}.mlp.hook_pre",
}


# =============================================================================
# LSC SEQUENCE STRUCTURE
# =============================================================================

# Raw sequence structure (21 tokens, no BOS)
SEQ_LEN: int = 21
SOURCE_POS: List[int] = list(range(0, 5))  # S1..S5 at positions 0-4
TARGET_POS: int = 5  # T at position 5
DISTRACTOR_POS: List[int] = list(range(6, 16))  # R1..R10 at positions 6-15
REPEAT_POS: List[int] = list(range(16, 21))  # S1..S5 at positions 16-20
PREDICTION_POS: int = 20  # Final S5, predict T

# BOS offset: TransformerLens prepend_bos=True shifts all positions by +1
BOS_OFFSET: int = 1
SEQ_LEN_WITH_BOS: int = SEQ_LEN + BOS_OFFSET  # 22

# Model-space positions (after BOS prepend)
MODEL_SOURCE_POS: List[int] = [p + BOS_OFFSET for p in SOURCE_POS]
MODEL_TARGET_POS: int = TARGET_POS + BOS_OFFSET
MODEL_DISTRACTOR_POS: List[int] = [p + BOS_OFFSET for p in DISTRACTOR_POS]
MODEL_REPEAT_POS: List[int] = [p + BOS_OFFSET for p in REPEAT_POS]
MODEL_PREDICTION_POS: int = PREDICTION_POS + BOS_OFFSET  # 21

# Induction cue: the previous occurrence of the current token.
# At prediction pos 20 (raw), the model sees S5. The previous occurrence of S5
# is at pos 4 (source block). An induction head attends to this position (or to
# pos 5 = T, the token to copy). We use pos 4 as the induction cue because:
#   - It's the "match" signal (same token as current position)
#   - Attention to T (pos 5) is already captured by target_fraction
INDUCTION_CUE_POS: int = 4  # Raw position: S5 in source block (matching token)
MODEL_INDUCTION_CUE_POS: int = INDUCTION_CUE_POS + BOS_OFFSET


# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================

N_PERMUTATIONS: int = 1000
CV_FOLDS: int = 5
K_NEIGHBORS: int = 10
PROBE_REGULARIZATION: float = 1.0
MLE_K_RANGE: tuple = (5, 20)  # k1, k2 for MLE intrinsic dimensionality

# Logit lens convergence threshold
CONVERGENCE_THRESHOLD: float = 0.9

# Head role classification thresholds
HEAD_ROLE_THRESHOLDS: Dict[str, float] = {
    "induction_min": 0.15,  # Min attention to induction cue
    "bos_sink_min": 0.3,  # Min attention to BOS
    "prev_token_min": 0.2,  # Min attention to previous position
    "entropy_diffuse_min": 3.0,  # Min entropy (bits) for "diffuse"
    "dominance_ratio": 1.5,  # Top role must be this much stronger than runner-up
}

# Extraction batch size
EXTRACTION_BATCH_SIZE: int = 64
