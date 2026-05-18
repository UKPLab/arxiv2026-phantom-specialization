"""
Data loading utilities for Phase Representational analysis.

Loads LSC datasets, pre-extracted activations, and cross-phase data
from Phases 1 (Functional) and 2 (Structural).
"""

import json
import importlib
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .constants import (
    MODELS,
    BANDS,
    DRAWS,
    FREQUENCY_BANDS,
    FREQUENCY_RANK,
    MODEL_DIR_NAMES,
    MODEL_CAPACITY,
    DATASETS_BASE,
    TOKEN_POOLS_BASE,
    ACTIVATIONS_DIR,
    CIRCUIT_ACTIVATIONS_DIR,
    EXTRACTION_DIR,
    ANALYSIS_DIR,
    DOMAIN_DIRS,
    get_domain_dirs,
    PHASE1_ANALYSIS_DIR,
    PHASE2_ANALYSIS_DIR,
)


# =============================================================================
# LSC DATASET LOADING
# =============================================================================


def load_lsc_dataset(draw: str, band: str, split: str = "test") -> Dict:
    """Load a single LSC dataset file.

    Args:
        draw: Draw name (e.g. 'draw_1').
        band: Frequency band name.
        split: Data split ('train', 'test', 'val').

    Returns:
        Dict with keys: examples, n_examples, sequence_structure, etc.
    """
    path = DATASETS_BASE / draw / band / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_all_lsc_datasets(split: str = "test") -> Dict[Tuple[str, str], Dict]:
    """Load all LSC datasets for all draws and bands.

    Args:
        split: Data split ('train', 'test', 'val').

    Returns:
        Dict keyed by (draw, band) tuples.
    """
    datasets = {}
    for draw in DRAWS:
        for band in BANDS:
            try:
                datasets[(draw, band)] = load_lsc_dataset(draw, band, split)
            except FileNotFoundError:
                pass
    return datasets


def get_token_ids_and_targets(dataset: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Extract token IDs and target IDs from a loaded dataset.

    Args:
        dataset: Loaded dataset dict.

    Returns:
        (input_ids, target_ids) where input_ids is (N, 21) and target_ids is (N,).
    """
    examples = dataset["examples"]
    input_ids = np.array([ex["token_ids"] for ex in examples])
    # Target is at position 5 (the token after source pattern)
    target_ids = input_ids[:, 5]
    return input_ids, target_ids


def load_token_pool(band: str) -> Dict:
    """Load a token pool file for a specific band.

    Args:
        band: Frequency band name.

    Returns:
        Dict with token pool information.
    """
    path = TOKEN_POOLS_BASE / f"lsc_pool_{band}.json"
    if not path.exists():
        raise FileNotFoundError(f"Token pool not found: {path}")
    with open(path) as f:
        return json.load(f)


# =============================================================================
# EXTRACTED ACTIVATIONS LOADING
# =============================================================================


def get_activation_path(model: str, band: str, draw: str) -> Path:
    """Get the path to a pre-extracted activation NPZ file.

    Args:
        model: Model name (e.g. 'pythia-70m').
        band: Frequency band name.
        draw: Draw name (e.g. 'draw_1').

    Returns:
        Path to NPZ file.
    """
    model_dir = MODEL_DIR_NAMES.get(model, model.replace("-", "_"))
    return ACTIVATIONS_DIR / f"{model_dir}_{band}_{draw}.npz"


def load_extracted_activations(
    model: str, band: str, draw: str
) -> Dict[str, np.ndarray]:
    """Load pre-extracted activations from NPZ file.

    Args:
        model: Model name.
        band: Frequency band name.
        draw: Draw name.

    Returns:
        Dict of numpy arrays keyed by array name.
    """
    path = get_activation_path(model, band, draw)
    if not path.exists():
        raise FileNotFoundError(f"Activations not found: {path}")
    return dict(np.load(path, allow_pickle=False))


def load_all_extracted_activations(models=None, bands=None, draws=None):
    """Load all extracted activations.

    Args:
        models: List of models (default: MODELS).
        bands: List of bands (default: BANDS).
        draws: List of draws (default: DRAWS).

    Returns:
        Nested dict: activations[model][band][draw] = {array_name: ndarray}.
    """
    models = models or MODELS
    bands = bands or BANDS
    draws = draws or DRAWS

    activations = {}
    for model in models:
        activations[model] = {}
        for band in bands:
            activations[model][band] = {}
            for draw in draws:
                try:
                    activations[model][band][draw] = load_extracted_activations(
                        model, band, draw
                    )
                except FileNotFoundError:
                    pass
    return activations


def load_extraction_summary() -> pd.DataFrame:
    """Load extraction summary CSV.

    Returns:
        DataFrame with one row per extracted configuration.
    """
    path = EXTRACTION_DIR / "extraction_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Extraction summary not found: {path}")
    return pd.read_csv(path)


def load_copy_scores(model: str) -> Dict[str, np.ndarray]:
    """Load pre-computed copy scores for a model.

    Args:
        model: Model name.

    Returns:
        Dict of numpy arrays.
    """
    model_dir = MODEL_DIR_NAMES.get(model, model.replace("-", "_"))
    path = ACTIVATIONS_DIR / f"copy_scores_{model_dir}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Copy scores not found: {path}")
    return dict(np.load(path, allow_pickle=False))


# =============================================================================
# CROSS-PHASE DATA LOADING
# =============================================================================


def load_functional_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Phase 1 functional analysis DataFrames.

    Returns:
        (df_circuit, df_transfer) tuple of DataFrames.
    """
    circuit_path = PHASE1_ANALYSIS_DIR / "full_circuit_data.csv"
    transfer_path = PHASE1_ANALYSIS_DIR / "full_transfer_data.csv"

    df_circuit = pd.read_csv(circuit_path) if circuit_path.exists() else pd.DataFrame()
    df_transfer = (
        pd.read_csv(transfer_path) if transfer_path.exists() else pd.DataFrame()
    )

    return df_circuit, df_transfer


def load_structural_data() -> pd.DataFrame:
    """Load Phase 2 structural analysis DataFrame.

    Returns:
        DataFrame with structural metrics per circuit.
    """
    path = PHASE2_ANALYSIS_DIR / "full_structure_data.csv"
    if not path.exists():
        # Try master summary
        path = PHASE2_ANALYSIS_DIR / "master_structural_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_jaccard_matrices() -> pd.DataFrame:
    """Load Phase 2 Jaccard similarity matrices.

    Returns:
        DataFrame with pairwise Jaccard similarities.
    """
    path = PHASE2_ANALYSIS_DIR / "band_jaccard.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_phase_csv(phase: int, filename: str) -> pd.DataFrame:
    """Generic loader for any analysis CSV from Phase 1 or 2.

    Args:
        phase: Phase number (1 or 2).
        filename: CSV filename.

    Returns:
        DataFrame.
    """
    if phase == 1:
        path = PHASE1_ANALYSIS_DIR / filename
    elif phase == 2:
        path = PHASE2_ANALYSIS_DIR / filename
    else:
        raise ValueError(f"Unknown phase: {phase}")

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


# =============================================================================
# REPRESENTATIONAL DATAFRAME BUILDER
# =============================================================================


def build_representational_df(metrics_list: List[Dict]) -> pd.DataFrame:
    """Build a 60-row DataFrame of representational metrics.

    Args:
        metrics_list: List of dicts with keys: model, band, draw, + metric columns.

    Returns:
        DataFrame with Categorical columns for model/band/draw.
    """
    df = pd.DataFrame(metrics_list)

    # Apply categorical ordering
    if "model" in df.columns:
        df["model"] = pd.Categorical(df["model"], categories=MODELS, ordered=True)
    if "band" in df.columns:
        df["band"] = pd.Categorical(df["band"], categories=BANDS, ordered=True)
    if "draw" in df.columns:
        df["draw"] = pd.Categorical(df["draw"], categories=DRAWS, ordered=True)

    # Add frequency rank
    if "band" in df.columns and "frequency_rank" not in df.columns:
        df["frequency_rank"] = df["band"].map(FREQUENCY_RANK)

    # Add model capacity
    if "model" in df.columns and "model_capacity" not in df.columns:
        df["model_capacity"] = df["model"].map(MODEL_CAPACITY)

    return df


# =============================================================================
# UTILITY
# =============================================================================


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy and torch types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        try:
            import torch

            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
        except ImportError:
            pass
        return super().default(obj)


def save_analysis(data, filename, analysis_dir=None):
    """Save analysis results to CSV or JSON.

    Args:
        data: DataFrame or dict.
        filename: Output filename.
        analysis_dir: Output directory (default: ANALYSIS_DIR).
    """
    if analysis_dir is None:
        analysis_dir = ANALYSIS_DIR
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    filepath = analysis_dir / filename

    if isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
    elif isinstance(data, dict):
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
    elif isinstance(data, np.ndarray):
        np.save(filepath, data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


# =============================================================================
# DOMAIN-AWARE LOADING (hierarchical output structure)
# =============================================================================


def load_domain_csv(domain: str, variant: str, filename: str) -> pd.DataFrame:
    """Load a CSV from the hierarchical output structure.

    Args:
        domain: Analysis domain (e.g. 'residual_stream').
        variant: 'base', 'circuit', or 'comparison'.
        filename: CSV filename.

    Returns:
        DataFrame.
    """
    analysis_dir, _ = get_domain_dirs(domain, variant)
    path = analysis_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path)


def save_domain_analysis(data, filename, domain: str, variant: str = "base"):
    """Save analysis results to domain-specific directory.

    Args:
        data: DataFrame or dict.
        filename: Output filename.
        domain: Analysis domain.
        variant: 'base', 'circuit', or 'comparison'.
    """
    analysis_dir, _ = get_domain_dirs(domain, variant)
    save_analysis(data, filename, analysis_dir=analysis_dir)
