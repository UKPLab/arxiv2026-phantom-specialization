"""
Derived metrics computation for Phase Functional analysis.

All metrics are computed from pre-loaded DataFrames -- no GPU required.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from .constants import (
    BANDS,
    FREQUENCY_BANDS,
    MODELS,
    LOW_FREQ_BANDS,
    HIGH_FREQ_BANDS,
    BAND_NAMES,
    MODEL_CAPACITY,
    FREQUENCY_RANK,
)


def compute_transfer_matrix(
    df_transfer: pd.DataFrame,
    model: str,
    metric: str = "circuit_accuracy",
    bands: List[str] = None,
) -> pd.DataFrame:
    """
    Compute cross-band transfer matrix for a specific model.

    Returns DataFrame with train_band as index, test_band as columns,
    values are mean metric across draws.
    """
    if bands is None:
        bands = BANDS
    data = df_transfer[df_transfer["model"] == model]
    matrix = data.pivot_table(
        values=metric, index="train_band", columns="test_band", aggfunc="mean"
    )
    matrix = matrix.reindex(index=bands, columns=bands)
    return matrix


def compute_generalization_gap(
    df_transfer: pd.DataFrame,
    model: str,
    metric: str = "circuit_accuracy",
) -> pd.DataFrame:
    """
    Compute generalization gap per (model, train_band, draw).

    Gap = same_band_metric - mean(cross_band_metric).
    """
    data = df_transfer[df_transfer["model"] == model].copy()
    results = []
    for (train_band, draw), group in data.groupby(
        ["train_band", "draw"], observed=True
    ):
        same = group[group["same_band"]][metric].values
        cross = group[~group["same_band"]][metric].values
        same_val = same[0] if len(same) > 0 else np.nan
        cross_mean = np.mean(cross) if len(cross) > 0 else np.nan
        results.append(
            {
                "model": model,
                "train_band": str(train_band),
                "draw": draw,
                "same_band_metric": same_val,
                "cross_band_mean": cross_mean,
                "gap": same_val - cross_mean,
                "transfer_ratio": cross_mean / same_val if same_val > 0 else np.nan,
            }
        )
    return pd.DataFrame(results)


def compute_asymmetric_transfer(
    df_transfer: pd.DataFrame,
    model: str,
    metric: str = "circuit_accuracy",
    low_bands: List[str] = None,
    high_bands: List[str] = None,
) -> Dict:
    """
    Compute asymmetric transfer: LF->HF vs HF->LF.

    Returns dict with individual values, means, asymmetry, and ratio.
    """
    if low_bands is None:
        low_bands = LOW_FREQ_BANDS
    if high_bands is None:
        high_bands = HIGH_FREQ_BANDS

    data = df_transfer[df_transfer["model"] == model]

    # LF->HF: circuits trained on low bands, tested on high bands
    lf_hf_mask = (data["train_band"].isin(low_bands)) & (
        data["test_band"].isin(high_bands)
    )
    lf_hf = data.loc[lf_hf_mask, metric].values

    # HF->LF: circuits trained on high bands, tested on low bands
    hf_lf_mask = (data["train_band"].isin(high_bands)) & (
        data["test_band"].isin(low_bands)
    )
    hf_lf = data.loc[hf_lf_mask, metric].values

    lf_hf_mean = float(np.mean(lf_hf)) if len(lf_hf) > 0 else np.nan
    hf_lf_mean = float(np.mean(hf_lf)) if len(hf_lf) > 0 else np.nan

    return {
        "lf_to_hf_values": lf_hf.tolist(),
        "hf_to_lf_values": hf_lf.tolist(),
        "lf_to_hf_mean": lf_hf_mean,
        "hf_to_lf_mean": hf_lf_mean,
        "asymmetry": lf_hf_mean - hf_lf_mean,
        "asymmetry_ratio": lf_hf_mean / hf_lf_mean if hf_lf_mean > 0 else np.nan,
    }


def compute_pairwise_asymmetry_matrix(
    df_transfer: pd.DataFrame,
    model: str,
    metric: str = "circuit_accuracy",
    bands: List[str] = None,
) -> pd.DataFrame:
    """
    Compute pairwise asymmetry matrix.

    Entry (i, j) = mean transfer(band_i -> band_j) - mean transfer(band_j -> band_i).
    Positive means band_i circuits transfer to band_j data better than the reverse.
    """
    if bands is None:
        bands = FREQUENCY_BANDS
    matrix = compute_transfer_matrix(df_transfer, model, metric, bands)
    asymm = pd.DataFrame(
        np.zeros((len(bands), len(bands))),
        index=bands,
        columns=bands,
    )
    for i, b1 in enumerate(bands):
        for j, b2 in enumerate(bands):
            if i != j:
                asymm.loc[b1, b2] = matrix.loc[b1, b2] - matrix.loc[b2, b1]
    return asymm


def compute_model_scaling_df(df_circuit: pd.DataFrame) -> pd.DataFrame:
    """Add model_size column for scaling analysis."""
    df = df_circuit.copy()
    df["model_size"] = df["model"].map(MODEL_CAPACITY)
    return df
