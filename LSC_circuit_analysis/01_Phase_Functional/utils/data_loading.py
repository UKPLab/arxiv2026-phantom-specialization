"""
Data loading for Phase Functional analysis.

Loads pre-computed metrics from 60 metrics.json files and builds
analysis-ready DataFrames. No GPU required.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .constants import (
    CIRCUITS_BASE,
    DISCOVERY_SUMMARY,
    MODELS,
    MODEL_DIR_NAMES,
    BANDS,
    DRAWS,
    FREQUENCY_RANK,
)


def load_single_metrics(model: str, band: str, draw: str) -> Dict:
    """Load a single metrics.json file."""
    model_dir = MODEL_DIR_NAMES[model]
    path = CIRCUITS_BASE / model_dir / band / draw / "metrics.json"
    with open(path, "r") as f:
        return json.load(f)


def load_all_metrics() -> List[Dict]:
    """Load all 60 metrics.json files. Returns list of raw dicts."""
    all_metrics = []
    for model in MODELS:
        for band in BANDS:
            for draw in DRAWS:
                try:
                    data = load_single_metrics(model, band, draw)
                    all_metrics.append(data)
                except FileNotFoundError as e:
                    print(f"WARNING: Missing {model}/{band}/{draw}: {e}")
    print(
        f"Loaded {len(all_metrics)} / {len(MODELS) * len(BANDS) * len(DRAWS)} metrics files"
    )
    return all_metrics


def load_summary() -> Dict:
    """Load the pre-computed discovery_summary.json."""
    with open(DISCOVERY_SUMMARY, "r") as f:
        return json.load(f)


def build_circuit_df(all_metrics: List[Dict]) -> pd.DataFrame:
    """
    Build the main circuit-level DataFrame (60 rows).

    One row per (model, band, draw) with base/circuit/ablation metrics
    plus derived metrics (retention, completeness, top1-top5 gap).
    """
    rows = []
    for m in all_metrics:
        row = {
            "model": m["model"],
            "band": m["band"],
            "draw": m["draw"],
            "threshold": m["threshold"],
            "n_edges": m["n_edges"],
            "total_edges": m["total_edges"],
            "size_fraction": m["size_fraction"],
        }
        # Base metrics
        for metric in [
            "accuracy",
            "top5_accuracy",
            "top10_accuracy",
            "mean_correct_prob",
        ]:
            row[f"base_{metric}"] = m["base_metrics"].get(metric, np.nan)
        # Circuit metrics (same-band = own test set)
        for metric in [
            "accuracy",
            "top5_accuracy",
            "top10_accuracy",
            "mean_correct_prob",
        ]:
            row[f"circuit_{metric}"] = m["circuit_metrics"].get(metric, np.nan)
        row["circuit_kl_div"] = m["circuit_metrics"].get("kl_div", np.nan)
        # Ablation metrics
        for metric in [
            "accuracy",
            "top5_accuracy",
            "top10_accuracy",
            "mean_correct_prob",
        ]:
            row[f"ablation_{metric}"] = m["ablation_metrics"].get(metric, np.nan)
        row["ablation_kl_div"] = m["ablation_metrics"].get("kl_div", np.nan)
        # Derived metrics
        base_acc = row["base_accuracy"]
        row["retention_ratio"] = (
            row["circuit_accuracy"] / base_acc if base_acc > 0 else np.nan
        )
        row["completeness"] = 1.0 - row["ablation_accuracy"]
        row["top1_top5_gap"] = row["circuit_top5_accuracy"] - row["circuit_accuracy"]
        # Frequency rank (None for control)
        row["frequency_rank"] = FREQUENCY_RANK.get(m["band"])
        # Necessity test
        row["necessity_pass"] = m.get("necessity_test") == "PASS"
        # Training time
        row["training_time_seconds"] = m.get("training_time_seconds", np.nan)

        rows.append(row)

    df = pd.DataFrame(rows)
    df["band"] = pd.Categorical(df["band"], categories=BANDS, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODELS, ordered=True)
    return df


def build_transfer_df(all_metrics: List[Dict]) -> pd.DataFrame:
    """
    Build the cross-band transfer DataFrame (60 circuits * 5 test bands = 300 rows).

    One row per (model, train_band, draw, test_band) with base and circuit
    metrics on that test band, plus frequency distance.
    """
    rows = []
    for m in all_metrics:
        for test_band, test_data in m["cross_band"].items():
            row = {
                "model": m["model"],
                "train_band": m["band"],
                "draw": m["draw"],
                "test_band": test_band,
                "same_band": m["band"] == test_band,
                # Base metrics on test band data
                "base_accuracy": test_data["base"]["accuracy"],
                "base_top5_accuracy": test_data["base"]["top5_accuracy"],
                "base_top10_accuracy": test_data["base"]["top10_accuracy"],
                "base_mean_correct_prob": test_data["base"]["mean_correct_prob"],
                # Circuit metrics (circuit trained on train_band, tested on test_band)
                "circuit_accuracy": test_data["circuit"]["accuracy"],
                "circuit_top5_accuracy": test_data["circuit"]["top5_accuracy"],
                "circuit_top10_accuracy": test_data["circuit"]["top10_accuracy"],
                "circuit_mean_correct_prob": test_data["circuit"]["mean_correct_prob"],
                "circuit_kl_div": test_data["circuit"].get("kl_div", np.nan),
            }
            # Frequency ranks and distance
            train_rank = FREQUENCY_RANK.get(m["band"])
            test_rank = FREQUENCY_RANK.get(test_band)
            row["train_rank"] = train_rank
            row["test_rank"] = test_rank
            if train_rank is not None and test_rank is not None:
                row["freq_distance"] = abs(train_rank - test_rank)
            else:
                row["freq_distance"] = np.nan
            # Retention on this test band
            base_acc = row["base_accuracy"]
            row["retention_ratio"] = (
                row["circuit_accuracy"] / base_acc if base_acc > 0 else np.nan
            )

            rows.append(row)

    df = pd.DataFrame(rows)
    df["train_band"] = pd.Categorical(df["train_band"], categories=BANDS, ordered=True)
    df["test_band"] = pd.Categorical(df["test_band"], categories=BANDS, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODELS, ordered=True)
    return df


def build_all_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict]]:
    """
    Load all data and return (df_circuit, df_transfer, raw_metrics).

    Main entry point for notebooks.
    """
    raw = load_all_metrics()
    df_circuit = build_circuit_df(raw)
    df_transfer = build_transfer_df(raw)
    return df_circuit, df_transfer, raw
