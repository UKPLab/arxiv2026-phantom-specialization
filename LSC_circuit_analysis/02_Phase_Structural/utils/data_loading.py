"""
Data loading for Phase Structural analysis.

Loads pre-extracted structural data from JSON/CSV files and builds
analysis-ready DataFrames. No GPU or PyTorch required.
"""

import os as _os
import json
from pathlib import Path as _Path
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _find_project_root() -> _Path:
    env = _os.environ.get("PROJECT_ROOT")
    if env:
        return _Path(env).resolve()
    for p in _Path(__file__).resolve().parents:
        if (p / "src" / "config.py").exists():
            return p
    return _Path(__file__).resolve().parents[1]


PROJECT_ROOT = _find_project_root()

from .constants import (
    ALL_CIRCUITS_JSON,
    CIRCUITS_SUMMARY_CSV,
    MODELS,
    BANDS,
    FREQUENCY_RANK,
    MODEL_INFO,
)


def load_extracted_data() -> Tuple[Dict, pd.DataFrame]:
    """
    Load all extracted circuit structures.

    Returns:
        (circuits_dict, df_structure)
        - circuits_dict: full JSON data keyed by circuit_id
        - df_structure: DataFrame with 60 rows (one per circuit)
    """
    with open(ALL_CIRCUITS_JSON, "r") as f:
        data = json.load(f)

    circuits = data["circuits"]
    n_total = data["metadata"]["n_circuits"]
    n_failed = data["metadata"]["n_failed"]

    print(f"Loaded {n_total} circuits ({n_failed} failed)")

    df = build_structure_df(circuits)
    return circuits, df


def build_structure_df(circuits: Dict) -> pd.DataFrame:
    """
    Build the main structural DataFrame (60 rows).

    One row per (model, band, draw) with edge counts, fractions,
    component breakdown, head participation, and derived metrics.
    """
    rows = []
    for circuit_id, c in circuits.items():
        if c.get("status") != "success":
            continue

        total = c["total_edges"]
        total_possible = c["total_possible_edges"]

        # Component type counts
        comp = c.get("by_component_type", {})
        comp_attn = comp.get("attn_in", 0)
        comp_mlp = comp.get("mlp_in", 0)
        comp_resid = comp.get("resid_post", 0)

        # Head stats
        by_head = c.get("by_head", {})
        n_heads = MODEL_INFO[c["model"]]["n_heads"]
        n_layers = MODEL_INFO[c["model"]]["n_layers"]
        total_heads = n_layers * n_heads
        active_heads = len(by_head)
        head_values = list(by_head.values()) if by_head else [0]

        row = {
            "circuit_id": circuit_id,
            "model": c["model"],
            "band": c["band"],
            "draw": c["draw"],
            "threshold": c.get("threshold"),
            # Edge counts
            "total_edges": total,
            "total_possible_edges": total_possible,
            "edge_fraction": c["edge_fraction"],
            # Edge categories
            "n_skip": c.get("n_skip_connections", 0),
            "n_input": c.get("n_input_edges", 0),
            "n_output": c.get("n_output_edges", 0),
            "n_local": c.get("n_local_edges", 0),
            "n_forward": c.get("n_forward_edges", 0),
            # Edge category fractions
            "skip_fraction": c.get("n_skip_connections", 0) / total if total > 0 else 0,
            "input_fraction": c.get("n_input_edges", 0) / total if total > 0 else 0,
            "output_fraction": c.get("n_output_edges", 0) / total if total > 0 else 0,
            "local_fraction": c.get("n_local_edges", 0) / total if total > 0 else 0,
            "forward_fraction": c.get("n_forward_edges", 0) / total if total > 0 else 0,
            # Component type counts
            "comp_attn_in": comp_attn,
            "comp_mlp_in": comp_mlp,
            "comp_resid_post": comp_resid,
            # Component type fractions
            "attn_fraction": comp_attn / total if total > 0 else 0,
            "mlp_fraction": comp_mlp / total if total > 0 else 0,
            "resid_fraction": comp_resid / total if total > 0 else 0,
            # Head participation
            "active_heads": active_heads,
            "total_heads": total_heads,
            "head_participation_rate": active_heads / total_heads
            if total_heads > 0
            else 0,
            "mean_edges_per_head": np.mean(head_values) if by_head else 0,
            "max_edges_per_head": max(head_values) if by_head else 0,
            # Frequency rank
            "frequency_rank": FREQUENCY_RANK.get(c["band"]),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["band"] = pd.Categorical(df["band"], categories=BANDS, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODELS, ordered=True)
    return df


def load_functional_data() -> pd.DataFrame:
    """
    Load Phase 1 functional data for structure-function linkage.

    Returns df_circuit from Phase 1 with retention_ratio, completeness, etc.
    Uses importlib to avoid namespace collision with Phase 2's utils package.
    """
    import sys
    import types
    import importlib
    import importlib.util

    phase1_utils = PROJECT_ROOT / "LSC_circuit_analysis/01_Phase_Functional/utils"
    pkg_name = "_phase1_utils"

    # Ensure Phase 1's constants are loaded (may already be from stats.py)
    if f"{pkg_name}.constants" not in sys.modules:
        _const_spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.constants", str(phase1_utils / "constants.py")
        )
        _const_mod = importlib.util.module_from_spec(_const_spec)
        _const_spec.loader.exec_module(_const_mod)

        _pkg = types.ModuleType(pkg_name)
        _pkg.__path__ = [str(phase1_utils)]
        _pkg.__package__ = pkg_name
        sys.modules[pkg_name] = _pkg
        sys.modules[f"{pkg_name}.constants"] = _const_mod

    # Load Phase 1's data_loading module
    _dl_spec = importlib.util.spec_from_file_location(
        f"{pkg_name}.data_loading", str(phase1_utils / "data_loading.py")
    )
    _dl_mod = importlib.util.module_from_spec(_dl_spec)
    _dl_mod.__package__ = pkg_name
    sys.modules[f"{pkg_name}.data_loading"] = _dl_mod
    _dl_spec.loader.exec_module(_dl_mod)

    df_circuit, _, _ = _dl_mod.build_all_dataframes()
    return df_circuit
