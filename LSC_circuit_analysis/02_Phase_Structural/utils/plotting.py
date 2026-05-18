"""
Visualization utilities for Phase Structural analysis.

Provides setup, save, and structural-specific plot functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

from .constants import (
    BAND_COLORS,
    MODEL_COLORS,
    COMPONENT_COLORS,
    EDGE_CATEGORY_COLORS,
    BAND_NAMES,
    FIGURE_DEFAULTS,
    VIZ_DIR,
)


def setup_plotting():
    """Configure matplotlib defaults."""
    matplotlib.rcParams["figure.figsize"] = FIGURE_DEFAULTS["figsize"]
    matplotlib.rcParams["font.size"] = FIGURE_DEFAULTS["font_size"]
    matplotlib.rcParams["axes.titlesize"] = FIGURE_DEFAULTS["title_size"]
    matplotlib.rcParams["axes.labelsize"] = FIGURE_DEFAULTS["label_size"]
    matplotlib.rcParams["figure.dpi"] = FIGURE_DEFAULTS["dpi"]
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")


def save_figure(fig, filename: str, viz_dir: Path = None):
    """Save figure to disk and close."""
    if viz_dir is None:
        viz_dir = VIZ_DIR
    viz_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(viz_dir / filename, dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {viz_dir / filename}")


def plot_layer_flow_heatmap(
    flow_matrix: Dict,
    model: str,
    n_layers: int,
    title: str = None,
    ax: plt.Axes = None,
):
    """
    Plot layer-to-layer flow heatmap.

    Args:
        flow_matrix: Dict from layer_flow['flow'] (str keys)
        model: Model name for title
        n_layers: Number of layers
        title: Optional title override
        ax: Optional matplotlib axes
    """
    # Build matrix: rows = src_layer (-1 to n_layers-1), cols = dst_layer (0 to n_layers-1)
    src_labels = ["embed"] + [f"L{i}" for i in range(n_layers)]
    dst_labels = [f"L{i}" for i in range(n_layers)]

    matrix = np.zeros((len(src_labels), len(dst_labels)))
    for src_str, dst_dict in flow_matrix.items():
        src_idx = int(src_str) + 1  # -1 -> 0, 0 -> 1, etc.
        if 0 <= src_idx < len(src_labels):
            for dst_str, count in dst_dict.items():
                dst_idx = int(dst_str)
                if 0 <= dst_idx < len(dst_labels):
                    matrix[src_idx, dst_idx] = count

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(8, n_layers * 0.6), max(6, (n_layers + 1) * 0.5))
        )

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlOrRd",
        xticklabels=dst_labels,
        yticklabels=src_labels,
        square=True,
        linewidths=0,
        linecolor="none",
        cbar_kws={"label": "Edge Count"},
    )
    ax.set_xlabel("Destination Layer")
    ax.set_ylabel("Source Layer")
    ax.set_title(title or f"Layer Flow: {model}")


def plot_head_participation_heatmap(
    by_head: Dict[str, int],
    model: str,
    n_layers: int,
    n_heads: int,
    title: str = None,
    ax: plt.Axes = None,
):
    """
    Plot head participation heatmap (heads x layers).

    Args:
        by_head: Dict mapping 'A{layer}.{head}' -> edge count
        model: Model name for title
        n_layers: Number of layers
        n_heads: Number of heads per layer
        title: Optional title override
        ax: Optional matplotlib axes
    """
    # Build matrix as (n_heads, n_layers) so heads=rows(Y), layers=cols(X)
    matrix = np.zeros((n_heads, n_layers))
    for head_key, count in by_head.items():
        if head_key.startswith("A"):
            parts = head_key[1:].split(".")
            layer, head = int(parts[0]), int(parts[1])
            if layer < n_layers and head < n_heads:
                matrix[head, layer] = count

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, n_layers * 0.5), max(4, n_heads * 0.4)))

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="Blues",
        xticklabels=[f"L{l}" for l in range(n_layers)],
        yticklabels=[f"H{h}" for h in range(n_heads)],
        square=True,
        linewidths=0,
        linecolor="none",
        cbar_kws={"label": "Incoming Edges"},
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head")
    ax.set_title(title or f"Head Participation: {model}")


def plot_jaccard_heatmap(
    matrix: np.ndarray,
    labels: List[str],
    model: str,
    title: str = None,
    ax: plt.Axes = None,
):
    """
    Plot Jaccard similarity heatmap.

    Args:
        matrix: Square similarity matrix
        labels: Labels for rows/columns
        model: Model name for title
        title: Optional title override
        ax: Optional matplotlib axes
    """
    if ax is None:
        n = len(labels)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.4), max(6, n * 0.35)))

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdYlBu_r",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0,
        linecolor="none",
        cbar_kws={"label": "Jaccard Similarity"},
    )
    ax.set_title(title or f"Jaccard Similarity: {model}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)


def plot_band_jaccard_heatmap(
    band_jaccard: Dict[str, Dict[str, float]],
    model: str,
    bands: List[str] = None,
    title: str = None,
    ax: plt.Axes = None,
):
    """
    Plot band-to-band mean Jaccard heatmap.
    """
    if bands is None:
        from .constants import BANDS

        bands = BANDS

    matrix = np.zeros((len(bands), len(bands)))
    for i, b1 in enumerate(bands):
        for j, b2 in enumerate(bands):
            matrix[i, j] = band_jaccard.get(b1, {}).get(b2, 0)

    display_labels = [BAND_NAMES.get(b, b) for b in bands]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdYlBu_r",
        xticklabels=display_labels,
        yticklabels=display_labels,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0,
        linecolor="none",
        cbar_kws={"label": "Mean Jaccard Similarity"},
    )
    ax.set_title(title or f"Band-to-Band Jaccard: {model}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


# =============================================================================
# DEEP STRUCTURAL ANALYSIS PLOTS
# =============================================================================


def plot_component_jaccard_comparison(
    results: Dict[str, Dict[str, List[float]]],
    model: str,
    title: str = None,
    ax: plt.Axes = None,
):
    """
    Bar plot comparing within/between-band Jaccard across component types.

    Args:
        results: {component: {'within': [...], 'between': [...]}}
    """
    components = list(results.keys())
    within_means = [np.mean(results[c]["within"]) for c in components]
    between_means = [np.mean(results[c]["between"]) for c in components]
    within_stds = [np.std(results[c]["within"]) for c in components]
    between_stds = [np.std(results[c]["between"]) for c in components]

    x = np.arange(len(components))
    width = 0.35

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(
        x - width / 2,
        within_means,
        width,
        yerr=within_stds,
        label="Within-band",
        color="#2ca02c",
        alpha=0.8,
        capsize=3,
    )
    bars2 = ax.bar(
        x + width / 2,
        between_means,
        width,
        yerr=between_stds,
        label="Between-band",
        color="#d62728",
        alpha=0.8,
        capsize=3,
    )

    ax.set_ylabel("Jaccard Similarity")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in components])
    ax.legend()
    ax.set_title(title or f"Component-Level Jaccard: {model}")


def plot_layer_sensitivity_profile(
    df: "pd.DataFrame",
    models: List[str] = None,
    title: str = None,
    ax: plt.Axes = None,
):
    """
    Line plot of mean Jaccard (y) vs layer (x): the band sensitivity profile.

    Low Jaccard = high sensitivity to frequency band.
    """
    from .constants import MODEL_COLORS

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    if models is None:
        models = df["model"].unique()

    for model in models:
        sub = df[df["model"] == model].sort_values("layer")
        color = MODEL_COLORS.get(model, None)
        ax.plot(
            sub["layer"],
            sub["mean_jaccard"],
            "o-",
            label=model,
            color=color,
            markersize=5,
            linewidth=2,
        )
        ax.fill_between(
            sub["layer"],
            sub["mean_jaccard"] - sub["std_jaccard"],
            sub["mean_jaccard"] + sub["std_jaccard"],
            alpha=0.15,
            color=color,
        )

    ax.set_xlabel("Destination Layer")
    ax.set_ylabel("Mean Pairwise Jaccard (between bands)")
    ax.set_title(title or "Layer-wise Band Sensitivity Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_edge_sharing_profile(
    df: "pd.DataFrame",
    model: str,
    property_col: str = "dst_type",
    title: str = None,
    ax: plt.Axes = None,
):
    """
    Stacked bar showing property breakdown by sharing level.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    pivot = df.groupby(["sharing_level", property_col]).size().unstack(fill_value=0)
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    pivot_pct.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_xlabel("Sharing Level (# bands)")
    ax.set_ylabel("Percentage")
    ax.set_title(title or f"Edge Properties by Sharing Level: {model}")
    ax.legend(title=property_col, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)


def plot_band_affinity_heatmap(
    matrix: np.ndarray,
    labels: List[str],
    model: str,
    title: str = None,
    ax: plt.Axes = None,
):
    """
    5x5 heatmap of non-universal edge sharing between bands.
    """
    display_labels = [BAND_NAMES.get(b, b) for b in labels]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlOrRd",
        xticklabels=display_labels,
        yticklabels=display_labels,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0,
        linecolor="none",
        cbar_kws={"label": "Jaccard (excl. universal)"},
    )
    ax.set_title(title or f"Band Affinity (non-universal edges): {model}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")


def plot_head_universality_map(
    df: "pd.DataFrame",
    model: str,
    n_layers: int,
    n_heads: int,
    title: str = None,
    ax: plt.Axes = None,
):
    """
    Head x layer heatmap colored by universality score (n_bands present).

    Average across draws. Heads on Y-axis, layers on X-axis.
    """
    # Average n_bands across draws
    avg = df.groupby(["layer", "head_idx"])["n_bands"].mean().reset_index()

    # Build matrix as (n_heads, n_layers) so heads=rows(Y), layers=cols(X)
    matrix = np.zeros((n_heads, n_layers))
    for _, row in avg.iterrows():
        matrix[int(row["head_idx"]), int(row["layer"])] = row["n_bands"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, n_layers * 0.5), max(4, n_heads * 0.4)))

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdYlGn",
        vmin=0,
        vmax=5,
        xticklabels=[f"L{l}" for l in range(n_layers)],
        yticklabels=[f"H{h}" for h in range(n_heads)],
        annot=False,
        square=True,
        linewidths=0,
        linecolor="none",
        cbar_kws={"label": "Bands Present (out of 5)"},
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head")
    ax.set_title(title or f"Head Universality: {model}")


def plot_graph_metrics_comparison(
    df: "pd.DataFrame",
    metrics: List[str] = None,
    title: str = None,
):
    """
    Multi-panel box plots of graph metrics by band, faceted by model.

    Returns figure for save_figure().
    """
    if metrics is None:
        metrics = ["diameter", "clustering_coefficient", "density", "avg_path_length"]

    n_metrics = len(metrics)
    n_models = df["model"].nunique()
    fig, axes = plt.subplots(
        n_metrics, n_models, figsize=(4 * n_models, 3.5 * n_metrics)
    )

    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    if n_models == 1:
        axes = axes.reshape(-1, 1)

    models = sorted(df["model"].unique())
    for j, model in enumerate(models):
        sub = df[df["model"] == model]
        for i, metric in enumerate(metrics):
            ax = axes[i, j]
            band_order = [
                b
                for b in ["low", "medium", "high", "very_high", "control"]
                if b in sub["band"].unique()
            ]
            sns.boxplot(
                data=sub,
                x="band",
                y=metric,
                order=band_order,
                ax=ax,
                palette={b: BAND_COLORS.get(b, "#999") for b in band_order},
            )
            if j == 0:
                ax.set_ylabel(metric.replace("_", " ").title())
            else:
                ax.set_ylabel("")
            if i == 0:
                ax.set_title(model)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=45)

    fig.suptitle(
        title or "Graph-Theoretic Metrics by Band and Model", fontsize=14, y=1.01
    )
    fig.tight_layout()
    return fig
