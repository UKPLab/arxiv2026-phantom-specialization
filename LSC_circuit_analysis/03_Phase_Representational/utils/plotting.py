"""
Plotting utilities for Phase Representational analysis.

Provides setup, save, and Phase-3-specific visualization functions.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

from .constants import (
    BAND_COLORS,
    MODEL_COLORS,
    BAND_NAMES,
    FIGURE_DEFAULTS,
    BANDS,
    FREQUENCY_BANDS,
    MODELS,
    VIZ_DIR,
)


def setup_plotting() -> None:
    """Configure matplotlib/seaborn defaults for analysis notebooks."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": FIGURE_DEFAULTS["figsize"],
            "figure.dpi": FIGURE_DEFAULTS["dpi"],
            "font.size": FIGURE_DEFAULTS["font_size"],
            "axes.titlesize": FIGURE_DEFAULTS["title_size"],
            "axes.labelsize": FIGURE_DEFAULTS["label_size"],
        }
    )


def save_figure(
    fig: Figure, filename: str, viz_dir: Optional[Union[str, Path]] = None
) -> None:
    """Save figure to viz directory and close.

    Args:
        fig: Matplotlib figure.
        filename: Filename (without directory).
        viz_dir: Output directory. Defaults to VIZ_DIR.
    """
    if viz_dir is None:
        viz_dir = VIZ_DIR
    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    filepath = viz_dir / filename
    fig.savefig(filepath, dpi=FIGURE_DEFAULTS["dpi"], bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# GENERAL-PURPOSE PLOTS
# =============================================================================


def plot_metric_heatmap(
    data: pd.DataFrame,
    title: str,
    xlabel: str = "Band",
    ylabel: str = "Model",
    fmt: str = ".2f",
    cmap: str = "YlOrRd",
    figsize: Tuple[float, float] = (10, 5),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Figure:
    """Plot a model x band heatmap.

    Args:
        data: DataFrame with models as index and bands as columns.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        square=True,
        linewidths=0,
        linecolor="none",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig


def plot_boxplot_by_band(
    data: pd.DataFrame,
    x: str = "band",
    y: Optional[str] = None,
    hue: Optional[str] = None,
    title: str = "",
    ylabel: str = "",
    figsize: Tuple[float, float] = (10, 6),
) -> Figure:
    """Boxplot with stripplot overlay, colored by band.

    Args:
        data: DataFrame.
        x: Column for x-axis (default: 'band').
        y: Column for y-axis.
        hue: Optional grouping column.
        title: Figure title.
        ylabel: Y-axis label.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    palette = {b: BAND_COLORS[b] for b in data[x].unique() if b in BAND_COLORS}

    sns.boxplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        order=[b for b in BANDS if b in data[x].unique()],
        ax=ax,
        fliersize=0,
    )
    sns.stripplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        order=[b for b in BANDS if b in data[x].unique()],
        ax=ax,
        alpha=0.4,
        size=4,
        dodge=True,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if hue and ax.get_legend():
        ax.get_legend().remove()
    return fig


# =============================================================================
# REPRESENTATIONAL-SPECIFIC PLOTS
# =============================================================================


def plot_probe_trajectory(
    trajectory_df: pd.DataFrame,
    model: Optional[str] = None,
    title: str = "Probe Accuracy Trajectory",
    figsize: Tuple[float, float] = (12, 6),
) -> Figure:
    """Plot linear probe accuracy across layers.

    Args:
        trajectory_df: DataFrame with columns: layer, band, accuracy (and optionally std).
        model: Model name for title.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    bands_to_plot = [b for b in BANDS if b in trajectory_df["band"].unique()]
    for band in bands_to_plot:
        band_data = trajectory_df[trajectory_df["band"] == band].sort_values("layer")
        color = BAND_COLORS.get(band, "gray")
        label = BAND_NAMES.get(band, band)

        ax.plot(
            band_data["layer"],
            band_data["accuracy"],
            color=color,
            label=label,
            marker="o",
            markersize=4,
        )

        if "std" in band_data.columns:
            ax.fill_between(
                band_data["layer"],
                band_data["accuracy"] - band_data["std"],
                band_data["accuracy"] + band_data["std"],
                color=color,
                alpha=0.15,
            )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe Accuracy")
    full_title = f"{title}: {model}" if model else title
    ax.set_title(full_title)
    ax.legend(loc="best")
    ax.set_ylim(bottom=0)
    return fig


def plot_logit_lens_heatmap(
    values: np.ndarray,
    row_labels: Sequence,
    col_labels: Sequence,
    title: str = "Logit Lens",
    cmap: str = "viridis",
    fmt: str = ".2f",
    figsize: Optional[Tuple[float, float]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    annot: bool = True,
) -> Figure:
    """Plot logit lens heatmap.

    Args:
        values: 2D array of shape (n_rows, n_cols).
        row_labels: List of row labels.
        col_labels: List of column labels.
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size tuple (auto-computed if None).

    Returns:
        Matplotlib figure.
    """
    if figsize is None:
        n_rows, n_cols = len(row_labels), len(col_labels)
        figsize = (max(6, n_cols * 0.6 + 2), max(4, n_rows * 0.8 + 2))
    fig, ax = plt.subplots(figsize=figsize)
    df = pd.DataFrame(values, index=row_labels, columns=col_labels)
    heatmap_kws = dict(cmap=cmap, square=True, linewidths=0, linecolor="none", ax=ax)
    if annot:
        heatmap_kws.update(annot=True, fmt=fmt)
    else:
        heatmap_kws.update(annot=False)
    sns.heatmap(df, **heatmap_kws)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    return fig


def plot_cka_matrix(
    cka_matrix: np.ndarray,
    labels: Sequence,
    title: str = "CKA Similarity",
    figsize: Tuple[float, float] = (8, 7),
) -> Figure:
    """Plot CKA similarity matrix.

    Args:
        cka_matrix: Square similarity matrix.
        labels: Labels for rows/columns.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    df = pd.DataFrame(cka_matrix, index=labels, columns=labels)
    sns.heatmap(
        df,
        annot=False,
        cmap="RdYlBu_r",
        square=True,
        linewidths=0,
        linecolor="none",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title(title)
    return fig


def plot_head_role_map(
    roles_df: pd.DataFrame,
    model: str,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Figure:
    """Plot head x layer heatmap of head role classifications.

    Args:
        roles_df: DataFrame with columns: layer, head, role.
        model: Model name.
        title: Figure title.
        figsize: Figure size tuple (auto-computed if None).

    Returns:
        Matplotlib figure.
    """
    role_to_int = {"induction": 3, "previous_token": 2, "bos_sink": 1, "diffuse": 0}
    role_colors = ["#d9d9d9", "#ff9896", "#aec7e8", "#98df8a"]

    pivot = roles_df.pivot(index="head", columns="layer", values="role")
    numeric = pivot.map(lambda x: role_to_int.get(x, -1))

    if figsize is None:
        n_heads, n_layers = numeric.shape
        figsize = (max(10, n_layers * 0.6 + 3), max(5, n_heads * 0.5))
    fig, ax = plt.subplots(figsize=figsize)
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap(role_colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    im = ax.imshow(numeric.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head")
    ax.set_xticks(range(numeric.shape[1]))
    ax.set_xticklabels(numeric.columns)
    ax.set_yticks(range(numeric.shape[0]))
    ax.set_yticklabels(numeric.index)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=role_colors[3], label="Induction"),
        Patch(facecolor=role_colors[2], label="Previous Token"),
        Patch(facecolor=role_colors[1], label="BOS Sink"),
        Patch(facecolor=role_colors[0], label="Diffuse"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.2, 1))

    if title is None:
        title = f"Head Role Classification: {model}"
    ax.set_title(title)
    ax.grid(False)
    return fig


def plot_knn_confusion(
    confusion_matrix: np.ndarray,
    bands: Sequence[str],
    title: str = "k-NN Band Confusion",
    figsize: Tuple[float, float] = (8, 7),
) -> Figure:
    """Plot k-NN band distribution confusion matrix.

    Args:
        confusion_matrix: Square matrix P(neighbor_band | token_band).
        bands: List of band names.
        title: Figure title.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    df = pd.DataFrame(confusion_matrix, index=bands, columns=bands)
    sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        square=True,
        linewidths=0,
        linecolor="none",
        vmin=0,
        ax=ax,
    )
    ax.set_xlabel("Neighbor Band")
    ax.set_ylabel("Token Band")
    ax.set_title(title)
    return fig


def plot_mi_trajectory(
    mi_df: pd.DataFrame,
    model: Optional[str] = None,
    title: str = "MI Trajectory",
    figsize: Tuple[float, float] = (12, 6),
) -> Figure:
    """Plot mutual information trajectory across layers.

    Args:
        mi_df: DataFrame with columns: layer, mi (and optionally method).
        model: Model name for title.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if "method" in mi_df.columns:
        for method, group in mi_df.groupby("method"):
            group = group.sort_values("layer")
            ax.plot(group["layer"], group["mi"], label=method, marker="o", markersize=4)
    else:
        mi_df = mi_df.sort_values("layer")
        ax.plot(mi_df["layer"], mi_df["mi"], marker="o", markersize=4)

    ax.set_xlabel("Layer")
    ax.set_ylabel("MI (bits)")
    full_title = f"{title}: {model}" if model else title
    ax.set_title(full_title)
    if "method" in mi_df.columns:
        ax.legend()
    ax.set_ylim(bottom=0)
    return fig


def plot_attribution_stacked(
    attribution_df: pd.DataFrame,
    model: Optional[str] = None,
    title: str = "Component Attribution",
    figsize: Tuple[float, float] = (12, 6),
) -> Figure:
    """Stacked area plot of attention vs MLP attribution across layers.

    Args:
        attribution_df: DataFrame with columns: layer, band, attn_frac, mlp_frac.
        model: Model name for title.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, len(FREQUENCY_BANDS), figsize=figsize, sharey=True)
    if len(FREQUENCY_BANDS) == 1:
        axes = [axes]

    for ax, band in zip(axes, FREQUENCY_BANDS):
        band_data = attribution_df[attribution_df["band"] == band].sort_values("layer")
        if len(band_data) == 0:
            continue

        layers = band_data["layer"].values
        attn = band_data["attn_frac"].values
        mlp = band_data["mlp_frac"].values

        ax.stackplot(
            layers,
            attn,
            mlp,
            labels=["Attention", "MLP"],
            colors=["#1f77b4", "#ff7f0e"],
            alpha=0.7,
        )
        ax.set_title(BAND_NAMES.get(band, band))
        ax.set_xlabel("Layer")
        if ax == axes[0]:
            ax.set_ylabel("Attribution Fraction")

    axes[-1].legend(loc="upper right")
    full_title = f"{title}: {model}" if model else title
    fig.suptitle(full_title, y=1.02)
    fig.tight_layout()
    return fig


def plot_scaling_panel(
    data: pd.DataFrame,
    metrics: List[str],
    titles: List[str],
    x: str = "model_capacity",
    figsize: Tuple[float, float] = (16, 5),
) -> Figure:
    """Multi-panel scaling plot (metric vs model size).

    Args:
        data: DataFrame with model_capacity column.
        metrics: List of column names to plot.
        titles: List of panel titles.
        x: X-axis column.

    Returns:
        Matplotlib figure.
    """
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, metric, title in zip(axes, metrics, titles):
        for band in BANDS:
            band_data = data[data["band"] == band]
            if len(band_data) == 0:
                continue
            color = BAND_COLORS.get(band, "gray")
            label = BAND_NAMES.get(band, band)
            ax.plot(
                band_data[x],
                band_data[metric],
                color=color,
                label=label,
                marker="o",
                markersize=5,
            )

        ax.set_xlabel("Model Capacity (M)")
        ax.set_title(title)
        ax.set_xscale("log")

    axes[0].legend(loc="best")
    fig.tight_layout()
    return fig
