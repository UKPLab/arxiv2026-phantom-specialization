"""
Plotting utilities for Phase Functional analysis.

All figures saved to disk (plt.savefig + plt.close). No plt.show().
Heatmaps: full square matrix, no gridlines.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from .constants import (
    BANDS,
    FREQUENCY_BANDS,
    MODELS,
    BAND_COLORS,
    MODEL_COLORS,
    BAND_NAMES,
    FIGURE_DEFAULTS,
    VIZ_DIR,
)


def setup_plotting():
    """Configure matplotlib with project defaults."""
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")
    plt.rcParams.update(
        {
            "figure.figsize": FIGURE_DEFAULTS["figsize"],
            "font.size": FIGURE_DEFAULTS["font_size"],
            "axes.titlesize": FIGURE_DEFAULTS["title_size"],
            "axes.labelsize": FIGURE_DEFAULTS["label_size"],
            "figure.dpi": FIGURE_DEFAULTS["dpi"],
        }
    )


def save_figure(fig, filename: str, viz_dir: Path = None, dpi: int = 150):
    """Save figure and close. No plt.show()."""
    if viz_dir is None:
        viz_dir = VIZ_DIR
    viz_dir.mkdir(parents=True, exist_ok=True)
    path = viz_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def get_band_labels(bands: List[str] = None) -> List[str]:
    """Get human-readable band labels."""
    if bands is None:
        bands = BANDS
    return [BAND_NAMES.get(b, b) for b in bands]


def plot_transfer_heatmap(
    matrix: pd.DataFrame,
    title: str,
    ax=None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "RdYlGn",
    annot_fmt: str = ".2f",
):
    """
    Plot cross-band transfer heatmap (square, no gridlines).

    matrix: DataFrame with train_band as index, test_band as columns.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    labels_x = get_band_labels(list(matrix.columns))
    labels_y = get_band_labels(list(matrix.index))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=annot_fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0,
        linecolor="none",
        ax=ax,
        xticklabels=labels_x,
        yticklabels=labels_y,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Test Band")
    ax.set_ylabel("Train Band")
    return ax


def plot_metric_heatmap(
    data: pd.DataFrame,
    values_col: str,
    title: str,
    ax=None,
    cmap: str = "RdYlGn",
    fmt: str = ".3f",
    vmin: float = None,
    vmax: float = None,
):
    """
    Plot a model x band heatmap for a single metric.

    data: DataFrame with 'model' and 'band' columns.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    pivot = data.pivot_table(
        values=values_col, index="model", columns="band", aggfunc="mean"
    )
    pivot = pivot.reindex(index=MODELS, columns=BANDS)

    labels_x = get_band_labels(list(pivot.columns))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0,
        linecolor="none",
        ax=ax,
        xticklabels=labels_x,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("Model")
    return ax


def plot_grouped_bars(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str,
    ax=None,
    ylabel: str = None,
    palette=None,
    add_errorbars: bool = True,
):
    """Generic grouped bar chart."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    if palette is None:
        palette = BAND_COLORS if hue == "band" else MODEL_COLORS

    if add_errorbars:
        sns.barplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            palette=palette,
            errorbar="sd",
            capsize=0.05,
        )
    else:
        sns.barplot(
            data=data,
            x=x,
            y=y,
            hue=hue,
            ax=ax,
            palette=palette,
            errorbar=None,
        )

    ax.set_title(title, fontsize=13)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(title=hue.capitalize(), bbox_to_anchor=(1.02, 1), loc="upper left")
    return ax


def plot_boxplot_by_band(
    data: pd.DataFrame,
    metric: str,
    title: str,
    ax=None,
    palette=None,
):
    """Boxplot of metric across bands."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    if palette is None:
        palette = BAND_COLORS

    sns.boxplot(
        data=data,
        x="band",
        y=metric,
        ax=ax,
        palette=palette,
        order=BANDS,
    )
    sns.stripplot(
        data=data,
        x="band",
        y=metric,
        ax=ax,
        color="black",
        alpha=0.5,
        size=4,
        order=BANDS,
    )
    ax.set_title(title, fontsize=13)
    ax.set_xticklabels(get_band_labels())
    ax.set_xlabel("Frequency Band")
    return ax


def plot_scaling_panel(
    data: pd.DataFrame,
    metrics: List[str],
    titles: List[str],
    ylabels: List[str],
    suptitle: str,
    ncols: int = 2,
):
    """Multi-panel scaling plot (metric vs model size, one line per band)."""
    from .constants import MODEL_CAPACITY

    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes)

    data = data.copy()
    data["model_size"] = data["model"].map(MODEL_CAPACITY)

    for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]

        for band in BANDS:
            band_data = data[data["band"] == band]
            agg = (
                band_data.groupby("model_size")[metric]
                .agg(["mean", "std"])
                .reset_index()
            )
            ax.errorbar(
                agg["model_size"],
                agg["mean"],
                yerr=agg["std"],
                marker="o",
                label=BAND_NAMES[band],
                color=BAND_COLORS[band],
                capsize=3,
                linewidth=1.5,
            )

        ax.set_xlabel("Model Size (M params)")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.set_xscale("log")
        ax.set_xticks([70, 160, 410, 1000])
        ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())

    # Legend on last subplot
    axes.flat[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    # Hide unused axes
    for idx in range(len(metrics), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle(suptitle, fontsize=15, y=1.02)
    fig.tight_layout()
    return fig
