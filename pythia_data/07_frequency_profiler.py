#!/usr/bin/env python3
"""
Frequency Distribution Profiler
================================
Pipeline step 07.  Run BEFORE defining any banding scheme.

PURPOSE
-------
Profile the continuous log-frequency distribution of the full BPE vocabulary
so that discretization decisions (how many bands, where to cut) are driven
by the data rather than guesswork.

This script answers four questions:

    Q1  SHAPE:      Is the distribution normal?  Skewed?  Heavy-tailed?
    Q2  MODALITY:   How many natural groups exist in the distribution?
    Q3  BREAKS:     Where are the natural cut points?
    Q4  TAILS:      How do the extremes differ from the bulk?

Each question is addressed by multiple independent statistical methods.
If the methods converge, we have high confidence in the answer.

INPUT
-----
all_tokens_complete.csv  (from 06_build_token_dataset.py)
Required column:  log_frequency

OUTPUTS
-------
frequency_profile.json             all statistics, machine-readable
frequency_profile_report.txt       human-readable summary with conclusions
fig_shape.png                      histogram + KDE + QQ-plot + CDF
fig_modality.png                   KDE modes, GMM components, silhouette
fig_breaks.png                     natural breaks overlaid on distribution
fig_tails.png                      tail analysis and density ratios

Usage:
    python 07_frequency_profiler.py
    python 07_frequency_profiler.py --csv-file path/to/tokens.csv
    python 07_frequency_profiler.py --max-k 15
"""

import json
import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


SCRIPT_DIR = Path(__file__).resolve().parent
STAGE0_CSV = SCRIPT_DIR / "token_dataset" / "all_tokens_complete.csv"
OUTPUT_DIR = SCRIPT_DIR / "frequency_profile"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ============================================================================
# Q1: DISTRIBUTION SHAPE
# ============================================================================


def compute_descriptive_stats(x: np.ndarray) -> Dict:
    """Basic descriptive statistics."""
    pctiles = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]
    pct_values = {f"p{p}": round(float(np.percentile(x, p)), 6) for p in pctiles}

    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))

    return OrderedDict(
        [
            ("n", int(len(x))),
            ("min", round(float(x.min()), 6)),
            ("max", round(float(x.max()), 6)),
            ("range", round(float(x.max() - x.min()), 6)),
            ("mean", round(float(x.mean()), 6)),
            ("median", round(float(np.median(x)), 6)),
            ("std", round(float(x.std()), 6)),
            ("var", round(float(x.var()), 6)),
            ("iqr", round(iqr, 6)),
            (
                "cv",
                round(
                    float(x.std() / abs(x.mean())) if x.mean() != 0 else float("inf"), 4
                ),
            ),
            ("percentiles", pct_values),
        ]
    )


def test_normality(x: np.ndarray) -> Dict:
    """
    Multiple normality tests.  If they all reject, the distribution is
    clearly non-normal.  If results are mixed, we report the ambiguity.

    Tests:
      - Shapiro-Wilk (best power for small-moderate n; subsampled for n>5000)
      - D'Agostino-Pearson (combines skewness + kurtosis tests)
      - Anderson-Darling (sensitive to tails)
      - Kolmogorov-Smirnov (against fitted normal)
    """
    results = OrderedDict()

    # Skewness and kurtosis (Fisher definition: normal = 0)
    skewness = float(stats.skew(x))
    kurt = float(stats.kurtosis(x))  # excess kurtosis
    results["skewness"] = round(skewness, 4)
    results["excess_kurtosis"] = round(kurt, 4)
    results["skewness_interpretation"] = (
        "approximately symmetric"
        if abs(skewness) < 0.5
        else "moderately skewed"
        if abs(skewness) < 1.0
        else "highly skewed"
    )
    results["kurtosis_interpretation"] = (
        "mesokurtic (normal-like tails)"
        if abs(kurt) < 1.0
        else "leptokurtic (heavy tails)"
        if kurt > 0
        else "platykurtic (light tails)"
    )

    # Shapiro-Wilk (subsample if large n: test is designed for n <= 5000)
    sw_n = min(5000, len(x))
    rng = np.random.RandomState(42)
    x_sw = rng.choice(x, size=sw_n, replace=False) if len(x) > sw_n else x
    sw_stat, sw_p = stats.shapiro(x_sw)
    results["shapiro_wilk"] = OrderedDict(
        [
            ("statistic", round(float(sw_stat), 6)),
            ("p_value", float(sw_p)),
            ("sample_size", sw_n),
            ("reject_at_005", bool(sw_p < 0.05)),
        ]
    )

    # D'Agostino-Pearson (requires n >= 20)
    if len(x) >= 20:
        da_stat, da_p = stats.normaltest(x)
        results["dagostino_pearson"] = OrderedDict(
            [
                ("statistic", round(float(da_stat), 4)),
                ("p_value", float(da_p)),
                ("reject_at_005", bool(da_p < 0.05)),
            ]
        )

    # Anderson-Darling
    ad = stats.anderson(x, dist="norm")
    ad_reject_5 = bool(ad.statistic > ad.critical_values[2])  # index 2 = 5%
    results["anderson_darling"] = OrderedDict(
        [
            ("statistic", round(float(ad.statistic), 4)),
            ("critical_value_5pct", round(float(ad.critical_values[2]), 4)),
            ("reject_at_005", ad_reject_5),
        ]
    )

    # Kolmogorov-Smirnov against fitted normal
    ks_stat, ks_p = stats.kstest(x, "norm", args=(x.mean(), x.std()))
    results["kolmogorov_smirnov"] = OrderedDict(
        [
            ("statistic", round(float(ks_stat), 6)),
            ("p_value", float(ks_p)),
            ("reject_at_005", bool(ks_p < 0.05)),
        ]
    )

    # Summary: count how many tests reject normality
    rejections = sum(
        [
            results["shapiro_wilk"]["reject_at_005"],
            results.get("dagostino_pearson", {}).get("reject_at_005", False),
            results["anderson_darling"]["reject_at_005"],
            results["kolmogorov_smirnov"]["reject_at_005"],
        ]
    )
    n_tests = 4 if "dagostino_pearson" in results else 3

    results["summary"] = OrderedDict(
        [
            ("tests_rejecting_normality", rejections),
            ("tests_total", n_tests),
            (
                "conclusion",
                "clearly non-normal"
                if rejections == n_tests
                else "likely non-normal"
                if rejections >= n_tests - 1
                else "ambiguous; mixed test results"
                if rejections >= 1
                else "consistent with normality",
            ),
        ]
    )

    return results


# ============================================================================
# Q2: MODALITY: how many natural groups?
# ============================================================================


def analyze_kde_modes(x: np.ndarray, n_eval: int = 2000) -> Dict:
    """
    Fit KDE and find modes (peaks) and antimodes (valleys).

    Valleys in the density are candidate natural break points.
    Multiple bandwidths are tested to check stability of mode count.
    """
    results = OrderedDict()

    x_grid = np.linspace(x.min() - 0.5, x.max() + 0.5, n_eval)

    # Silverman's rule of thumb bandwidth
    silverman_bw = 1.06 * x.std() * len(x) ** (-1 / 5)
    results["silverman_bandwidth"] = round(silverman_bw, 4)

    # Test multiple bandwidth multipliers to check mode stability
    bw_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    mode_counts = OrderedDict()

    primary_modes = []
    primary_valleys = []
    primary_density = None

    for mult in bw_multipliers:
        bw = silverman_bw * mult
        try:
            kde = stats.gaussian_kde(x, bw_method=bw / x.std())
            density = kde(x_grid)

            # Find peaks (modes)
            # Require prominence > 2% of max density to filter noise
            peak_idx, peak_props = find_peaks(density, prominence=0.02 * density.max())
            # Find valleys (antimodes) : peaks in negated density
            valley_idx, valley_props = find_peaks(
                -density, prominence=0.02 * density.max()
            )

            n_modes = len(peak_idx)
            mode_counts[f"bw_x{mult}"] = n_modes

            # Store primary result (1.0x bandwidth)
            if mult == 1.0:
                primary_density = density
                primary_modes = [
                    {
                        "position": round(float(x_grid[i]), 4),
                        "density": round(float(density[i]), 6),
                    }
                    for i in peak_idx
                ]
                primary_valleys = [
                    {
                        "position": round(float(x_grid[i]), 4),
                        "density": round(float(density[i]), 6),
                    }
                    for i in valley_idx
                ]
        except Exception as e:
            mode_counts[f"bw_x{mult}"] = f"error: {e}"

    results["mode_counts_by_bandwidth"] = mode_counts
    results["modes"] = primary_modes
    results["valleys"] = primary_valleys
    results["n_modes"] = len(primary_modes)
    results["n_valleys"] = len(primary_valleys)

    # Stability assessment: are mode counts consistent across bandwidths?
    valid_counts = [v for v in mode_counts.values() if isinstance(v, int)]
    if valid_counts:
        results["mode_count_range"] = [min(valid_counts), max(valid_counts)]
        results["modes_stable"] = (max(valid_counts) - min(valid_counts)) <= 1

    return results, x_grid, primary_density


def fit_gaussian_mixtures(
    x: np.ndarray, max_k: int = 12, max_samples: int = 10000
) -> Dict:
    """
    Fit Gaussian Mixture Models for k=1..max_k components.
    Compare with BIC and AIC to find optimal number of components.

    Subsamples to max_samples for performance.  For density estimation
    tasks the subsample gives equivalent model selection results.
    """
    rng = np.random.RandomState(42)
    if len(x) > max_samples:
        x_fit = rng.choice(x, size=max_samples, replace=False)
    else:
        x_fit = x
    X = x_fit.reshape(-1, 1)

    results = OrderedDict()
    results["n_samples_used"] = len(x_fit)
    fit_results = []

    bics = []
    aics = []

    for k in range(1, max_k + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            n_init=3,
            max_iter=200,
            random_state=42,
        )
        gmm.fit(X)

        bic = float(gmm.bic(X))
        aic = float(gmm.aic(X))
        bics.append(bic)
        aics.append(aic)

        # Extract component parameters
        components = []
        for j in range(k):
            components.append(
                {
                    "mean": round(float(gmm.means_[j, 0]), 4),
                    "std": round(float(np.sqrt(gmm.covariances_[j, 0, 0])), 4),
                    "weight": round(float(gmm.weights_[j]), 4),
                }
            )
        # Sort components by mean
        components.sort(key=lambda c: c["mean"])

        fit_results.append(
            OrderedDict(
                [
                    ("k", k),
                    ("bic", round(bic, 1)),
                    ("aic", round(aic, 1)),
                    ("converged", bool(gmm.converged_)),
                    ("components", components),
                ]
            )
        )

    # Optimal k by BIC and AIC (lower is better)
    optimal_bic_k = int(np.argmin(bics) + 1)
    optimal_aic_k = int(np.argmin(aics) + 1)

    # BIC elbow: largest drop in BIC
    bic_drops = [bics[i] - bics[i + 1] for i in range(len(bics) - 1)]
    elbow_k = int(np.argmax(bic_drops) + 2) if bic_drops else 1  # k after biggest drop

    results["fits"] = fit_results
    results["optimal_k_bic"] = optimal_bic_k
    results["optimal_k_aic"] = optimal_aic_k
    results["elbow_k_bic"] = elbow_k
    results["bic_values"] = [round(b, 1) for b in bics]
    results["aic_values"] = [round(a, 1) for a in aics]

    # Component boundaries for optimal BIC model
    # Boundary = midpoint between adjacent component means, weighted by std
    opt_fit = fit_results[optimal_bic_k - 1]
    if optimal_bic_k > 1:
        comps = opt_fit["components"]
        boundaries = []
        for i in range(len(comps) - 1):
            # Weighted midpoint: closer to the narrower component
            m1, s1 = comps[i]["mean"], comps[i]["std"]
            m2, s2 = comps[i + 1]["mean"], comps[i + 1]["std"]
            if s1 + s2 > 0:
                boundary = (m1 * s2 + m2 * s1) / (s1 + s2)
            else:
                boundary = (m1 + m2) / 2
            boundaries.append(round(float(boundary), 4))
        results["gmm_boundaries_bic_optimal"] = boundaries

    return results


def compute_kmeans_analysis(
    x: np.ndarray, max_k: int = 12, max_samples: int = 10000
) -> Dict:
    """
    K-means clustering for k=2..max_k.
    Compute inertia (elbow method), silhouette scores, and
    Goodness of Variance Fit (GVF = 1 - inertia/total_var).

    GVF is the same quality metric as Jenks natural breaks.
    For 1D data, k-means with optimal k ≈ Jenks.

    Subsamples to max_samples for performance.
    """
    rng = np.random.RandomState(42)
    if len(x) > max_samples:
        x_fit = rng.choice(x, size=max_samples, replace=False)
    else:
        x_fit = x
    X = x_fit.reshape(-1, 1)
    total_var = float(np.var(x_fit) * len(x_fit))

    results = OrderedDict()
    results["n_samples_used"] = len(x_fit)
    k_results = []

    inertias = []
    silhouettes = []
    gvfs = []

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=42)
        labels = km.fit_predict(X)
        inertia = float(km.inertia_)
        inertias.append(inertia)

        sil = float(silhouette_score(X, labels))
        silhouettes.append(sil)

        gvf = 1.0 - inertia / total_var if total_var > 0 else 0.0
        gvfs.append(gvf)

        # Extract break points (boundaries between clusters)
        centers = sorted(km.cluster_centers_[:, 0])
        breaks = []
        for i in range(len(centers) - 1):
            breaks.append(round(float((centers[i] + centers[i + 1]) / 2), 4))

        k_results.append(
            OrderedDict(
                [
                    ("k", k),
                    ("inertia", round(inertia, 2)),
                    ("silhouette", round(sil, 4)),
                    ("gvf", round(gvf, 4)),
                    ("centers", [round(float(c), 4) for c in centers]),
                    ("breaks", breaks),
                ]
            )
        )

    # Optimal k by silhouette (higher is better)
    optimal_sil_k = int(np.argmax(silhouettes) + 2)

    # Elbow detection on inertia: largest relative drop
    if len(inertias) >= 2:
        relative_drops = [
            (inertias[i] - inertias[i + 1]) / inertias[i] if inertias[i] > 0 else 0
            for i in range(len(inertias) - 1)
        ]
        elbow_k = int(np.argmax(relative_drops) + 2)  # k value (not index)
    else:
        elbow_k = 2

    # GVF threshold: first k where GVF > 0.90 (90% variance explained)
    gvf_90_k = None
    for i, g in enumerate(gvfs):
        if g >= 0.90:
            gvf_90_k = i + 2  # k value
            break

    results["fits"] = k_results
    results["optimal_k_silhouette"] = optimal_sil_k
    results["elbow_k_inertia"] = elbow_k
    results["gvf_90_threshold_k"] = gvf_90_k
    results["silhouette_values"] = [round(s, 4) for s in silhouettes]
    results["gvf_values"] = [round(g, 4) for g in gvfs]

    return results


# ============================================================================
# Q3: NATURAL BREAK POINTS: where to cut?
# ============================================================================


def find_natural_breaks(
    x: np.ndarray, kde_results: Dict, gmm_results: Dict, kmeans_results: Dict
) -> Dict:
    """
    Synthesize break point suggestions from all methods.

    Each method independently proposes break points.  We check for
    convergence: if multiple methods agree on a break location (within
    tolerance), that's a high-confidence boundary.
    """
    results = OrderedDict()

    all_breaks = []

    # 1. KDE valleys
    kde_breaks = [v["position"] for v in kde_results.get("valleys", [])]
    results["kde_valleys"] = kde_breaks
    for b in kde_breaks:
        all_breaks.append(("kde_valley", b))

    # 2. GMM boundaries (from BIC-optimal model)
    gmm_breaks = gmm_results.get("gmm_boundaries_bic_optimal", [])
    results["gmm_boundaries"] = gmm_breaks
    for b in gmm_breaks:
        all_breaks.append(("gmm", b))

    # 3. K-means breaks (from silhouette-optimal k)
    opt_sil_k = gmm_results.get("optimal_k_bic", 3)  # fallback
    km_opt_k = kmeans_results.get("optimal_k_silhouette", 3)
    for fit in kmeans_results.get("fits", []):
        if fit["k"] == km_opt_k:
            km_breaks = fit["breaks"]
            results["kmeans_breaks"] = km_breaks
            for b in km_breaks:
                all_breaks.append(("kmeans", b))
            break

    # 4. Find convergent breaks (points where multiple methods agree)
    if len(all_breaks) >= 2:
        tolerance = float(np.std([b for _, b in all_breaks]) * 0.15)
        tolerance = max(tolerance, 0.1)  # minimum tolerance

        # Cluster the break points themselves
        break_values = sorted(set(b for _, b in all_breaks))
        convergent = []

        visited = set()
        for bv in break_values:
            if bv in visited:
                continue
            # Find all breaks within tolerance
            nearby = [
                (method, val)
                for method, val in all_breaks
                if abs(val - bv) <= tolerance
            ]
            if len(nearby) >= 2:
                methods = list(set(m for m, _ in nearby))
                mean_pos = float(np.mean([v for _, v in nearby]))
                convergent.append(
                    OrderedDict(
                        [
                            ("position", round(mean_pos, 4)),
                            ("n_methods_agreeing", len(methods)),
                            ("methods", methods),
                            (
                                "spread",
                                round(
                                    float(
                                        max(v for _, v in nearby)
                                        - min(v for _, v in nearby)
                                    ),
                                    4,
                                ),
                            ),
                        ]
                    )
                )
                for _, v in nearby:
                    visited.add(v)

        results["convergent_breaks"] = convergent
        results["tolerance_used"] = round(tolerance, 4)
    else:
        results["convergent_breaks"] = []

    return results


# ============================================================================
# Q4: TAIL ANALYSIS
# ============================================================================


def analyze_tails(x: np.ndarray) -> Dict:
    """
    Characterize tail behavior and identify where tails start.

    Methods:
      - IQR fences (Tukey): mild outlier at 1.5xIQR, extreme at 3xIQR
      - MAD (Median Absolute Deviation): robust outlier detection
      - Percentile density: compare density in tail vs bulk regions
      - Tail weight: fraction of tokens and freq-mass in each tail
    """
    results = OrderedDict()

    q1 = float(np.percentile(x, 25))
    q3 = float(np.percentile(x, 75))
    iqr = q3 - q1
    median = float(np.median(x))

    # Tukey fences
    inner_lo = q1 - 1.5 * iqr
    inner_hi = q3 + 1.5 * iqr
    outer_lo = q1 - 3.0 * iqr
    outer_hi = q3 + 3.0 * iqr

    n = len(x)
    n_mild_lo = int((x < inner_lo).sum())
    n_mild_hi = int((x > inner_hi).sum())
    n_extreme_lo = int((x < outer_lo).sum())
    n_extreme_hi = int((x > outer_hi).sum())

    results["tukey_fences"] = OrderedDict(
        [
            ("inner_fence_lo", round(inner_lo, 4)),
            ("inner_fence_hi", round(inner_hi, 4)),
            ("outer_fence_lo", round(outer_lo, 4)),
            ("outer_fence_hi", round(outer_hi, 4)),
            ("n_mild_outliers_lo", n_mild_lo),
            ("n_mild_outliers_hi", n_mild_hi),
            ("n_extreme_outliers_lo", n_extreme_lo),
            ("n_extreme_outliers_hi", n_extreme_hi),
            ("pct_mild_outliers_lo", round(n_mild_lo / n * 100, 2)),
            ("pct_mild_outliers_hi", round(n_mild_hi / n * 100, 2)),
        ]
    )

    # MAD-based outlier detection
    mad = float(np.median(np.abs(x - median)))
    # Modified Z-score: |x - median| / (1.4826 * MAD)
    if mad > 0:
        mad_threshold_lo = median - 3.5 * 1.4826 * mad
        mad_threshold_hi = median + 3.5 * 1.4826 * mad
        n_mad_lo = int((x < mad_threshold_lo).sum())
        n_mad_hi = int((x > mad_threshold_hi).sum())
    else:
        mad_threshold_lo = mad_threshold_hi = median
        n_mad_lo = n_mad_hi = 0

    results["mad_analysis"] = OrderedDict(
        [
            ("mad", round(mad, 4)),
            ("threshold_lo", round(mad_threshold_lo, 4)),
            ("threshold_hi", round(mad_threshold_hi, 4)),
            ("n_outliers_lo", n_mad_lo),
            ("n_outliers_hi", n_mad_hi),
        ]
    )

    # Density comparison: tails vs bulk
    # Split into percentile regions and compare local density
    regions = [
        ("bottom_1pct", 0, 1),
        ("bottom_5pct", 0, 5),
        ("bottom_10pct", 0, 10),
        ("bulk_25_75", 25, 75),
        ("top_10pct", 90, 100),
        ("top_5pct", 95, 100),
        ("top_1pct", 99, 100),
    ]
    density_comparison = OrderedDict()
    for name, lo_pct, hi_pct in regions:
        lo_val = float(np.percentile(x, lo_pct))
        hi_val = float(np.percentile(x, hi_pct))
        region_data = x[(x >= lo_val) & (x <= hi_val)]
        n_region = len(region_data)
        pct_span = hi_pct - lo_pct
        width = hi_val - lo_val if hi_val > lo_val else 1e-10

        # Density = fraction of tokens / width of region in log-freq space
        density = (n_region / n) / width if width > 0 else 0

        density_comparison[name] = OrderedDict(
            [
                ("pct_range", [lo_pct, hi_pct]),
                ("log_freq_range", [round(lo_val, 4), round(hi_val, 4)]),
                ("n_tokens", n_region),
                ("pct_of_vocab", round(n_region / n * 100, 2)),
                ("width", round(width, 4)),
                ("density", round(density, 4)),
            ]
        )

    results["density_comparison"] = density_comparison

    # Tail asymmetry: compare bottom vs top
    bulk_density = density_comparison["bulk_25_75"]["density"]
    if bulk_density > 0:
        bottom_ratio = density_comparison["bottom_5pct"]["density"] / bulk_density
        top_ratio = density_comparison["top_5pct"]["density"] / bulk_density
        results["tail_vs_bulk_density_ratio"] = OrderedDict(
            [
                ("bottom_5pct_vs_bulk", round(bottom_ratio, 4)),
                ("top_5pct_vs_bulk", round(top_ratio, 4)),
                (
                    "interpretation",
                    "bottom tail is sparser than bulk"
                    if bottom_ratio < 0.5
                    else "bottom tail density similar to bulk"
                    if bottom_ratio < 1.5
                    else "bottom tail is denser than bulk",
                ),
            ]
        )

    return results


# ============================================================================
# LOCAL DENSITY PROFILE
# ============================================================================


def compute_local_density(x: np.ndarray, n_bins: int = 50) -> Dict:
    """
    Compute density at regular intervals across the distribution.
    Identifies sparse and dense regions.
    """
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)

    counts, _ = np.histogram(x, bins=edges)
    densities = counts / (len(x) * widths)

    # Find sparse regions (density < 20% of median density)
    median_density = float(np.median(densities[densities > 0]))
    sparse_threshold = median_density * 0.2
    sparse_regions = []
    for i, (c, d) in enumerate(zip(centers, densities)):
        if d < sparse_threshold:
            sparse_regions.append(round(float(c), 4))

    # Find dense regions (density > 2x median)
    dense_regions = []
    for i, (c, d) in enumerate(zip(centers, densities)):
        if d > median_density * 2:
            dense_regions.append(round(float(c), 4))

    return OrderedDict(
        [
            ("n_bins", n_bins),
            ("bin_width", round(float(widths[0]), 4)),
            ("median_density", round(median_density, 4)),
            ("sparse_threshold", round(sparse_threshold, 4)),
            ("n_sparse_bins", len(sparse_regions)),
            ("n_dense_bins", len(dense_regions)),
            ("sparse_regions", sparse_regions),
            ("dense_regions", dense_regions),
            (
                "bin_centers",
                [round(float(c), 4) for c in centers],
            ),
            ("bin_densities", [round(float(d), 6) for d in densities]),
        ]
    )


# ============================================================================
# OPTIMAL HISTOGRAM BINNING
# ============================================================================


def compare_binning_rules(x: np.ndarray) -> Dict:
    """
    Compare standard binning rules.  These give a data-driven estimate
    of the distribution's "resolution" > how fine-grained the structure is.
    """
    n = len(x)
    r = float(x.max() - x.min())
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))

    # Sturges: assumes approximate normality
    sturges = int(np.ceil(np.log2(n) + 1))

    # Freedman-Diaconis: robust, based on IQR
    fd_width = 2 * iqr * n ** (-1 / 3) if iqr > 0 else r / sturges
    freedman_diaconis = int(np.ceil(r / fd_width)) if fd_width > 0 else sturges

    # Scott: assumes normality, based on std
    scott_width = 3.49 * float(x.std()) * n ** (-1 / 3) if x.std() > 0 else fd_width
    scott = int(np.ceil(r / scott_width)) if scott_width > 0 else sturges

    # Rice: simple rule, tends to slightly oversmooth
    rice = int(np.ceil(2 * n ** (1 / 3)))

    # Doane: extension of Sturges for skewed distributions
    se_skew = np.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
    skewness = float(stats.skew(x))
    doane = max(1, int(np.ceil(1 + np.log2(n) + np.log2(1 + abs(skewness) / se_skew))))

    return OrderedDict(
        [
            ("sturges", sturges),
            ("freedman_diaconis", freedman_diaconis),
            ("scott", scott),
            ("rice", rice),
            ("doane", doane),
            (
                "median_recommendation",
                int(np.median([sturges, freedman_diaconis, scott, rice, doane])),
            ),
            (
                "interpretation",
                f"Binning rules suggest {min(sturges, freedman_diaconis, scott)}-"
                f"{max(sturges, freedman_diaconis, scott)} bins. "
                f"Large spread indicates the distribution has multi-scale structure.",
            ),
        ]
    )


def plot_shape(
    x: np.ndarray,
    kde_grid: np.ndarray,
    kde_density: np.ndarray,
    normality: Dict,
    descriptive: Dict,
    path: Path,
) -> None:
    """Figure 1: Distribution shape: histogram, KDE, QQ-plot, CDF."""
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

    # --- 1. Histogram + KDE ---
    ax1 = fig.add_subplot(gs[0, 0])
    n_bins = max(80, len(x) // 200)
    ax1.hist(
        x,
        bins=n_bins,
        density=True,
        alpha=0.5,
        color="#4285f4",
        edgecolor="white",
        linewidth=0.3,
        label="Histogram",
    )
    if kde_density is not None:
        ax1.plot(
            kde_grid, kde_density, color="#ea4335", linewidth=2, label="KDE (Silverman)"
        )
    # Normal overlay
    xn = np.linspace(x.min(), x.max(), 500)
    ax1.plot(
        xn,
        stats.norm.pdf(xn, x.mean(), x.std()),
        color="#34a853",
        linewidth=1.5,
        linestyle="--",
        label=f"Normal fit (μ={x.mean():.2f}, σ={x.std():.2f})",
    )
    ax1.set_xlabel("log₁₀(freq per million)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title(
        "Distribution with KDE and Normal Overlay", fontsize=12, fontweight="bold"
    )
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # --- 2. QQ-plot ---
    ax2 = fig.add_subplot(gs[0, 1])
    theoretical_q = stats.norm.ppf((np.arange(1, len(x) + 1) - 0.5) / len(x))
    sorted_x = np.sort(x)
    # Standardize
    z_scores = (sorted_x - sorted_x.mean()) / sorted_x.std()
    ax2.scatter(theoretical_q, z_scores, s=1, alpha=0.3, color="#4285f4")
    lims = [
        min(theoretical_q.min(), z_scores.min()),
        max(theoretical_q.max(), z_scores.max()),
    ]
    ax2.plot(lims, lims, "r--", linewidth=1.5, label="Perfect normal")
    ax2.set_xlabel("Theoretical quantiles (normal)", fontsize=11)
    ax2.set_ylabel("Sample quantiles (standardized)", fontsize=11)
    ax2.set_title("Q-Q Plot vs Normal Distribution", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # --- 3. CDF comparison ---
    ax3 = fig.add_subplot(gs[1, 0])
    sorted_x = np.sort(x)
    empirical_cdf = np.arange(1, len(x) + 1) / len(x)
    ax3.plot(
        sorted_x, empirical_cdf, color="#4285f4", linewidth=1.5, label="Empirical CDF"
    )
    ax3.plot(
        sorted_x,
        stats.norm.cdf(sorted_x, x.mean(), x.std()),
        color="#34a853",
        linewidth=1.5,
        linestyle="--",
        label="Normal CDF",
    )
    ax3.set_xlabel("log₁₀(freq per million)", fontsize=11)
    ax3.set_ylabel("Cumulative probability", fontsize=11)
    ax3.set_title("Empirical vs Normal CDF", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # --- 4. Summary text ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    summary_lines = [
        "SHAPE SUMMARY",
        "─" * 40,
        f"n = {descriptive['n']:,}",
        f"Range: [{descriptive['min']:.3f}, {descriptive['max']:.3f}]",
        f"Mean: {descriptive['mean']:.4f}   Std: {descriptive['std']:.4f}",
        f"Median: {descriptive['median']:.4f}   IQR: {descriptive['iqr']:.4f}",
        "",
        f"Skewness: {normality['skewness']:.4f}  "
        f"({normality['skewness_interpretation']})",
        f"Excess kurtosis: {normality['excess_kurtosis']:.4f}  "
        f"({normality['kurtosis_interpretation']})",
        "",
        "NORMALITY TESTS",
        "─" * 40,
        f"Shapiro-Wilk:     p={normality['shapiro_wilk']['p_value']:.2e}  "
        f"{'REJECT' if normality['shapiro_wilk']['reject_at_005'] else 'PASS'}",
    ]
    if "dagostino_pearson" in normality:
        summary_lines.append(
            f"D'Agostino:       p={normality['dagostino_pearson']['p_value']:.2e}  "
            f"{'REJECT' if normality['dagostino_pearson']['reject_at_005'] else 'PASS'}"
        )
    summary_lines.extend(
        [
            f"Anderson-Darling: stat={normality['anderson_darling']['statistic']:.4f}  "
            f"{'REJECT' if normality['anderson_darling']['reject_at_005'] else 'PASS'}",
            f"K-S test:         p={normality['kolmogorov_smirnov']['p_value']:.2e}  "
            f"{'REJECT' if normality['kolmogorov_smirnov']['reject_at_005'] else 'PASS'}",
            "",
            f"Conclusion: {normality['summary']['conclusion'].upper()}",
        ]
    )
    ax4.text(
        0.05,
        0.95,
        "\n".join(summary_lines),
        transform=ax4.transAxes,
        fontsize=10,
        fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_modality(
    x: np.ndarray,
    kde_grid: np.ndarray,
    kde_density: np.ndarray,
    kde_results: Dict,
    gmm_results: Dict,
    kmeans_results: Dict,
    path: Path,
) -> None:
    """Figure 2: Modality. KDE modes, GMM components, silhouette/BIC."""
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

    # --- 1. KDE with modes and valleys ---
    ax1 = fig.add_subplot(gs[0, 0])
    if kde_density is not None:
        ax1.plot(kde_grid, kde_density, color="#4285f4", linewidth=2)
        ax1.fill_between(kde_grid, kde_density, alpha=0.15, color="#4285f4")
    for m in kde_results.get("modes", []):
        ax1.axvline(
            m["position"], color="#34a853", linewidth=1.5, linestyle="--", alpha=0.8
        )
        ax1.annotate(
            f"mode\n{m['position']:.2f}",
            xy=(m["position"], m["density"]),
            fontsize=8,
            ha="center",
            color="#34a853",
        )
    for v in kde_results.get("valleys", []):
        ax1.axvline(
            v["position"], color="#ea4335", linewidth=1.5, linestyle=":", alpha=0.8
        )
        ax1.annotate(
            f"valley\n{v['position']:.2f}",
            xy=(v["position"], v["density"]),
            fontsize=8,
            ha="center",
            color="#ea4335",
        )
    ax1.set_xlabel("log₁₀(freq per million)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title(
        f"KDE Modes ({kde_results.get('n_modes', '?')}) and "
        f"Valleys ({kde_results.get('n_valleys', '?')})",
        fontsize=12,
        fontweight="bold",
    )
    ax1.grid(alpha=0.3)

    # --- 2. GMM optimal fit overlaid ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(
        x,
        bins=100,
        density=True,
        alpha=0.4,
        color="#9e9e9e",
        edgecolor="white",
        linewidth=0.3,
    )
    opt_k = gmm_results.get("optimal_k_bic", 1)
    if opt_k <= len(gmm_results.get("fits", [])):
        comps = gmm_results["fits"][opt_k - 1]["components"]
        x_plot = np.linspace(x.min(), x.max(), 500)
        total_density = np.zeros_like(x_plot)
        colors = plt.cm.Set2(np.linspace(0, 1, len(comps)))
        for j, (comp, color) in enumerate(zip(comps, colors)):
            comp_density = comp["weight"] * stats.norm.pdf(
                x_plot, comp["mean"], comp["std"]
            )
            total_density += comp_density
            ax2.plot(
                x_plot,
                comp_density,
                color=color,
                linewidth=1.5,
                linestyle="--",
                label=f"k={j + 1}: μ={comp['mean']:.2f} σ={comp['std']:.2f} "
                f"w={comp['weight']:.2f}",
            )
        ax2.plot(x_plot, total_density, color="black", linewidth=2, label="GMM total")
    ax2.set_xlabel("log₁₀(freq per million)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title(
        f"Gaussian Mixture Model (optimal k={opt_k} by BIC)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(alpha=0.3)

    # --- 3. BIC / AIC ---
    ax3 = fig.add_subplot(gs[1, 0])
    ks = list(range(1, len(gmm_results.get("bic_values", [])) + 1))
    if ks:
        ax3.plot(
            ks,
            gmm_results["bic_values"],
            "o-",
            color="#4285f4",
            linewidth=2,
            markersize=6,
            label="BIC",
        )
        ax3.plot(
            ks,
            gmm_results["aic_values"],
            "s--",
            color="#ea4335",
            linewidth=2,
            markersize=6,
            label="AIC",
        )
        ax3.axvline(
            gmm_results["optimal_k_bic"],
            color="#4285f4",
            linewidth=1,
            linestyle=":",
            alpha=0.6,
            label=f"BIC optimal: k={gmm_results['optimal_k_bic']}",
        )
        ax3.axvline(
            gmm_results["optimal_k_aic"],
            color="#ea4335",
            linewidth=1,
            linestyle=":",
            alpha=0.6,
            label=f"AIC optimal: k={gmm_results['optimal_k_aic']}",
        )
    ax3.set_xlabel("Number of components (k)", fontsize=11)
    ax3.set_ylabel("Information criterion", fontsize=11)
    ax3.set_title("GMM Model Selection: BIC and AIC", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    # --- 4. Silhouette + GVF ---
    ax4 = fig.add_subplot(gs[1, 1])
    ks_km = list(range(2, 2 + len(kmeans_results.get("silhouette_values", []))))
    if ks_km:
        color_sil = "#4285f4"
        color_gvf = "#34a853"
        ax4.plot(
            ks_km,
            kmeans_results["silhouette_values"],
            "o-",
            color=color_sil,
            linewidth=2,
            markersize=6,
            label="Silhouette",
        )
        ax4.set_ylabel("Silhouette score", fontsize=11, color=color_sil)
        ax4.tick_params(axis="y", labelcolor=color_sil)

        ax4b = ax4.twinx()
        ax4b.plot(
            ks_km,
            kmeans_results["gvf_values"],
            "s--",
            color=color_gvf,
            linewidth=2,
            markersize=6,
            label="GVF",
        )
        ax4b.axhline(
            0.9,
            color=color_gvf,
            linewidth=0.8,
            linestyle=":",
            alpha=0.5,
            label="GVF=0.90 threshold",
        )
        ax4b.set_ylabel("Goodness of Variance Fit", fontsize=11, color=color_gvf)
        ax4b.tick_params(axis="y", labelcolor=color_gvf)

        opt_sil = kmeans_results["optimal_k_silhouette"]
        ax4.axvline(opt_sil, color=color_sil, linewidth=1, linestyle=":", alpha=0.6)
        ax4.annotate(
            f"sil opt: k={opt_sil}",
            xy=(opt_sil, 0),
            fontsize=9,
            ha="center",
            color=color_sil,
        )

        # Combined legend
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4b.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

    ax4.set_xlabel("Number of clusters (k)", fontsize=11)
    ax4.set_title(
        "K-Means: Silhouette and Variance Explained", fontsize=12, fontweight="bold"
    )
    ax4.grid(alpha=0.3)

    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_breaks(
    x: np.ndarray,
    kde_grid: np.ndarray,
    kde_density: np.ndarray,
    natural_breaks: Dict,
    path: Path,
) -> None:
    """Figure 3: Natural break points overlaid on distribution."""
    fig, axes = plt.subplots(3, 1, figsize=(18, 16), sharex=True)

    for ax in axes:
        ax.hist(
            x,
            bins=150,
            density=True,
            alpha=0.3,
            color="#9e9e9e",
            edgecolor="white",
            linewidth=0.2,
        )
        if kde_density is not None:
            ax.plot(kde_grid, kde_density, color="#4285f4", linewidth=1.5, alpha=0.7)

    # Panel 1: KDE valleys
    ax = axes[0]
    for b in natural_breaks.get("kde_valleys", []):
        ax.axvline(b, color="#ea4335", linewidth=2, linestyle="--", alpha=0.8)
        ax.annotate(
            f"{b:.2f}", xy=(b, 0), fontsize=8, ha="center", color="#ea4335", rotation=45
        )
    ax.set_title(
        f"KDE Valley Break Points ({len(natural_breaks.get('kde_valleys', []))})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)

    # Panel 2: GMM boundaries
    ax = axes[1]
    for b in natural_breaks.get("gmm_boundaries", []):
        ax.axvline(b, color="#fbbc04", linewidth=2, linestyle="--", alpha=0.8)
        ax.annotate(
            f"{b:.2f}", xy=(b, 0), fontsize=8, ha="center", color="#fbbc04", rotation=45
        )
    ax.set_title(
        f"GMM Component Boundaries ({len(natural_breaks.get('gmm_boundaries', []))})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)

    # Panel 3: K-means breaks
    ax = axes[2]
    for b in natural_breaks.get("kmeans_breaks", []):
        ax.axvline(b, color="#34a853", linewidth=2, linestyle="--", alpha=0.8)
        ax.annotate(
            f"{b:.2f}", xy=(b, 0), fontsize=8, ha="center", color="#34a853", rotation=45
        )
    ax.set_title(
        f"K-Means Break Points ({len(natural_breaks.get('kmeans_breaks', []))})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Density")
    ax.set_xlabel("log₁₀(freq per million)", fontsize=11)
    ax.grid(alpha=0.3)

    # Mark convergent breaks on all panels
    for ax in axes:
        for cb in natural_breaks.get("convergent_breaks", []):
            ax.axvline(
                cb["position"], color="black", linewidth=2.5, linestyle="-", alpha=0.4
            )

    # Convergent breaks annotation at bottom
    conv = natural_breaks.get("convergent_breaks", [])
    if conv:
        conv_str = "  |  ".join(
            f"{cb['position']:.2f} ({cb['n_methods_agreeing']} methods)" for cb in conv
        )
        fig.text(
            0.5,
            0.01,
            f"CONVERGENT BREAKS (black lines):  {conv_str}",
            ha="center",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def plot_tails(
    x: np.ndarray, tail_results: Dict, density_profile: Dict, path: Path
) -> None:
    """Figure 4: Tail analysis. Density ratios, outlier bounds, local density."""
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(1, 2, wspace=0.3)

    # --- 1. Local density profile ---
    ax1 = fig.add_subplot(gs[0, 0])
    centers = density_profile.get("bin_centers", [])
    densities = density_profile.get("bin_densities", [])
    if centers and densities:
        ax1.bar(
            centers,
            densities,
            width=density_profile.get("bin_width", 0.1),
            color="#4285f4",
            alpha=0.6,
            edgecolor="white",
            linewidth=0.3,
        )
        ax1.axhline(
            density_profile["median_density"],
            color="black",
            linewidth=1,
            linestyle="--",
            label="Median density",
        )
        ax1.axhline(
            density_profile["sparse_threshold"],
            color="#ea4335",
            linewidth=1,
            linestyle=":",
            label="Sparse threshold (20% median)",
        )

    # Overlay Tukey fences
    tf = tail_results.get("tukey_fences", {})
    for key, color, label in [
        ("inner_fence_lo", "#fbbc04", "1.5xIQR fence"),
        ("inner_fence_hi", "#fbbc04", None),
        ("outer_fence_lo", "#ea4335", "3xIQR fence"),
        ("outer_fence_hi", "#ea4335", None),
    ]:
        if key in tf:
            ax1.axvline(
                tf[key],
                color=color,
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
                label=label,
            )
    ax1.set_xlabel("log₁₀(freq per million)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title(
        "Local Density Profile with Outlier Fences", fontsize=12, fontweight="bold"
    )
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(alpha=0.3)

    # --- 2. Density comparison bar chart ---
    ax2 = fig.add_subplot(gs[0, 1])
    dc = tail_results.get("density_comparison", {})
    if dc:
        names = list(dc.keys())
        values = [dc[n]["density"] for n in names]
        colors = []
        for n in names:
            if "bulk" in n:
                colors.append("#34a853")
            elif "bottom" in n:
                colors.append("#4285f4")
            else:
                colors.append("#ea4335")
        ax2.barh(
            range(len(names)),
            values,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
        )
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=9)
        ax2.set_xlabel("Density (tokens per unit log-freq width)", fontsize=11)
        ax2.set_title("Density by Region", fontsize=12, fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)

    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")


def write_report(profile: Dict, path: Path) -> None:
    """Human-readable summary with conclusions."""
    lines = []

    def section(title):
        lines.append("")
        lines.append("=" * 80)
        lines.append(title)
        lines.append("=" * 80)

    def subsection(title):
        lines.append("")
        lines.append(f"--- {title} ---")

    section("FREQUENCY DISTRIBUTION PROFILE REPORT")
    lines.append(f"Generated: {profile['created_at']}")
    lines.append(f"Source:    {profile['source_csv']}")

    # Q1: Shape
    section("Q1: DISTRIBUTION SHAPE")
    desc = profile["descriptive"]
    lines.append(f"n = {desc['n']:,}")
    lines.append(
        f"Range:    [{desc['min']:.4f}, {desc['max']:.4f}]  "
        f"(span = {desc['range']:.4f})"
    )
    lines.append(f"Mean:     {desc['mean']:.4f}")
    lines.append(f"Median:   {desc['median']:.4f}")
    lines.append(f"Std:      {desc['std']:.4f}")
    lines.append(f"IQR:      {desc['iqr']:.4f}")

    norm = profile["normality"]
    subsection("Normality Tests")
    lines.append(
        f"Skewness:        {norm['skewness']:.4f}  ({norm['skewness_interpretation']})"
    )
    lines.append(
        f"Excess kurtosis: {norm['excess_kurtosis']:.4f}  "
        f"({norm['kurtosis_interpretation']})"
    )
    lines.append(
        f"Shapiro-Wilk:    p = {norm['shapiro_wilk']['p_value']:.2e}  "
        f"{'-> REJECT H0' if norm['shapiro_wilk']['reject_at_005'] else '-> FAIL TO REJECT'}"
    )
    if "dagostino_pearson" in norm:
        lines.append(
            f"D'Agostino:      p = {norm['dagostino_pearson']['p_value']:.2e}  "
            f"{'-> REJECT H0' if norm['dagostino_pearson']['reject_at_005'] else '-> FAIL TO REJECT'}"
        )
    lines.append(
        f"Anderson-Darling: stat = {norm['anderson_darling']['statistic']:.4f}  "
        f"(crit 5% = {norm['anderson_darling']['critical_value_5pct']:.4f})  "
        f"{'-> REJECT H0' if norm['anderson_darling']['reject_at_005'] else '-> FAIL TO REJECT'}"
    )
    lines.append(
        f"K-S test:        p = {norm['kolmogorov_smirnov']['p_value']:.2e}  "
        f"{'-> REJECT H0' if norm['kolmogorov_smirnov']['reject_at_005'] else '-> FAIL TO REJECT'}"
    )
    lines.append("")
    lines.append(f"CONCLUSION: {norm['summary']['conclusion'].upper()}")
    lines.append(
        f"  ({norm['summary']['tests_rejecting_normality']}/{norm['summary']['tests_total']} "
        f"tests reject normality at α=0.05)"
    )

    subsection("Percentile Table")
    for k, v in desc["percentiles"].items():
        lines.append(f"  {k:>6s}: {v:.4f}")

    # Q2: Modality
    section("Q2: HOW MANY NATURAL GROUPS?")

    subsection("KDE Mode Analysis")
    kde = profile["kde_modes"]
    lines.append(
        f"Modes found:  {kde['n_modes']} "
        f"(stable across bandwidths: {'YES' if kde.get('modes_stable') else 'NO'})"
    )
    lines.append(f"Mode count by bandwidth: {dict(kde['mode_counts_by_bandwidth'])}")
    for m in kde.get("modes", []):
        lines.append(
            f"  Mode at log-freq = {m['position']:.4f}  (density = {m['density']:.6f})"
        )
    for v in kde.get("valleys", []):
        lines.append(
            f"  Valley at log-freq = {v['position']:.4f}  "
            f"(density = {v['density']:.6f})"
        )

    subsection("Gaussian Mixture Models")
    gmm = profile["gmm"]
    lines.append(f"Optimal k by BIC:   {gmm['optimal_k_bic']}")
    lines.append(f"Optimal k by AIC:   {gmm['optimal_k_aic']}")
    lines.append(f"Elbow k by BIC:     {gmm['elbow_k_bic']}")
    if "gmm_boundaries_bic_optimal" in gmm:
        lines.append(f"GMM boundaries:     {gmm['gmm_boundaries_bic_optimal']}")

    subsection("K-Means Clustering")
    km = profile["kmeans"]
    lines.append(f"Optimal k by silhouette:  {km['optimal_k_silhouette']}")
    lines.append(f"Elbow k by inertia:       {km['elbow_k_inertia']}")
    lines.append(f"First k with GVF >= 0.90:  {km['gvf_90_threshold_k']}")
    lines.append("")
    lines.append("Silhouette scores:")
    for fit in km["fits"]:
        lines.append(
            f"  k={fit['k']:>2d}:  sil={fit['silhouette']:.4f}  GVF={fit['gvf']:.4f}"
        )

    subsection("Method Consensus on Number of Groups")
    methods_k = {
        "KDE modes": kde["n_modes"],
        "GMM BIC": gmm["optimal_k_bic"],
        "GMM AIC": gmm["optimal_k_aic"],
        "GMM elbow": gmm["elbow_k_bic"],
        "K-Means silhouette": km["optimal_k_silhouette"],
        "K-Means elbow": km["elbow_k_inertia"],
    }
    if km["gvf_90_threshold_k"]:
        methods_k["K-Means GVF>=0.90"] = km["gvf_90_threshold_k"]
    for name, k in methods_k.items():
        lines.append(f"  {name:<25s}: k = {k}")
    ks = [v for v in methods_k.values() if isinstance(v, (int, float))]
    if ks:
        lines.append(f"  {'Median':>25s}: k = {int(np.median(ks))}")
        lines.append(f"  {'Range':>25s}: [{min(ks)}, {max(ks)}]")

    # Q3: Break points
    section("Q3: WHERE ARE THE NATURAL BREAK POINTS?")
    nb = profile["natural_breaks"]
    lines.append(f"KDE valleys:     {nb.get('kde_valleys', [])}")
    lines.append(f"GMM boundaries:  {nb.get('gmm_boundaries', [])}")
    lines.append(f"K-Means breaks:  {nb.get('kmeans_breaks', [])}")
    lines.append("")
    conv = nb.get("convergent_breaks", [])
    if conv:
        lines.append(
            f"CONVERGENT BREAKS (tolerance = {nb.get('tolerance_used', '?')}):"
        )
        for cb in conv:
            lines.append(
                f"  log-freq = {cb['position']:.4f}  "
                f"({cb['n_methods_agreeing']} methods: "
                f"{', '.join(cb['methods'])})"
            )
    else:
        lines.append("No convergent break points found across methods.")

    # Q4: Tails
    section("Q4: TAIL ANALYSIS")
    tails = profile["tails"]
    tf = tails.get("tukey_fences", {})
    lines.append(
        f"Tukey inner fences:  [{tf.get('inner_fence_lo', '?'):.4f}, "
        f"{tf.get('inner_fence_hi', '?'):.4f}]"
    )
    lines.append(
        f"  Mild outliers below:   {tf.get('n_mild_outliers_lo', '?'):,} "
        f"({tf.get('pct_mild_outliers_lo', '?')}%)"
    )
    lines.append(
        f"  Mild outliers above:   {tf.get('n_mild_outliers_hi', '?'):,} "
        f"({tf.get('pct_mild_outliers_hi', '?')}%)"
    )

    mad = tails.get("mad_analysis", {})
    lines.append(
        f"MAD thresholds:      [{mad.get('threshold_lo', '?')}, "
        f"{mad.get('threshold_hi', '?')}]"
    )
    lines.append(f"  Outliers below:        {mad.get('n_outliers_lo', '?')}")
    lines.append(f"  Outliers above:        {mad.get('n_outliers_hi', '?')}")

    if "tail_vs_bulk_density_ratio" in tails:
        tr = tails["tail_vs_bulk_density_ratio"]
        lines.append("")
        lines.append(f"Tail density vs bulk (IQR):")
        lines.append(f"  Bottom 5%: {tr['bottom_5pct_vs_bulk']:.2f}x bulk density")
        lines.append(f"  Top 5%:    {tr['top_5pct_vs_bulk']:.2f}x bulk density")
        lines.append(f"  -> {tr['interpretation']}")

    # Binning
    section("BINNING RECOMMENDATIONS")
    binning = profile["binning"]
    for rule, val in binning.items():
        if rule not in ("interpretation", "median_recommendation"):
            lines.append(f"  {rule:<25s}: {val}")
    lines.append(f"  {'RECOMMENDED':>25s}: {binning['median_recommendation']}")
    lines.append(f"  {binning['interpretation']}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile the log-frequency distribution before discretization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=STAGE0_CSV,
        help="Path to all_tokens_complete.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=12,
        help="Maximum k to test for GMM and K-Means (default: 12)",
    )
    args = parser.parse_args()

    if not args.csv_file.exists():
        logger.error(f"CSV not found: {args.csv_file}")
        return 1

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load ----
    logger.info(f"Loading: {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    if "log_frequency" not in df.columns:
        logger.error(f"CSV must contain 'log_frequency'.  Found: {list(df.columns)}")
        return 1
    x = df["log_frequency"].values
    logger.info(f"  n = {len(x):,}   range = [{x.min():.4f}, {x.max():.4f}]")

    # ---- Q1: Shape ----
    logger.info("\n=== Q1: Distribution Shape ===")
    descriptive = compute_descriptive_stats(x)
    logger.info(
        f"  Mean={descriptive['mean']:.4f}  Std={descriptive['std']:.4f}  "
        f"Skew={descriptive.get('cv', '?')}"
    )

    normality = test_normality(x)
    logger.info(f"  Normality: {normality['summary']['conclusion']}")

    binning = compare_binning_rules(x)

    # ---- Q2: Modality ----
    logger.info("\n=== Q2: Modality ===")
    kde_results, kde_grid, kde_density = analyze_kde_modes(x)
    logger.info(
        f"  KDE modes: {kde_results['n_modes']}  "
        f"valleys: {kde_results['n_valleys']}  "
        f"stable: {kde_results.get('modes_stable', '?')}"
    )

    logger.info("  Fitting GMMs (k=1..%d)...", args.max_k)
    gmm_results = fit_gaussian_mixtures(x, max_k=args.max_k)
    logger.info(
        f"  GMM optimal k: BIC={gmm_results['optimal_k_bic']}  "
        f"AIC={gmm_results['optimal_k_aic']}"
    )

    logger.info("  Running K-Means (k=2..%d)...", args.max_k)
    kmeans_results = compute_kmeans_analysis(x, max_k=args.max_k)
    logger.info(
        f"  K-Means optimal k: silhouette={kmeans_results['optimal_k_silhouette']}  "
        f"elbow={kmeans_results['elbow_k_inertia']}"
    )

    # ---- Q3: Natural breaks ----
    logger.info("\n=== Q3: Natural Break Points ===")
    natural_breaks = find_natural_breaks(x, kde_results, gmm_results, kmeans_results)
    n_conv = len(natural_breaks.get("convergent_breaks", []))
    logger.info(f"  Convergent breaks: {n_conv}")

    # ---- Q4: Tails ----
    logger.info("\n=== Q4: Tail Analysis ===")
    tail_results = analyze_tails(x)
    density_profile = compute_local_density(x)
    tf = tail_results.get("tukey_fences", {})
    logger.info(
        f"  Tukey mild outliers: "
        f"{tf.get('n_mild_outliers_lo', 0)} below, "
        f"{tf.get('n_mild_outliers_hi', 0)} above"
    )

    # ---- Assemble profile ----
    profile = OrderedDict(
        [
            ("created_at", datetime.now().isoformat()),
            ("source_csv", str(args.csv_file)),
            ("n_tokens", int(len(x))),
            ("descriptive", descriptive),
            ("normality", normality),
            ("binning", binning),
            ("kde_modes", kde_results),
            ("gmm", gmm_results),
            ("kmeans", kmeans_results),
            ("natural_breaks", natural_breaks),
            ("tails", tail_results),
            ("density_profile", density_profile),
        ]
    )

    # ---- Save ----
    logger.info("\n=== Saving Outputs ===")
    with open(out / "frequency_profile.json", "w") as f:
        json.dump(profile, f, indent=2, default=str)
    logger.info(f"Saved: {out / 'frequency_profile.json'}")

    write_report(profile, out / "frequency_profile_report.txt")

    plot_shape(x, kde_grid, kde_density, normality, descriptive, out / "fig_shape.png")
    plot_modality(
        x,
        kde_grid,
        kde_density,
        kde_results,
        gmm_results,
        kmeans_results,
        out / "fig_modality.png",
    )
    plot_breaks(x, kde_grid, kde_density, natural_breaks, out / "fig_breaks.png")
    plot_tails(x, tail_results, density_profile, out / "fig_tails.png")

    # ---- Final summary ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("PROFILE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Shape:        {normality['summary']['conclusion']}")
    logger.info(f"  Skewness:     {normality['skewness']:.4f}")
    logger.info(f"  KDE modes:    {kde_results['n_modes']}")
    logger.info(f"  GMM optimal:  k={gmm_results['optimal_k_bic']} (BIC)")
    logger.info(
        f"  K-Means opt:  k={kmeans_results['optimal_k_silhouette']} (silhouette)"
    )
    logger.info(f"  Convergent:   {n_conv} break points")
    logger.info("")
    logger.info("Read the profile before defining any banding scheme.")
    logger.info(f"Outputs in: {out}")

    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    sys.exit(main())
