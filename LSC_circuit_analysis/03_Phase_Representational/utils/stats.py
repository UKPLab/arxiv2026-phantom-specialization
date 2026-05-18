"""
Re-export Phase 1 statistical utilities + Phase 3 additions.

Sets up a synthetic package context so Phase 1's relative imports
(from .constants import ...) resolve correctly.
"""

import sys
import types
import importlib
import importlib.util
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


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


_PHASE1_UTILS = PROJECT_ROOT / "LSC_circuit_analysis/01_Phase_Functional/utils"
_PKG_NAME = "_phase1_utils_p3"

# 1. Load Phase 1's constants module
_const_spec = importlib.util.spec_from_file_location(
    f"{_PKG_NAME}.constants", str(_PHASE1_UTILS / "constants.py")
)
_const_mod = importlib.util.module_from_spec(_const_spec)
_const_spec.loader.exec_module(_const_mod)

# 2. Create synthetic parent package so relative imports work
_pkg = types.ModuleType(_PKG_NAME)
_pkg.__path__ = [str(_PHASE1_UTILS)]
_pkg.__package__ = _PKG_NAME
sys.modules[_PKG_NAME] = _pkg
sys.modules[f"{_PKG_NAME}.constants"] = _const_mod

# 3. Load Phase 1's stats module with package context
_stats_spec = importlib.util.spec_from_file_location(
    f"{_PKG_NAME}.stats", str(_PHASE1_UTILS / "stats.py")
)
_stats_mod = importlib.util.module_from_spec(_stats_spec)
_stats_mod.__package__ = _PKG_NAME
sys.modules[f"{_PKG_NAME}.stats"] = _stats_mod
_stats_spec.loader.exec_module(_stats_mod)

# Re-export Phase 1 stats
# Effect sizes
cohens_d = _stats_mod.cohens_d
cohens_d_paired = _stats_mod.cohens_d_paired
rank_biserial = _stats_mod.rank_biserial
eta_squared = _stats_mod.eta_squared
cliff_delta = _stats_mod.cliff_delta
interpret_d = _stats_mod.interpret_d
interpret_r = _stats_mod.interpret_r
interpret_eta2 = _stats_mod.interpret_eta2
# Bootstrap
bootstrap_ci = _stats_mod.bootstrap_ci
bootstrap_ci_diff = _stats_mod.bootstrap_ci_diff
# Safe test wrappers
safe_kruskal = _stats_mod.safe_kruskal
safe_mannwhitneyu = _stats_mod.safe_mannwhitneyu
safe_wilcoxon = _stats_mod.safe_wilcoxon
safe_spearmanr = _stats_mod.safe_spearmanr
jonckheere_terpstra = _stats_mod.jonckheere_terpstra
# Test accumulator
TestAccumulator = _stats_mod.TestAccumulator


# =============================================================================
# REPRESENTATIONAL TEST HELPERS
# =============================================================================


def permutation_test_metric(
    observed: float,
    labels: Sequence,
    metric_fn: Callable[[np.ndarray], float],
    n_permutations: int = 1000,
    seed: int = 42,
    alternative: str = "greater",
) -> Tuple[float, np.ndarray]:
    """Generic permutation test for any metric.

    Shuffles labels and recomputes metric_fn(data, shuffled_labels)
    to build a null distribution.

    Args:
        observed: Observed metric value.
        labels: Array of group labels.
        metric_fn: Callable(labels_shuffled) -> float.
        n_permutations: Number of permutations.
        seed: Random seed.
        alternative: 'greater', 'less', or 'two-sided'.

    Returns:
        (p_value, null_distribution) tuple.
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    null_dist = np.zeros(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(labels)
        null_dist[i] = metric_fn(shuffled)

    if alternative == "greater":
        p = np.mean(null_dist >= observed)
    elif alternative == "less":
        p = np.mean(null_dist <= observed)
    else:
        p = np.mean(np.abs(null_dist) >= np.abs(observed))

    return float(p), null_dist


def partial_correlation(
    x: np.ndarray, y: np.ndarray, covariates: np.ndarray
) -> Tuple[float, float]:
    """Partial correlation between x and y, controlling for covariates.

    Uses linear regression to residualize both x and y on covariates,
    then computes Spearman correlation of residuals.

    Args:
        x: Array of shape (n,).
        y: Array of shape (n,).
        covariates: Array of shape (n, p) or (n,).

    Returns:
        (rho, p_value) tuple.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    covariates = np.atleast_2d(np.asarray(covariates, dtype=float))
    if covariates.shape[0] == 1 and covariates.shape[1] == len(x):
        covariates = covariates.T

    x_resid = residualize(x, covariates)
    y_resid = residualize(y, covariates)

    mask = ~(np.isnan(x_resid) | np.isnan(y_resid))
    if mask.sum() < 3:
        return np.nan, np.nan

    rho, p = sp_stats.spearmanr(x_resid[mask], y_resid[mask])
    return float(rho), float(p)


def residualize(target: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Remove linear effects of covariates from target via OLS.

    Args:
        target: Array of shape (n,).
        covariates: Array of shape (n, p).

    Returns:
        Residuals array of shape (n,).
    """
    target = np.asarray(target, dtype=float).ravel()
    covariates = np.atleast_2d(np.asarray(covariates, dtype=float))
    if covariates.shape[0] == 1 and covariates.shape[1] == len(target):
        covariates = covariates.T

    # Add intercept
    X = np.column_stack([np.ones(len(target)), covariates])

    mask = ~np.isnan(target) & ~np.any(np.isnan(X), axis=1)
    if mask.sum() < X.shape[1] + 1:
        return np.full_like(target, np.nan)

    coeffs, _, _, _ = np.linalg.lstsq(X[mask], target[mask], rcond=None)
    predicted = X @ coeffs
    residuals = target - predicted
    residuals[~mask] = np.nan
    return residuals


def mahalanobis_separation(
    activations: np.ndarray,
    labels: Sequence,
    bands: Optional[List[str]] = None,
) -> "pd.DataFrame":
    """Pairwise Mahalanobis distances between band centroids.

    Args:
        activations: Array of shape (n_samples, n_features).
        labels: Array of band labels, shape (n_samples,).
        bands: List of bands to include. If None, uses unique labels.

    Returns:
        DataFrame of pairwise Mahalanobis distances.
    """
    import pandas as pd

    labels = np.asarray(labels)
    if bands is None:
        bands = sorted(set(labels))

    # Compute pooled covariance
    pooled_cov = np.cov(activations.T)
    try:
        cov_inv = np.linalg.pinv(pooled_cov)
    except np.linalg.LinAlgError:
        return pd.DataFrame(np.nan, index=bands, columns=bands)

    # Compute centroids
    centroids = {}
    for band in bands:
        mask = labels == band
        if mask.sum() > 0:
            centroids[band] = activations[mask].mean(axis=0)

    # Pairwise distances
    n = len(bands)
    dist_matrix = np.zeros((n, n))
    for i, b1 in enumerate(bands):
        for j, b2 in enumerate(bands):
            if i < j and b1 in centroids and b2 in centroids:
                diff = centroids[b1] - centroids[b2]
                d = np.sqrt(diff @ cov_inv @ diff)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

    return pd.DataFrame(dist_matrix, index=bands, columns=bands)


def hedges_g(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, float, float]:
    """Hedges' g with small-sample bias correction and 95% CI.

    Args:
        group1: First group values.
        group2: Second group values.

    Returns:
        (g, ci_lower, ci_upper) tuple.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan

    # Cohen's d
    var1 = np.var(g1, ddof=1)
    var2 = np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0, 0.0, 0.0

    d = (np.mean(g1) - np.mean(g2)) / pooled_std

    # Bias correction factor (Hedges & Olkin, 1985)
    df = n1 + n2 - 2
    correction = 1 - 3 / (4 * df - 1)
    g = d * correction

    # Approximate SE for Hedges' g
    se = np.sqrt((n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2)))

    ci_lower = g - 1.96 * se
    ci_upper = g + 1.96 * se

    return float(g), float(ci_lower), float(ci_upper)
