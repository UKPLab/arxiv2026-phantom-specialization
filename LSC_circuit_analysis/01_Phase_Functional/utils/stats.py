"""
Statistical testing utilities for Phase Functional analysis.

Non-parametric tests by default, BH-FDR correction, effect sizes + CIs.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy import stats as sp_stats
from scipy.stats import mannwhitneyu, kruskal, wilcoxon, spearmanr

from .constants import ALPHA, N_BOOTSTRAP, RANDOM_SEED


# =============================================================================
# EFFECT SIZE FUNCTIONS
# =============================================================================


def cohens_d(group1, group2) -> float:
    """Cohen's d for independent samples (pooled SD)."""
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    if len(g1) < 2 or len(g2) < 2:
        return np.nan
    n1, n2 = len(g1), len(g2)
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(g1) - np.mean(g2)) / pooled_std)


def cohens_d_paired(differences) -> float:
    """Cohen's d for paired samples."""
    d = np.asarray(differences, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 2:
        return np.nan
    sd = np.std(d, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(d) / sd)


def rank_biserial(group1, group2) -> float:
    """Rank-biserial correlation r for Mann-Whitney U."""
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    if len(g1) == 0 or len(g2) == 0:
        return np.nan
    try:
        U, _ = mannwhitneyu(g1, g2, alternative="two-sided")
        n1, n2 = len(g1), len(g2)
        return float(2 * U / (n1 * n2) - 1)
    except Exception:
        return np.nan


def eta_squared(groups) -> float:
    """Eta-squared effect size for Kruskal-Wallis H test."""
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[~np.isnan(g)] for g in groups]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan
    try:
        H, _ = kruskal(*groups)
        N = sum(len(g) for g in groups)
        k = len(groups)
        if N - k == 0:
            return np.nan
        return max(0.0, float((H - k + 1) / (N - k)))
    except Exception:
        return np.nan


def cliff_delta(group1, group2) -> float:
    """Cliff's delta - nonparametric effect size."""
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    if len(g1) == 0 or len(g2) == 0:
        return np.nan
    more = 0
    less = 0
    for x in g1:
        for y in g2:
            if x > y:
                more += 1
            elif x < y:
                less += 1
    n = len(g1) * len(g2)
    return float((more - less) / n) if n > 0 else np.nan


def interpret_d(d) -> str:
    """Interpret Cohen's d magnitude."""
    if np.isnan(d):
        return "N/A"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    elif ad < 0.5:
        return "small"
    elif ad < 0.8:
        return "medium"
    else:
        return "large"


def interpret_r(r) -> str:
    """Interpret rank-biserial r magnitude."""
    if np.isnan(r):
        return "N/A"
    ar = abs(r)
    if ar < 0.1:
        return "negligible"
    elif ar < 0.3:
        return "small"
    elif ar < 0.5:
        return "medium"
    else:
        return "large"


def interpret_eta2(eta2) -> str:
    """Interpret eta-squared magnitude."""
    if np.isnan(eta2):
        return "N/A"
    if eta2 < 0.01:
        return "negligible"
    elif eta2 < 0.06:
        return "small"
    elif eta2 < 0.14:
        return "medium"
    else:
        return "large"


# =============================================================================
# BOOTSTRAP FUNCTIONS
# =============================================================================


def bootstrap_ci(
    data, statistic=np.mean, n_boot=N_BOOTSTRAP, ci=0.95, seed=RANDOM_SEED
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval. Returns (lower, upper, point_estimate)."""
    data = np.asarray(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.RandomState(seed)
    boot_stats = np.array(
        [
            statistic(rng.choice(data, size=len(data), replace=True))
            for _ in range(n_boot)
        ]
    )
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_stats, 100 * alpha))
    upper = float(np.percentile(boot_stats, 100 * (1 - alpha)))
    point = float(statistic(data))
    return (lower, upper, point)


def bootstrap_ci_diff(
    group1, group2, n_boot=N_BOOTSTRAP, ci=0.95, seed=RANDOM_SEED
) -> Tuple[float, float, float]:
    """Bootstrap CI for difference in means between two groups."""
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    if len(g1) == 0 or len(g2) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.RandomState(seed)
    diffs = np.array(
        [
            np.mean(rng.choice(g1, size=len(g1), replace=True))
            - np.mean(rng.choice(g2, size=len(g2), replace=True))
            for _ in range(n_boot)
        ]
    )
    alpha = (1 - ci) / 2
    lower = float(np.percentile(diffs, 100 * alpha))
    upper = float(np.percentile(diffs, 100 * (1 - alpha)))
    point = float(np.mean(g1) - np.mean(g2))
    return (lower, upper, point)


# =============================================================================
# SAFE TEST WRAPPERS
# =============================================================================


def safe_kruskal(*groups):
    """Kruskal-Wallis with NaN handling."""
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[~np.isnan(g)] for g in groups]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan, np.nan
    if all(np.var(g) == 0 for g in groups):
        return 0.0, 1.0
    try:
        H, p = kruskal(*groups)
        return float(H), float(p)
    except Exception:
        return np.nan, np.nan


def safe_mannwhitneyu(g1, g2, alternative="two-sided"):
    """Mann-Whitney U with NaN handling."""
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    if len(g1) == 0 or len(g2) == 0:
        return np.nan, np.nan
    try:
        U, p = mannwhitneyu(g1, g2, alternative=alternative)
        return float(U), float(p)
    except Exception:
        return np.nan, np.nan


def safe_wilcoxon(d, alternative="two-sided"):
    """Wilcoxon signed-rank with NaN handling."""
    d = np.asarray(d, dtype=float)
    d = d[~np.isnan(d)]
    d = d[d != 0]
    if len(d) < 3:
        return np.nan, np.nan
    try:
        W, p = wilcoxon(d, alternative=alternative)
        return float(W), float(p)
    except Exception:
        return np.nan, np.nan


def safe_spearmanr(x, y):
    """Spearman correlation with NaN handling."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan, np.nan
    try:
        rho, p = spearmanr(x, y)
        return float(rho), float(p)
    except Exception:
        return np.nan, np.nan


def jonckheere_terpstra(groups, alternative="two-sided"):
    """
    Jonckheere-Terpstra trend test for ordered groups.

    Tests whether there is an increasing/decreasing trend across
    ordered groups. Returns (Z-statistic, p-value).
    """
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[~np.isnan(g)] for g in groups]
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return np.nan, np.nan

    # Compute J statistic: count concordant pairs across ordered groups
    k = len(groups)
    J = 0
    for i in range(k - 1):
        for j in range(i + 1, k):
            for x in groups[i]:
                for y in groups[j]:
                    if y > x:
                        J += 1
                    elif y == x:
                        J += 0.5

    # Expected value and variance under H0
    ns = [len(g) for g in groups]
    N = sum(ns)
    E_J = sum(ns[i] * ns[j] for i in range(k - 1) for j in range(i + 1, k)) / 2

    # Variance (handling ties)
    all_vals = np.concatenate(groups)
    _, tie_counts = np.unique(all_vals, return_counts=True)

    term1 = N * N * (2 * N + 3)
    term2 = sum(n * n * (2 * n + 3) for n in ns)
    term3 = sum(t * t * (2 * t + 3) for t in tie_counts)

    var_num = (term1 - term2) * (term1 - term3)
    var_denom = 72 * N * (N - 1)

    if var_denom == 0:
        return np.nan, np.nan

    # Additional tie correction terms
    tie_a = sum(t * (t - 1) for t in tie_counts)
    tie_b = sum(n * (n - 1) for n in ns)
    tie_c = sum(t * (t - 1) * (t - 2) for t in tie_counts)
    tie_d = sum(n * (n - 1) * (n - 2) for n in ns)

    var_J = var_num / var_denom
    var_J += (tie_a * tie_b) / (4 * N * (N - 1))
    if N >= 3:
        var_J += (tie_c * tie_d) / (18 * N * (N - 1) * (N - 2))

    if var_J <= 0:
        return np.nan, np.nan

    Z = (J - E_J) / np.sqrt(var_J)

    if alternative == "two-sided":
        p = 2 * sp_stats.norm.sf(abs(Z))
    elif alternative == "increasing":
        p = sp_stats.norm.sf(Z)
    elif alternative == "decreasing":
        p = sp_stats.norm.cdf(Z)
    else:
        p = 2 * sp_stats.norm.sf(abs(Z))

    return float(Z), float(p)


# =============================================================================
# TEST RESULT ACCUMULATOR
# =============================================================================


class TestAccumulator:
    """Accumulates statistical test results with metadata."""

    def __init__(self):
        self.tests = []

    def add_test(
        self,
        domain: str,
        hypothesis: str,
        model: str,
        test_name: str,
        comparison: str,
        statistic: float,
        p_value: float,
        effect_size: float,
        effect_type: str,
        n1: int = None,
        n2: int = None,
        ci_lower: float = None,
        ci_upper: float = None,
        **kwargs,
    ):
        """Add a single test result."""
        if effect_type == "cohens_d":
            interp = interpret_d(effect_size)
        elif effect_type in ("rank_biserial", "spearman_rho"):
            interp = interpret_r(effect_size)
        elif effect_type == "eta_squared":
            interp = interpret_eta2(effect_size)
        else:
            interp = "N/A"

        result = {
            "domain": domain,
            "hypothesis": hypothesis,
            "model": model,
            "test": test_name,
            "comparison": comparison,
            "statistic": statistic,
            "p_value": p_value,
            "effect_size": effect_size,
            "effect_type": effect_type,
            "interpretation": interp,
            "significant_uncorrected": p_value < ALPHA
            if not np.isnan(p_value)
            else False,
            "n1": n1,
            "n2": n2,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
        result.update(kwargs)
        self.tests.append(result)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all accumulated tests to a DataFrame."""
        return pd.DataFrame(self.tests)

    def apply_fdr_correction(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Apply Benjamini-Hochberg FDR correction to p-values."""
        from statsmodels.stats.multitest import multipletests

        if df is None:
            df = self.to_dataframe()

        valid = df["p_value"].notna()
        if valid.sum() == 0:
            df["p_value_bh"] = np.nan
            df["significant_bh"] = False
            return df

        _, pvals_corrected, _, _ = multipletests(
            df.loc[valid, "p_value"], alpha=ALPHA, method="fdr_bh"
        )
        df["p_value_bh"] = np.nan
        df.loc[valid, "p_value_bh"] = pvals_corrected
        df["significant_bh"] = df["p_value_bh"] < ALPHA
        return df

    def summary(self, df: pd.DataFrame = None) -> str:
        """Print a summary of test results by domain."""
        if df is None:
            df = self.to_dataframe()
        lines = []
        for domain in df["domain"].unique():
            d = df[df["domain"] == domain]
            col = (
                "significant_bh"
                if "significant_bh" in d.columns
                else "significant_uncorrected"
            )
            n_sig = d[col].sum()
            lines.append(f"{domain}: {n_sig}/{len(d)} significant")
        return "\n".join(lines)
