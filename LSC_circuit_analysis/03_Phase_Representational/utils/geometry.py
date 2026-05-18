"""
Embedding and activation geometry metrics for representational analysis.

Provides centroid analysis, dimensionality metrics, similarity measures,
k-NN analysis, and permutation-based significance tests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import spearmanr

from .constants import FREQUENCY_BANDS, BANDS, K_NEIGHBORS, N_PERMUTATIONS, RANDOM_SEED


# =============================================================================
# CENTROID ANALYSIS
# =============================================================================


def compute_band_centroids(
    activations: np.ndarray, labels: np.ndarray, bands: List[str] = None
) -> Dict[str, np.ndarray]:
    """Compute mean representation per frequency band.

    Args:
        activations: Array of shape (n_samples, n_features).
        labels: Array of band labels, shape (n_samples,).
        bands: List of bands to include.

    Returns:
        Dict mapping band name to centroid array (n_features,).
    """
    labels = np.asarray(labels)
    if bands is None:
        bands = sorted(set(labels))

    centroids = {}
    for band in bands:
        mask = labels == band
        if mask.sum() > 0:
            centroids[band] = activations[mask].mean(axis=0)
    return centroids


def compute_centroid_distances(
    centroids: Dict[str, np.ndarray], metric: str = "euclidean"
) -> pd.DataFrame:
    """Compute pairwise distances between band centroids.

    Args:
        centroids: Dict mapping band name to centroid array.
        metric: Distance metric ('euclidean' or 'cosine').

    Returns:
        DataFrame of pairwise distances.
    """
    bands = list(centroids.keys())
    n = len(bands)
    matrix = np.zeros((n, n))

    vecs = np.array([centroids[b] for b in bands])
    dist = squareform(pdist(vecs, metric=metric))

    return pd.DataFrame(dist, index=bands, columns=bands)


def compute_within_band_spread(
    activations: np.ndarray, labels: np.ndarray, bands: List[str] = None
) -> Dict[str, float]:
    """Compute within-band spread (mean standard deviation).

    Args:
        activations: Array of shape (n_samples, n_features).
        labels: Array of band labels.
        bands: List of bands.

    Returns:
        Dict mapping band to mean within-band std.
    """
    labels = np.asarray(labels)
    if bands is None:
        bands = sorted(set(labels))

    spreads = {}
    for band in bands:
        mask = labels == band
        if mask.sum() > 1:
            spreads[band] = float(np.mean(np.std(activations[mask], axis=0)))
        else:
            spreads[band] = 0.0
    return spreads


def compute_separation_ratio(
    centroid_distances: pd.DataFrame, within_spreads: Dict[str, float]
) -> float:
    """Compute scale-invariant separation ratio.

    separation_ratio = mean(between-band distances) / mean(within-band spreads)

    Args:
        centroid_distances: DataFrame of pairwise distances.
        within_spreads: Dict of within-band spreads.

    Returns:
        Separation ratio (higher = more separated).
    """
    # Mean off-diagonal distance
    mask = np.ones(centroid_distances.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    mean_dist = centroid_distances.values[mask].mean()

    # Mean within spread
    mean_spread = np.mean(list(within_spreads.values()))

    if mean_spread == 0:
        return np.inf
    return float(mean_dist / mean_spread)


def compute_band_geometry(
    activations: np.ndarray, labels: np.ndarray, bands: List[str] = None
) -> Dict:
    """All-in-one geometry analysis.

    Returns dict with centroids, distances, spreads, separation_ratio.
    """
    if bands is None:
        bands = sorted(set(labels))

    centroids = compute_band_centroids(activations, labels, bands)
    distances = compute_centroid_distances(centroids)
    spreads = compute_within_band_spread(activations, labels, bands)
    sep_ratio = compute_separation_ratio(distances, spreads)

    return {
        "centroids": centroids,
        "centroid_distances": distances,
        "within_spreads": spreads,
        "separation_ratio": sep_ratio,
    }


# =============================================================================
# DIMENSIONALITY METRICS
# =============================================================================


def compute_participation_ratio(activations: np.ndarray) -> float:
    """Compute participation ratio (effective dimensionality).

    PR = (sum(eigenvalues))^2 / sum(eigenvalues^2)

    Args:
        activations: Array of shape (n_samples, n_features).

    Returns:
        Participation ratio.
    """
    cov = np.cov(activations.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]

    if len(eigenvalues) == 0:
        return 0.0

    return float(np.sum(eigenvalues) ** 2 / np.sum(eigenvalues**2))


def compute_isotropy(
    activations: np.ndarray, n_pairs: int = 1000, seed: int = RANDOM_SEED
) -> float:
    """Compute isotropy: 1 - mean(|cosine_similarity|) across random pairs.

    Isotropy = 1 means uniformly distributed directions.
    Isotropy = 0 means all vectors point in the same direction.

    Args:
        activations: Array of shape (n_samples, n_features).
        n_pairs: Number of random pairs to sample.
        seed: Random seed.

    Returns:
        Isotropy score in [0, 1].
    """
    n = len(activations)
    if n < 2:
        return np.nan

    rng = np.random.RandomState(seed)
    norms = np.linalg.norm(activations, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    normalized = activations / norms

    # Sample random pairs
    n_pairs = min(n_pairs, n * (n - 1) // 2)
    idx1 = rng.randint(0, n, size=n_pairs)
    idx2 = rng.randint(0, n, size=n_pairs)
    # Ensure different indices
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    if len(idx1) == 0:
        return np.nan

    cos_sims = np.sum(normalized[idx1] * normalized[idx2], axis=1)
    return float(1.0 - np.mean(np.abs(cos_sims)))


def mle_intrinsic_dim(X: np.ndarray, k1: int = 5, k2: int = 20) -> float:
    """Maximum Likelihood Estimation of intrinsic dimensionality (Levina-Bickel).

    Args:
        X: Array of shape (n_samples, n_features).
        k1: Minimum number of neighbors.
        k2: Maximum number of neighbors.

    Returns:
        Estimated intrinsic dimensionality.
    """
    from sklearn.neighbors import NearestNeighbors

    n = len(X)
    k2 = min(k2, n - 1)
    if k2 < k1 or n < k2 + 1:
        return np.nan

    nn = NearestNeighbors(n_neighbors=k2 + 1).fit(X)
    distances, _ = nn.kneighbors(X)
    # Remove self-distance (column 0)
    distances = distances[:, 1:]

    estimates = []
    for k in range(k1, k2 + 1):
        # Levina-Bickel estimator
        dk = distances[:, k - 1]
        dk = dk[dk > 0]
        if len(dk) == 0:
            continue
        log_ratios = np.log(dk[:, None] / distances[: len(dk), : k - 1])
        log_ratios = log_ratios[np.isfinite(log_ratios)]
        if len(log_ratios) == 0:
            continue
        m_hat = (k - 1) / np.mean(log_ratios)
        estimates.append(m_hat)

    if not estimates:
        return np.nan
    return float(np.mean(estimates))


# =============================================================================
# SIMILARITY MEASURES
# =============================================================================


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment (feature-space formula).

    CKA = ||X^T Y||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    where X and Y are centered.

    Args:
        X: Array of shape (n_samples, p).
        Y: Array of shape (n_samples, q).

    Returns:
        CKA similarity in [0, 1].
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    XtY = X.T @ Y
    XtX = X.T @ X
    YtY = Y.T @ Y

    num = np.linalg.norm(XtY, "fro") ** 2
    denom = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")

    if denom == 0:
        return 0.0
    return float(num / denom)


def compute_svcca_similarity(
    X: np.ndarray, Y: np.ndarray, variance_threshold: float = 0.99
) -> float:
    """Singular Vector Canonical Correlation Analysis.

    SVD-truncates both matrices to retain variance_threshold fraction,
    then computes mean canonical correlation.

    Args:
        X, Y: Activation matrices of shape (n_samples, n_features).
        variance_threshold: Fraction of variance to retain.

    Returns:
        Mean canonical correlation in [0, 1].
    """
    from sklearn.cross_decomposition import CCA

    def _truncate(M, threshold):
        U, s, Vt = np.linalg.svd(M - M.mean(axis=0), full_matrices=False)
        cumvar = np.cumsum(s**2) / np.sum(s**2)
        k = np.searchsorted(cumvar, threshold) + 1
        return U[:, :k] * s[:k]

    X_trunc = _truncate(X, variance_threshold)
    Y_trunc = _truncate(Y, variance_threshold)

    n_components = min(X_trunc.shape[1], Y_trunc.shape[1], len(X_trunc) - 1)
    if n_components < 1:
        return np.nan

    cca = CCA(n_components=n_components)
    try:
        X_c, Y_c = cca.fit_transform(X_trunc, Y_trunc)
        correlations = [
            np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)
        ]
        return float(np.mean(correlations))
    except Exception:
        return np.nan


def compute_principal_angles(
    X: np.ndarray, Y: np.ndarray, n_components: int = 10
) -> np.ndarray:
    """Compute principal (canonical) angles between two subspaces.

    Args:
        X, Y: Activation matrices of shape (n_samples, n_features).
        n_components: Number of principal components to use.

    Returns:
        Array of principal angles in degrees.
    """
    from sklearn.decomposition import PCA

    n_comp = min(n_components, X.shape[1], Y.shape[1], len(X) - 1)
    if n_comp < 1:
        return np.array([])

    pca_x = PCA(n_components=n_comp).fit(X)
    pca_y = PCA(n_components=n_comp).fit(Y)

    # SVD of the inner product of basis vectors
    M = pca_x.components_ @ pca_y.components_.T
    _, s, _ = np.linalg.svd(M)
    # Clip to valid range for arccos
    s = np.clip(s, -1, 1)
    angles = np.degrees(np.arccos(s))

    return angles


# =============================================================================
# RSA (Representational Similarity Analysis)
# =============================================================================


def compute_rsa_similarity(rdm1: np.ndarray, rdm2: np.ndarray) -> Tuple[float, float]:
    """Spearman correlation between two representational distance matrices.

    Args:
        rdm1, rdm2: Square distance matrices.

    Returns:
        (rho, p_value) tuple.
    """
    # Extract upper triangle
    n = rdm1.shape[0]
    idx = np.triu_indices(n, k=1)
    v1 = rdm1[idx]
    v2 = rdm2[idx]

    if len(v1) < 3:
        return np.nan, np.nan

    rho, p = spearmanr(v1, v2)
    return float(rho), float(p)


def rsa_permutation_test(
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> Dict:
    """RSA with permutation test (band-level resampling).

    Permutes rows and columns of rdm2 simultaneously to preserve
    within-band structure.

    Args:
        rdm1: Representational distance matrix 1.
        rdm2: Representational distance matrix 2.
        n_permutations: Number of permutations.
        seed: Random seed.

    Returns:
        Dict with rho, p_value, null_distribution.
    """
    rng = np.random.RandomState(seed)

    observed_rho, _ = compute_rsa_similarity(rdm1, rdm2)
    n = rdm1.shape[0]
    null_rhos = np.zeros(n_permutations)

    for i in range(n_permutations):
        perm = rng.permutation(n)
        rdm2_perm = rdm2[np.ix_(perm, perm)]
        null_rhos[i], _ = compute_rsa_similarity(rdm1, rdm2_perm)

    p_value = float(np.mean(null_rhos >= observed_rho))

    return {
        "rho": observed_rho,
        "p_value": p_value,
        "null_distribution": null_rhos,
    }


def bootstrap_rsa_ci(
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for RSA correlation.

    Args:
        rdm1, rdm2: Distance matrices.
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence level.
        seed: Random seed.

    Returns:
        (lower, upper, observed) tuple.
    """
    rng = np.random.RandomState(seed)
    n = rdm1.shape[0]
    observed, _ = compute_rsa_similarity(rdm1, rdm2)

    boot_rhos = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        rdm1_boot = rdm1[np.ix_(idx, idx)]
        rdm2_boot = rdm2[np.ix_(idx, idx)]
        boot_rhos[i], _ = compute_rsa_similarity(rdm1_boot, rdm2_boot)

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_rhos, 100 * alpha))
    upper = float(np.percentile(boot_rhos, 100 * (1 - alpha)))

    return lower, upper, observed


# =============================================================================
# k-NN ANALYSIS
# =============================================================================


def compute_knn_purity(
    embeddings: np.ndarray, labels: np.ndarray, k: int = K_NEIGHBORS
) -> float:
    """Fraction of k-nearest neighbors that share the same band label.

    Args:
        embeddings: Array of shape (n_samples, n_features).
        labels: Array of band labels.
        k: Number of neighbors.

    Returns:
        Purity score in [0, 1].
    """
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(labels)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    # Remove self (first column)
    neighbor_indices = indices[:, 1:]

    same_label = (labels[neighbor_indices] == labels[:, None]).mean()
    return float(same_label)


def compute_knn_band_distribution(
    embeddings: np.ndarray,
    labels: np.ndarray,
    k: int = K_NEIGHBORS,
    bands: List[str] = None,
) -> pd.DataFrame:
    """Full band distribution of k-nearest neighbors.

    Returns confusion matrix: P(neighbor_band | token_band).

    Args:
        embeddings: Array of shape (n_samples, n_features).
        labels: Array of band labels.
        k: Number of neighbors.
        bands: Ordered list of bands for matrix rows/columns.

    Returns:
        DataFrame confusion matrix.
    """
    from sklearn.neighbors import NearestNeighbors

    labels = np.asarray(labels)
    if bands is None:
        bands = sorted(set(labels))

    nn = NearestNeighbors(n_neighbors=k + 1).fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    neighbor_indices = indices[:, 1:]
    neighbor_labels = labels[neighbor_indices]

    confusion = np.zeros((len(bands), len(bands)))
    band_to_idx = {b: i for i, b in enumerate(bands)}

    for i, band in enumerate(labels):
        if band not in band_to_idx:
            continue
        row = band_to_idx[band]
        for j in range(k):
            nb = neighbor_labels[i, j]
            if nb in band_to_idx:
                confusion[row, band_to_idx[nb]] += 1

    # Normalize rows
    row_sums = confusion.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1)
    confusion = confusion / row_sums

    return pd.DataFrame(confusion, index=bands, columns=bands)


def permutation_test_purity(
    embeddings: np.ndarray,
    labels: np.ndarray,
    observed_purity: float,
    n_permutations: int = N_PERMUTATIONS,
    k: int = K_NEIGHBORS,
    seed: int = RANDOM_SEED,
) -> Tuple[float, np.ndarray]:
    """Permutation test for k-NN purity significance.

    Args:
        embeddings: Array of shape (n_samples, n_features).
        labels: Array of band labels.
        observed_purity: Observed purity value.
        n_permutations: Number of permutations.
        k: Number of neighbors.
        seed: Random seed.

    Returns:
        (p_value, null_distribution) tuple.
    """
    rng = np.random.RandomState(seed)
    null_purities = np.zeros(n_permutations)

    for i in range(n_permutations):
        shuffled = rng.permutation(labels)
        null_purities[i] = compute_knn_purity(embeddings, shuffled, k=k)

    p_value = float(np.mean(null_purities >= observed_purity))
    return p_value, null_purities
