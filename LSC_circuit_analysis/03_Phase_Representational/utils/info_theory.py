"""
Information-theoretic utilities for representational analysis.

Provides MI estimation (KSG and probe-based), conditional entropy,
per-layer information gain (delta-MI), and coding efficiency.

All entropy/MI values are in bits (log base 2) unless otherwise noted.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial import KDTree
from scipy.special import digamma

from .constants import K_NEIGHBORS, CV_FOLDS, RANDOM_SEED


# =============================================================================
# KSG MUTUAL INFORMATION ESTIMATOR
# =============================================================================


def estimate_mi_ksg(
    X: np.ndarray,
    labels: np.ndarray,
    k: int = K_NEIGHBORS,
) -> float:
    """Estimate MI(X; Y) using the mixed continuous-discrete KSG estimator.

    Based on Ross (2014) "Mutual Information between Discrete and Continuous
    Data Sets" and the Kraskov, Stogbauer & Grassberger (2004) framework.

    MI(X; Y) = psi(k) - <psi(n_y)> + psi(N) - <psi(m)>

    where:
    - For each point i with class y_i, eps_i is the distance to the k-th
      nearest neighbor AMONG SAME-CLASS points.
    - n_y is the total count of points with same label y_i.
    - m_i is the count of ALL points (any class) within distance eps_i,
      excluding self.

    Args:
        X: Continuous features, shape (n_samples, n_features).
        labels: Discrete labels (strings or ints), shape (n_samples,).
        k: Number of nearest neighbors.

    Returns:
        MI estimate in bits.
    """
    n = len(X)
    if n < k + 1:
        return 0.0

    # Encode labels to integers
    unique_labels, label_ints = np.unique(labels, return_inverse=True)
    n_classes = len(unique_labels)

    if n_classes < 2:
        return 0.0

    # Build KD-tree for the FULL X space (used for m_i counting)
    tree_full = KDTree(X)

    label_counts = np.bincount(label_ints)
    n_y = label_counts[label_ints]  # (n,)

    # For each point, find the k-th NN among SAME-CLASS points
    eps_i = np.zeros(n)
    for c in range(n_classes):
        mask = label_ints == c
        X_c = X[mask]
        n_c = len(X_c)
        k_c = min(k + 1, n_c)  # k+1 because query includes self
        if k_c < 2:
            continue
        tree_c = KDTree(X_c)
        dists_c, _ = tree_c.query(X_c, k=k_c)
        eps_i[mask] = dists_c[:, -1]  # k-th same-class NN distance

    # Count ALL points (any class) within eps_i in full space (excluding self)
    m_i = np.zeros(n)
    for i in range(n):
        if eps_i[i] > 0:
            m_i[i] = (
                tree_full.query_ball_point(X[i], eps_i[i] - 1e-10, return_length=True)
                - 1
            )
        else:
            # Degenerate case: multiple identical points
            m_i[i] = k

    # KSG formula (mixed variant)
    mi_nats = (
        digamma(k)
        - np.mean(digamma(n_y))
        + digamma(n)
        - np.mean(digamma(np.maximum(m_i, 1)))
    )

    # Convert nats to bits
    mi_bits = max(0.0, mi_nats / np.log(2))
    return float(mi_bits)


# =============================================================================
# PROBE-BASED MI ESTIMATION
# =============================================================================


def probe_based_mi(
    predictions: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Estimate MI(X; Y) from probe predictions using confusion matrix.

    MI(X; Y) = H(Y) - H(Y|X_hat) where X_hat are probe predictions.
    This is a lower bound on the true MI(X; Y).

    Args:
        predictions: Predicted labels from probe, shape (n_samples,).
        true_labels: True labels, shape (n_samples,).

    Returns:
        MI estimate in bits.
    """
    n = len(true_labels)
    unique_true = np.unique(true_labels)
    unique_pred = np.unique(predictions)

    # H(Y) - marginal entropy of true labels
    true_counts = np.array([np.sum(true_labels == c) for c in unique_true])
    p_y = true_counts / n
    h_y = -np.sum(p_y * np.log2(np.clip(p_y, 1e-10, 1.0)))

    # H(Y|X_hat) - conditional entropy of Y given predictions
    h_y_given_xhat = 0.0
    for pred_val in unique_pred:
        mask = predictions == pred_val
        n_pred = np.sum(mask)
        if n_pred == 0:
            continue
        p_pred = n_pred / n

        # P(Y | X_hat = pred_val)
        true_in_pred = true_labels[mask]
        cond_counts = np.array([np.sum(true_in_pred == c) for c in unique_true])
        cond_probs = cond_counts / n_pred
        cond_probs = np.clip(cond_probs, 1e-10, 1.0)

        h_cond = -np.sum(cond_probs * np.log2(cond_probs))
        h_y_given_xhat += p_pred * h_cond

    mi = max(0.0, h_y - h_y_given_xhat)
    return float(mi)


def mi_from_accuracy(accuracy: float, n_classes: int) -> float:
    """Quick MI lower bound from classification accuracy.

    Uses Fano's inequality bound:
        MI >= H(Y) - H_b(error_rate) - error_rate * log2(n_classes - 1)

    where H_b is binary entropy.

    Args:
        accuracy: Classification accuracy (0 to 1).
        n_classes: Number of classes.

    Returns:
        MI lower bound in bits.
    """
    if n_classes < 2 or accuracy <= 1.0 / n_classes:
        return 0.0

    h_y = np.log2(n_classes)  # Assumes uniform prior
    error_rate = 1.0 - accuracy

    if error_rate <= 0:
        return h_y
    if error_rate >= 1:
        return 0.0

    # Binary entropy of error rate
    h_b = -error_rate * np.log2(error_rate) - (1 - error_rate) * np.log2(1 - error_rate)

    mi = h_y - h_b - error_rate * np.log2(max(n_classes - 1, 1))
    return float(max(0.0, mi))


# =============================================================================
# CONDITIONAL ENTROPY
# =============================================================================


def compute_conditional_entropy(
    X: np.ndarray,
    labels: np.ndarray,
    k: int = K_NEIGHBORS,
) -> float:
    """Compute H(Y | X) = H(Y) - MI(X; Y).

    Args:
        X: Continuous features, shape (n_samples, n_features).
        labels: Discrete labels, shape (n_samples,).
        k: Number of nearest neighbors for MI estimation.

    Returns:
        Conditional entropy in bits.
    """
    # H(Y) - marginal entropy
    unique, counts = np.unique(labels, return_counts=True)
    p_y = counts / len(labels)
    h_y = -np.sum(p_y * np.log2(np.clip(p_y, 1e-10, 1.0)))

    # MI(X; Y)
    mi = estimate_mi_ksg(X, labels, k=k)

    return float(max(0.0, h_y - mi))


# =============================================================================
# TRAJECTORY ANALYSIS
# =============================================================================


def compute_mi_trajectory(
    layer_activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    k: int = K_NEIGHBORS,
    method: str = "ksg",
    probe_predictions: Optional[Dict[int, np.ndarray]] = None,
) -> List[Dict]:
    """Compute MI(activations; band) at each layer.

    Args:
        layer_activations: Dict mapping layer index to activation array
            of shape (n_samples, n_features).
        labels: Band labels, shape (n_samples,).
        k: Number of nearest neighbors for KSG.
        method: 'ksg' for KSG estimator, 'probe' for probe-based.
        probe_predictions: Required if method='probe'. Dict mapping layer
            to predicted labels.

    Returns:
        List of dicts with layer, mi, method.
    """
    results = []
    for layer in sorted(layer_activations.keys()):
        X = layer_activations[layer]

        if method == "ksg":
            mi = estimate_mi_ksg(X, labels, k=k)
        elif method == "probe":
            if probe_predictions is None or layer not in probe_predictions:
                continue
            mi = probe_based_mi(probe_predictions[layer], labels)
        else:
            raise ValueError(f"Unknown method: {method}")

        results.append(
            {
                "layer": layer,
                "mi": mi,
                "method": method,
            }
        )

    return results


def compute_delta_mi(
    mi_trajectory: List[Dict],
) -> List[Dict]:
    """Compute per-layer MI increment (information gain).

    delta_MI[L] = MI[L] - MI[L-1]

    Args:
        mi_trajectory: List of dicts with 'layer' and 'mi' keys,
            sorted by layer.

    Returns:
        List of dicts with layer, delta_mi, cumulative_mi.
    """
    sorted_traj = sorted(mi_trajectory, key=lambda x: x["layer"])

    results = []
    prev_mi = 0.0
    for entry in sorted_traj:
        delta = entry["mi"] - prev_mi
        results.append(
            {
                "layer": entry["layer"],
                "delta_mi": delta,
                "cumulative_mi": entry["mi"],
            }
        )
        prev_mi = entry["mi"]

    return results


# =============================================================================
# CODING EFFICIENCY
# =============================================================================


def compute_coding_efficiency(mi: float, n_dims: int) -> float:
    """Compute coding efficiency: bits of band information per dimension.

    Args:
        mi: Mutual information in bits.
        n_dims: Number of dimensions (e.g., d_model).

    Returns:
        Bits per dimension.
    """
    if n_dims <= 0:
        return 0.0
    return float(mi / n_dims)


def compute_coding_efficiency_trajectory(
    mi_trajectory: List[Dict],
    n_dims: int,
) -> List[Dict]:
    """Compute coding efficiency at each layer.

    Args:
        mi_trajectory: List of dicts with 'layer' and 'mi' keys.
        n_dims: Number of dimensions at each layer.

    Returns:
        List of dicts with layer, efficiency.
    """
    return [
        {
            "layer": entry["layer"],
            "efficiency": compute_coding_efficiency(entry["mi"], n_dims),
            "mi": entry["mi"],
        }
        for entry in mi_trajectory
    ]


# =============================================================================
# COMPONENT-WISE MI
# =============================================================================


def compute_component_mi(
    attn_out: Dict[int, np.ndarray],
    mlp_out: Dict[int, np.ndarray],
    labels: np.ndarray,
    k: int = K_NEIGHBORS,
) -> List[Dict]:
    """Compute MI for attention and MLP outputs separately at each layer.

    Args:
        attn_out: Dict mapping layer to attention output, (n_samples, d_model).
        mlp_out: Dict mapping layer to MLP output, (n_samples, d_model).
        labels: Band labels, shape (n_samples,).
        k: Number of nearest neighbors.

    Returns:
        List of dicts with layer, attn_mi, mlp_mi.
    """
    layers = sorted(set(attn_out.keys()) & set(mlp_out.keys()))
    results = []

    for layer in layers:
        attn_mi = estimate_mi_ksg(attn_out[layer], labels, k=k)
        mlp_mi = estimate_mi_ksg(mlp_out[layer], labels, k=k)

        results.append(
            {
                "layer": layer,
                "attn_mi": attn_mi,
                "mlp_mi": mlp_mi,
            }
        )

    return results
