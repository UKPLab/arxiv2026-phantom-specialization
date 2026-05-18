"""
Linear and MLP probing utilities for frequency band classification.

StandardScaler inside CV pipeline (no data leakage),
StratifiedKFold, balanced class weights.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report

from .constants import CV_FOLDS, RANDOM_SEED, PROBE_REGULARIZATION, N_PERMUTATIONS


def create_probe_pipeline(
    C: float = PROBE_REGULARIZATION, max_iter: int = 2000
) -> Pipeline:
    """Create a linear probe pipeline with StandardScaler.

    Scaler is inside the pipeline to prevent data leakage during CV.

    Args:
        C: Regularization strength (inverse).
        max_iter: Maximum iterations for solver.

    Returns:
        sklearn Pipeline.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=max_iter,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def train_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    n_folds: int = CV_FOLDS,
    C: float = PROBE_REGULARIZATION,
    return_predictions: bool = False,
) -> Dict:
    """Train a linear probe with cross-validated evaluation.

    Args:
        activations: Array of shape (n_samples, n_features).
        labels: Array of band labels (string or int).
        n_folds: Number of CV folds.
        C: Regularization strength.
        return_predictions: Whether to return per-sample predictions.

    Returns:
        Dict with accuracy, std, cv_scores, and optionally predictions.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)

    pipeline = create_probe_pipeline(C=C)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    scores = []
    all_preds = np.zeros_like(y)

    for train_idx, test_idx in cv.split(activations, y):
        X_train, X_test = activations[train_idx], activations[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        scores.append(accuracy_score(y_test, preds))
        all_preds[test_idx] = preds

    result = {
        "accuracy": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "cv_scores": [float(s) for s in scores],
        "classes": le.classes_.tolist(),
    }

    if return_predictions:
        result["predictions"] = le.inverse_transform(all_preds)
        result["true_labels"] = le.inverse_transform(y)

    return result


def train_probe_trajectory(
    layer_activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    n_folds: int = CV_FOLDS,
) -> List[Dict]:
    """Train probes at each layer and return trajectory.

    Args:
        layer_activations: Dict mapping layer index to activation array.
        labels: Array of band labels.
        n_folds: Number of CV folds.

    Returns:
        List of dicts with layer, accuracy, std.
    """
    results = []
    for layer in sorted(layer_activations.keys()):
        probe_result = train_probe(layer_activations[layer], labels, n_folds=n_folds)
        results.append(
            {
                "layer": layer,
                "accuracy": probe_result["accuracy"],
                "std": probe_result["std"],
            }
        )
    return results


def train_mlp_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    n_folds: int = CV_FOLDS,
    hidden_sizes: tuple = (128, 64),
) -> Dict:
    """Train a non-linear MLP probe for comparison with linear probe.

    Args:
        activations: Array of shape (n_samples, n_features).
        labels: Array of band labels.
        n_folds: Number of CV folds.
        hidden_sizes: MLP hidden layer sizes.

    Returns:
        Dict with accuracy, std, cv_scores.
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=hidden_sizes,
                    max_iter=500,
                    random_state=RANDOM_SEED,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    scores = []

    for train_idx, test_idx in cv.split(activations, y):
        X_train, X_test = activations[train_idx], activations[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        scores.append(accuracy_score(y_test, preds))

    return {
        "accuracy": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "cv_scores": [float(s) for s in scores],
    }


def train_probe_with_permutation(
    activations: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
    n_folds: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
) -> Dict:
    """Train probe and compute permutation null for significance.

    Args:
        activations: Array of shape (n_samples, n_features).
        labels: Array of band labels.
        n_permutations: Number of permutations.
        n_folds: Number of CV folds.
        seed: Random seed.

    Returns:
        Dict with observed accuracy, p_value, null_distribution.
    """
    observed = train_probe(activations, labels, n_folds=n_folds)

    rng = np.random.RandomState(seed)
    null_accs = np.zeros(n_permutations)

    for i in range(n_permutations):
        shuffled = rng.permutation(labels)
        result = train_probe(activations, shuffled, n_folds=n_folds)
        null_accs[i] = result["accuracy"]

    p_value = float(np.mean(null_accs >= observed["accuracy"]))

    return {
        "accuracy": observed["accuracy"],
        "std": observed["std"],
        "p_value": p_value,
        "null_mean": float(np.mean(null_accs)),
        "null_std": float(np.std(null_accs)),
        "null_distribution": null_accs,
    }
