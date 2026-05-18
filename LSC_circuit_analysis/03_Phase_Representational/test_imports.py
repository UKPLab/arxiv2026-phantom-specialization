#!/usr/bin/env python3
"""test of utils package imports and basic functionality."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("Testing utils imports...")

# Test constants
from utils.constants import (
    MODELS,
    BANDS,
    DRAWS,
    MODEL_INFO,
    MODEL_D_MODEL,
    HF_MODEL_NAMES,
    DATASETS_BASE,
    ACTIVATIONS_DIR,
    SEQ_LEN,
    SEQ_LEN_WITH_BOS,
    BOS_OFFSET,
    MODEL_PREDICTION_POS,
    MODEL_SOURCE_POS,
    MODEL_TARGET_POS,
    HEAD_ROLE_THRESHOLDS,
    BAND_COLORS,
)

print(f"  constants: OK (MODELS={MODELS}, SEQ_LEN_WITH_BOS={SEQ_LEN_WITH_BOS})")

# Test data loading
from utils.data_loading import load_lsc_dataset, get_token_ids_and_targets

ds = load_lsc_dataset("draw_1", "low", "test")
input_ids, target_ids = get_token_ids_and_targets(ds)
print(
    f"  data_loading: OK (input_ids shape={input_ids.shape}, target_ids shape={target_ids.shape})"
)

# Test geometry
import numpy as np
from utils.geometry import (
    compute_band_centroids,
    compute_separation_ratio,
    linear_cka,
    compute_knn_purity,
    compute_participation_ratio,
)

X = np.random.randn(100, 64).astype(np.float32)
labels = np.array(["low"] * 25 + ["medium"] * 25 + ["high"] * 25 + ["very_high"] * 25)
centroids = compute_band_centroids(X, labels)
purity = compute_knn_purity(X, labels, k=5)
pr = compute_participation_ratio(X)
print(f"  geometry: OK (centroids={len(centroids)}, purity={purity:.3f}, PR={pr:.1f})")

# Test probing
from utils.probing import train_probe

result = train_probe(X, labels, n_folds=3)
print(f"  probing: OK (accuracy={result['accuracy']:.3f})")

# Test attention
from utils.attention import compute_attention_entropy, classify_head_role

probs = np.random.dirichlet(np.ones(22), size=(4, 6, 8))
entropy = compute_attention_entropy(probs)
print(
    f"  attention: OK (entropy shape={entropy.shape}, mean={entropy.mean():.2f} bits)"
)

# Test logit lens (numpy path)
from utils.logit_lens import compute_convergence_layer

prob_correct = np.random.rand(10, 6)
conv = compute_convergence_layer(prob_correct)
print(f"  logit_lens: OK (convergence shape={conv.shape})")

# Test info theory
from utils.info_theory import estimate_mi_ksg, mi_from_accuracy

mi = estimate_mi_ksg(X[:50], labels[:50], k=3)
mi_bound = mi_from_accuracy(0.8, 4)
print(f"  info_theory: OK (KSG MI={mi:.3f} bits, Fano bound={mi_bound:.3f} bits)")

# Test stats
from utils.stats import permutation_test_metric, hedges_g

g, ci_lo, ci_hi = hedges_g(np.random.randn(30), np.random.randn(30))
print(f"  stats: OK (Hedges' g={g:.3f}, 95% CI=[{ci_lo:.3f}, {ci_hi:.3f}])")

# Test plotting
from utils.plotting import setup_plotting

setup_plotting()
print(f"  plotting: OK")

# Test dataset paths exist
n_found = 0
for draw in DRAWS:
    for band in BANDS:
        path = DATASETS_BASE / draw / band / "test.json"
        if path.exists():
            n_found += 1
print(f"\nDataset files found: {n_found}/15")

# Check output directories
ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir ready: {ACTIVATIONS_DIR}")

print("\nAll imports and basic tests passed!")
