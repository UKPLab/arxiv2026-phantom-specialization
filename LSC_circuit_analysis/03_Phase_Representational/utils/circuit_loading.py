"""
Circuit loading utilities for Phase Representational analysis.

Loads ACDC prune_scores, extracts circuit masks (which heads/MLPs are in-circuit),
builds mean activation caches for ablation, and provides circuit summaries.
"""

import re
import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

from .constants import (
    MODELS,
    BANDS,
    DRAWS,
    MODEL_DIR_NAMES,
    MODEL_LAYERS,
    MODEL_HEADS,
    CIRCUITS_DIR,
    CIRCUIT_ACTIVATIONS_DIR,
)


# =============================================================================
# PRUNE SCORES LOADING
# =============================================================================


def load_prune_scores(model: str, band: str, draw: str) -> Dict:
    """Load ACDC prune scores for a circuit.

    Args:
        model: Model name (e.g. 'pythia-70m').
        band: Frequency band name.
        draw: Draw name (e.g. 'draw_1').

    Returns:
        Dict mapping hook_name -> score tensor.
        In-circuit edges have score = inf, pruned edges have finite scores.
    """
    model_dir = MODEL_DIR_NAMES.get(model, model.replace("-", "_"))
    path = CIRCUITS_DIR / model_dir / band / draw / "prune_scores.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Prune scores not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# CIRCUIT EDGE EXTRACTION
# =============================================================================


def get_circuit_edges(prune_scores: Dict) -> List[Tuple[str, int, Optional[int]]]:
    """Extract in-circuit edges from prune scores.

    Args:
        prune_scores: Dict mapping hook_name -> score tensor.

    Returns:
        List of (hook_name, src_idx, head_idx_or_None) for in-circuit edges.
    """
    edges = []
    for hook_name, scores in prune_scores.items():
        scores_np = (
            scores.detach().cpu().numpy()
            if hasattr(scores, "detach")
            else np.array(scores)
        )
        mask = np.isinf(scores_np) & (scores_np > 0)

        if mask.ndim == 2:
            # attn_in: shape [n_heads, n_src]
            heads, srcs = np.where(mask)
            for h, s in zip(heads, srcs):
                edges.append((hook_name, int(s), int(h)))
        elif mask.ndim == 1:
            # mlp_in or resid_post: shape [n_src]
            srcs = np.where(mask)[0]
            for s in srcs:
                edges.append((hook_name, int(s), None))

    return edges


def get_circuit_mask(
    prune_scores: Dict,
    n_layers: int,
    n_heads: int,
) -> Dict[str, np.ndarray]:
    """Convert prune scores to boolean masks per component.

    Determines which attention heads and MLP layers are "in-circuit"
    (i.e., have at least one in-circuit edge connecting to them).

    Args:
        prune_scores: Dict mapping hook_name -> score tensor.
        n_layers: Number of transformer layers.
        n_heads: Number of attention heads per layer.

    Returns:
        Dict with:
        - 'head_mask': (n_layers, n_heads) bool: which heads are in-circuit
        - 'mlp_mask': (n_layers,) bool: which MLPs are in-circuit
        - 'head_edge_counts': (n_layers, n_heads) int: edges per head
        - 'mlp_edge_counts': (n_layers,) int: edges per MLP
    """
    head_mask = np.zeros((n_layers, n_heads), dtype=bool)
    mlp_mask = np.zeros(n_layers, dtype=bool)
    head_edge_counts = np.zeros((n_layers, n_heads), dtype=int)
    mlp_edge_counts = np.zeros(n_layers, dtype=int)

    attn_in_pattern = re.compile(r"^blocks\.(\d+)\.hook_attn_in$")
    mlp_in_pattern = re.compile(r"^blocks\.(\d+)\.hook_mlp_in$")

    for hook_name, scores in prune_scores.items():
        scores_np = (
            scores.detach().cpu().numpy()
            if hasattr(scores, "detach")
            else np.array(scores)
        )
        mask = np.isinf(scores_np) & (scores_np > 0)

        m = attn_in_pattern.match(hook_name)
        if m:
            layer = int(m.group(1))
            # mask shape: [n_heads, n_src]
            for h in range(min(mask.shape[0], n_heads)):
                n_edges = int(mask[h].sum())
                if n_edges > 0:
                    head_mask[layer, h] = True
                    head_edge_counts[layer, h] = n_edges
            continue

        m = mlp_in_pattern.match(hook_name)
        if m:
            layer = int(m.group(1))
            n_edges = int(mask.sum())
            if n_edges > 0:
                mlp_mask[layer] = True
                mlp_edge_counts[layer] = n_edges
            continue

    return {
        "head_mask": head_mask,
        "mlp_mask": mlp_mask,
        "head_edge_counts": head_edge_counts,
        "mlp_edge_counts": mlp_edge_counts,
    }


def get_circuit_summary(prune_scores: Dict) -> Dict[str, Any]:
    """Get summary statistics for a circuit.

    Args:
        prune_scores: Dict mapping hook_name -> score tensor.

    Returns:
        Dict with total_edges, attn_edges, mlp_edges, resid_edges, n_hooks.
    """
    total_edges = 0
    attn_edges = 0
    mlp_edges = 0
    resid_edges = 0

    for hook_name, scores in prune_scores.items():
        scores_np = (
            scores.detach().cpu().numpy()
            if hasattr(scores, "detach")
            else np.array(scores)
        )
        mask = np.isinf(scores_np) & (scores_np > 0)
        n = int(mask.sum())
        total_edges += n

        if "attn_in" in hook_name:
            attn_edges += n
        elif "mlp_in" in hook_name:
            mlp_edges += n
        elif "resid_post" in hook_name:
            resid_edges += n

    return {
        "total_edges": total_edges,
        "attn_edges": attn_edges,
        "mlp_edges": mlp_edges,
        "resid_edges": resid_edges,
        "n_hooks": len(prune_scores),
    }


# =============================================================================
# MEAN ACTIVATION CACHE
# =============================================================================


def build_mean_cache(
    model, dataset: Dict, batch_size: int = 64
) -> Dict[str, "torch.Tensor"]:
    """Build mean activation cache for ablation baseline.

    Runs forward pass over the dataset and computes mean activations
    at every hook point. Non-circuit edges will be replaced with these means.

    Args:
        model: Loaded HookedTransformer model (with fold_ln matching circuit discovery).
        dataset: Dict with 'input_ids' (N, 21) and 'target_ids' (N,).
        batch_size: Batch size for forward passes.

    Returns:
        Dict mapping hook_name -> mean activation tensor on same device as model.
    """
    import torch

    device = next(model.parameters()).device
    input_ids = dataset["input_ids"]
    n_examples = len(input_ids)

    # Prepend BOS
    bos_id = model.tokenizer.bos_token_id
    input_ids_with_bos = np.concatenate(
        [np.full((n_examples, 1), bos_id, dtype=input_ids.dtype), input_ids],
        axis=1,
    )

    # Collect all hook names from the model
    hook_names = []
    for name, _ in model.hook_dict.items():
        hook_names.append(name)

    # Accumulate mean activations
    mean_cache = {}
    n_batches = (n_examples + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_examples)
            batch_size_actual = end - start

            batch_input = torch.tensor(
                input_ids_with_bos[start:end],
                dtype=torch.long,
                device=device,
            )

            _, cache = model.run_with_cache(
                batch_input,
                prepend_bos=False,
            )

            for name in cache.keys():
                val = cache[name]
                if name not in mean_cache:
                    mean_cache[name] = val.sum(dim=0)
                else:
                    mean_cache[name] = mean_cache[name] + val.sum(dim=0)

            del cache
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Divide by N to get mean
    for name in mean_cache:
        mean_cache[name] = mean_cache[name] / n_examples

    return mean_cache


# =============================================================================
# CIRCUIT ACTIVATION FILE I/O
# =============================================================================


def get_circuit_activation_path(model: str, band: str, draw: str) -> Path:
    """Get path for a circuit activation NPZ file.

    Args:
        model: Model name.
        band: Frequency band name.
        draw: Draw name.

    Returns:
        Path to circuit NPZ file.
    """
    model_dir = MODEL_DIR_NAMES.get(model, model.replace("-", "_"))
    return CIRCUIT_ACTIVATIONS_DIR / f"{model_dir}_{band}_{draw}_circuit.npz"


def load_circuit_activations(model: str, band: str, draw: str) -> Dict[str, np.ndarray]:
    """Load circuit-mode extracted activations.

    Args:
        model: Model name.
        band: Frequency band name.
        draw: Draw name.

    Returns:
        Dict of numpy arrays keyed by array name.
    """
    path = get_circuit_activation_path(model, band, draw)
    if not path.exists():
        raise FileNotFoundError(f"Circuit activations not found: {path}")
    return dict(np.load(path, allow_pickle=False))


def load_base_and_circuit(model: str, band: str, draw: str) -> Tuple[Dict, Dict]:
    """Load both base and circuit activations for comparison.

    Args:
        model: Model name.
        band: Frequency band name.
        draw: Draw name.

    Returns:
        (base_activations, circuit_activations) tuple.
    """
    from .data_loading import load_extracted_activations

    base = load_extracted_activations(model, band, draw)
    circuit = load_circuit_activations(model, band, draw)
    return base, circuit
