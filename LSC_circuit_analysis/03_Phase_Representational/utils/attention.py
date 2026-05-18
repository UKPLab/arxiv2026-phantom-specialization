"""
Attention pattern analysis utilities.

Head role classification, copy scores, entropy (in bits), and positional
attention statistics for LSC induction task.

- Entropy always in bits (base=2), not nats
- Head role classification uses dominance ratio, not simple if-elif
- Copy scores use correct target_id, never re-encode
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .constants import (
    HEAD_ROLE_THRESHOLDS,
    MODEL_PREDICTION_POS,
    MODEL_SOURCE_POS,
    MODEL_REPEAT_POS,
    MODEL_INDUCTION_CUE_POS,
    MODEL_TARGET_POS,
    BOS_OFFSET,
    SEQ_LEN_WITH_BOS,
)


# =============================================================================
# ATTENTION ENTROPY
# =============================================================================


def compute_attention_entropy(
    attention_probs: np.ndarray, base: int = 2, eps: float = 1e-10
) -> np.ndarray:
    """Compute entropy of attention distributions in bits.

    Args:
        attention_probs: Attention probabilities with shape [..., seq_len].
        base: Logarithm base. Default 2 for bits.
        eps: Small value for numerical stability.

    Returns:
        Entropy values with same shape except last dimension.
    """
    probs = np.clip(attention_probs, eps, 1.0)
    if base == 2:
        entropy = -np.sum(probs * np.log2(probs), axis=-1)
    else:
        entropy = -np.sum(probs * np.log(probs), axis=-1) / np.log(base)
    return entropy


# =============================================================================
# POSITIONAL ATTENTION STATISTICS
# =============================================================================


def compute_positional_attention_stats(
    attn_from_predpos: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute positional attention statistics from prediction position.

    Args:
        attn_from_predpos: Attention weights FROM prediction position TO all positions.
            Shape: (n_examples, n_layers, n_heads, seq_len_with_bos) or
                   (n_layers, n_heads, seq_len_with_bos).

    Returns:
        Dict with attention fractions, each shaped like input without last dim.
    """
    # Ensure 4D
    squeeze = False
    if attn_from_predpos.ndim == 3:
        attn_from_predpos = attn_from_predpos[np.newaxis]
        squeeze = True

    stats = {}

    # BOS attention (position 0, which is the BOS token)
    stats["bos_fraction"] = attn_from_predpos[..., 0]

    # Self attention (to prediction position itself)
    stats["self_fraction"] = attn_from_predpos[..., MODEL_PREDICTION_POS]

    # Previous token attention (position just before prediction)
    stats["prev_fraction"] = attn_from_predpos[..., MODEL_PREDICTION_POS - 1]

    # Source pattern attention (S1..S5 at model positions 1-5)
    source_start = MODEL_SOURCE_POS[0]
    source_end = MODEL_SOURCE_POS[-1] + 1
    stats["source_fraction"] = attn_from_predpos[..., source_start:source_end].sum(
        axis=-1
    )

    # Target attention (T at model position 6)
    stats["target_fraction"] = attn_from_predpos[..., MODEL_TARGET_POS]

    # Repeat pattern attention (S1..S5 repeated at model positions 17-21)
    repeat_start = MODEL_REPEAT_POS[0]
    repeat_end = MODEL_REPEAT_POS[-1] + 1
    stats["repeat_fraction"] = attn_from_predpos[..., repeat_start:repeat_end].sum(
        axis=-1
    )

    # Induction cue: attention to the previous occurrence of the current token
    # (S5 in source block at model pos 5). Key diagnostic for induction heads.
    stats["induction_score"] = attn_from_predpos[..., MODEL_INDUCTION_CUE_POS]

    # Distractor attention (positions 7-16 in model coords)
    dist_start = MODEL_SOURCE_POS[-1] + 2  # Skip target
    dist_end = MODEL_REPEAT_POS[0]
    if dist_end > dist_start:
        stats["distractor_fraction"] = attn_from_predpos[..., dist_start:dist_end].sum(
            axis=-1
        )

    # Entropy (bits)
    stats["entropy"] = compute_attention_entropy(attn_from_predpos)

    if squeeze:
        stats = {k: v.squeeze(0) for k, v in stats.items()}

    return stats


# =============================================================================
# HEAD ROLE CLASSIFICATION
# =============================================================================


def classify_head_role(
    attn_stats: Dict[str, np.ndarray],
    thresholds: Dict[str, float] = None,
) -> str:
    """Classify a single attention head's role based on positional statistics.

    Uses dominance ratio: the top-scoring role must be sufficiently stronger
    than the runner-up to avoid ambiguous classifications.

    Args:
        attn_stats: Dict with mean attention fractions for this head:
            induction_score, bos_fraction, prev_fraction, entropy.
        thresholds: Classification thresholds. Defaults to HEAD_ROLE_THRESHOLDS.

    Returns:
        Role string: 'induction', 'previous_token', 'bos_sink', or 'diffuse'.
    """
    if thresholds is None:
        thresholds = HEAD_ROLE_THRESHOLDS

    induction = float(attn_stats.get("induction_score", 0))
    bos = float(attn_stats.get("bos_fraction", 0))
    prev_tok = float(attn_stats.get("prev_fraction", 0))
    entropy = float(attn_stats.get("entropy", 0))

    # Candidate scores (higher = more likely that role)
    candidates = {
        "induction": induction / max(thresholds["induction_min"], 1e-10),
        "bos_sink": bos / max(thresholds["bos_sink_min"], 1e-10),
        "previous_token": prev_tok / max(thresholds["prev_token_min"], 1e-10),
    }

    # Sort by score
    sorted_roles = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    top_role, top_score = sorted_roles[0]
    runner_up_score = sorted_roles[1][1] if len(sorted_roles) > 1 else 0

    # Check minimum thresholds
    passes_threshold = {
        "induction": induction >= thresholds["induction_min"],
        "bos_sink": bos >= thresholds["bos_sink_min"],
        "previous_token": prev_tok >= thresholds["prev_token_min"],
    }

    # Dominance check: top role must be sufficiently stronger than runner-up
    if passes_threshold.get(top_role, False) and (
        runner_up_score == 0
        or top_score / max(runner_up_score, 1e-10) >= thresholds["dominance_ratio"]
    ):
        return top_role

    # Check diffuse (high entropy, no dominant pattern)
    if entropy >= thresholds["entropy_diffuse_min"]:
        return "diffuse"

    # Fallback: if top role passes threshold but not dominance, still assign it
    if passes_threshold.get(top_role, False):
        return top_role

    return "diffuse"


def classify_all_heads(
    attn_from_predpos: np.ndarray,
    n_layers: int,
    n_heads: int,
) -> List[Dict]:
    """Classify all heads in a model.

    Args:
        attn_from_predpos: Attention from prediction position.
            Shape: (n_examples, n_layers, n_heads, seq_len_with_bos).
        n_layers: Number of layers.
        n_heads: Number of heads per layer.

    Returns:
        List of dicts with layer, head, role, and attention stats.
    """
    # Compute mean stats across examples
    mean_attn = attn_from_predpos.mean(axis=0)  # (n_layers, n_heads, seq_len)

    results = []
    for layer in range(n_layers):
        for head in range(n_heads):
            head_attn = mean_attn[layer, head]

            stats = compute_positional_attention_stats(
                head_attn[np.newaxis, np.newaxis, np.newaxis, :]
            )
            # Flatten stats to scalars
            head_stats = {k: float(v.item()) for k, v in stats.items()}

            role = classify_head_role(head_stats)

            results.append(
                {
                    "layer": layer,
                    "head": head,
                    "role": role,
                    **head_stats,
                }
            )

    return results


# =============================================================================
# COPY SCORES
# =============================================================================


def compute_copy_scores(
    model, token_pairs: List[Tuple[int, int]] = None, n_pairs: int = 200, seed: int = 42
) -> np.ndarray:
    """Compute OV copy scores for all heads.

    copy_score(layer, head) = mean over token pairs of:
        W_U[target]^T @ W_OV @ W_E[source]

    where W_OV = W_V @ W_O for that head.

    Args:
        model: HookedTransformer model.
        token_pairs: List of (source_token_id, target_token_id) pairs.
            If None, samples random pairs.
        n_pairs: Number of random pairs if token_pairs is None.
        seed: Random seed for random pair sampling.

    Returns:
        Array of shape (n_layers, n_heads) with mean copy scores.
    """
    import torch

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head

    W_E = model.W_E.detach()  # (d_vocab, d_model)
    W_U = model.W_U.detach()  # (d_model, d_vocab)

    if token_pairs is None:
        rng = np.random.RandomState(seed)
        d_vocab = model.cfg.d_vocab
        sources = rng.randint(0, d_vocab, size=n_pairs)
        targets = rng.randint(0, d_vocab, size=n_pairs)
        token_pairs = list(zip(sources.tolist(), targets.tolist()))

    source_ids = torch.tensor([p[0] for p in token_pairs], device=W_E.device)
    target_ids = torch.tensor([p[1] for p in token_pairs], device=W_E.device)

    # Source embeddings: (n_pairs, d_model)
    source_embeds = W_E[source_ids]
    # Target unembed directions: (n_pairs, d_model)
    target_unembed = W_U[:, target_ids].T  # (n_pairs, d_model)

    scores = np.zeros((n_layers, n_heads))

    with torch.no_grad():
        for layer in range(n_layers):
            W_V = model.blocks[layer].attn.W_V.detach()  # (n_heads, d_model, d_head)
            W_O = model.blocks[layer].attn.W_O.detach()  # (n_heads, d_head, d_model)

            for head in range(n_heads):
                # W_OV = W_V[head] @ W_O[head]: (d_model, d_model)
                W_OV = W_V[head] @ W_O[head]  # (d_model, d_model)

                # Copy score: target_unembed^T @ W_OV @ source_embed
                # (n_pairs, d_model) @ (d_model, d_model) @ (d_model, n_pairs) -> diagonal
                intermediate = source_embeds @ W_OV.T  # (n_pairs, d_model)
                copy = (target_unembed * intermediate).sum(dim=1)  # (n_pairs,)

                scores[layer, head] = copy.mean().item()

    return scores


def compute_copy_scores_by_band(
    model, datasets: Dict, bands: List[str], n_pairs_per_band: int = 100, seed: int = 42
) -> Dict[str, np.ndarray]:
    """Compute copy scores separately for each frequency band.

    Uses actual source-target token pairs from LSC datasets.

    Args:
        model: HookedTransformer model.
        datasets: Dict mapping band to loaded dataset.
        bands: List of band names.
        n_pairs_per_band: Number of pairs per band.
        seed: Random seed.

    Returns:
        Dict mapping band to copy score array (n_layers, n_heads).
    """
    rng = np.random.RandomState(seed)
    results = {}

    for band in bands:
        if band not in datasets:
            continue

        examples = datasets[band]["examples"]
        # Sample pairs: (source_token = S5 at pos 4, target = T at pos 5)
        n = min(n_pairs_per_band, len(examples))
        indices = rng.choice(len(examples), size=n, replace=False)

        pairs = []
        for idx in indices:
            ex = examples[idx]
            source_id = ex["token_ids"][4]  # S5
            target_id = ex["token_ids"][5]  # T
            pairs.append((source_id, target_id))

        results[band] = compute_copy_scores(model, token_pairs=pairs)

    return results
