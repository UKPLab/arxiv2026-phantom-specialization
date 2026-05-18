"""
Logit lens utilities for Pythia models.

Computes per-layer P(correct), rank(correct), and KL divergence from
final-layer distribution using the residual stream.

design decisions:
- Applies ln_final -> W_U only to the FULL residual stream (never to partial
  sums like attention-only or MLP-only output), since LayerNorm is nonlinear
  and only meaningful on the complete residual.
- Pythia uses parallel blocks: attention and MLP operate on the same
  layer-normalized input. The residual stream is:
    resid_post[L] = resid_pre[L] + attn_out[L] + mlp_out[L]
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .constants import CONVERGENCE_THRESHOLD


# =============================================================================
# LOGIT LENS CORE
# =============================================================================


def logit_lens_from_residuals(
    resid_post: np.ndarray,
    ln_final_weight: np.ndarray,
    ln_final_bias: np.ndarray,
    W_U: np.ndarray,
    b_U: Optional[np.ndarray] = None,
    eps: float = 1e-5,
) -> np.ndarray:
    """Apply logit lens: ln_final -> W_U to residual stream.

    Args:
        resid_post: Residual stream after each layer.
            Shape: (n_examples, n_layers, d_model) or (n_layers, d_model).
        ln_final_weight: LayerNorm final weight, shape (d_model,).
        ln_final_bias: LayerNorm final bias, shape (d_model,).
        W_U: Unembedding matrix, shape (d_model, d_vocab).
        b_U: Unembedding bias, shape (d_vocab,) or None.
        eps: LayerNorm epsilon.

    Returns:
        Logits at each layer, shape (n_examples, n_layers, d_vocab) or
        (n_layers, d_vocab).
    """
    squeeze = False
    if resid_post.ndim == 2:
        resid_post = resid_post[np.newaxis]
        squeeze = True

    # Apply LayerNorm: normalize, scale, shift
    mean = resid_post.mean(axis=-1, keepdims=True)
    var = resid_post.var(axis=-1, keepdims=True)
    normalized = (resid_post - mean) / np.sqrt(var + eps)
    ln_out = normalized * ln_final_weight + ln_final_bias

    # Apply unembedding
    logits = ln_out @ W_U
    if b_U is not None:
        logits = logits + b_U

    if squeeze:
        logits = logits.squeeze(0)
    return logits


def compute_logit_lens_metrics(
    logits: np.ndarray,
    target_ids: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute logit lens metrics from pre-computed logits.

    Args:
        logits: Logit values, shape (n_examples, n_layers, d_vocab).
        target_ids: Correct target token IDs, shape (n_examples,).

    Returns:
        Dict with:
            prob_correct: P(correct) per example per layer, (n_examples, n_layers).
            rank_correct: Rank of correct token (0-indexed), (n_examples, n_layers).
            kl_from_final: KL(final || layer), (n_examples, n_layers).
            logit_correct: Raw logit for correct token, (n_examples, n_layers).
    """
    n_examples, n_layers, d_vocab = logits.shape

    # Softmax (numerically stable)
    logits_max = logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    # P(correct) at each layer
    prob_correct = np.zeros((n_examples, n_layers))
    logit_correct = np.zeros((n_examples, n_layers))
    for i in range(n_examples):
        prob_correct[i] = probs[i, :, target_ids[i]]
        logit_correct[i] = logits[i, :, target_ids[i]]

    # Rank of correct token (0 = top rank)
    rank_correct = np.zeros((n_examples, n_layers), dtype=np.int32)
    for i in range(n_examples):
        for layer in range(n_layers):
            # Count how many tokens have higher logit than the correct one
            rank_correct[i, layer] = (logits[i, layer] > logit_correct[i, layer]).sum()

    # KL(final_layer || current_layer) per example
    final_probs = probs[:, -1, :]  # (n_examples, d_vocab)
    eps = 1e-10
    kl_from_final = np.zeros((n_examples, n_layers))
    for layer in range(n_layers):
        layer_probs = np.clip(probs[:, layer, :], eps, 1.0)
        final_clipped = np.clip(final_probs, eps, 1.0)
        # KL(final || layer) = sum final * log(final / layer)
        kl_from_final[:, layer] = np.sum(
            final_clipped * (np.log(final_clipped) - np.log(layer_probs)),
            axis=-1,
        )

    return {
        "prob_correct": prob_correct,
        "rank_correct": rank_correct,
        "kl_from_final": kl_from_final,
        "logit_correct": logit_correct,
    }


# =============================================================================
# CONVERGENCE ANALYSIS
# =============================================================================


def compute_convergence_layer(
    prob_correct: np.ndarray,
    threshold: float = CONVERGENCE_THRESHOLD,
) -> np.ndarray:
    """Find the layer where P(correct) first exceeds threshold x final P(correct).

    Args:
        prob_correct: P(correct) per example per layer, shape (n_examples, n_layers).
        threshold: Fraction of final probability to use as convergence criterion.

    Returns:
        Convergence layer per example, shape (n_examples,).
        Returns n_layers (= never converged) if threshold is never reached.
    """
    n_examples, n_layers = prob_correct.shape
    final_prob = prob_correct[:, -1]  # (n_examples,)
    target_prob = threshold * final_prob

    convergence = np.full(n_examples, n_layers, dtype=np.int32)
    for i in range(n_examples):
        exceeds = np.where(prob_correct[i] >= target_prob[i])[0]
        if len(exceeds) > 0:
            convergence[i] = exceeds[0]

    return convergence


# =============================================================================
# COMPONENT ATTRIBUTION (LOGIT DECOMPOSITION)
# =============================================================================


def compute_component_attribution(
    attn_out: np.ndarray,
    mlp_out: np.ndarray,
    W_U: np.ndarray,
    target_ids: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Decompose the logit for the correct token into attention and MLP contributions.

    At each layer, the residual stream accumulates:
        resid += attn_out + mlp_out  (Pythia parallel blocks)

    The attribution to the correct logit direction is:
        attn_contrib = attn_out · W_U[:, target_id]
        mlp_contrib  = mlp_out  · W_U[:, target_id]

    IMPORTANT: We do NOT apply ln_final to individual components. ln_final is
    a nonlinear operation that can only be meaningfully applied to the full
    residual stream.

    Args:
        attn_out: Attention output at prediction position.
            Shape: (n_examples, n_layers, d_model).
        mlp_out: MLP output at prediction position.
            Shape: (n_examples, n_layers, d_model).
        W_U: Unembedding matrix, shape (d_model, d_vocab).
        target_ids: Correct target token IDs, shape (n_examples,).

    Returns:
        Dict with:
            attn_logit: Attention contribution to correct logit, (n_examples, n_layers).
            mlp_logit: MLP contribution, (n_examples, n_layers).
            attn_frac: Attention fraction (of |attn| + |mlp|), (n_examples, n_layers).
            mlp_frac: MLP fraction, (n_examples, n_layers).
    """
    n_examples, n_layers, d_model = attn_out.shape

    # Target unembedding directions: (n_examples, d_model)
    target_dirs = W_U[:, target_ids].T  # (n_examples, d_model)

    # Dot product attribution per layer
    attn_logit = np.zeros((n_examples, n_layers))
    mlp_logit = np.zeros((n_examples, n_layers))

    for layer in range(n_layers):
        # (n_examples, d_model) · (n_examples, d_model) -> (n_examples,)
        attn_logit[:, layer] = np.sum(attn_out[:, layer] * target_dirs, axis=-1)
        mlp_logit[:, layer] = np.sum(mlp_out[:, layer] * target_dirs, axis=-1)

    # Attribution fractions (using absolute values to handle negative contributions)
    abs_attn = np.abs(attn_logit)
    abs_mlp = np.abs(mlp_logit)
    total = abs_attn + abs_mlp + 1e-10

    attn_frac = abs_attn / total
    mlp_frac = abs_mlp / total

    return {
        "attn_logit": attn_logit,
        "mlp_logit": mlp_logit,
        "attn_frac": attn_frac,
        "mlp_frac": mlp_frac,
    }


# =============================================================================
# FULL LOGIT LENS PIPELINE (TORCH-BASED, FOR EXTRACTION)
# =============================================================================


def compute_logit_lens_torch(model, resid_post, target_ids):
    """Compute logit lens metrics using PyTorch model weights.

    This is the preferred method during extraction when the model is loaded.
    Uses the model's actual ln_final and W_U for accuracy.

    Args:
        model: HookedTransformer model.
        resid_post: Residual stream tensor, shape (n_examples, n_layers, d_model).
        target_ids: Target token IDs tensor, shape (n_examples,).

    Returns:
        Dict with numpy arrays:
            prob_correct: (n_examples, n_layers)
            rank_correct: (n_examples, n_layers)
            kl_from_final: (n_examples, n_layers)
    """
    import torch

    device = resid_post.device
    n_examples, n_layers, d_model = resid_post.shape

    # Extract model components
    ln_final = model.ln_final
    W_U = model.W_U  # (d_model, d_vocab)
    b_U = model.b_U if hasattr(model, "b_U") and model.b_U is not None else None

    prob_correct = torch.zeros(n_examples, n_layers, device=device)
    rank_correct = torch.zeros(n_examples, n_layers, dtype=torch.long, device=device)
    logit_correct = torch.zeros(n_examples, n_layers, device=device)

    with torch.no_grad():
        all_probs = []
        for layer in range(n_layers):
            # Apply ln_final to the full residual stream at this layer
            normed = ln_final(resid_post[:, layer])  # (n_examples, d_model)
            logits = normed @ W_U  # (n_examples, d_vocab)
            if b_U is not None:
                logits = logits + b_U

            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs)

            # Extract metrics for correct tokens
            for i in range(n_examples):
                tid = target_ids[i]
                prob_correct[i, layer] = probs[i, tid]
                logit_correct[i, layer] = logits[i, tid]
                rank_correct[i, layer] = (logits[i] > logits[i, tid]).sum()

        # KL from final layer
        all_probs = torch.stack(all_probs, dim=1)  # (n_examples, n_layers, d_vocab)
        final_probs = all_probs[:, -1, :]  # (n_examples, d_vocab)

        eps = 1e-10
        kl_from_final = torch.zeros(n_examples, n_layers, device=device)
        for layer in range(n_layers):
            layer_probs = all_probs[:, layer, :].clamp(min=eps)
            final_clipped = final_probs.clamp(min=eps)
            kl = (final_clipped * (final_clipped.log() - layer_probs.log())).sum(dim=-1)
            kl_from_final[:, layer] = kl

    return {
        "prob_correct": prob_correct.cpu().numpy(),
        "rank_correct": rank_correct.cpu().numpy(),
        "kl_from_final": kl_from_final.cpu().numpy(),
    }
