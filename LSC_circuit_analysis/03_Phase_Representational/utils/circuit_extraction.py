"""
Circuit-mode activation extraction for Phase Representational analysis.

Extracts activations from Pythia models with mean ablation applied:
non-circuit edges are replaced with their dataset-mean activations.
This isolates the computational path through the discovered circuit.

design decisions:
- fold_ln=True to match circuit discovery setting
- Mean ablation: non-circuit edge outputs -> dataset mean
- Same output format as base extraction for direct comparison
- Prediction position only (no full-sequence extraction)
"""

import re
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from functools import partial

from .constants import (
    MODELS,
    BANDS,
    DRAWS,
    MODEL_DIR_NAMES,
    HF_MODEL_NAMES,
    MODEL_D_MODEL,
    MODEL_D_MLP,
    DATASETS_BASE,
    CIRCUIT_ACTIVATIONS_DIR,
    EXTRACTION_DIR,
    SEQ_LEN,
    SEQ_LEN_WITH_BOS,
    BOS_OFFSET,
    MODEL_PREDICTION_POS,
    TARGET_POS,
    EXTRACTION_BATCH_SIZE,
    CIRCUIT_FOLD_LN,
)
from .circuit_loading import (
    load_prune_scores,
    get_circuit_mask,
    get_circuit_summary,
    build_mean_cache,
    get_circuit_activation_path,
)
from .extraction import (
    get_device,
    _patch_gptneox_config,
    load_dataset_for_extraction,
    save_extraction,
    save_extraction_summary,
)


# =============================================================================
# MODEL LOADING (with fold_ln=True for circuit mode)
# =============================================================================


def load_model_circuit_mode(model_name: str, device: str = None, verbose: bool = True):
    """Load a HookedTransformer model with fold_ln=True for circuit analysis.

    Circuit discovery used fold_ln=True, so we must match this setting
    to ensure mean ablation values are in the correct space.

    Args:
        model_name: Short model name (e.g. 'pythia-70m').
        device: Device string. Auto-detected if None.
        verbose: Print loading progress.

    Returns:
        Loaded HookedTransformer model with fold_ln=True.
    """
    from transformer_lens import HookedTransformer

    _patch_gptneox_config()

    if device is None:
        device = get_device(verbose=verbose)

    hf_name = HF_MODEL_NAMES.get(model_name, f"EleutherAI/{model_name}")

    if verbose:
        print(f"\nLoading {hf_name} (fold_ln={CIRCUIT_FOLD_LN})...")

    model = HookedTransformer.from_pretrained(
        hf_name,
        device=device,
        fold_ln=CIRCUIT_FOLD_LN,
        center_writing_weights=False,
        center_unembed=False,
    )

    if verbose:
        print(
            f"  Layers: {model.cfg.n_layers}, Heads: {model.cfg.n_heads}, "
            f"d_model: {model.cfg.d_model}, fold_ln: {CIRCUIT_FOLD_LN}"
        )

    return model


# =============================================================================
# ABLATION HOOK CREATION
# =============================================================================


def create_ablation_hooks(
    prune_scores: Dict,
    mean_cache: Dict,
    n_heads: int,
) -> List[Tuple[str, callable]]:
    """Create hook functions for source-level mean ablation.

    Determines which source components (attention heads, MLPs) have ANY
    outgoing in-circuit edges to downstream destinations. Components with
    zero outgoing in-circuit edges are fully ablated (output replaced
    with mean). This is source-level ablation: more granular than
    destination-level but tractable with TransformerLens hooks.

    Args:
        prune_scores: Dict mapping hook_name -> score tensor.
            Keys like 'blocks.{L}.hook_attn_in' (shape: n_heads, n_sources)
            and 'blocks.{L}.hook_mlp_in' (shape: n_sources,).
            Source encoding: 0=embed, then (n_heads+1) per layer.
        mean_cache: Dict mapping hook_name -> mean activation tensor.
        n_heads: Number of attention heads.

    Returns:
        List of (hook_name, hook_fn) tuples for model.add_hook().
    """
    import torch

    hooks = []

    attn_in_pattern = re.compile(r"^blocks\.(\d+)\.hook_attn_in$")
    mlp_in_pattern = re.compile(r"^blocks\.(\d+)\.hook_mlp_in$")
    resid_pattern = re.compile(r"^blocks\.(\d+)\.hook_resid_post$")

    n_sources_per_layer = n_heads + 1  # n_heads attn + 1 MLP

    def decode_source(src_idx):
        """Decode source index: 0=embed, then (n_heads+1) per layer."""
        if src_idx == 0:
            return ("embed", -1, -1)
        adjusted = src_idx - 1
        src_layer = adjusted // n_sources_per_layer
        src_component = adjusted % n_sources_per_layer
        if src_component < n_heads:
            return ("head", src_layer, src_component)
        else:
            return ("mlp", src_layer, -1)

    def register_sources(in_circuit, is_2d=False):
        """Mark source components that have outgoing in-circuit edges."""
        if is_2d:
            # shape: (n_heads, n_sources): iterate over all heads
            for h in range(in_circuit.shape[0]):
                for src_idx in range(in_circuit.shape[1]):
                    if in_circuit[h, src_idx]:
                        stype, slayer, scomp = decode_source(src_idx)
                        if stype == "head":
                            head_has_outgoing.add((slayer, scomp))
                        elif stype == "mlp":
                            mlp_has_outgoing.add(slayer)
        else:
            # shape: (n_sources,)
            for src_idx in range(len(in_circuit)):
                if in_circuit[src_idx]:
                    stype, slayer, scomp = decode_source(src_idx)
                    if stype == "head":
                        head_has_outgoing.add((slayer, scomp))
                    elif stype == "mlp":
                        mlp_has_outgoing.add(slayer)

    # Scan all prune_scores to find which sources have outgoing circuit edges
    head_has_outgoing = set()  # set of (layer, head)
    mlp_has_outgoing = set()  # set of layer

    for dest_hook, scores in prune_scores.items():
        scores_np = (
            scores.detach().cpu().numpy()
            if hasattr(scores, "detach")
            else np.array(scores)
        )
        in_circuit = np.isinf(scores_np) & (scores_np > 0)

        m = attn_in_pattern.match(dest_hook)
        if m:
            register_sources(in_circuit, is_2d=True)
            continue

        m = mlp_in_pattern.match(dest_hook)
        if m:
            register_sources(in_circuit, is_2d=False)
            continue

        m = resid_pattern.match(dest_hook)
        if m:
            register_sources(in_circuit, is_2d=False)

    # Collect all layers referenced in prune_scores
    all_layers = set()
    for hook_name in prune_scores:
        m = (
            attn_in_pattern.match(hook_name)
            or mlp_in_pattern.match(hook_name)
            or resid_pattern.match(hook_name)
        )
        if m:
            all_layers.add(int(m.group(1)))

    # Create hooks for attention heads without outgoing circuit edges
    # Use hook_z (pre-W_O, always cached) instead of hook_result (needs use_attn_result=True).
    # Mean ablation at hook_z is equivalent to hook_result since W_O is linear:
    # mean(z) @ W_O = mean(z @ W_O).
    for layer in sorted(all_layers):
        ablated_heads = [
            h for h in range(n_heads) if (layer, h) not in head_has_outgoing
        ]

        if not ablated_heads:
            continue

        hook_name = f"blocks.{layer}.attn.hook_z"
        if hook_name not in mean_cache:
            continue

        mean_val = mean_cache[hook_name]

        def attn_hook(activation, hook, _mean=mean_val, _heads=ablated_heads):
            # activation shape: (batch, pos, n_heads, d_head)
            # _mean shape: (pos, n_heads, d_head): broadcasts over batch
            for h in _heads:
                activation[:, :, h, :] = _mean[:, h, :]
            return activation

        hooks.append((hook_name, attn_hook))

    # Create hooks for MLPs without outgoing circuit edges
    for layer in sorted(all_layers):
        if layer in mlp_has_outgoing:
            continue

        hook_name = f"blocks.{layer}.hook_mlp_out"
        if hook_name not in mean_cache:
            continue

        mean_val = mean_cache[hook_name]

        def mlp_hook(activation, hook, _mean=mean_val):
            return _mean.unsqueeze(0).expand_as(activation)

        hooks.append((hook_name, mlp_hook))

    return hooks


# =============================================================================
# CIRCUIT-MODE EXTRACTION
# =============================================================================


def extract_circuit_activations(
    model,
    dataset: Dict,
    prune_scores: Dict,
    mean_cache: Dict,
    batch_size: int = EXTRACTION_BATCH_SIZE,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Extract activations with mean ablation applied.

    Runs forward pass with ablation hooks active: non-circuit edges
    have their outputs replaced with dataset-mean activations. Extracts
    the same activation types as base extraction (at prediction position).

    Args:
        model: Loaded HookedTransformer (fold_ln=True).
        dataset: Dict with 'input_ids' (N, 21) and 'target_ids' (N,).
        prune_scores: Dict from load_prune_scores().
        mean_cache: Dict from build_mean_cache().
        batch_size: Batch size for forward passes.
        verbose: Print progress.

    Returns:
        Dict of numpy arrays ready to be saved as NPZ.
    """
    import torch

    device = next(model.parameters()).device
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_mlp = model.cfg.d_mlp

    input_ids = dataset["input_ids"]
    target_ids = dataset["target_ids"]
    n_examples = len(input_ids)
    pred_pos = MODEL_PREDICTION_POS

    # Create ablation hooks
    ablation_hooks = create_ablation_hooks(prune_scores, mean_cache, n_heads)
    if verbose:
        print(f"  Created {len(ablation_hooks)} ablation hooks")

    # Get circuit mask for metadata
    circuit_mask = get_circuit_mask(prune_scores, n_layers, n_heads)

    # Pre-allocate output arrays
    resid_post_predpos = np.zeros((n_examples, n_layers, d_model), dtype=np.float32)
    attn_pattern_predpos = np.zeros(
        (n_examples, n_layers, n_heads, SEQ_LEN_WITH_BOS), dtype=np.float32
    )
    attn_out_predpos = np.zeros((n_examples, n_layers, d_model), dtype=np.float32)
    mlp_out_predpos = np.zeros((n_examples, n_layers, d_model), dtype=np.float32)
    mlp_pre_predpos = np.zeros((n_examples, n_layers, d_mlp), dtype=np.float32)

    # Prepend BOS
    bos_id = model.tokenizer.bos_token_id
    input_ids_with_bos = np.concatenate(
        [np.full((n_examples, 1), bos_id, dtype=input_ids.dtype), input_ids],
        axis=1,
    )

    # Build hook names to extract (in addition to ablation hooks)
    extract_names = []
    for layer in range(n_layers):
        extract_names.extend(
            [
                f"blocks.{layer}.hook_resid_post",
                f"blocks.{layer}.attn.hook_pattern",
                f"blocks.{layer}.hook_attn_out",
                f"blocks.{layer}.hook_mlp_out",
                f"blocks.{layer}.mlp.hook_pre",
            ]
        )

    # Register ablation hooks on the model (persist across batches)
    model.reset_hooks()
    for hook_name, hook_fn in ablation_hooks:
        model.add_hook(hook_name, hook_fn)

    n_batches = (n_examples + batch_size - 1) // batch_size

    try:
        with torch.no_grad():
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_examples)
                batch_input = torch.tensor(
                    input_ids_with_bos[start:end],
                    dtype=torch.long,
                    device=device,
                )

                # Run with cache: ablation hooks are already registered on model,
                # run_with_cache adds its own cache hooks on top.
                # reset_hooks_end=False keeps ablation hooks for next batch.
                _, cache = model.run_with_cache(
                    batch_input,
                    names_filter=extract_names,
                    reset_hooks_end=False,
                    prepend_bos=False,
                )

                for layer in range(n_layers):
                    resid = cache[f"blocks.{layer}.hook_resid_post"]
                    resid_post_predpos[start:end, layer] = (
                        resid[:, pred_pos, :].cpu().numpy()
                    )

                    attn_pat = cache[f"blocks.{layer}.attn.hook_pattern"]
                    attn_pattern_predpos[start:end, layer] = (
                        attn_pat[:, :, pred_pos, :].cpu().numpy()
                    )

                    attn_out_predpos[start:end, layer] = (
                        cache[f"blocks.{layer}.hook_attn_out"][:, pred_pos, :]
                        .cpu()
                        .numpy()
                    )
                    mlp_out_predpos[start:end, layer] = (
                        cache[f"blocks.{layer}.hook_mlp_out"][:, pred_pos, :]
                        .cpu()
                        .numpy()
                    )
                    mlp_pre_predpos[start:end, layer] = (
                        cache[f"blocks.{layer}.mlp.hook_pre"][:, pred_pos, :]
                        .cpu()
                        .numpy()
                    )

                del cache
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                if verbose and (batch_idx + 1) % 2 == 0:
                    print(f"  Batch {batch_idx + 1}/{n_batches} done")
    finally:
        # Always clean up hooks
        model.reset_hooks()

    # Compute logit lens under ablation
    from .logit_lens import compute_logit_lens_torch

    resid_tensor = torch.tensor(resid_post_predpos, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
    logit_lens = compute_logit_lens_torch(model, resid_tensor, target_tensor)

    del resid_tensor, target_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Get circuit summary
    summary = get_circuit_summary(prune_scores)

    results = {
        "resid_post_predpos": resid_post_predpos,
        "attn_pattern_predpos": attn_pattern_predpos,
        "attn_out_predpos": attn_out_predpos,
        "mlp_out_predpos": mlp_out_predpos,
        "mlp_pre_predpos": mlp_pre_predpos,
        "logit_lens_prob_correct": logit_lens["prob_correct"].astype(np.float32),
        "logit_lens_rank_correct": logit_lens["rank_correct"].astype(np.int32),
        "logit_lens_kl_from_final": logit_lens["kl_from_final"].astype(np.float32),
        "target_ids": target_ids.astype(np.int32),
        "input_ids": input_ids.astype(np.int32),
        "circuit_edge_count": np.array(summary["total_edges"], dtype=np.int32),
        "circuit_head_mask": circuit_mask["head_mask"],
        "circuit_mlp_mask": circuit_mask["mlp_mask"],
    }

    return results


# =============================================================================
# FULL CIRCUIT EXTRACTION PIPELINE
# =============================================================================


def run_circuit_extraction_pipeline(
    models: List[str] = None,
    bands: List[str] = None,
    draws: List[str] = None,
    batch_size: int = EXTRACTION_BATCH_SIZE,
    skip_existing: bool = True,
    verbose: bool = True,
) -> List[Dict]:
    """Run the full circuit extraction pipeline.

    For each model:
    1. Load model with fold_ln=True
    2. Build mean cache (one forward pass over representative dataset)
    3. For each band x draw: load prune_scores, apply ablation, extract
    4. Free model, load next

    Args:
        models: List of model names. Defaults to MODELS.
        bands: List of band names. Defaults to BANDS.
        draws: List of draw names. Defaults to DRAWS.
        batch_size: Batch size for forward passes.
        skip_existing: Skip configurations with existing NPZ files.
        verbose: Print progress.

    Returns:
        List of summary dicts.
    """
    import torch
    import gc

    models = models or MODELS
    bands = bands or BANDS
    draws = draws or DRAWS

    CIRCUIT_ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device(verbose=verbose)
    summary_rows = []
    total_configs = len(models) * len(bands) * len(draws)
    config_idx = 0

    for model_name in models:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name} (circuit mode, fold_ln={CIRCUIT_FOLD_LN})")
        print(f"{'=' * 70}")

        model = load_model_circuit_mode(model_name, device=device, verbose=verbose)

        # Build mean cache over ALL bands and draws so the ablation mean is not
        # biased toward any single band's token distribution.
        if verbose:
            print(f"\n  Building mean activation cache (all bands x draws)...")
        all_input_ids = []
        all_target_ids = []
        for draw in draws:
            for band in bands:
                ds = load_dataset_for_extraction(draw, band)
                all_input_ids.append(ds["input_ids"])
                all_target_ids.append(ds["target_ids"])
        cache_dataset = {
            "input_ids": np.concatenate(all_input_ids, axis=0),
            "target_ids": np.concatenate(all_target_ids, axis=0),
            "n_examples": sum(a.shape[0] for a in all_input_ids),
        }
        if verbose:
            print(
                f"  Combined dataset: {cache_dataset['n_examples']} examples "
                f"({len(bands)} bands x {len(draws)} draws)"
            )
        mean_cache = build_mean_cache(model, cache_dataset, batch_size=batch_size)
        if verbose:
            print(f"  Mean cache built: {len(mean_cache)} hooks cached")

        for band in bands:
            for draw in draws:
                config_idx += 1
                output_path = get_circuit_activation_path(model_name, band, draw)

                if skip_existing and output_path.exists():
                    if verbose:
                        print(
                            f"\n[{config_idx}/{total_configs}] SKIP {model_name}/{band}/{draw} (exists)"
                        )
                    summary_rows.append(
                        {
                            "model": model_name,
                            "band": band,
                            "draw": draw,
                            "status": "skipped",
                            "filename": output_path.name,
                        }
                    )
                    continue

                if verbose:
                    print(
                        f"\n[{config_idx}/{total_configs}] {model_name}/{band}/{draw}"
                    )

                t0 = time.time()

                try:
                    # Load dataset
                    dataset = load_dataset_for_extraction(draw, band)

                    # Load circuit
                    prune_scores = load_prune_scores(model_name, band, draw)
                    summary = get_circuit_summary(prune_scores)
                    if verbose:
                        print(
                            f"  Circuit: {summary['total_edges']} edges "
                            f"({summary['attn_edges']} attn, {summary['mlp_edges']} mlp)"
                        )

                    # Extract with ablation
                    results = extract_circuit_activations(
                        model,
                        dataset,
                        prune_scores,
                        mean_cache,
                        batch_size=batch_size,
                        verbose=verbose,
                    )

                    # Save
                    save_extraction(results, output_path)
                    elapsed = time.time() - t0
                    file_size_mb = output_path.stat().st_size / (1024 * 1024)

                    row = {
                        "model": model_name,
                        "band": band,
                        "draw": draw,
                        "status": "success",
                        "n_examples": dataset["n_examples"],
                        "circuit_edges": summary["total_edges"],
                        "file_size_mb": round(file_size_mb, 1),
                        "extraction_time_s": round(elapsed, 1),
                        "filename": output_path.name,
                        "timestamp": datetime.now().isoformat(),
                    }
                    summary_rows.append(row)

                    if verbose:
                        print(
                            f"  OK: {dataset['n_examples']} examples, "
                            f"{file_size_mb:.1f} MB, {elapsed:.1f}s"
                        )

                except Exception as e:
                    elapsed = time.time() - t0
                    import traceback

                    if verbose:
                        print(f"  FAILED: {str(e)}")
                        traceback.print_exc()

                    summary_rows.append(
                        {
                            "model": model_name,
                            "band": band,
                            "draw": draw,
                            "status": "error",
                            "error": str(e),
                            "filename": output_path.name,
                            "extraction_time_s": round(elapsed, 1),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

        # Free model and mean cache
        del model, mean_cache
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()
        if verbose:
            print(f"\nFreed {model_name} from GPU memory")

    # Save summary
    save_extraction_summary(summary_rows, output_dir=EXTRACTION_DIR)
    if verbose:
        n_success = sum(1 for r in summary_rows if r.get("status") == "success")
        n_skip = sum(1 for r in summary_rows if r.get("status") == "skipped")
        n_fail = sum(1 for r in summary_rows if r.get("status") == "error")
        print(f"\n{'=' * 70}")
        print(f"CIRCUIT EXTRACTION COMPLETE")
        print(f"  Success: {n_success}, Skipped: {n_skip}, Failed: {n_fail}")
        print(f"{'=' * 70}")

    return summary_rows
