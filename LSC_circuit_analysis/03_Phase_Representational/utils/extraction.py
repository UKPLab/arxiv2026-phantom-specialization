"""
Activation extraction logic for Phase Representational analysis.

Extracts token embeddings, residual stream activations, attention patterns,
MLP outputs, and logit lens metrics from Pythia models using TransformerLens.

design decisions:
- prepend_bos=True: BOS is prepended, so all positions shift +1
- Prediction position = 21 in model coords (= position 20 in raw + BOS offset)
- One model at a time to manage GPU memory
- Batched processing with configurable batch size
- Saves NPZ files per model x band x draw configuration
"""

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .constants import (
    MODELS,
    BANDS,
    DRAWS,
    MODEL_DIR_NAMES,
    MODEL_INFO,
    HF_MODEL_NAMES,
    MODEL_D_MODEL,
    MODEL_D_MLP,
    DATASETS_BASE,
    ACTIVATIONS_DIR,
    EXTRACTION_DIR,
    SEQ_LEN,
    SEQ_LEN_WITH_BOS,
    BOS_OFFSET,
    MODEL_PREDICTION_POS,
    TARGET_POS,
    EXTRACTION_BATCH_SIZE,
)


# =============================================================================
# DEVICE SETUP
# =============================================================================


def get_device(verbose: bool = True) -> str:
    """Get the best available GPU device."""
    import torch

    if not torch.cuda.is_available():
        if verbose:
            print("CUDA not available, using CPU")
        return "cpu"

    import subprocess

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError("nvidia-smi failed")

        gpu_info = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpu_info.append(
                    {
                        "index": int(parts[0]),
                        "utilization": int(parts[1]),
                        "memory_used": int(parts[2]),
                        "memory_total": int(parts[3]),
                    }
                )

        if not gpu_info:
            raise RuntimeError("No GPU info parsed")

        idle_gpus = [g for g in gpu_info if g["utilization"] == 0]
        if idle_gpus:
            best = min(idle_gpus, key=lambda g: g["memory_used"])
        else:
            best = min(gpu_info, key=lambda g: g["memory_used"])

        device = f"cuda:{best['index']}"
        if verbose:
            print(
                f"Using GPU {best['index']} "
                f"(memory: {best['memory_used']}/{best['memory_total']} MiB)"
            )
        return device

    except Exception as e:
        if verbose:
            print(f"Could not query GPU status ({e}), using cuda:0")
        return "cuda:0"


def _patch_gptneox_config():
    """Patch GPTNeoXConfig to restore rotary_pct for TransformerLens compatibility.

    transformers >= 5.x moved rotary_pct into rope_parameters.partial_rotary_factor,
    but TransformerLens 2.x still expects hf_config.rotary_pct as a direct attribute.
    """
    try:
        from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig

        if hasattr(GPTNeoXConfig, "rotary_pct"):
            return  # Already has it (older transformers)

        @property
        def _rotary_pct(self):
            rope = getattr(self, "rope_parameters", None)
            if rope and "partial_rotary_factor" in rope:
                return rope["partial_rotary_factor"]
            return 0.25  # Pythia default

        GPTNeoXConfig.rotary_pct = _rotary_pct
    except ImportError:
        pass


def load_model(model_name: str, device: str = None, verbose: bool = True):
    """Load a HookedTransformer model.

    Args:
        model_name: Short model name (e.g. 'pythia-70m').
        device: Device string. Auto-detected if None.
        verbose: Print loading progress.

    Returns:
        Loaded HookedTransformer model.
    """
    from transformer_lens import HookedTransformer

    # Patch for transformers >= 5.x compatibility
    _patch_gptneox_config()

    if device is None:
        device = get_device(verbose=verbose)

    hf_name = HF_MODEL_NAMES.get(model_name, f"EleutherAI/{model_name}")

    if verbose:
        print(f"\nLoading {hf_name}...")

    model = HookedTransformer.from_pretrained(
        hf_name,
        device=device,
        fold_ln=True,  # Match circuit discovery and circuit extraction settings
        center_writing_weights=False,
        center_unembed=False,
    )

    if verbose:
        print(
            f"  Layers: {model.cfg.n_layers}, Heads: {model.cfg.n_heads}, "
            f"d_model: {model.cfg.d_model}, Vocab: {model.cfg.d_vocab}"
        )

    return model


# =============================================================================
# DATASET LOADING
# =============================================================================


def load_dataset_for_extraction(draw: str, band: str, split: str = "test") -> Dict:
    """Load an LSC dataset and prepare for extraction.

    Args:
        draw: Draw name (e.g. 'draw_1').
        band: Frequency band name.
        split: Data split.

    Returns:
        Dict with input_ids (N, 21), target_ids (N,), n_examples.
    """
    path = DATASETS_BASE / draw / band / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path) as f:
        data = json.load(f)

    examples = data["examples"]
    input_ids = np.array([ex["token_ids"] for ex in examples])  # (N, 21)
    target_ids = input_ids[:, TARGET_POS]  # T at position 5

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "n_examples": len(examples),
    }


# =============================================================================
# CORE EXTRACTION
# =============================================================================


def extract_single_configuration(
    model,
    dataset: Dict,
    batch_size: int = EXTRACTION_BATCH_SIZE,
    extract_full_sequence: bool = False,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Extract all activations for one model x band x draw configuration.

    Args:
        model: Loaded HookedTransformer model.
        dataset: Dict from load_dataset_for_extraction().
        batch_size: Batch size for forward passes.
        extract_full_sequence: If True, extract resid_post at all positions
            (can be very large for pythia-1b). If False, only prediction pos.
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

    input_ids = dataset["input_ids"]  # (N, 21) without BOS
    target_ids = dataset["target_ids"]  # (N,)
    n_examples = dataset["n_examples"]

    pred_pos = MODEL_PREDICTION_POS  # 21 (after BOS)

    # Pre-allocate output arrays
    token_embeddings = np.zeros((n_examples, SEQ_LEN, d_model), dtype=np.float32)
    resid_post_predpos = np.zeros((n_examples, n_layers, d_model), dtype=np.float32)
    attn_pattern_predpos = np.zeros(
        (n_examples, n_layers, n_heads, SEQ_LEN_WITH_BOS), dtype=np.float32
    )
    attn_out_predpos = np.zeros((n_examples, n_layers, d_model), dtype=np.float32)
    mlp_out_predpos = np.zeros((n_examples, n_layers, d_model), dtype=np.float32)
    mlp_pre_predpos = np.zeros((n_examples, n_layers, d_mlp), dtype=np.float32)

    if extract_full_sequence:
        resid_post_all = np.zeros(
            (n_examples, n_layers, SEQ_LEN_WITH_BOS, d_model), dtype=np.float32
        )

    # Manually prepend BOS token.
    # Note: prepend_bos=True in run_with_cache does NOT work for tensor inputs,
    # so we prepend BOS ourselves and use prepend_bos=False.
    bos_id = model.tokenizer.bos_token_id
    input_ids_with_bos = np.concatenate(
        [np.full((n_examples, 1), bos_id, dtype=input_ids.dtype), input_ids],
        axis=1,
    )  # (N, 22)

    # Build hook names to extract
    hook_names = ["hook_embed"]
    for layer in range(n_layers):
        hook_names.extend(
            [
                f"blocks.{layer}.hook_resid_post",
                f"blocks.{layer}.attn.hook_pattern",
                f"blocks.{layer}.hook_attn_out",  # Total attention output (d_model)
                f"blocks.{layer}.hook_mlp_out",  # Total MLP output (d_model)
                f"blocks.{layer}.mlp.hook_pre",  # MLP pre-activation (d_mlp)
            ]
        )

    n_batches = (n_examples + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, n_examples)
            batch_input = torch.tensor(
                input_ids_with_bos[start:end],
                dtype=torch.long,
                device=device,
            )

            # Run with cache; BOS already prepended manually
            _, cache = model.run_with_cache(
                batch_input,
                names_filter=hook_names,
                prepend_bos=False,
            )

            # Token embeddings (without BOS, positions 1..21 in model space)
            embed = (
                cache["hook_embed"][:, BOS_OFFSET:, :].cpu().numpy()
            )  # (batch, 21, d_model)
            token_embeddings[start:end] = embed

            for layer in range(n_layers):
                # Residual stream at prediction position
                resid = cache[f"blocks.{layer}.hook_resid_post"]
                resid_post_predpos[start:end, layer] = (
                    resid[:, pred_pos, :].cpu().numpy()
                )

                if extract_full_sequence:
                    resid_post_all[start:end, layer] = resid.cpu().numpy()

                # Attention patterns FROM prediction position TO all positions
                attn_pat = cache[
                    f"blocks.{layer}.attn.hook_pattern"
                ]  # (batch, n_heads, seq, seq)
                attn_pattern_predpos[start:end, layer] = (
                    attn_pat[:, :, pred_pos, :].cpu().numpy()
                )

                # Attention output at prediction position
                # hook_attn_out shape: (batch, seq, d_model): total attention contribution
                attn_out_predpos[start:end, layer] = (
                    cache[f"blocks.{layer}.hook_attn_out"][:, pred_pos, :].cpu().numpy()
                )

                # MLP output at prediction position
                # hook_mlp_out shape: (batch, seq, d_model): total MLP contribution
                mlp_out_predpos[start:end, layer] = (
                    cache[f"blocks.{layer}.hook_mlp_out"][:, pred_pos, :].cpu().numpy()
                )

                # MLP pre-activation at prediction position
                mlp_pre_predpos[start:end, layer] = (
                    cache[f"blocks.{layer}.mlp.hook_pre"][:, pred_pos, :].cpu().numpy()
                )

            # Free cache memory
            del cache
            if device != "cpu":
                torch.cuda.empty_cache()

            if verbose and (batch_idx + 1) % 2 == 0:
                print(f"  Batch {batch_idx + 1}/{n_batches} done")

    # Compute logit lens metrics using the model's ln_final and W_U
    from .logit_lens import compute_logit_lens_torch

    resid_tensor = torch.tensor(resid_post_predpos, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)

    logit_lens = compute_logit_lens_torch(model, resid_tensor, target_tensor)

    del resid_tensor, target_tensor
    if device != "cpu":
        torch.cuda.empty_cache()

    # Assemble results
    results = {
        "token_embeddings": token_embeddings,
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
    }

    if extract_full_sequence:
        results["resid_post_all"] = resid_post_all

    return results


# =============================================================================
# COPY SCORE EXTRACTION
# =============================================================================


def extract_copy_scores(
    model,
    n_pairs: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """Extract OV copy scores for all heads using random token pairs.

    copy_score(layer, head) = mean over pairs of:
        W_U[target]^T @ W_OV @ W_E[source]

    Args:
        model: Loaded HookedTransformer model.
        n_pairs: Number of random token pairs.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Array of shape (n_layers, n_heads) with mean copy scores.
    """
    from .attention import compute_copy_scores

    if verbose:
        print(f"  Computing copy scores ({n_pairs} pairs)...")
    return compute_copy_scores(model, n_pairs=n_pairs, seed=seed)


def get_activation_filename(model: str, band: str, draw: str) -> str:
    """Get NPZ filename for a configuration."""
    model_dir = MODEL_DIR_NAMES.get(model, model.replace("-", "_"))
    return f"{model_dir}_{band}_{draw}.npz"


def save_extraction(results: Dict[str, np.ndarray], output_path: Path):
    """Save extraction results to NPZ file.

    Args:
        results: Dict of numpy arrays.
        output_path: Path to save NPZ file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **results)


def save_extraction_summary(summary_rows: List[Dict], output_dir: Path = None):
    """Save extraction summary CSV.

    Writes both a timestamped version (never overwritten) and a latest version.
    Previous runs' summaries are preserved as extraction_summary_<timestamp>.csv.

    Args:
        summary_rows: List of dicts with model, band, draw, shapes, timing.
        output_dir: Output directory. Defaults to EXTRACTION_DIR.
    """
    import csv

    if output_dir is None:
        output_dir = EXTRACTION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if not summary_rows:
        return

    # Collect all unique keys across all rows (some rows may have different fields)
    all_keys = []
    seen = set()
    for row in summary_rows:
        for k in row.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    def _write_csv(p):
        with open(p, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary_rows)

    # Write timestamped copy (preserves history)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_path = output_dir / f"extraction_summary_{ts}.csv"
    _write_csv(ts_path)

    # Write latest (for easy access)
    latest_path = output_dir / "extraction_summary.csv"
    _write_csv(latest_path)


# =============================================================================
# FULL EXTRACTION PIPELINE
# =============================================================================


def run_extraction_pipeline(
    models: List[str] = None,
    bands: List[str] = None,
    draws: List[str] = None,
    batch_size: int = EXTRACTION_BATCH_SIZE,
    extract_full_sequence: bool = False,
    skip_existing: bool = True,
    verbose: bool = True,
) -> List[Dict]:
    """Run the full extraction pipeline.

    Loads one model at a time, processes all band x draw configurations,
    then frees the model before loading the next.

    Args:
        models: List of model names. Defaults to MODELS.
        bands: List of band names. Defaults to BANDS.
        draws: List of draw names. Defaults to DRAWS.
        batch_size: Batch size for forward passes.
        extract_full_sequence: Extract full-sequence residual stream.
        skip_existing: Skip configurations with existing NPZ files.
        verbose: Print progress.

    Returns:
        List of summary dicts for extraction_summary.csv.
    """
    import torch
    import gc

    models = models or MODELS
    bands = bands or BANDS
    draws = draws or DRAWS

    ACTIVATIONS_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device(verbose=verbose)
    summary_rows = []
    total_configs = len(models) * len(bands) * len(draws)
    config_idx = 0

    for model_name in models:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name}")
        print(f"{'=' * 70}")

        model = load_model(model_name, device=device, verbose=verbose)

        # Extract copy scores for this model
        copy_scores = extract_copy_scores(model, verbose=verbose)
        model_dir = MODEL_DIR_NAMES.get(model_name, model_name.replace("-", "_"))
        copy_path = ACTIVATIONS_DIR / f"copy_scores_{model_dir}.npz"
        np.savez_compressed(copy_path, copy_scores=copy_scores)
        if verbose:
            print(f"  Saved copy scores: {copy_path}")

        for band in bands:
            for draw in draws:
                config_idx += 1
                filename = get_activation_filename(model_name, band, draw)
                output_path = ACTIVATIONS_DIR / filename

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
                            "filename": filename,
                        }
                    )
                    continue

                if verbose:
                    print(
                        f"\n[{config_idx}/{total_configs}] {model_name}/{band}/{draw}"
                    )

                t0 = time.time()

                try:
                    dataset = load_dataset_for_extraction(draw, band)
                    results = extract_single_configuration(
                        model,
                        dataset,
                        batch_size=batch_size,
                        extract_full_sequence=extract_full_sequence,
                        verbose=verbose,
                    )
                    save_extraction(results, output_path)
                    elapsed = time.time() - t0

                    # Compute file size
                    file_size_mb = output_path.stat().st_size / (1024 * 1024)

                    row = {
                        "model": model_name,
                        "band": band,
                        "draw": draw,
                        "status": "success",
                        "n_examples": dataset["n_examples"],
                        "n_layers": model.cfg.n_layers,
                        "n_heads": model.cfg.n_heads,
                        "d_model": model.cfg.d_model,
                        "file_size_mb": round(file_size_mb, 1),
                        "extraction_time_s": round(elapsed, 1),
                        "filename": filename,
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

                    error_msg = f"{str(e)}\n{traceback.format_exc()}"
                    if verbose:
                        print(f"  FAILED: {str(e)}")

                    summary_rows.append(
                        {
                            "model": model_name,
                            "band": band,
                            "draw": draw,
                            "status": "error",
                            "error": str(e),
                            "filename": filename,
                            "extraction_time_s": round(elapsed, 1),
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

        # Free model memory
        del model
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()
        if verbose:
            print(f"\nFreed {model_name} from GPU memory")

    # Save summary
    save_extraction_summary(summary_rows)
    if verbose:
        n_success = sum(1 for r in summary_rows if r.get("status") == "success")
        n_skip = sum(1 for r in summary_rows if r.get("status") == "skipped")
        n_fail = sum(1 for r in summary_rows if r.get("status") == "error")
        print(f"\n{'=' * 70}")
        print(f"EXTRACTION COMPLETE")
        print(f"  Success: {n_success}, Skipped: {n_skip}, Failed: {n_fail}")
        print(f"  Summary: {EXTRACTION_DIR / 'extraction_summary.csv'}")
        print(f"{'=' * 70}")

    return summary_rows
