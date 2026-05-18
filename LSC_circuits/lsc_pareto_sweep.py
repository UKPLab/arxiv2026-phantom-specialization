#!/usr/bin/env python3
"""
LSC Pareto Sweep: Threshold Selection for Circuit Discovery
============================================================

Find the optimal ACDC threshold τ* per model via Pareto analysis on the
CONTROL band.  Each threshold requires a SEPARATE ACDC run (post-hoc
thresholding is invalid).


WORKFLOW
--------
For each model (ascending size: 70m -> 12b):
    1. Load model ONCE
    2. Compute base logits on VAL ONCE
    3. For each threshold τ_i:
        a. Run ACDC on TRAIN split -> prune_scores
        b. Evaluate circuit on VAL -> KL, accuracy (reuse base_logits)
        c. Evaluate ablation on VAL -> accuracy
    4. Free model
    Pareto frontier reported (human selects τ*)

Models are processed smallest-first so pipeline errors surface quickly.

DATA SPLITS
-------------------------------
    train (1050 examples, 256 sampled) -> ACDC edge pruning (seed-controlled)
    val   (225 examples, full split)   -> Pareto sweep evaluation (reliable metrics)
    test  (225 examples)               -> HELD OUT for Phase 2 (never touched here)

LSC SEQUENCE FORMAT
-------------------
Token ID lists (21 tokens, no BOS).  With BOS prepended = 22 tokens.

    [BOS] [S1 S2 S3 S4 S5] [T] [R1..R10] [S1 S2 S3 S4 S5]
     0     1  2  3  4  5    6   7..16      17 18 19 20 21

Prediction: at position 21 (last S5), model predicts T.
Corrupt prompts: positions 17-21 replaced with different tokens.
diverge_idx = 17.

THRESHOLD DISTRIBUTION
---------------------
UNIFORM log-spaced thresholds for all models (scientifically correct):
  - Same τ applied to all models -> unbiased Pareto discovery
  - Log-uniform spacing from 10⁻² to 10⁻⁶ (equal density per decade)
  - Different models will have different valid regions (data reveals this)

Thresholds processed in DESCENDING order (large τ first):
  - Smaller circuits converge faster in ACDC
  - Early failure detection (empty circuit = τ too large)

Usage:
    python lsc_pareto_sweep.py                          # Model-specific thresholds
    python lsc_pareto_sweep.py --models pythia-70m      # Single model
    python lsc_pareto_sweep.py --thresholds 0.001 0.01  # Override with uniform τ
    python lsc_pareto_sweep.py --gpus 0 1 2 3
    python lsc_pareto_sweep.py --eval-val-size 0        # Use full VAL split
    python lsc_pareto_sweep.py --bf16                   # Enable BF16 (default: FP32)
    python lsc_pareto_sweep.py --analyze-only
    python lsc_pareto_sweep.py --force
"""

import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import pickle
import random
import argparse
import traceback
import fcntl
import math
import gc
import time
import threading
import logging
import multiprocessing as mp
from multiprocessing import Queue, Process, Manager
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import OrderedDict, defaultdict

import numpy as np
import torch as t
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
ISC_ROOT = SCRIPT_DIR.parent  # repo root

AUTOCIRCUIT_PATH = os.environ.get("AUTOCIRCUIT_PATH") or str(
    ISC_ROOT / "circuit_discovery" / "auto-circuit"
)
sys.path.insert(0, AUTOCIRCUIT_PATH)


# Models in ASCENDING size order (smallest first -> fast error detection)
DEFAULT_MODELS = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
]

# UNIFORM THRESHOLDS for unbiased Pareto analysis
#
# Scientific approach: Use the same thresholds for all models.
# Let the data reveal each model's valid region, don't assume it.
#
# LOG-UNIFORM spacing: τ spans 4 decades (10^-2 to 10^-6).
# Equal density per decade ensures unbiased sampling across scales.
#
# DESCENDING order (large τ first) for faster ACDC convergence.
#
# BASE MODEL PERFORMANCE on control band (for reference):
#   pythia-70m: 58.7% (borderline) | pythia-160m+: 95-100% (solves task)
#
# 11 log-spaced thresholds from 0.01 to 0.000001:
DEFAULT_THRESHOLDS = [
    1e-2,  # 0.01
    3.98e-3,  # ~0.004
    1.58e-3,  # ~0.0016
    6.31e-4,  # ~0.00063
    2.51e-4,  # ~0.00025
    1e-4,  # 0.0001
    3.98e-5,  # ~0.00004
    1.58e-5,  # ~0.000016
    6.31e-6,  # ~0.0000063
    2.51e-6,  # ~0.0000025
    1e-6,  # 0.000001
]

# Optimized batch sizes per model for A100 80GB (increased from conservative defaults)
MODEL_BATCH_SIZES = {
    "pythia-70m": 256,
    "pythia-160m": 256,
    "pythia-410m": 128,
    "pythia-1b": 96,
    "pythia-1.4b": 64,
}

# Model size ordering (for sorting smallest-first)
MODEL_SIZE_ORDER = {
    "pythia-70m": 0,
    "pythia-160m": 1,
    "pythia-410m": 2,
    "pythia-1b": 3,
    "pythia-1.4b": 4,
}

# Estimated ACDC runtime in minutes (for scheduling)
ESTIMATED_MINUTES = {
    "pythia-70m": 5,
    "pythia-160m": 30,
    "pythia-410m": 150,
    "pythia-1b": 45,
    "pythia-1.4b": 150,
}

# LSC sequence constants
N_SOURCE = 5
N_DISTRACT = 10
RAW_SEQ_LEN = N_SOURCE + 1 + N_DISTRACT + N_SOURCE  # 21
SEQ_LEN_WITH_BOS = RAW_SEQ_LEN + 1  # 22
DIVERGE_IDX = N_SOURCE + 1 + N_DISTRACT + 1  # 17 (with BOS)


@dataclass
class SweepConfig:
    # Paths (relative to ISC_ROOT by default)
    data_dir: str = field(default_factory=lambda: str(ISC_ROOT / "LSC_data"))
    pool_dir: str = field(
        default_factory=lambda: str(
            ISC_ROOT / "LSC_data" / "lsc_token_pools" / "matched"
        )
    )
    output_dir: str = field(default_factory=lambda: str(SCRIPT_DIR / "pareto_sweep"))
    # Dataset structure: datasets/{variant}/{draw}/{band}/{split}.json
    variant: str = "matched"
    draw: str = "draw_1"
    # Experiment settings
    models: List[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    sweep_band: str = "control"
    thresholds: List[float] = field(default_factory=lambda: list(DEFAULT_THRESHOLDS))
    # ACDC uses sampled TRAIN split
    acdc_train_size: int = 256  # Sampled from train split for ACDC
    # Evaluation uses FULL VAL split for reliable metrics
    eval_val_size: int = 0  # 0 = use full val split (225 samples)
    # Use BF16 for evaluation forward passes (disabled by default - BF16 causes accuracy issues with pythia models)
    use_bf16_eval: bool = False
    factorized: bool = True
    separate_qkv: bool = False
    slice_output: str = "last_seq"
    acdc_seed: int = 42
    eval_seed: int = 123
    gpus: List[int] = field(default_factory=list)
    registry_file: str = "registry.json"


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False


def cleanup_gpu():
    gc.collect()
    if t.cuda.is_available():
        t.cuda.synchronize()
        t.cuda.empty_cache()


def safe_delete_model(model):
    if model is not None:
        try:
            model.cpu()
        except Exception:
            pass
        del model
    cleanup_gpu()


def setup_logging(output_dir: Path, debug: bool = False) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"sweep_{timestamp}.log"

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(process)d] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("pareto_sweep")
    logger.info(f"Log file: {log_file}")
    return logger


def sort_models_by_size(models: List[str]) -> List[str]:
    """Sort models by size (smallest first) for early error detection."""
    return sorted(models, key=lambda m: MODEL_SIZE_ORDER.get(m, 999))


def threshold_to_tao(threshold: float) -> Tuple[int, float]:
    """threshold = tao_base x 10^tao_exp"""
    exponent = math.floor(math.log10(threshold))
    base = round(threshold / (10**exponent), 6)
    return exponent, base


def tao_to_str(threshold: float) -> str:
    exp = math.floor(math.log10(threshold))
    base = threshold / (10**exp)
    return f"{base:.2f}em{abs(exp):02d}".replace(".", "_")


def load_pool(band: str, pool_dir: Path) -> dict:
    """Load token pool JSON for a frequency band."""
    # Pool files are named lsc_pool_{band}.json
    with open(pool_dir / f"lsc_pool_{band}.json") as f:
        return json.load(f)


def load_dataset(
    band: str, split: str, data_dir: Path, variant: str, draw: str
) -> dict:
    """Load LSC dataset JSON for a band and split."""
    # Path: data_dir/datasets/{variant}/{draw}/{band}/{split}.json
    with open(data_dir / "datasets" / variant / draw / band / f"{split}.json") as f:
        return json.load(f)


def prepare_dataloader(
    dataset: dict,
    pool: dict,
    bos_token_id: int,
    n_samples: int,
    batch_size: int,
    seed: int,
    device: str,
) -> Tuple[Any, List[int]]:
    """
    Build AutoCircuit PromptDataLoader from token-ID-based LSC data.
    Samples n_samples examples with seed-controlled randomness.

    Returns:
        dataloader: PromptDataLoader for AutoCircuit
        sampled_indices: List of original example indices used (for reproducibility)
    """
    from auto_circuit.data import PromptDataset, PromptDataLoader

    examples = dataset["examples"]
    rng = random.Random(seed)

    # Create shuffled indices for reproducibility tracking
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    if len(indices) > n_samples:
        indices = indices[:n_samples]

    pool_ids = [tok["token_id"] for tok in pool["tokens"]]

    clean_prompts, corrupt_prompts, answers, wrong_answers = [], [], [], []

    for idx in indices:
        ex = examples[idx]
        token_ids = ex["token_ids"]
        clean = [bos_token_id] + token_ids

        used_set = set(token_ids)
        available = [tid for tid in pool_ids if tid not in used_set]
        if len(available) >= N_SOURCE:
            replacements = rng.sample(available, N_SOURCE)
        else:
            replacements = rng.sample(pool_ids, N_SOURCE)

        corrupt = [bos_token_id] + token_ids[:16] + replacements
        assert len(corrupt) == SEQ_LEN_WITH_BOS

        wrong_pool = [tid for tid in pool_ids if tid != ex["target_token_id"]]

        clean_prompts.append(t.tensor(clean, dtype=t.long, device=device))
        corrupt_prompts.append(t.tensor(corrupt, dtype=t.long, device=device))
        answers.append(t.tensor([ex["target_token_id"]], dtype=t.long, device=device))
        wrong_answers.append(
            t.tensor([rng.choice(wrong_pool)], dtype=t.long, device=device)
        )

    ds = PromptDataset(
        clean_prompts=clean_prompts,
        corrupt_prompts=corrupt_prompts,
        answers=answers,
        wrong_answers=wrong_answers,
    )
    dataloader = PromptDataLoader(
        prompt_dataset=ds,
        seq_len=SEQ_LEN_WITH_BOS,
        diverge_idx=DIVERGE_IDX,
        batch_size=batch_size,
    )
    return dataloader, indices


def prepare_eval_dataloader(
    dataset: dict,
    pool: dict,
    bos_token_id: int,
    batch_size: int,
    seed: int,
    device: str,
    n_samples: int = 0,
) -> Tuple[Any, List[int], List[int]]:
    """
    Build AutoCircuit PromptDataLoader for evaluation.

    Args:
        n_samples: Number of samples to use. 0 = use all examples.

    Returns:
        dataloader: PromptDataLoader for AutoCircuit
        example_indices: List of indices used (for reproducibility)
        answer_ids: List of target token IDs (for metric computation)
    """
    from auto_circuit.data import PromptDataset, PromptDataLoader

    examples = dataset["examples"]
    rng = random.Random(seed)

    # Create shuffled indices
    indices = list(range(len(examples)))
    rng.shuffle(indices)

    # Sample if requested
    if n_samples > 0 and n_samples < len(indices):
        indices = indices[:n_samples]

    pool_ids = [tok["token_id"] for tok in pool["tokens"]]

    clean_prompts, corrupt_prompts, answers, wrong_answers = [], [], [], []
    answer_ids = []

    for idx in indices:
        ex = examples[idx]
        token_ids = ex["token_ids"]
        clean = [bos_token_id] + token_ids

        used_set = set(token_ids)
        available = [tid for tid in pool_ids if tid not in used_set]
        if len(available) >= N_SOURCE:
            replacements = rng.sample(available, N_SOURCE)
        else:
            replacements = rng.sample(pool_ids, N_SOURCE)

        corrupt = [bos_token_id] + token_ids[:16] + replacements
        assert len(corrupt) == SEQ_LEN_WITH_BOS

        wrong_pool = [tid for tid in pool_ids if tid != ex["target_token_id"]]

        clean_prompts.append(t.tensor(clean, dtype=t.long, device=device))
        corrupt_prompts.append(t.tensor(corrupt, dtype=t.long, device=device))
        answers.append(t.tensor([ex["target_token_id"]], dtype=t.long, device=device))
        wrong_answers.append(
            t.tensor([rng.choice(wrong_pool)], dtype=t.long, device=device)
        )
        answer_ids.append(ex["target_token_id"])

    ds = PromptDataset(
        clean_prompts=clean_prompts,
        corrupt_prompts=corrupt_prompts,
        answers=answers,
        wrong_answers=wrong_answers,
    )
    # PromptDataLoader uses drop_last=True internally, so batch_size must evenly divide
    # n_samples to avoid silently dropping the remainder. Find the largest divisor of
    # n_samples that fits within the model's batch_size limit (for GPU memory).
    n = len(indices)
    actual_batch_size = min(batch_size, n)
    while actual_batch_size > 1 and n % actual_batch_size != 0:
        actual_batch_size -= 1
    dataloader = PromptDataLoader(
        prompt_dataset=ds,
        seq_len=SEQ_LEN_WITH_BOS,
        diverge_idx=DIVERGE_IDX,
        batch_size=actual_batch_size,
    )
    return dataloader, indices, answer_ids


def _patch_gpt_neox_config():
    """Compatibility patch for transformers >=4.48 (rotary_pct moved to rope_parameters)."""
    from transformers import GPTNeoXConfig

    if getattr(GPTNeoXConfig, "_tl_compat_patched", False):
        return

    _MAP = {
        "rotary_pct": ("partial_rotary_factor", 0.25),
        "rotary_emb_base": ("base", 10000),
    }
    original = getattr(GPTNeoXConfig, "__getattr__", None)

    def _patched(self, name):
        if name in _MAP:
            key, default = _MAP[name]
            rp = object.__getattribute__(self, "__dict__").get("rope_parameters", {})
            return rp.get(key, default)
        if original is not None:
            return original(self, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    GPTNeoXConfig.__getattr__ = _patched
    GPTNeoXConfig._tl_compat_patched = True


def load_model(model_name: str, device: str):
    import transformer_lens as tl

    _patch_gpt_neox_config()
    model = tl.HookedTransformer.from_pretrained(
        model_name,
        device=device,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
    )
    model.cfg.use_attn_result = True
    model.cfg.use_attn_in = True
    model.cfg.use_hook_mlp_in = True
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def get_batch_size(model_name: str) -> int:
    return MODEL_BATCH_SIZES.get(model_name, 32)


def compute_metrics(
    logits: t.Tensor,
    answer_ids: List[int],
    base_logits: Optional[t.Tensor] = None,
) -> Dict[str, Any]:
    if len(logits.shape) == 3:
        logits = logits[:, -1, :]
    if base_logits is not None and len(base_logits.shape) == 3:
        base_logits = base_logits[:, -1, :]

    probs = F.softmax(logits, dim=-1)
    n = min(len(answer_ids), logits.shape[0])

    top1 = top5 = top10 = 0
    correct_probs = []
    for i in range(n):
        topk = t.topk(logits[i], k=10).indices.tolist()
        if topk[0] == answer_ids[i]:
            top1 += 1
        if answer_ids[i] in topk[:5]:
            top5 += 1
        if answer_ids[i] in topk:
            top10 += 1
        correct_probs.append(probs[i, answer_ids[i]].item())

    result = {
        "accuracy": top1 / n if n else 0.0,
        "top5_accuracy": top5 / n if n else 0.0,
        "top10_accuracy": top10 / n if n else 0.0,
        "mean_correct_prob": float(np.mean(correct_probs)) if correct_probs else 0.0,
        "n_samples": n,
    }

    if base_logits is not None:
        base_probs = F.softmax(base_logits, dim=-1)
        k = min(probs.shape[0], base_probs.shape[0])
        eps = 1e-10
        kl = (
            (base_probs[:k] * (t.log(base_probs[:k] + eps) - t.log(probs[:k] + eps)))
            .sum(-1)
            .mean()
        )
        result["kl_div"] = kl.item()
    else:
        result["kl_div"] = 0.0

    return result


def run_model_sweep(
    model_name: str,
    thresholds: List[float],
    config: SweepConfig,
    device: str,
    output_dir: Path,
    registry: "Registry",
) -> List[Dict[str, Any]]:
    """
    Run all thresholds for a single model with ONE model load.
    """
    from auto_circuit.prune_algos.ACDC import acdc_prune_scores
    from auto_circuit.prune import run_circuits
    from auto_circuit.types import PatchType, AblationType
    from auto_circuit.utils.graph_utils import patchable_model

    logger = logging.getLogger("pareto_sweep")
    band = config.sweep_band
    batch_size = get_batch_size(model_name)
    model_safe = model_name.replace("/", "_").replace("-", "_")

    logger.info(
        f"[{device}] Starting model sweep: {model_name} / {band} / {len(thresholds)} thresholds"
    )

    model = None
    patchable = None
    results = []

    try:
        # =====================================================================
        # PHASE 1: Load model and data ONCE
        # =====================================================================
        set_all_seeds(config.acdc_seed)

        # Load model (FP32 for ACDC precision)
        model = load_model(model_name, device)
        bos_id = model.tokenizer.bos_token_id

        # Load data files once
        pool_dir = Path(config.pool_dir)
        data_dir = Path(config.data_dir)
        pool = load_pool(band, pool_dir)
        train_data = load_dataset(band, "train", data_dir, config.variant, config.draw)
        val_data = load_dataset(band, "val", data_dir, config.variant, config.draw)

        # Create patchable model once
        patchable = patchable_model(
            model=model,
            factorized=config.factorized,
            slice_output=config.slice_output,
            seq_len=None,
            separate_qkv=config.separate_qkv,
            device=device,
        )
        total_edges = len(patchable.edges)

        # Create ACDC training dataloader once (same for all thresholds)
        # PromptDataLoader uses drop_last=True, so batch_size must not exceed n_samples
        n_train = min(config.acdc_train_size, len(train_data["examples"]))
        train_batch_size = min(batch_size, n_train)
        train_loader, train_indices = prepare_dataloader(
            train_data,
            pool,
            bos_id,
            n_samples=config.acdc_train_size,
            batch_size=train_batch_size,
            seed=config.acdc_seed,
            device=device,
        )

        # =====================================================================
        # PHASE 2: Compute base logits ONCE (reused for all thresholds)
        # =====================================================================
        set_all_seeds(config.eval_seed)
        # Determine actual val size (0 = full split)
        n_val = (
            len(val_data["examples"])
            if config.eval_val_size == 0
            else min(config.eval_val_size, len(val_data["examples"]))
        )
        # prepare_eval_dataloader handles drop_last internally (finds largest divisor)
        val_batch_size = min(batch_size, n_val)
        val_loader, val_indices, val_answer_ids = prepare_eval_dataloader(
            val_data,
            pool,
            bos_id,
            batch_size=val_batch_size,
            seed=config.eval_seed,
            device=device,
            n_samples=config.eval_val_size,  # 0 = full val split
        )

        logger.info(
            f"[{device}] Computing base logits on {len(val_answer_ids)} val samples..."
        )

        # Use BF16 for evaluation if enabled (safe for forward pass)
        eval_dtype = t.bfloat16 if config.use_bf16_eval else t.float32
        base_logits_list = []
        # CRITICAL: Collect answer_ids from batches to ensure alignment with logits
        # (PromptDataLoader may not yield batches in the same order as val_answer_ids)
        aligned_answer_ids = []

        with (
            t.no_grad(),
            t.autocast(
                device_type="cuda", dtype=eval_dtype, enabled=config.use_bf16_eval
            ),
        ):
            for batch in val_loader:
                logits = model(batch.clean)
                if len(logits.shape) == 3:
                    logits = logits[:, -1, :]
                # Keep in FP32 for metrics computation
                base_logits_list.append(logits.float())
                # Extract answers from batch (tensor shape: [batch_size, 1])
                aligned_answer_ids.extend(batch.answers.squeeze(-1).tolist())

        base_logits = t.cat(base_logits_list, dim=0)
        base_metrics = compute_metrics(base_logits, aligned_answer_ids)
        logger.info(f"[{device}] Base accuracy: {base_metrics['accuracy']:.1%}")

        # =====================================================================
        # PHASE 3: Run ACDC for each threshold (model stays loaded)
        # =====================================================================
        scores_dir = output_dir / "sweep_results" / model_safe / "prune_scores"
        scores_dir.mkdir(parents=True, exist_ok=True)

        for threshold in thresholds:
            task_id = Registry.make_task_id(model_name, threshold)
            tao_exp, tao_base = threshold_to_tao(threshold)

            logger.info(f"[{device}] ACDC: tau={threshold} ({tao_base}x10^{tao_exp})")

            try:
                t_start = time.time()

                # Run ACDC (stays in FP32 for precision with small thresholds)
                prune_scores = acdc_prune_scores(
                    model=patchable,
                    dataloader=train_loader,
                    official_edges=None,
                    tao_exps=[tao_exp],
                    tao_bases=[tao_base],
                    faithfulness_target="kl_div",
                    test_mode=False,
                    show_graphs=False,
                )

                training_time = time.time() - t_start

                # Count edges (batch the .item() calls for efficiency)
                n_edges = sum(t.isinf(s).sum() for s in prune_scores.values()).item()
                total_possible = sum(s.numel() for s in prune_scores.values())
                size_fraction = n_edges / total_possible if total_possible else 0.0

                logger.info(
                    f"[{device}] Circuit: {n_edges}/{total_possible} ({size_fraction:.1%}) in {training_time:.1f}s"
                )

                # Save prune scores (non-blocking CPU transfer)
                scores_file = scores_dir / f"tau_{tao_to_str(threshold)}.pkl"
                prune_scores_cpu = {k: v.cpu() for k, v in prune_scores.items()}

                # Async save in background thread
                def save_scores(path, data):
                    with open(path, "wb") as f:
                        pickle.dump(data, f)

                threading.Thread(
                    target=save_scores,
                    args=(scores_file, prune_scores_cpu),
                    daemon=True,
                ).start()

                # =====================================================================
                # PHASE 4: Evaluate circuit (reuse val_loader and base_logits)
                # =====================================================================
                if n_edges > 0:
                    # Non-blocking transfer to GPU
                    prune_scores_dev = {
                        k: v.to(device, non_blocking=True)
                        for k, v in prune_scores_cpu.items()
                    }
                    t.cuda.synchronize()

                    # Recreate val_loader with same seed for deterministic batching
                    set_all_seeds(config.eval_seed)
                    val_loader_eval, _, _ = prepare_eval_dataloader(
                        val_data,
                        pool,
                        bos_id,
                        batch_size=val_batch_size,
                        seed=config.eval_seed,
                        device=device,
                        n_samples=config.eval_val_size,
                    )

                    with t.autocast(
                        device_type="cuda",
                        dtype=eval_dtype,
                        enabled=config.use_bf16_eval,
                    ):
                        circuit_outputs = run_circuits(
                            model=patchable,
                            dataloader=val_loader_eval,
                            test_edge_counts=[n_edges],
                            prune_scores=prune_scores_dev,
                            patch_type=PatchType.TREE_PATCH,
                            ablation_type=AblationType.RESAMPLE,
                        )

                    # Extract circuit logits aligned by batch.key
                    set_all_seeds(config.eval_seed)
                    val_loader_keys, _, _ = prepare_eval_dataloader(
                        val_data,
                        pool,
                        bos_id,
                        batch_size=val_batch_size,
                        seed=config.eval_seed,
                        device=device,
                        n_samples=config.eval_val_size,
                    )
                    circ_logits_list = []
                    circ_answer_ids = []
                    for batch in val_loader_keys:
                        logits = circuit_outputs[n_edges][batch.key]
                        if len(logits.shape) == 3:
                            logits = logits[:, -1, :]
                        circ_logits_list.append(logits.float())
                        circ_answer_ids.extend(batch.answers.squeeze(-1).tolist())

                    circuit_logits = t.cat(circ_logits_list, dim=0)
                    circuit_metrics = compute_metrics(
                        circuit_logits, circ_answer_ids, base_logits
                    )

                    # =====================================================================
                    # PHASE 5: Evaluate ablation (complement circuit)
                    # =====================================================================
                    ablation_scores = {}
                    for name, scores in prune_scores_dev.items():
                        inv = scores.clone()
                        is_circuit = t.isinf(scores)
                        inv[is_circuit] = 0.0
                        inv[~is_circuit] = float("inf")
                        ablation_scores[name] = inv

                    n_abl_edges = sum(
                        t.isinf(s).sum() for s in ablation_scores.values()
                    ).item()

                    if n_abl_edges > 0:
                        set_all_seeds(config.eval_seed)
                        val_loader_abl, _, _ = prepare_eval_dataloader(
                            val_data,
                            pool,
                            bos_id,
                            batch_size=val_batch_size,
                            seed=config.eval_seed,
                            device=device,
                            n_samples=config.eval_val_size,
                        )

                        with t.autocast(
                            device_type="cuda",
                            dtype=eval_dtype,
                            enabled=config.use_bf16_eval,
                        ):
                            abl_outputs = run_circuits(
                                model=patchable,
                                dataloader=val_loader_abl,
                                test_edge_counts=[n_abl_edges],
                                prune_scores=ablation_scores,
                                patch_type=PatchType.TREE_PATCH,
                                ablation_type=AblationType.RESAMPLE,
                            )

                        # Extract ablation logits aligned by batch.key
                        set_all_seeds(config.eval_seed)
                        val_loader_abl_keys, _, _ = prepare_eval_dataloader(
                            val_data,
                            pool,
                            bos_id,
                            batch_size=val_batch_size,
                            seed=config.eval_seed,
                            device=device,
                            n_samples=config.eval_val_size,
                        )
                        abl_logits_list = []
                        abl_answer_ids = []
                        for batch in val_loader_abl_keys:
                            logits = abl_outputs[n_abl_edges][batch.key]
                            if len(logits.shape) == 3:
                                logits = logits[:, -1, :]
                            abl_logits_list.append(logits.float())
                            abl_answer_ids.extend(batch.answers.squeeze(-1).tolist())

                        abl_logits = t.cat(abl_logits_list, dim=0)
                        ablation_metrics = compute_metrics(
                            abl_logits, abl_answer_ids, base_logits
                        )
                    else:
                        ablation_metrics = {
                            "accuracy": 0.0,
                            "kl_div": float("inf"),
                            "n_samples": 0,
                            "error": "empty_ablation",
                        }

                    # Cleanup intermediate tensors
                    del prune_scores_dev, circuit_outputs, ablation_scores
                    if n_abl_edges > 0:
                        del abl_outputs
                else:
                    circuit_metrics = {
                        "accuracy": 0.0,
                        "kl_div": float("inf"),
                        "n_samples": 0,
                        "top5_accuracy": 0.0,
                        "top10_accuracy": 0.0,
                        "mean_correct_prob": 0.0,
                        "error": "empty_circuit",
                    }
                    ablation_metrics = {
                        "accuracy": 0.0,
                        "kl_div": float("inf"),
                        "n_samples": 0,
                        "error": "empty_circuit",
                    }

                # Build result
                result = {
                    "model": model_name,
                    "band": band,
                    "threshold": threshold,
                    "tao_exp": tao_exp,
                    "tao_base": tao_base,
                    "n_edges": n_edges,
                    "total_edges": total_possible,
                    "size_fraction": size_fraction,
                    "base_metrics": base_metrics,
                    "circuit_metrics": circuit_metrics,
                    "ablation_metrics": ablation_metrics,
                    "training_time_seconds": training_time,
                    "prune_scores_file": str(scores_file),
                    "batch_size": batch_size,
                    "acdc_seed": config.acdc_seed,
                    "eval_seed": config.eval_seed,
                    "acdc_train_size": config.acdc_train_size,
                    "train_indices": train_indices,
                    "val_n_samples": len(val_answer_ids),
                    "eval_val_size": config.eval_val_size,
                    "use_bf16_eval": config.use_bf16_eval,
                    "status": "completed",
                    "completed_at": datetime.now().isoformat(),
                }

                logger.info(
                    f"[{device}] tau={threshold}: {n_edges} edges ({size_fraction:.1%}), "
                    f"KL={circuit_metrics.get('kl_div', '?'):.4f}, "
                    f"acc={circuit_metrics.get('accuracy', '?'):.1%}, "
                    f"abl={ablation_metrics.get('accuracy', '?'):.1%}"
                )

                # Save to registry immediately
                registry.add_task(task_id, result)
                results.append(result)

                # Cleanup prune_scores
                del prune_scores, prune_scores_cpu

            except Exception as e:
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                logger.error(f"[{device}] FAILED tau={threshold}: {error_msg}")
                err_result = {
                    "model": model_name,
                    "band": band,
                    "threshold": threshold,
                    "status": "failed",
                    "error": error_msg,
                    "failed_at": datetime.now().isoformat(),
                }
                registry.add_task(task_id, err_result)
                results.append(err_result)

            # Light cleanup between thresholds (keep model loaded)
            gc.collect()
            t.cuda.empty_cache()

        logger.info(
            f"[{device}] Completed {model_name}: {len(results)} thresholds processed"
        )
        return results

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"[{device}] FAILED model {model_name}: {error_msg}")
        # Return error for all remaining thresholds
        for threshold in thresholds:
            if not any(r.get("threshold") == threshold for r in results):
                task_id = Registry.make_task_id(model_name, threshold)
                err_result = {
                    "model": model_name,
                    "band": config.sweep_band,
                    "threshold": threshold,
                    "status": "failed",
                    "error": error_msg,
                    "failed_at": datetime.now().isoformat(),
                }
                registry.add_task(task_id, err_result)
                results.append(err_result)
        return results

    finally:
        # Full cleanup at model switch
        if patchable is not None:
            del patchable
        safe_delete_model(model)
        cleanup_gpu()


class Registry:
    def __init__(self, path: Path):
        self.path = path
        self.lock_path = path.with_suffix(".lock")

    def _lock(self, timeout=30.0):
        lf = open(self.lock_path, "w")
        t0 = time.time()
        while True:
            try:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return lf
            except (IOError, OSError):
                if time.time() - t0 > timeout:
                    lf.close()
                    raise TimeoutError(f"Lock timeout: {self.lock_path}")
                time.sleep(0.1)

    def _unlock(self, lf):
        fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        lf.close()

    def load(self) -> dict:
        lf = self._lock()
        try:
            if self.path.exists():
                with open(self.path) as f:
                    return json.load(f)
            return {"metadata": {"created_at": datetime.now().isoformat()}, "tasks": {}}
        finally:
            self._unlock(lf)

    def add_task(self, task_id: str, data: dict):
        lf = self._lock()
        try:
            if self.path.exists():
                with open(self.path) as _f:
                    reg = json.load(_f)
            else:
                reg = {
                    "metadata": {"created_at": datetime.now().isoformat()},
                    "tasks": {},
                }
            reg["tasks"][task_id] = data
            reg["metadata"]["last_updated"] = datetime.now().isoformat()
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(reg, f, indent=2, default=str)
            tmp.rename(self.path)
        finally:
            self._unlock(lf)

    def get_completed_ids(self) -> Set[str]:
        reg = self.load()
        return {
            tid
            for tid, d in reg.get("tasks", {}).items()
            if d.get("status") == "completed"
        }

    @staticmethod
    def make_task_id(model: str, threshold: float) -> str:
        model_safe = model.replace("/", "_").replace("-", "_")
        return f"{model_safe}__tau_{tao_to_str(threshold)}"


# =============================================================================
# PARETO ANALYSIS
# =============================================================================
#
# Pareto objective: minimize (edge_fraction, KL_divergence)
#
# Why KL (not accuracy)?
#   - KL measures FAITHFULNESS: how well the circuit reproduces the full model
#   - Accuracy measures TASK PERFORMANCE: whether the circuit gets the right answer
#   - For interpretability, we want faithfulness (understand what the model does)
#   - Accuracy is reported but not used for selection
#
# No automatic threshold selection:
#   - Full Pareto frontier is reported for human inspection
#   - Avoids arbitrary algorithmic choices (e.g., Kneedle S parameter)
#   - Human selects τ* based on size/KL trade-off for their use case
#


def compute_pareto_frontier(points: List[Tuple[float, float]]) -> List[int]:
    """
    Compute Pareto frontier indices (minimize both size and KL).

    Returns indices of non-dominated points sorted by size.
    """
    indexed = sorted(enumerate(points), key=lambda x: x[1][0])
    frontier = []
    min_kl = float("inf")
    for orig_idx, (size, kl) in indexed:
        if kl < min_kl:
            frontier.append(orig_idx)
            min_kl = kl
    return frontier


def run_pareto_analysis(
    task_results: List[dict],
    model_name: str,
    output_dir: Path,
) -> Optional[dict]:
    """
    Analyze sweep results and report Pareto frontier.

    NO automatic threshold selection - reports full frontier for human decision.
    All metrics reported: KL (faithfulness), accuracy, ablation accuracy.
    """
    logger = logging.getLogger("pareto_sweep")

    valid = [
        r
        for r in task_results
        if r.get("status") == "completed"
        and r.get("circuit_metrics", {}).get("kl_div") is not None
    ]

    if len(valid) < 2:
        logger.warning(f"Only {len(valid)} valid results for {model_name} -- need >=2")
        return None

    valid.sort(key=lambda r: r["threshold"])

    thresholds = [r["threshold"] for r in valid]
    sizes = [r["size_fraction"] for r in valid]
    kls = [r["circuit_metrics"]["kl_div"] for r in valid]
    accs = [r["circuit_metrics"]["accuracy"] for r in valid]
    abl_accs = [r["ablation_metrics"].get("accuracy", 0) for r in valid]
    base_acc = valid[0].get("base_metrics", {}).get("accuracy", 0)

    # Compute Pareto frontier (minimize size AND KL)
    points = list(zip(sizes, kls))
    frontier_idx = compute_pareto_frontier(points)

    # Build sweep points with all metrics
    sweep_points = []
    for i in range(len(valid)):
        retention = accs[i] / base_acc if base_acc > 0 else 0
        sweep_points.append(
            {
                "threshold": thresholds[i],
                "size_fraction": sizes[i],
                "n_edges": valid[i]["n_edges"],
                "total_edges": valid[i]["total_edges"],
                "kl_div": kls[i],
                "accuracy": accs[i],
                "retention": retention,
                "ablation_accuracy": abl_accs[i],
                "is_pareto_optimal": i in frontier_idx,
                "training_time_seconds": valid[i]["training_time_seconds"],
            }
        )

    # Save
    model_safe = model_name.replace("/", "_").replace("-", "_")
    result_dir = output_dir / "sweep_results" / model_safe
    result_dir.mkdir(parents=True, exist_ok=True)

    pareto_data = {
        "model": model_name,
        "band": "control",
        "base_accuracy": base_acc,
        "n_thresholds_tested": len(valid),
        "n_pareto_optimal": len(frontier_idx),
        "pareto_frontier_indices": frontier_idx,
        "sweep_points": sweep_points,
        "analyzed_at": datetime.now().isoformat(),
        # NO automatic selection - human decides based on frontier
    }

    with open(result_dir / "pareto_analysis.json", "w") as f:
        json.dump(pareto_data, f, indent=2)

    # Plot
    try:
        _plot_pareto(
            sizes,
            kls,
            accs,
            abl_accs,
            thresholds,
            frontier_idx,
            base_acc,
            model_name,
            result_dir / "pareto_plot.png",
        )
    except Exception as e:
        logger.warning(f"Plot failed: {e}")

    # Log Pareto frontier summary
    logger.info(f"Pareto {model_name}: {len(frontier_idx)} optimal points on frontier")
    logger.info(
        f"  {'tau':<12} {'size%':<8} {'KL':<10} {'acc%':<8} {'ret%':<8} {'abl%':<8}"
    )
    for idx in frontier_idx:
        pt = sweep_points[idx]
        logger.info(
            f"  {pt['threshold']:<12.6f} {pt['size_fraction'] * 100:<8.2f} "
            f"{pt['kl_div']:<10.4f} {pt['accuracy'] * 100:<8.1f} "
            f"{pt['retention'] * 100:<8.1f} {pt['ablation_accuracy'] * 100:<8.1f}"
        )

    return pareto_data


def _plot_pareto(
    sizes,
    kls,
    accs,
    abl_accs,
    thresholds,
    frontier_idx,
    base_acc,
    model_name,
    save_path,
):
    """Plot Pareto analysis results (no auto-selection highlighted)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Pareto (size vs KL)
    ax = axes[0]
    ax.scatter(sizes, kls, c="steelblue", s=40, zorder=3, label="All thresholds")
    f_s = [sizes[i] for i in frontier_idx]
    f_k = [kls[i] for i in frontier_idx]
    ax.plot(f_s, f_k, "r-o", markersize=8, label="Pareto frontier", zorder=4)
    for i, tau in enumerate(thresholds):
        ax.annotate(
            f"{tau:.1e}",
            (sizes[i], kls[i]),
            fontsize=6,
            alpha=0.6,
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Circuit size (fraction)")
    ax.set_ylabel("KL(base || circuit)")
    ax.set_title(f"{model_name} / control\nPareto: Size vs KL")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Accuracy vs size
    ax = axes[1]
    ax.scatter(sizes, accs, c="steelblue", s=40, label="Circuit acc")
    ax.scatter(sizes, abl_accs, c="coral", s=40, marker="^", label="Ablation acc")
    ax.axhline(
        base_acc, color="green", ls="--", alpha=0.7, label=f"Base ({base_acc:.1%})"
    )
    # Highlight Pareto-optimal points
    for i in frontier_idx:
        ax.scatter([sizes[i]], [accs[i]], c="red", s=80, edgecolors="black", zorder=5)
    ax.set_xlabel("Circuit size (fraction)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Size\n(red = Pareto-optimal)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Metrics vs threshold
    ax = axes[2]
    ax.semilogx(thresholds, accs, "o-", color="steelblue", label="Circuit acc")
    ax.semilogx(thresholds, abl_accs, "^-", color="coral", label="Ablation acc")
    ax.axhline(base_acc, color="green", ls="--", alpha=0.7, label="Base acc")
    ax2 = ax.twinx()
    ax2.semilogx(thresholds, kls, "s-", color="purple", alpha=0.7, label="KL div")
    ax2.set_ylabel("KL divergence", color="purple")
    ax.set_xlabel("Threshold tau")
    ax.set_ylabel("Accuracy")
    ax.set_title("Metrics vs Threshold")
    ax.legend(loc="lower left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# SINGLE THRESHOLD RUNNER (for threshold-level parallelization)
# =============================================================================


def run_single_threshold(
    model_name: str,
    threshold: float,
    config: SweepConfig,
    device: str,
    output_dir: Path,
    registry: "Registry",
) -> Dict[str, Any]:
    """
    Run ACDC for a SINGLE (model, threshold) pair.

    Each call loads the model, processes one threshold, and releases.
    Less efficient than run_model_sweep() for sequential runs, but enables
    true multi-GPU parallelization across thresholds.
    """
    from auto_circuit.prune_algos.ACDC import acdc_prune_scores
    from auto_circuit.prune import run_circuits
    from auto_circuit.types import PatchType, AblationType
    from auto_circuit.utils.graph_utils import patchable_model

    logger = logging.getLogger("pareto_sweep")
    band = config.sweep_band
    batch_size = get_batch_size(model_name)
    model_safe = model_name.replace("/", "_").replace("-", "_")
    task_id = Registry.make_task_id(model_name, threshold)
    tao_exp, tao_base = threshold_to_tao(threshold)

    logger.info(f"[{device}] Starting: {model_name} tau={threshold}")

    model = None
    patchable = None

    try:
        # Load model
        set_all_seeds(config.acdc_seed)
        model = load_model(model_name, device)
        bos_id = model.tokenizer.bos_token_id

        # Load data
        pool_dir = Path(config.pool_dir)
        data_dir = Path(config.data_dir)
        pool = load_pool(band, pool_dir)
        train_data = load_dataset(band, "train", data_dir, config.variant, config.draw)
        val_data = load_dataset(band, "val", data_dir, config.variant, config.draw)

        # Create patchable model
        patchable = patchable_model(
            model=model,
            factorized=config.factorized,
            slice_output=config.slice_output,
            seq_len=None,
            separate_qkv=config.separate_qkv,
            device=device,
        )
        total_edges = len(patchable.edges)

        # Create ACDC training dataloader
        n_train = min(config.acdc_train_size, len(train_data["examples"]))
        train_batch_size = min(batch_size, n_train)
        train_loader, train_indices = prepare_dataloader(
            train_data,
            pool,
            bos_id,
            n_samples=config.acdc_train_size,
            batch_size=train_batch_size,
            seed=config.acdc_seed,
            device=device,
        )

        # Compute base logits
        set_all_seeds(config.eval_seed)
        n_val = (
            len(val_data["examples"])
            if config.eval_val_size == 0
            else min(config.eval_val_size, len(val_data["examples"]))
        )
        val_batch_size = min(batch_size, n_val)
        val_loader, val_indices, val_answer_ids = prepare_eval_dataloader(
            val_data,
            pool,
            bos_id,
            batch_size=val_batch_size,
            seed=config.eval_seed,
            device=device,
            n_samples=config.eval_val_size,
        )

        eval_dtype = t.bfloat16 if config.use_bf16_eval else t.float32
        base_logits_list = []
        # CRITICAL: Collect answer_ids from batches to ensure alignment with logits
        aligned_answer_ids = []

        with (
            t.no_grad(),
            t.autocast(
                device_type="cuda", dtype=eval_dtype, enabled=config.use_bf16_eval
            ),
        ):
            for batch in val_loader:
                logits = model(batch.clean)
                if len(logits.shape) == 3:
                    logits = logits[:, -1, :]
                base_logits_list.append(logits.float())
                aligned_answer_ids.extend(batch.answers.squeeze(-1).tolist())

        base_logits = t.cat(base_logits_list, dim=0)
        base_metrics = compute_metrics(base_logits, aligned_answer_ids)

        # Run ACDC
        logger.info(f"[{device}] ACDC: tau={threshold} ({tao_base}x10^{tao_exp})")
        t_start = time.time()

        prune_scores = acdc_prune_scores(
            model=patchable,
            dataloader=train_loader,
            official_edges=None,
            tao_exps=[tao_exp],
            tao_bases=[tao_base],
            faithfulness_target="kl_div",
            test_mode=False,
            show_graphs=False,
        )

        training_time = time.time() - t_start

        # Count edges
        n_edges = sum(t.isinf(s).sum() for s in prune_scores.values()).item()
        total_possible = sum(s.numel() for s in prune_scores.values())
        size_fraction = n_edges / total_possible if total_possible else 0.0

        logger.info(
            f"[{device}] Circuit: {n_edges}/{total_possible} ({size_fraction:.1%}) in {training_time:.1f}s"
        )

        # Save prune scores
        scores_dir = output_dir / "sweep_results" / model_safe / "prune_scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        scores_file = scores_dir / f"tau_{tao_to_str(threshold)}.pkl"
        prune_scores_cpu = {k: v.cpu() for k, v in prune_scores.items()}

        def save_scores(path, data):
            with open(path, "wb") as f:
                pickle.dump(data, f)

        threading.Thread(
            target=save_scores, args=(scores_file, prune_scores_cpu), daemon=True
        ).start()

        # Evaluate circuit
        if n_edges > 0:
            prune_scores_dev = {
                k: v.to(device, non_blocking=True) for k, v in prune_scores_cpu.items()
            }
            t.cuda.synchronize()

            set_all_seeds(config.eval_seed)
            val_loader_eval, _, _ = prepare_eval_dataloader(
                val_data,
                pool,
                bos_id,
                batch_size=val_batch_size,
                seed=config.eval_seed,
                device=device,
                n_samples=config.eval_val_size,
            )

            with t.autocast(
                device_type="cuda", dtype=eval_dtype, enabled=config.use_bf16_eval
            ):
                circuit_outputs = run_circuits(
                    model=patchable,
                    dataloader=val_loader_eval,
                    test_edge_counts=[n_edges],
                    prune_scores=prune_scores_dev,
                    patch_type=PatchType.TREE_PATCH,
                    ablation_type=AblationType.RESAMPLE,
                )

            # Extract circuit logits aligned by batch.key
            set_all_seeds(config.eval_seed)
            val_loader_keys, _, _ = prepare_eval_dataloader(
                val_data,
                pool,
                bos_id,
                batch_size=val_batch_size,
                seed=config.eval_seed,
                device=device,
                n_samples=config.eval_val_size,
            )
            circ_logits_list = []
            circ_answer_ids = []
            for batch in val_loader_keys:
                logits = circuit_outputs[n_edges][batch.key]
                if len(logits.shape) == 3:
                    logits = logits[:, -1, :]
                circ_logits_list.append(logits.float())
                circ_answer_ids.extend(batch.answers.squeeze(-1).tolist())

            circuit_logits = t.cat(circ_logits_list, dim=0)
            circuit_metrics = compute_metrics(
                circuit_logits, circ_answer_ids, base_logits
            )

            # Evaluate ablation
            ablation_scores = {}
            for name, scores in prune_scores_dev.items():
                inv = scores.clone()
                is_circuit = t.isinf(scores)
                inv[is_circuit] = 0.0
                inv[~is_circuit] = float("inf")
                ablation_scores[name] = inv

            n_abl_edges = sum(t.isinf(s).sum() for s in ablation_scores.values()).item()

            if n_abl_edges > 0:
                set_all_seeds(config.eval_seed)
                val_loader_abl, _, _ = prepare_eval_dataloader(
                    val_data,
                    pool,
                    bos_id,
                    batch_size=val_batch_size,
                    seed=config.eval_seed,
                    device=device,
                    n_samples=config.eval_val_size,
                )

                with t.autocast(
                    device_type="cuda", dtype=eval_dtype, enabled=config.use_bf16_eval
                ):
                    abl_outputs = run_circuits(
                        model=patchable,
                        dataloader=val_loader_abl,
                        test_edge_counts=[n_abl_edges],
                        prune_scores=ablation_scores,
                        patch_type=PatchType.TREE_PATCH,
                        ablation_type=AblationType.RESAMPLE,
                    )

                # Extract ablation logits aligned by batch.key
                set_all_seeds(config.eval_seed)
                val_loader_abl_keys, _, _ = prepare_eval_dataloader(
                    val_data,
                    pool,
                    bos_id,
                    batch_size=val_batch_size,
                    seed=config.eval_seed,
                    device=device,
                    n_samples=config.eval_val_size,
                )
                abl_logits_list = []
                abl_answer_ids = []
                for batch in val_loader_abl_keys:
                    logits = abl_outputs[n_abl_edges][batch.key]
                    if len(logits.shape) == 3:
                        logits = logits[:, -1, :]
                    abl_logits_list.append(logits.float())
                    abl_answer_ids.extend(batch.answers.squeeze(-1).tolist())

                abl_logits = t.cat(abl_logits_list, dim=0)
                ablation_metrics = compute_metrics(
                    abl_logits, abl_answer_ids, base_logits
                )
            else:
                ablation_metrics = {
                    "accuracy": 0.0,
                    "kl_div": float("inf"),
                    "n_samples": 0,
                    "error": "empty_ablation",
                }

            del prune_scores_dev, circuit_outputs, ablation_scores
            if n_abl_edges > 0:
                del abl_outputs
        else:
            circuit_metrics = {
                "accuracy": 0.0,
                "kl_div": float("inf"),
                "n_samples": 0,
                "top5_accuracy": 0.0,
                "top10_accuracy": 0.0,
                "mean_correct_prob": 0.0,
                "error": "empty_circuit",
            }
            ablation_metrics = {
                "accuracy": 0.0,
                "kl_div": float("inf"),
                "n_samples": 0,
                "error": "empty_circuit",
            }

        # Build result
        result = {
            "model": model_name,
            "band": band,
            "threshold": threshold,
            "tao_exp": tao_exp,
            "tao_base": tao_base,
            "n_edges": n_edges,
            "total_edges": total_possible,
            "size_fraction": size_fraction,
            "base_metrics": base_metrics,
            "circuit_metrics": circuit_metrics,
            "ablation_metrics": ablation_metrics,
            "training_time_seconds": training_time,
            "prune_scores_file": str(scores_file),
            "batch_size": batch_size,
            "acdc_seed": config.acdc_seed,
            "eval_seed": config.eval_seed,
            "acdc_train_size": config.acdc_train_size,
            "train_indices": train_indices,
            "val_n_samples": len(val_answer_ids),
            "eval_val_size": config.eval_val_size,
            "use_bf16_eval": config.use_bf16_eval,
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
        }

        logger.info(
            f"[{device}] tau={threshold}: {n_edges} edges ({size_fraction:.1%}), "
            f"KL={circuit_metrics.get('kl_div', '?'):.4f}, "
            f"acc={circuit_metrics.get('accuracy', '?'):.1%}, "
            f"abl={ablation_metrics.get('accuracy', '?'):.1%}"
        )

        registry.add_task(task_id, result)
        return result

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"[{device}] FAILED tau={threshold}: {error_msg}")
        err_result = {
            "model": model_name,
            "band": band,
            "threshold": threshold,
            "status": "failed",
            "error": error_msg,
            "failed_at": datetime.now().isoformat(),
        }
        registry.add_task(task_id, err_result)
        return err_result

    finally:
        if patchable is not None:
            del patchable
        safe_delete_model(model)
        cleanup_gpu()


def gpu_worker_threshold(
    gpu_id,
    task_queue,
    result_queue,
    config_dict,
    output_dir_str,
    progress_dict,
    worker_id,
    heartbeat_dict,
):
    """
    THRESHOLD-LEVEL PARALLELIZATION: Each task is a single (model, threshold) pair.
    Model is loaded for each threshold (trades efficiency for parallelism).
    """
    import queue as queue_module  # For Empty exception

    config = SweepConfig(**config_dict)
    output_dir = Path(output_dir_str)
    device = f"cuda:{gpu_id}"
    logger = logging.getLogger("pareto_sweep")

    try:
        t.cuda.set_device(gpu_id)
    except Exception as e:
        error_msg = f"GPU {gpu_id} init failed: {e}\n{traceback.format_exc()}"
        logger.error(error_msg)
        progress_dict[worker_id] = f"GPU {gpu_id}: FAILED (init)"
        heartbeat_dict[worker_id] = -1  # Signal fatal error
        return

    registry = Registry(output_dir / config.registry_file)
    tasks_completed = 0

    logger.info(f"[Worker {worker_id}] Started on GPU {gpu_id}")
    heartbeat_dict[worker_id] = time.time()

    while True:
        # Update heartbeat while waiting for task
        heartbeat_dict[worker_id] = time.time()

        try:
            task = task_queue.get(timeout=5)
        except queue_module.Empty:
            # Queue is empty but no poison pill yet - keep waiting
            continue
        except Exception as e:
            # Unexpected error getting from queue
            logger.error(f"[Worker {worker_id}] Queue error: {e}")
            continue

        if task is None:
            # Poison pill - clean shutdown
            logger.info(
                f"[Worker {worker_id}] Received shutdown signal. Completed {tasks_completed} tasks."
            )
            progress_dict[worker_id] = (
                f"GPU {gpu_id}: shutdown (completed {tasks_completed})"
            )
            break

        model_name = task["model"]
        threshold = task["threshold"]
        task_id = Registry.make_task_id(model_name, threshold)
        progress_dict[worker_id] = f"GPU {gpu_id}: {model_name} tau={threshold:.2e}"
        heartbeat_dict[worker_id] = time.time()

        logger.info(
            f"[Worker {worker_id}] Starting task: {model_name} tau={threshold:.2e}"
        )

        try:
            result = run_single_threshold(
                model_name=model_name,
                threshold=threshold,
                config=config,
                device=device,
                output_dir=output_dir,
                registry=registry,
            )
            result_queue.put(("success", task, result, None))
            tasks_completed += 1
            logger.info(
                f"[Worker {worker_id}] Completed: {model_name} tau={threshold:.2e} ({tasks_completed} total)"
            )

        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(
                f"[Worker {worker_id}] FAILED: {model_name} tau={threshold:.2e}\n{error_msg}"
            )

            try:
                err = {
                    "model": model_name,
                    "threshold": threshold,
                    "status": "failed",
                    "error": error_msg,
                    "failed_at": datetime.now().isoformat(),
                }
                registry.add_task(task_id, err)
            except Exception as reg_err:
                logger.error(f"[Worker {worker_id}] Registry write failed: {reg_err}")

            result_queue.put(("error", task, None, error_msg))
            cleanup_gpu()

        progress_dict[worker_id] = f"GPU {gpu_id}: idle (completed {tasks_completed})"
        heartbeat_dict[worker_id] = time.time()

    # Final cleanup
    cleanup_gpu()
    logger.info(f"[Worker {worker_id}] Exiting cleanly")


def gpu_worker(
    gpu_id,
    task_queue,
    result_queue,
    config_dict,
    output_dir_str,
    progress_dict,
    worker_id,
):
    """
    MODEL-LEVEL: Each task is a MODEL with all its pending thresholds.
    Model is loaded once and all thresholds are processed together.
    Used when n_models >= n_gpus (more efficient).
    """
    config = SweepConfig(**config_dict)
    output_dir = Path(output_dir_str)
    device = f"cuda:{gpu_id}"
    t.cuda.set_device(gpu_id)
    registry = Registry(output_dir / config.registry_file)

    while True:
        try:
            task = task_queue.get(timeout=2)
        except Exception:
            if task_queue.empty():
                break
            continue
        if task is None:
            break

        model_name = task["model"]
        thresholds = task["thresholds"]
        progress_dict[worker_id] = (
            f"GPU {gpu_id}: {model_name} ({len(thresholds)} thresholds)"
        )

        try:
            results = run_model_sweep(
                model_name=model_name,
                thresholds=thresholds,
                config=config,
                device=device,
                output_dir=output_dir,
                registry=registry,
            )
            result_queue.put((task, results, None))
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            # Mark all thresholds as failed
            for threshold in thresholds:
                task_id = Registry.make_task_id(model_name, threshold)
                err = {
                    "model": model_name,
                    "threshold": threshold,
                    "status": "failed",
                    "error": error_msg,
                    "failed_at": datetime.now().isoformat(),
                }
                registry.add_task(task_id, err)
            result_queue.put((task, [], error_msg))
            cleanup_gpu()

        progress_dict[worker_id] = f"GPU {gpu_id}: idle"


def run_all_tasks(
    config: SweepConfig, output_dir: Path, gpus: List[int], force: bool = False
) -> Dict[str, int]:
    """
    Each (model, threshold) pair is a separate task.

    Returns:
        Dict with 'completed', 'failed', 'total' counts
    """
    logger = logging.getLogger("pareto_sweep")
    registry = Registry(output_dir / config.registry_file)

    # Get completed task IDs
    completed = set() if force else registry.get_completed_ids()

    # Sort models by size (smallest first) for early error detection
    sorted_models = sort_models_by_size(config.models)
    logger.info(f"Model order (smallest first): {sorted_models}")

    # Build list of ALL pending (model, threshold) tasks
    threshold_tasks = []
    total_all = 0

    for model in sorted_models:
        model_thresholds = config.thresholds
        total_all += len(model_thresholds)

        for threshold in model_thresholds:
            task_id = Registry.make_task_id(model, threshold)
            if task_id not in completed:
                threshold_tasks.append(
                    {
                        "model": model,
                        "threshold": threshold,
                    }
                )

    already_completed = total_all - len(threshold_tasks)
    logger.info(f"Total threshold tasks: {total_all}")
    logger.info(f"Already completed: {already_completed}")
    logger.info(f"Remaining: {len(threshold_tasks)} (model, threshold) pairs")

    if not threshold_tasks:
        return {
            "completed": already_completed,
            "failed": 0,
            "total": total_all,
            "skipped": 0,
        }

    n_gpus = len(gpus)

    # -- Single GPU: sequential with model reuse (efficient) --
    if n_gpus <= 1:
        device = f"cuda:{gpus[0]}" if gpus else "cpu"
        logger.info(f"Sequential execution on {device} (single GPU -> model reuse)")

        # Group by model for efficient sequential processing
        by_model = defaultdict(list)
        for task in threshold_tasks:
            by_model[task["model"]].append(task["threshold"])

        t_start = time.time()
        model_list = [m for m in sorted_models if m in by_model]

        seq_completed = 0
        seq_failed = 0

        for i, model_name in enumerate(model_list):
            thresholds = by_model[model_name]

            if i > 0:
                elapsed = time.time() - t_start
                avg_per_model = elapsed / i
                eta = timedelta(seconds=int(avg_per_model * (len(model_list) - i)))
                logger.info(
                    f"Progress: {i}/{len(model_list)} models ({100 * i / len(model_list):.0f}%) ETA: {eta}"
                )

            try:
                results = run_model_sweep(
                    model_name=model_name,
                    thresholds=thresholds,
                    config=config,
                    device=device,
                    output_dir=output_dir,
                    registry=registry,
                )
                for r in results:
                    if r.get("status") == "completed":
                        seq_completed += 1
                    else:
                        seq_failed += 1
            except Exception as e:
                logger.error(
                    f"Model sweep failed for {model_name}: {e}\n{traceback.format_exc()}"
                )
                seq_failed += len(thresholds)

        return {
            "completed": already_completed + seq_completed,
            "failed": seq_failed,
            "total": total_all,
            "skipped": 0,
        }

    # -- Multi-GPU: THRESHOLD-LEVEL parallelization --
    # Each GPU processes individual (model, threshold) pairs
    # Model is loaded per-threshold (trades efficiency for parallelism)
    logger.info(f"THRESHOLD-LEVEL parallel execution on {n_gpus} GPUs: {gpus}")
    logger.info(f"  Each GPU processes individual (model, threshold) pairs")
    logger.info(
        f"  Model loaded per-threshold (enables all GPUs to work on same model)"
    )

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    manager = ctx.Manager()
    progress_dict = manager.dict()
    heartbeat_dict = manager.dict()  # Track worker liveness

    # Queue individual (model, threshold) tasks
    for task in threshold_tasks:
        task_queue.put(task)
    for _ in gpus:
        task_queue.put(None)  # Poison pills

    config_dict = asdict(config)
    workers = []
    for i, gpu_id in enumerate(gpus):
        progress_dict[i] = f"GPU {gpu_id}: starting"
        heartbeat_dict[i] = time.time()
        p = ctx.Process(
            target=gpu_worker_threshold,
            args=(
                gpu_id,
                task_queue,
                result_queue,
                config_dict,
                str(output_dir),
                progress_dict,
                i,
                heartbeat_dict,
            ),
        )
        p.start()
        workers.append(p)

    # Robust result collection with worker health monitoring
    completed_tasks = 0
    failed_tasks = 0
    total_tasks = len(threshold_tasks)
    last_progress_log = time.time()
    # Large models (410m+) can take 4+ hours per threshold in ACDC
    # Heartbeat only updates between tasks, not during ACDC
    # So we set a very long timeout and rely on process.is_alive() instead
    HEARTBEAT_TIMEOUT = 14400  # 4 hours - ACDC doesn't update heartbeat during run
    PROGRESS_LOG_INTERVAL = 60  # Log progress every minute
    heartbeat_warned = set()  # Only warn once per worker

    logger.info(f"Waiting for {total_tasks} tasks across {n_gpus} workers...")

    while completed_tasks + failed_tasks < total_tasks:
        # Check for results (non-blocking with short timeout)
        try:
            result = result_queue.get(timeout=10)
            status, task, data, error = result

            if status == "success":
                completed_tasks += 1
                logger.info(
                    f"Task completed: {task['model']} tau={task['threshold']:.2e} "
                    f"[{completed_tasks}/{total_tasks}]"
                )
            else:
                failed_tasks += 1
                logger.error(
                    f"Task FAILED: {task['model']} tau={task['threshold']:.2e} "
                    f"[failures: {failed_tasks}] - {error[:200] if error else 'unknown'}"
                )

        except Exception:
            # No result yet - check worker health
            pass

        # Check worker health
        current_time = time.time()
        all_workers_dead = True
        workers_stuck = []

        for i, p in enumerate(workers):
            if p.is_alive():
                all_workers_dead = False
                # Check heartbeat (but ACDC doesn't update it during long runs)
                last_heartbeat = heartbeat_dict.get(i, 0)
                if last_heartbeat == -1:
                    logger.error(f"Worker {i} reported fatal error")
                elif current_time - last_heartbeat > HEARTBEAT_TIMEOUT:
                    workers_stuck.append(i)
                    # Only warn once per worker to avoid log spam
                    if i not in heartbeat_warned:
                        heartbeat_warned.add(i)
                        logger.warning(
                            f"Worker {i} heartbeat stale ({current_time - last_heartbeat:.0f}s) - "
                            f"this is normal for large models during ACDC"
                        )

        # Log progress periodically
        if current_time - last_progress_log >= PROGRESS_LOG_INTERVAL:
            last_progress_log = current_time
            pct = 100 * (completed_tasks + failed_tasks) / total_tasks
            logger.info(
                f"Progress: {completed_tasks + failed_tasks}/{total_tasks} ({pct:.1f}%) "
                f"[completed={completed_tasks}, failed={failed_tasks}]"
            )
            for wid, st in progress_dict.items():
                hb = heartbeat_dict.get(wid, 0)
                hb_age = current_time - hb if hb > 0 else float("inf")
                logger.info(f"  Worker {wid}: {st} (heartbeat: {hb_age:.0f}s ago)")

        # If all workers are dead but we haven't received all results, something went wrong
        if all_workers_dead:
            remaining = total_tasks - completed_tasks - failed_tasks
            if remaining > 0:
                logger.error(f"All workers died but {remaining} tasks not completed!")
                # Try to drain any remaining results
                try:
                    while True:
                        result = result_queue.get(timeout=1)
                        status, task, data, error = result
                        if status == "success":
                            completed_tasks += 1
                        else:
                            failed_tasks += 1
                except Exception:
                    pass
            break

    # Summary
    logger.info(f"\nTask execution finished:")
    logger.info(f"  Completed: {completed_tasks}/{total_tasks}")
    logger.info(f"  Failed: {failed_tasks}/{total_tasks}")

    # Wait for workers to finish cleanly (they should exit after processing poison pills)
    logger.info("Waiting for workers to exit...")
    for i, p in enumerate(workers):
        p.join(timeout=60)
        if p.is_alive():
            logger.warning(f"Worker {i} still alive after 60s, terminating...")
            p.terminate()
            p.join(timeout=10)

    return {
        "completed": already_completed + completed_tasks,
        "failed": failed_tasks,
        "total": total_all,
        "skipped": 0,
    }


def generate_threshold_summary(config: SweepConfig, output_dir: Path):
    logger = logging.getLogger("pareto_sweep")
    registry = Registry(output_dir / config.registry_file)
    reg = registry.load()

    # Group by model
    by_model = defaultdict(list)
    for tid, data in reg.get("tasks", {}).items():
        if data.get("status") == "completed":
            by_model[data["model"]].append(data)

    summary = OrderedDict()
    summary["created_at"] = datetime.now().isoformat()
    summary["config"] = {
        "models": config.models,
        "sweep_band": config.sweep_band,
        "thresholds": config.thresholds,
    }
    summary["pareto_results"] = {}

    for model in config.models:
        results = by_model.get(model, [])
        if not results:
            continue
        logger.info(f"Pareto analysis: {model} ({len(results)} thresholds)")
        pareto_data = run_pareto_analysis(results, model, output_dir)
        if pareto_data:
            summary["pareto_results"][model] = pareto_data

    summary_path = output_dir / "sweep_results" / "pareto_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary table (Pareto frontiers, no auto-selection)
    logger.info("\n" + "=" * 80)
    logger.info("PARETO ANALYSIS SUMMARY (no auto-selection, human decides)")
    logger.info("=" * 80)

    for model, data in summary["pareto_results"].items():
        logger.info(f"\n{model} (base_acc={data['base_accuracy']:.1%}):")
        logger.info(
            f"  {'tau':<12} {'size%':<8} {'KL':<10} {'acc%':<8} {'ret%':<8} {'abl%':<8}"
        )
        for pt in data["sweep_points"]:
            marker = " *" if pt["is_pareto_optimal"] else "  "
            logger.info(
                f"{marker}{pt['threshold']:<12.2e} {pt['size_fraction'] * 100:<8.2f} "
                f"{pt['kl_div']:<10.4f} {pt['accuracy'] * 100:<8.1f} "
                f"{pt['retention'] * 100:<8.1f} {pt['ablation_accuracy'] * 100:<8.1f}"
            )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="LSC Pareto Sweep: threshold selection on control band",
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
    parser.add_argument("--gpus", nargs="+", default=["auto"])
    parser.add_argument(
        "--train-size",
        type=int,
        default=256,
        help="ACDC training examples sampled from train split (default: 256)",
    )
    parser.add_argument(
        "--eval-val-size",
        type=int,
        default=0,
        help="Validation examples for sweep evaluation (default: 0=full split)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable BF16 for evaluation (default: FP32, BF16 causes accuracy issues)",
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--pool-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--variant",
        type=str,
        default="matched",
        help="Dataset variant: 'matched' or 'unmatched'",
    )
    parser.add_argument(
        "--draw",
        type=str,
        default="draw_1",
        help="Draw index: 'draw_1', 'draw_2', or 'draw_3'",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    config = SweepConfig()
    if args.models:
        config.models = args.models
    if args.thresholds:
        config.thresholds = args.thresholds
    config.acdc_train_size = args.train_size
    config.eval_val_size = args.eval_val_size
    config.use_bf16_eval = args.bf16
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.pool_dir:
        config.pool_dir = args.pool_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    config.variant = args.variant
    config.draw = args.draw

    if args.gpus == ["auto"]:
        config.gpus = (
            list(range(t.cuda.device_count())) if t.cuda.is_available() else []
        )
    else:
        config.gpus = [int(g) for g in args.gpus]

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir, args.debug)

    logger.info("=" * 70)
    logger.info("LSC PARETO SWEEP (Phase 1 - Threshold Selection) [OPTIMIZED]")
    logger.info("=" * 70)
    logger.info(f"Models:     {config.models}")
    logger.info(f"Band:       {config.sweep_band}")

    # Show threshold info
    if args.thresholds:
        logger.info(f"Thresholds: {config.thresholds} (CLI override)")
    else:
        logger.info(f"Thresholds: {config.thresholds} (uniform for all models)")
    logger.info(f"GPUs:       {config.gpus}")
    logger.info(f"ACDC train: {config.acdc_train_size} samples")
    eval_desc = (
        f"{config.eval_val_size} samples" if config.eval_val_size > 0 else "FULL"
    )
    logger.info(f"Eval VAL:   {eval_desc}")
    logger.info(f"BF16 eval:  {config.use_bf16_eval}")
    logger.info(f"Data:       {config.data_dir}")
    logger.info(f"Output:     {config.output_dir}")
    logger.info("-" * 70)
    logger.info("OPTIMIZATIONS ENABLED:")
    logger.info("  - Model reuse: Load once per model, run all thresholds")
    logger.info("  - Base logits: Computed once, reused for all thresholds")
    logger.info("  - Dataloader:  Created once, reused")
    logger.info(
        f"  - BF16 eval:   {'Yes (faster)' if config.use_bf16_eval else 'No (FP32)'}"
    )
    logger.info(f"  - Val samples: {eval_desc} (smaller = faster Pareto ranking)")
    n_tasks = len(config.models) * len(config.thresholds)
    logger.info(f"Tasks:      {n_tasks}")
    logger.info("=" * 70)

    if args.analyze_only:
        generate_threshold_summary(config, output_dir)
        return

    # Validate data exists
    pool_path = Path(config.pool_dir) / f"lsc_pool_{config.sweep_band}.json"
    train_path = (
        Path(config.data_dir)
        / "datasets"
        / config.variant
        / config.draw
        / config.sweep_band
        / "train.json"
    )
    val_path = (
        Path(config.data_dir)
        / "datasets"
        / config.variant
        / config.draw
        / config.sweep_band
        / "val.json"
    )
    for p in [pool_path, train_path, val_path]:
        if not p.exists():
            logger.error(f"Missing: {p}")
            return 1

    task_stats = run_all_tasks(config, output_dir, config.gpus, force=args.force)

    logger.info("\n" + "=" * 70)
    logger.info("PARETO ANALYSIS")
    logger.info("=" * 70)
    generate_threshold_summary(config, output_dir)

    logger.info("\n" + "=" * 70)
    if task_stats and task_stats.get("failed", 0) > 0:
        logger.warning(f"SWEEP FINISHED WITH ERRORS")
        logger.warning(f"  Completed: {task_stats['completed']}/{task_stats['total']}")
        logger.warning(f"  Failed: {task_stats['failed']}/{task_stats['total']}")
    elif task_stats and task_stats["completed"] < task_stats["total"]:
        logger.warning(f"SWEEP INCOMPLETE")
        logger.warning(f"  Completed: {task_stats['completed']}/{task_stats['total']}")
    else:
        logger.info("SWEEP COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output:  {output_dir}")
    logger.info(f"Summary: {output_dir / 'sweep_results' / 'pareto_summary.json'}")

    return 1 if (task_stats and task_stats.get("failed", 0) > 0) else 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main() or 0)
