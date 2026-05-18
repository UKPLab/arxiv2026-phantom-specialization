#!/usr/bin/env python3
"""
LSC Circuit Discovery (Phase 2)
================================
Discover circuits across ALL frequency bands using per-model thresholds
selected by lsc_pareto_sweep.py (Phase 1).

Prerequisites:
  - lsc_base_eval.py must be run first to pre-compute base model metrics
  - lsc_pareto_sweep.py (Phase 1) must be run to select thresholds

Workflow per task (model, band, draw):
  1. Read τ* from threshold_summary.json
  2. Run fresh ACDC at τ* on TRAIN split (sampled, seed-controlled) ->  prune_scores
  3. Load pre-computed base metrics from lsc_base_eval.py
  4. Compute base logits on FULL TEST split (needed for KL divergence)
  5. Evaluate circuit on FULL TEST split  ->  circuit_metrics + KL
  6. Evaluate ablation on FULL TEST split ->  ablation_metrics + KL
  7. Cross-band: evaluate circuit on each other band's FULL TEST  ->  transfer_matrix
  8. Save prune_scores + register results with reproducibility info

Multiple draws (independent dataset re-samplings) give statistical robustness.
Aggregation computes mean +/- std across draws and builds cross-band transfer heatmaps.

DATA SPLITS (ML Best Practices - No Leakage):
  - TRAIN split: Used for ACDC circuit pruning (sampled, seed-controlled)
  - VAL split: Used ONLY in Phase 1 for threshold selection (never touched here)
  - TEST split: Used for final evaluation (FULL - no sampling for consistency)

Inputs:
  - threshold_summary.json     from lsc_pareto_sweep.py
  - base_metrics/{model}/{draw}/{band}.json from lsc_base_eval.py
  - LSC_data/datasets/{variant}/{draw}/{band}/{split}.json
  - LSC_data/lsc_token_pools/matched/lsc_pool_{band}.json

Outputs:
  circuits/{model}/{band}/{draw}/prune_scores.pkl
  registry.json          (all task results)
  summary/
    discovery_summary.json
    cross_band_transfer.json
    plots/  (per-model heatmaps, accuracy bar charts)

PRECISION:
  - FP32 evaluation by default (BF16 causes accuracy issues with Pythia models)
  - Enable BF16 with --bf16 flag if needed for speed

Usage:
    python lsc_acdc_circuit.py
    python lsc_acdc_circuit.py --models pythia-70m --bands low medium high --draws draw_1 draw_2
    python lsc_acdc_circuit.py --threshold 0.001       # override for all models
    python lsc_acdc_circuit.py --analyze-only           # just aggregate existing results
    python lsc_acdc_circuit.py --no-cross-band          # skip cross-band evaluation
    python lsc_acdc_circuit.py --gpus 0 1 2 3
    python lsc_acdc_circuit.py --bf16                   # Enable BF16 (default: FP32)
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
import queue as queue_module
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, OrderedDict

import numpy as np
import torch as t
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
ISC_ROOT = SCRIPT_DIR.parent  # repo root

AUTOCIRCUIT_PATH = os.environ.get("AUTOCIRCUIT_PATH") or str(
    ISC_ROOT / "circuit_discovery" / "auto-circuit"
)
sys.path.insert(0, AUTOCIRCUIT_PATH)

ALL_BANDS = [
    "low",
    "medium",
    "high",
    "very_high",
    "control",
]

DEFAULT_MODELS = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
]

# Model size ordering (for sorting smallest-first -> early error detection)
MODEL_SIZE_ORDER = {
    "pythia-70m": 0,
    "pythia-160m": 1,
    "pythia-410m": 2,
    "pythia-1b": 3,
    "pythia-1.4b": 4,
}

# Optimized batch sizes per model for A100 80GB (matched with sweep script)
MODEL_BATCH_SIZES = {
    "pythia-70m": 256,
    "pythia-160m": 256,
    "pythia-410m": 128,
    "pythia-1b": 96,
    "pythia-1.4b": 64,
}

# Estimated ACDC runtime in minutes (for progress reporting)
ESTIMATED_MINUTES = {
    "pythia-70m": 5,
    "pythia-160m": 30,
    "pythia-410m": 150,
    "pythia-1b": 45,
    "pythia-1.4b": 150,
}

# LSC sequence structure
N_SOURCE = 5
N_DISTRACT = 10
RAW_SEQ_LEN = N_SOURCE + 1 + N_DISTRACT + N_SOURCE  # 21
SEQ_LEN_WITH_BOS = RAW_SEQ_LEN + 1  # 22
DIVERGE_IDX = N_SOURCE + 1 + N_DISTRACT + 1  # 17 (with BOS)


@dataclass
class DiscoveryConfig:
    """Configuration for Phase 2 circuit discovery."""

    # Paths (relative to ISC_ROOT by default)
    data_dir: str = field(default_factory=lambda: str(ISC_ROOT / "LSC_data"))
    pool_dir: str = field(
        default_factory=lambda: str(
            ISC_ROOT / "LSC_data" / "lsc_token_pools" / "matched"
        )
    )
    sweep_dir: str = field(default_factory=lambda: str(SCRIPT_DIR / "pareto_sweep"))
    output_dir: str = field(
        default_factory=lambda: str(SCRIPT_DIR / "circuit_discovery")
    )
    base_metrics_dir: str = field(
        default_factory=lambda: str(SCRIPT_DIR / "base_metrics")
    )
    # Dataset structure: datasets/{variant}/{draw}/{band}/{split}.json
    variant: str = "matched"
    draws: List[str] = field(default_factory=lambda: ["draw_1", "draw_2", "draw_3"])

    # Experiment grid
    models: List[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    bands: List[str] = field(default_factory=lambda: list(ALL_BANDS))

    # Optional threshold override (None = read from threshold_summary.json)
    threshold_override: Optional[float] = None

    # ACDC settings
    acdc_train_size: int = 256  # Sampled from train split for ACDC
    # Note: Evaluation uses FULL TEST split (no sampling) for consistency with base_eval
    factorized: bool = True
    separate_qkv: bool = False
    slice_output: str = "last_seq"

    # Seeds: acdc_seed is fixed (draws provide the variation); eval_seed is fixed for fair comparison
    acdc_seed: int = 42
    eval_seed: int = 123

    # Cross-band evaluation
    cross_band: bool = True

    # Use BF16 for evaluation (disabled by default - BF16 causes accuracy issues with Pythia)
    use_bf16_eval: bool = False

    # GPU
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
    log_file = log_dir / f"discovery_{timestamp}.log"

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
    logger = logging.getLogger("circuit_discovery")
    logger.info(f"Log file: {log_file}")
    return logger


def threshold_to_tao(threshold: float) -> Tuple[int, float]:
    """threshold = tao_base x 10^tao_exp"""
    exponent = math.floor(math.log10(threshold))
    base = round(threshold / (10**exponent), 6)
    return exponent, base


def tao_to_str(threshold: float) -> str:
    """Convert threshold to filesystem-safe string."""
    exp = math.floor(math.log10(threshold))
    base = threshold / (10**exp)
    return f"{base:.2f}em{abs(exp):02d}".replace(".", "_")


def sort_models_by_size(models: List[str]) -> List[str]:
    """Sort models by size (smallest first) for early error detection."""
    return sorted(models, key=lambda m: MODEL_SIZE_ORDER.get(m, 999))


def model_safe_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def get_batch_size(model_name: str) -> int:
    return MODEL_BATCH_SIZES.get(model_name, 32)


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
    n_actual = min(len(indices), n_samples)
    if len(indices) > n_actual:
        indices = indices[:n_actual]

    pool_ids = [tok["token_id"] for tok in pool["tokens"]]

    clean_prompts, corrupt_prompts, answers, wrong_answers = [], [], [], []

    for idx in indices:
        ex = examples[idx]
        token_ids = ex["token_ids"]  # length 21, no BOS
        clean = [bos_token_id] + token_ids

        # Corrupt: replace repetition segment (positions 16-20, 0-indexed) with random tokens
        used_set = set(token_ids)
        available = [tid for tid in pool_ids if tid not in used_set]
        if len(available) >= N_SOURCE:
            replacements = rng.sample(available, N_SOURCE)
        else:
            replacements = rng.sample(pool_ids, N_SOURCE)

        corrupt = [bos_token_id] + token_ids[:16] + replacements
        assert len(corrupt) == SEQ_LEN_WITH_BOS, (
            f"corrupt len={len(corrupt)} != {SEQ_LEN_WITH_BOS}"
        )

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
    # PromptDataLoader uses drop_last=True, so batch_size must not exceed n_samples
    actual_batch_size = min(batch_size, len(indices))
    dataloader = PromptDataLoader(
        prompt_dataset=ds,
        seq_len=SEQ_LEN_WITH_BOS,
        diverge_idx=DIVERGE_IDX,
        batch_size=actual_batch_size,
    )
    return dataloader, indices


def prepare_full_dataloader(
    dataset: dict,
    pool: dict,
    bos_token_id: int,
    batch_size: int,
    seed: int,
    device: str,
) -> Tuple[Any, List[int]]:
    """
    Build AutoCircuit PromptDataLoader using ALL examples (no sampling).
    Used for evaluation to ensure consistency with base_eval.py.

    Returns:
        dataloader: PromptDataLoader for AutoCircuit
        example_indices: List of all indices in deterministic order
    """
    from auto_circuit.data import PromptDataset, PromptDataLoader

    examples = dataset["examples"]
    rng = random.Random(seed)

    # Use all examples in deterministic order
    indices = list(range(len(examples)))
    rng.shuffle(indices)  # Deterministic shuffle for consistent batch ordering

    pool_ids = [tok["token_id"] for tok in pool["tokens"]]

    clean_prompts, corrupt_prompts, answers, wrong_answers = [], [], [], []

    for idx in indices:
        ex = examples[idx]
        token_ids = ex["token_ids"]  # length 21, no BOS
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
    return dataloader, indices


def _patch_gpt_neox_config():
    """Compatibility patch for transformers >=4.48 (rotary_pct -> rope_parameters)."""
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
    """Load Pythia model via TransformerLens with hooks for circuit analysis."""
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


def load_base_metrics(
    model_name: str, band: str, split: str, base_metrics_dir: Path, draw: str = "draw_1"
) -> Dict[str, Any]:
    """
    Load pre-computed base model metrics from lsc_base_eval.py output.
    Returns metrics dict with accuracy, top5_accuracy, etc.

    Path: base_metrics_dir / {model} / {draw} / {band}.json
    """
    m_safe = model_safe_name(model_name)
    metrics_file = base_metrics_dir / m_safe / draw / f"{band}.json"

    if not metrics_file.exists():
        raise FileNotFoundError(
            f"Base metrics not found: {metrics_file}\n"
            f"Run lsc_base_eval.py first to pre-compute base model metrics."
        )

    with open(metrics_file) as f:
        data = json.load(f)

    splits = data.get("splits", {})
    if split not in splits:
        raise KeyError(f"Split '{split}' not found in {metrics_file}")

    return splits[split]


def compute_accuracy_metrics(
    logits: t.Tensor,
    answer_ids: List[int],
) -> Dict[str, Any]:
    """
    Compute accuracy metrics (top-1/5/10, mean P(correct)).
    Does NOT compute KL divergence - use compute_kl_divergence() for that.
    """
    if len(logits.shape) == 3:
        logits = logits[:, -1, :]

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

    return {
        "accuracy": top1 / n if n else 0.0,
        "top5_accuracy": top5 / n if n else 0.0,
        "top10_accuracy": top10 / n if n else 0.0,
        "mean_correct_prob": float(np.mean(correct_probs)) if correct_probs else 0.0,
        "n_samples": n,
    }


def compute_kl_divergence(
    circuit_logits: t.Tensor,
    base_logits: t.Tensor,
) -> float:
    """
    Compute KL(base || circuit) - measures circuit faithfulness to base model.
    Lower KL = more faithful circuit.
    """
    if len(circuit_logits.shape) == 3:
        circuit_logits = circuit_logits[:, -1, :]
    if len(base_logits.shape) == 3:
        base_logits = base_logits[:, -1, :]

    circuit_probs = F.softmax(circuit_logits, dim=-1)
    base_probs = F.softmax(base_logits, dim=-1)

    k = min(circuit_probs.shape[0], base_probs.shape[0])
    eps = 1e-10

    kl = (
        (
            base_probs[:k]
            * (t.log(base_probs[:k] + eps) - t.log(circuit_probs[:k] + eps))
        )
        .sum(-1)
        .mean()
    )
    return kl.item()


def compute_base_logits(
    model,
    dataset: dict,
    pool: dict,
    bos_id: int,
    batch_size: int,
    eval_seed: int,
    device: str,
    use_bf16_eval: bool = False,
) -> Tuple[t.Tensor, List[int], List[int]]:
    """
    Run full model on FULL dataset (no sampling).
    Returns (last-position logits in FP32, answer token IDs aligned with logits, example indices).
    """
    loader, indices = prepare_full_dataloader(
        dataset, pool, bos_id, batch_size, eval_seed, device
    )
    logits_list = []
    # CRITICAL: Collect answer_ids from batches to ensure alignment with logits
    aligned_answer_ids = []

    eval_dtype = t.bfloat16 if use_bf16_eval else t.float32
    with (
        t.no_grad(),
        t.autocast(device_type="cuda", dtype=eval_dtype, enabled=use_bf16_eval),
    ):
        for batch in loader:
            logits = model(batch.clean)
            if len(logits.shape) == 3:
                logits = logits[:, -1, :]
            # Keep in FP32 for metrics computation
            logits_list.append(logits.float())
            # Extract answers from batch (tensor shape: [batch_size, 1])
            aligned_answer_ids.extend(batch.answers.squeeze(-1).tolist())

    return t.cat(logits_list, dim=0), aligned_answer_ids, indices


def run_circuit_and_collect(
    patchable,
    prune_scores_dev: dict,
    n_edges: int,
    dataset: dict,
    pool: dict,
    bos_id: int,
    batch_size: int,
    eval_seed: int,
    device: str,
    use_bf16_eval: bool = False,
) -> Tuple[t.Tensor, List[int]]:
    """
    Run circuit via AutoCircuit on FULL dataset (no sampling).
    Returns (circuit logits in FP32, answer_ids aligned with logits).

    Creates two dataloaders with the same seed: one consumed by run_circuits,
    one to match batch keys to answer tokens.
    """
    from auto_circuit.prune import run_circuits
    from auto_circuit.types import PatchType, AblationType

    set_all_seeds(eval_seed)
    loader1, _ = prepare_full_dataloader(
        dataset, pool, bos_id, batch_size, eval_seed, device
    )

    eval_dtype = t.bfloat16 if use_bf16_eval else t.float32
    with t.autocast(device_type="cuda", dtype=eval_dtype, enabled=use_bf16_eval):
        outputs = run_circuits(
            model=patchable,
            dataloader=loader1,
            test_edge_counts=[n_edges],
            prune_scores=prune_scores_dev,
            patch_type=PatchType.TREE_PATCH,
            ablation_type=AblationType.RESAMPLE,
        )

    # Second loader (same seed) to extract answer IDs in batch order
    set_all_seeds(eval_seed)
    loader2, _ = prepare_full_dataloader(
        dataset, pool, bos_id, batch_size, eval_seed, device
    )

    # Extract circuit logits aligned by batch.key (hash of clean+corrupt tensors)
    logits_list = []
    aligned_answer_ids = []

    for batch in loader2:
        logits = outputs[n_edges][batch.key]
        if len(logits.shape) == 3:
            logits = logits[:, -1, :]
        # Keep in FP32 for metrics computation
        logits_list.append(logits.float())
        # Extract answers from batch (tensor shape: [batch_size, 1])
        aligned_answer_ids.extend(batch.answers.squeeze(-1).tolist())

    return t.cat(logits_list, dim=0), aligned_answer_ids


def invert_prune_scores(prune_scores: dict) -> Tuple[dict, int]:
    """
    Invert prune scores for ablation: circuit edges -> 0, pruned -> inf.
    Returns (inverted scores, number of edges in complement).
    """
    inverted = {}
    for name, scores in prune_scores.items():
        inv = scores.clone()
        is_circuit = t.isinf(scores)
        inv[is_circuit] = 0.0
        inv[~is_circuit] = float("inf")
        inverted[name] = inv
    n_abl = sum(t.isinf(s).sum().item() for s in inverted.values())
    return inverted, n_abl


def evaluate_on_band(
    model,
    patchable,
    prune_scores_dev: dict,
    n_edges: int,
    band: str,
    model_name: str,
    config: DiscoveryConfig,
    device: str,
    draw: str = "draw_1",
) -> Dict[str, Any]:
    """
    Evaluate a circuit on a single band's FULL TEST data (no sampling).
    Returns dict with base_metrics (from pre-computed file) and circuit_metrics.
    """
    pool_dir = Path(config.pool_dir)
    data_dir = Path(config.data_dir)
    base_metrics_dir = Path(config.base_metrics_dir)
    bos_id = model.tokenizer.bos_token_id
    batch_size = get_batch_size(model_name)

    try:
        pool = load_pool(band, pool_dir)
        test_data = load_dataset(band, "test", data_dir, config.variant, draw)
        # Load pre-computed base metrics
        base_metrics = load_base_metrics(
            model_name, band, "test", base_metrics_dir, draw=draw
        )
    except FileNotFoundError as e:
        return {"error": str(e)}

    # Base model logits on FULL test (needed for KL divergence computation)
    base_logits, _, _ = compute_base_logits(
        model,
        test_data,
        pool,
        bos_id,
        batch_size,
        config.eval_seed,
        device,
        use_bf16_eval=config.use_bf16_eval,
    )

    # Circuit logits on FULL test
    if n_edges > 0:
        circ_logits, circ_answers = run_circuit_and_collect(
            patchable,
            prune_scores_dev,
            n_edges,
            test_data,
            pool,
            bos_id,
            batch_size,
            config.eval_seed,
            device,
            use_bf16_eval=config.use_bf16_eval,
        )
        circuit_metrics = compute_accuracy_metrics(circ_logits, circ_answers)
        circuit_metrics["kl_div"] = compute_kl_divergence(circ_logits, base_logits)
    else:
        circuit_metrics = {
            "accuracy": 0.0,
            "kl_div": float("inf"),
            "n_samples": 0,
            "error": "empty_circuit",
        }

    return {"base": base_metrics, "circuit": circuit_metrics}


def load_thresholds(sweep_dir: Path, models: List[str]) -> Dict[str, float]:
    """
    Load per-model selected thresholds from Phase 1 threshold_summary.json.
    Returns {model_name: threshold} for models that have selections.
    Models without thresholds are warned about and skipped.
    """
    logger = logging.getLogger("circuit_discovery")
    summary_path = sweep_dir / "sweep_results" / "threshold_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"threshold_summary.json not found at {summary_path}.\n"
            f"Run lsc_pareto_sweep.py first, or use --threshold to override."
        )

    with open(summary_path) as f:
        summary = json.load(f)

    selections = summary.get("selections", {})
    thresholds = {}
    skipped = []
    for model in models:
        if model in selections:
            thresholds[model] = selections[model]["threshold"]
        else:
            skipped.append(model)

    if skipped:
        logger.warning(
            f"No threshold for {skipped} in {summary_path}; "
            f"these models will be skipped. "
            f"Available: {list(selections.keys())}"
        )

    if not thresholds:
        raise KeyError(
            f"No thresholds found for any requested model in {summary_path}.\n"
            f"Requested: {models}\nAvailable: {list(selections.keys())}"
        )

    return thresholds


# =============================================================================
# SINGLE DISCOVERY TASK
# =============================================================================


def run_discovery_task(
    model_name: str,
    band: str,
    draw: str,
    threshold: float,
    config: DiscoveryConfig,
    device: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Single circuit discovery task.  Loads model ONCE, then:
      1. ACDC on TRAIN split at τ*      ->  prune_scores
      2. Base model on TEST split        ->  base_metrics
      3. Circuit on TEST split           ->  circuit_metrics
      4. Ablation on TEST split          ->  ablation_metrics
      5. Cross-band transfer (optional)  ->  transfer[other_band]

    Uses `config.acdc_seed` for ACDC training randomness (draws provide variation).
    Uses `config.eval_seed` for all evaluations (consistent test set).
    """
    from auto_circuit.prune_algos.ACDC import acdc_prune_scores
    from auto_circuit.utils.graph_utils import patchable_model

    logger = logging.getLogger("circuit_discovery")
    batch_size = get_batch_size(model_name)
    tao_exp, tao_base = threshold_to_tao(threshold)

    logger.info(f"[{device}] Task: {model_name} / {band} / {draw} / tau={threshold}")

    model = None
    patchable = None
    prune_scores_cpu = None

    try:
        # ==== 1. ACDC on TRAIN split (sampled with seed control) ====
        set_all_seeds(config.acdc_seed)
        model = load_model(model_name, device)
        bos_id = model.tokenizer.bos_token_id

        pool_dir = Path(config.pool_dir)
        data_dir = Path(config.data_dir)
        pool = load_pool(band, pool_dir)
        train_data = load_dataset(band, "train", data_dir, config.variant, draw)

        train_loader, train_indices = prepare_dataloader(
            train_data,
            pool,
            bos_id,
            n_samples=config.acdc_train_size,
            batch_size=batch_size,
            seed=config.acdc_seed,
            device=device,
        )

        patchable = patchable_model(
            model=model,
            factorized=config.factorized,
            slice_output=config.slice_output,
            seq_len=None,
            separate_qkv=config.separate_qkv,
            device=device,
        )
        total_edges = len(patchable.edges)

        logger.info(
            f"[{device}] ACDC: {band}/{draw}, tau={threshold} "
            f"({tao_base}x10^{tao_exp}), {total_edges} edges"
        )
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
        n_edges = sum(t.isinf(s).sum().item() for s in prune_scores.values())
        total_possible = sum(s.numel() for s in prune_scores.values())
        size_fraction = n_edges / total_possible if total_possible else 0.0

        logger.info(
            f"[{device}] ACDC done: {n_edges}/{total_possible} edges "
            f"({size_fraction:.1%}), {training_time:.1f}s"
        )

        # Save prune scores to CPU, free GPU copy
        prune_scores_cpu = {k: v.cpu() for k, v in prune_scores.items()}
        del prune_scores
        cleanup_gpu()

        # Persist to disk (async save in background thread)
        m_safe = model_safe_name(model_name)
        circuit_dir = output_dir / "circuits" / m_safe / band / draw
        circuit_dir.mkdir(parents=True, exist_ok=True)
        scores_path = circuit_dir / "prune_scores.pkl"

        def save_scores(path, data):
            with open(path, "wb") as f:
                pickle.dump(data, f)

        threading.Thread(
            target=save_scores, args=(scores_path, prune_scores_cpu), daemon=True
        ).start()

        # ==== 2-4. Evaluate on same band (FULL TEST split) ====
        # Non-blocking transfer to GPU
        prune_scores_dev = {
            k: v.to(device, non_blocking=True) for k, v in prune_scores_cpu.items()
        }
        t.cuda.synchronize()

        # Load pre-computed base metrics
        base_metrics_dir = Path(config.base_metrics_dir)
        base_metrics = load_base_metrics(
            model_name, band, "test", base_metrics_dir, draw=draw
        )

        # Base model logits on FULL test (needed for KL divergence computation)
        test_data = load_dataset(band, "test", data_dir, config.variant, draw)
        base_logits, _, test_indices = compute_base_logits(
            model,
            test_data,
            pool,
            bos_id,
            batch_size,
            config.eval_seed,
            device,
            use_bf16_eval=config.use_bf16_eval,
        )

        # Circuit on FULL test
        if n_edges > 0:
            circ_logits, circ_answers = run_circuit_and_collect(
                patchable,
                prune_scores_dev,
                n_edges,
                test_data,
                pool,
                bos_id,
                batch_size,
                config.eval_seed,
                device,
                use_bf16_eval=config.use_bf16_eval,
            )
            circuit_metrics = compute_accuracy_metrics(circ_logits, circ_answers)
            circuit_metrics["kl_div"] = compute_kl_divergence(circ_logits, base_logits)
        else:
            circuit_metrics = {
                "accuracy": 0.0,
                "kl_div": float("inf"),
                "n_samples": 0,
                "error": "empty_circuit",
            }

        # Ablation (complement circuit) on FULL test
        abl_scores, n_abl_edges = invert_prune_scores(prune_scores_dev)
        if n_abl_edges > 0:
            abl_logits, abl_answers = run_circuit_and_collect(
                patchable,
                abl_scores,
                n_abl_edges,
                test_data,
                pool,
                bos_id,
                batch_size,
                config.eval_seed,
                device,
                use_bf16_eval=config.use_bf16_eval,
            )
            ablation_metrics = compute_accuracy_metrics(abl_logits, abl_answers)
            ablation_metrics["kl_div"] = compute_kl_divergence(abl_logits, base_logits)
        else:
            ablation_metrics = {
                "accuracy": 0.0,
                "kl_div": float("inf"),
                "n_samples": 0,
                "error": "empty_ablation",
            }

        # Necessity check: ablation accuracy should be < 50% of base accuracy
        necessity = (
            "PASS"
            if ablation_metrics.get("accuracy", 1) < (base_metrics["accuracy"] * 0.5)
            else "WARN"
        )

        logger.info(
            f"[{device}] Same-band {band}: "
            f"base={base_metrics['accuracy']:.1%}, "
            f"circuit={circuit_metrics.get('accuracy', 0):.1%}, "
            f"KL={circuit_metrics.get('kl_div', '?'):.4f}, "
            f"ablation={ablation_metrics.get('accuracy', 0):.1%}, "
            f"necessity={necessity}"
        )

        # ==== 5. Cross-band transfer ====
        transfer = {}
        if config.cross_band:
            for other_band in config.bands:
                if other_band == band:
                    transfer[other_band] = {
                        "base": base_metrics,
                        "circuit": circuit_metrics,
                    }
                    continue

                logger.debug(f"[{device}] Cross-band: {band}->{other_band}")
                xb_result = evaluate_on_band(
                    model,
                    patchable,
                    prune_scores_dev,
                    n_edges,
                    other_band,
                    model_name,
                    config,
                    device,
                    draw=draw,
                )
                transfer[other_band] = xb_result

            # Log cross-band summary
            for ob, xb in transfer.items():
                if ob == band or "error" in xb:
                    continue
                xb_acc = xb.get("circuit", {}).get("accuracy", 0)
                xb_base = xb.get("base", {}).get("accuracy", 0)
                logger.info(
                    f"[{device}]   {band}->{ob}: "
                    f"circuit={xb_acc:.1%}, base={xb_base:.1%}"
                )

        # ==== Package result ====
        result = {
            "model": model_name,
            "band": band,
            "draw": draw,
            "threshold": threshold,
            "n_edges": n_edges,
            "total_edges": total_possible,
            "size_fraction": size_fraction,
            "training_time_seconds": training_time,
            "base_metrics": base_metrics,
            "circuit_metrics": circuit_metrics,
            "ablation_metrics": ablation_metrics,
            "necessity_test": necessity,
            "prune_scores_file": str(scores_path.relative_to(output_dir)),
            # Reproducibility tracking
            "acdc_seed": config.acdc_seed,
            "eval_seed": config.eval_seed,
            "acdc_train_size": config.acdc_train_size,
            "train_indices": train_indices,  # Sampled indices for ACDC
            "test_n_samples": len(test_indices),  # Full TEST split size
            "full_test_evaluation": True,  # Flag indicating no sampling for eval
            "use_bf16_eval": config.use_bf16_eval,
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
        }
        if transfer:
            result["cross_band"] = transfer

        # Save per-task metrics alongside prune scores
        with open(circuit_dir / "metrics.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        return result

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"[{device}] FAILED: {model_name}/{band}/{draw}: {error_msg}")
        return {
            "model": model_name,
            "band": band,
            "draw": draw,
            "threshold": threshold,
            "status": "failed",
            "error": error_msg,
            "failed_at": datetime.now().isoformat(),
        }

    finally:
        if patchable is not None:
            del patchable
        safe_delete_model(model)
        if prune_scores_cpu is not None:
            del prune_scores_cpu
        cleanup_gpu()


class Registry:
    """Atomic JSON registry with file locking for concurrent GPU workers."""

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
    def make_task_id(model: str, band: str, draw: str) -> str:
        m = model_safe_name(model)
        return f"{m}__{band}__{draw}"


def gpu_worker(
    gpu_id,
    task_queue,
    result_queue,
    config_dict,
    thresholds_dict,
    output_dir_str,
    progress_dict,
    worker_id,
    heartbeat_dict,
):
    """
    Process tasks from queue on a single GPU.

    """
    config = DiscoveryConfig(**config_dict)
    output_dir = Path(output_dir_str)
    device = f"cuda:{gpu_id}"
    logger = logging.getLogger("circuit_discovery")

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

        if task is None:  # poison pill
            logger.info(
                f"[Worker {worker_id}] Received shutdown signal. Completed {tasks_completed} tasks."
            )
            progress_dict[worker_id] = (
                f"GPU {gpu_id}: shutdown (completed {tasks_completed})"
            )
            break

        progress_dict[worker_id] = (
            f"GPU {gpu_id}: {task['model']}/{task['band']}/{task['draw']}"
        )
        heartbeat_dict[worker_id] = time.time()

        logger.info(
            f"[Worker {worker_id}] Starting task: {task['model']}/{task['band']}/{task['draw']}"
        )

        try:
            result = run_discovery_task(
                model_name=task["model"],
                band=task["band"],
                draw=task["draw"],
                threshold=thresholds_dict[task["model"]],
                config=config,
                device=device,
                output_dir=output_dir,
            )
            registry.add_task(task["id"], result)
            result_queue.put(("success", task, result, None))
            tasks_completed += 1
            logger.info(
                f"[Worker {worker_id}] Completed: {task['model']}/{task['band']}/{task['draw']} ({tasks_completed} total)"
            )

        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(
                f"[Worker {worker_id}] FAILED: {task['model']}/{task['band']}/{task['draw']}\n{error_msg}"
            )

            try:
                err = {
                    "model": task["model"],
                    "band": task["band"],
                    "draw": task["draw"],
                    "status": "failed",
                    "error": error_msg,
                    "failed_at": datetime.now().isoformat(),
                }
                registry.add_task(task["id"], err)
            except Exception as reg_err:
                logger.error(f"[Worker {worker_id}] Registry write failed: {reg_err}")

            result_queue.put(("error", task, None, error_msg))
            cleanup_gpu()

        progress_dict[worker_id] = f"GPU {gpu_id}: idle (completed {tasks_completed})"
        heartbeat_dict[worker_id] = time.time()

    # Final cleanup
    cleanup_gpu()
    logger.info(f"[Worker {worker_id}] Exiting cleanly")


def run_all_tasks(
    config: DiscoveryConfig,
    thresholds: Dict[str, float],
    output_dir: Path,
    gpus: List[int],
    force: bool = False,
) -> Dict[str, int]:
    """
    Generate and execute all (model, band, draw) tasks.

    Returns:
        Dict with 'completed', 'failed', 'total' counts
    """
    logger = logging.getLogger("circuit_discovery")
    registry = Registry(output_dir / config.registry_file)

    # Sort models by size (smallest first) for early error detection
    sorted_models = sort_models_by_size(config.models)
    logger.info(f"Model order (smallest first): {sorted_models}")

    # Generate tasks: draws (outermost) x models (smallest first) x bands
    # Completes all models/bands for draw_1 before moving to draw_2, etc.
    all_tasks = []
    for draw in config.draws:
        for model in sorted_models:
            for band in config.bands:
                all_tasks.append(
                    {
                        "model": model,
                        "band": band,
                        "draw": draw,
                        "id": Registry.make_task_id(model, band, draw),
                        "estimated_minutes": ESTIMATED_MINUTES.get(model, 60),
                    }
                )

    logger.info(f"Total tasks: {len(all_tasks)}")

    # Filter completed
    if force:
        tasks = all_tasks
    else:
        completed_ids = registry.get_completed_ids()
        tasks = [tk for tk in all_tasks if tk["id"] not in completed_ids]
        logger.info(f"Already completed: {len(all_tasks) - len(tasks)}")

    already_completed = len(all_tasks) - len(tasks)
    logger.info(f"Remaining: {len(tasks)}")
    if not tasks:
        return {"completed": already_completed, "failed": 0, "total": len(all_tasks)}

    n_gpus = len(gpus)

    # ---- Single GPU: sequential ----
    if n_gpus <= 1:
        device = f"cuda:{gpus[0]}" if gpus else "cpu"
        logger.info(f"Sequential execution on {device}")
        t_start = time.time()

        seq_completed = 0
        seq_failed = 0

        for i, task in enumerate(tasks):
            if i > 0:
                elapsed = time.time() - t_start
                avg = elapsed / i
                eta = timedelta(seconds=int(avg * (len(tasks) - i)))
                logger.info(
                    f"Progress: {i}/{len(tasks)} ({100 * i / len(tasks):.0f}%) ETA: {eta}"
                )

            try:
                result = run_discovery_task(
                    model_name=task["model"],
                    band=task["band"],
                    draw=task["draw"],
                    threshold=thresholds[task["model"]],
                    config=config,
                    device=device,
                    output_dir=output_dir,
                )
                registry.add_task(task["id"], result)
                if result.get("status") == "completed":
                    seq_completed += 1
                else:
                    seq_failed += 1
            except Exception as e:
                logger.error(
                    f"Task failed: {task['model']}/{task['band']}/{task['draw']}: {e}"
                )
                seq_failed += 1

        return {
            "completed": already_completed + seq_completed,
            "failed": seq_failed,
            "total": len(all_tasks),
        }

    # ---- Multi-GPU: parallel ----
    logger.info(f"Parallel execution on {n_gpus} GPUs: {gpus}")
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    manager = ctx.Manager()
    progress_dict = manager.dict()
    heartbeat_dict = manager.dict()  # Track worker liveness

    for task in tasks:
        task_queue.put(task)
    for _ in gpus:
        task_queue.put(None)  # poison pills

    config_dict = asdict(config)
    workers = []
    for i, gpu_id in enumerate(gpus):
        progress_dict[i] = f"GPU {gpu_id}: starting"
        heartbeat_dict[i] = time.time()
        p = ctx.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                task_queue,
                result_queue,
                config_dict,
                thresholds,
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
    total_tasks = len(tasks)
    last_progress_log = time.time()
    # Large models can take 4+ hours per task in ACDC
    # Heartbeat only updates between tasks, not during ACDC
    HEARTBEAT_TIMEOUT = 14400  # 4 hours
    PROGRESS_LOG_INTERVAL = 60  # Log progress every minute
    heartbeat_warned = set()

    logger.info(f"Waiting for {total_tasks} tasks across {n_gpus} workers...")

    while completed_tasks + failed_tasks < total_tasks:
        # Check for results (non-blocking with short timeout)
        try:
            result = result_queue.get(timeout=10)
            status, task, data, error = result

            if status == "success":
                completed_tasks += 1
                logger.info(
                    f"Task completed: {task['model']}/{task['band']}/{task['draw']} "
                    f"[{completed_tasks}/{total_tasks}]"
                )
            else:
                failed_tasks += 1
                logger.error(
                    f"Task FAILED: {task['model']}/{task['band']}/{task['draw']} "
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

    # Wait for workers to finish cleanly
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
        "total": len(all_tasks),
    }


def aggregate_results(config: DiscoveryConfig, output_dir: Path) -> dict:
    """
    Aggregate registry results across draws.
    Computes mean +/- std for all metrics per (model, band).
    Builds cross-band transfer matrix.
    """
    logger = logging.getLogger("circuit_discovery")
    registry = Registry(output_dir / config.registry_file)
    reg = registry.load()

    completed = {
        tid: d
        for tid, d in reg.get("tasks", {}).items()
        if d.get("status") == "completed"
    }
    failed = {
        tid: d for tid, d in reg.get("tasks", {}).items() if d.get("status") == "failed"
    }

    logger.info(f"Registry: {len(completed)} completed, {len(failed)} failed")

    # Group by (model, band)
    by_mb = defaultdict(list)
    for tid, data in completed.items():
        by_mb[(data["model"], data["band"])].append(data)

    summary = OrderedDict()
    summary["created_at"] = datetime.now().isoformat()
    summary["n_completed"] = len(completed)
    summary["n_failed"] = len(failed)

    def _agg(values):
        """Aggregate a list of numbers to mean +/- std."""
        if not values:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n": len(values),
        }

    model_band_results = OrderedDict()
    transfer_matrices = {}  # model -> {train_band -> {test_band -> accuracy_stats}}

    for model in config.models:
        model_results = OrderedDict()
        transfer_matrices[model] = {}

        for band in config.bands:
            results = by_mb.get((model, band), [])
            if not results:
                continue

            entry = OrderedDict()
            entry["n_draws"] = len(results)
            entry["threshold"] = results[0].get("threshold")

            # Same-band metrics (across draws)
            entry["circuit_accuracy"] = _agg(
                [r["circuit_metrics"]["accuracy"] for r in results]
            )
            entry["circuit_kl"] = _agg(
                [r["circuit_metrics"].get("kl_div", 0) for r in results]
            )
            entry["circuit_top5"] = _agg(
                [r["circuit_metrics"].get("top5_accuracy", 0) for r in results]
            )
            entry["base_accuracy"] = _agg(
                [r["base_metrics"]["accuracy"] for r in results]
            )
            entry["ablation_accuracy"] = _agg(
                [r["ablation_metrics"].get("accuracy", 0) for r in results]
            )
            entry["size_fraction"] = _agg([r["size_fraction"] for r in results])
            entry["n_edges"] = _agg([r["n_edges"] for r in results])
            entry["training_time"] = _agg([r["training_time_seconds"] for r in results])

            # Necessity test summary
            necessity_pass = sum(
                1 for r in results if r.get("necessity_test") == "PASS"
            )
            entry["necessity_pass_rate"] = (
                necessity_pass / len(results) if results else 0
            )

            # Cross-band transfer
            if config.cross_band:
                transfer_row = {}
                for other_band in config.bands:
                    xb_accs = []
                    for r in results:
                        xb = r.get("cross_band", {}).get(other_band, {})
                        acc = xb.get("circuit", {}).get("accuracy")
                        if acc is not None:
                            xb_accs.append(acc)
                    transfer_row[other_band] = _agg(xb_accs)
                entry["transfer"] = transfer_row
                transfer_matrices[model][band] = transfer_row

            model_results[band] = entry

        if model_results:
            model_band_results[model] = model_results

    summary["results"] = model_band_results

    # Save
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    with open(summary_dir / "discovery_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if transfer_matrices:
        with open(summary_dir / "cross_band_transfer.json", "w") as f:
            json.dump(transfer_matrices, f, indent=2)

    return summary


def print_summary_table(summary: dict, config: DiscoveryConfig):
    """Print concise results table to logger."""
    logger = logging.getLogger("circuit_discovery")

    logger.info("\n" + "=" * 100)
    logger.info("CIRCUIT DISCOVERY SUMMARY")
    logger.info("=" * 100)

    results = summary.get("results", {})
    for model in config.models:
        model_data = results.get(model)
        if not model_data:
            continue

        logger.info(f"\n--- {model} ---")
        logger.info(
            f"  {'Band':<14s} {'Draws':>5s} {'Size%':>7s} "
            f"{'CircAcc':>8s} {'KL':>8s} {'AblAcc':>8s} "
            f"{'BaseAcc':>8s} {'Nec%':>5s} {'Edges':>8s}"
        )
        logger.info("  " + "-" * 85)

        for band in config.bands:
            entry = model_data.get(band)
            if not entry:
                continue

            logger.info(
                f"  {band:<14s} {entry['n_draws']:>5d} "
                f"{entry['size_fraction']['mean']:>6.1%} "
                f"{entry['circuit_accuracy']['mean']:>7.1%} "
                f"{entry['circuit_kl']['mean']:>8.4f} "
                f"{entry['ablation_accuracy']['mean']:>7.1%} "
                f"{entry['base_accuracy']['mean']:>7.1%} "
                f"{entry['necessity_pass_rate']:>4.0%} "
                f"{entry['n_edges']['mean']:>7.0f}"
            )


def plot_summary(summary: dict, config: DiscoveryConfig, output_dir: Path):
    """Generate summary plots: per-model accuracy bars and transfer heatmaps."""
    logger = logging.getLogger("circuit_discovery")
    plot_dir = output_dir / "summary" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = summary.get("results", {})

    for model in config.models:
        model_data = results.get(model)
        if not model_data:
            continue

        m_safe = model_safe_name(model)
        bands_present = [b for b in config.bands if b in model_data]
        if not bands_present:
            continue

        # ---- Plot 1: Accuracy bar chart ----
        try:
            fig, ax = plt.subplots(figsize=(12, 5))

            x = np.arange(len(bands_present))
            w = 0.25

            base_means = [model_data[b]["base_accuracy"]["mean"] for b in bands_present]
            circ_means = [
                model_data[b]["circuit_accuracy"]["mean"] for b in bands_present
            ]
            circ_stds = [
                model_data[b]["circuit_accuracy"]["std"] for b in bands_present
            ]
            abl_means = [
                model_data[b]["ablation_accuracy"]["mean"] for b in bands_present
            ]
            abl_stds = [
                model_data[b]["ablation_accuracy"]["std"] for b in bands_present
            ]

            ax.bar(
                x - w, base_means, w, label="Base model", color="forestgreen", alpha=0.8
            )
            ax.bar(
                x,
                circ_means,
                w,
                yerr=circ_stds,
                label="Circuit",
                color="steelblue",
                alpha=0.8,
                capsize=3,
            )
            ax.bar(
                x + w,
                abl_means,
                w,
                yerr=abl_stds,
                label="Ablation",
                color="coral",
                alpha=0.8,
                capsize=3,
            )

            ax.set_xticks(x)
            ax.set_xticklabels(bands_present, rotation=45, ha="right")
            ax.set_ylabel("Top-1 Accuracy")
            ax.set_title(f"{model}: Circuit Performance by Band")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(0, 1.05)

            fig.tight_layout()
            fig.savefig(
                plot_dir / f"{m_safe}_accuracy.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Accuracy plot failed for {model}: {e}")

        # ---- Plot 2: Cross-band transfer heatmap ----
        if not config.cross_band:
            continue

        try:
            # Build transfer matrix
            matrix = np.zeros((len(bands_present), len(bands_present)))
            for i, train_band in enumerate(bands_present):
                transfer = model_data[train_band].get("transfer", {})
                for j, test_band in enumerate(bands_present):
                    cell = transfer.get(test_band, {})
                    matrix[i, j] = cell.get("mean", 0)

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(
                matrix,
                annot=True,
                fmt=".1%",
                cmap="YlOrRd",
                xticklabels=bands_present,
                yticklabels=bands_present,
                vmin=0,
                vmax=1,
                ax=ax,
            )
            ax.set_xlabel("Test Band")
            ax.set_ylabel("Training Band")
            ax.set_title(f"{model}: Cross-Band Transfer (Circuit Accuracy)")

            fig.tight_layout()
            fig.savefig(
                plot_dir / f"{m_safe}_transfer.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Transfer heatmap failed for {model}: {e}")

        # ---- Plot 3: Circuit size comparison ----
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Left: edge count by band
            ax = axes[0]
            edge_means = [model_data[b]["n_edges"]["mean"] for b in bands_present]
            edge_stds = [model_data[b]["n_edges"]["std"] for b in bands_present]
            ax.barh(
                bands_present,
                edge_means,
                xerr=edge_stds,
                color="steelblue",
                alpha=0.8,
                capsize=3,
            )
            ax.set_xlabel("Number of Edges")
            ax.set_title(f"{model}: Circuit Size")
            ax.grid(axis="x", alpha=0.3)

            # Right: size fraction by band
            ax = axes[1]
            frac_means = [model_data[b]["size_fraction"]["mean"] for b in bands_present]
            frac_stds = [model_data[b]["size_fraction"]["std"] for b in bands_present]
            ax.barh(
                bands_present,
                frac_means,
                xerr=frac_stds,
                color="coral",
                alpha=0.8,
                capsize=3,
            )
            ax.set_xlabel("Edge Fraction")
            ax.set_title(f"{model}: Circuit Size Fraction")
            ax.grid(axis="x", alpha=0.3)

            fig.tight_layout()
            fig.savefig(plot_dir / f"{m_safe}_size.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Size plot failed for {model}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="LSC Circuit Discovery; Phase 2: discover circuits across all bands",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to process (default: all Pythia 70m-1b)",
    )
    parser.add_argument(
        "--bands", nargs="+", default=None, help="Bands to process (default: all 8)"
    )
    parser.add_argument(
        "--draws",
        nargs="+",
        type=str,
        default=None,
        help="Dataset draws for replication (default: draw_1 draw_2 draw_3)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override: use this threshold for ALL models",
    )
    parser.add_argument(
        "--sweep-dir",
        type=str,
        default=None,
        help="Directory with threshold_summary.json from Phase 1",
    )
    parser.add_argument(
        "--gpus", nargs="+", default=["auto"], help="GPU devices: 'auto' or list of IDs"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=256,
        help="ACDC training examples sampled from train split (default: 256)",
    )
    parser.add_argument(
        "--no-cross-band",
        action="store_true",
        help="Skip cross-band transfer evaluation",
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--pool-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--base-metrics-dir",
        type=str,
        default=None,
        help="Directory with pre-computed base metrics from lsc_base_eval.py",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="matched",
        help="Dataset variant: 'matched' or 'unmatched'",
    )
    parser.add_argument(
        "--force", action="store_true", help="Recompute all tasks (ignore registry)"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only aggregate and plot existing results",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable BF16 for evaluation (default: FP32, BF16 causes accuracy issues)",
    )
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # ---- Build config ----
    config = DiscoveryConfig()
    if args.models:
        config.models = args.models
    if args.bands:
        config.bands = args.bands
    if args.draws:
        config.draws = args.draws
    if args.threshold is not None:
        config.threshold_override = args.threshold
    if args.sweep_dir:
        config.sweep_dir = args.sweep_dir
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.pool_dir:
        config.pool_dir = args.pool_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.base_metrics_dir:
        config.base_metrics_dir = args.base_metrics_dir
    config.variant = args.variant
    config.acdc_train_size = args.train_size
    config.cross_band = not args.no_cross_band
    config.use_bf16_eval = args.bf16

    if args.gpus == ["auto"]:
        config.gpus = (
            list(range(t.cuda.device_count())) if t.cuda.is_available() else []
        )
    else:
        config.gpus = [int(g) for g in args.gpus]

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir, args.debug)

    # ---- Load per-model thresholds ----
    if config.threshold_override is not None:
        thresholds = {m: config.threshold_override for m in config.models}
        logger.info(f"Threshold override: {config.threshold_override} for all models")
    else:
        sweep_dir = Path(config.sweep_dir)
        thresholds = load_thresholds(sweep_dir, config.models)
        # Filter models to only those with thresholds
        skipped = [m for m in config.models if m not in thresholds]
        if skipped:
            logger.warning(f"Skipping models without thresholds: {skipped}")
            config.models = [m for m in config.models if m in thresholds]
        logger.info(f"Loaded thresholds from {sweep_dir}")

    # ---- Header ----
    n_tasks = len(config.models) * len(config.bands) * len(config.draws)

    logger.info("=" * 70)
    logger.info("LSC CIRCUIT DISCOVERY (Phase 2 - Final Evaluation)")
    logger.info("=" * 70)
    logger.info(f"Models:     {config.models}")
    logger.info(f"Bands:      {config.bands}")
    logger.info(f"Draws:      {config.draws}")
    logger.info(f"Cross-band: {config.cross_band}")
    logger.info(f"Precision:  {'BF16' if config.use_bf16_eval else 'FP32'}")
    logger.info(f"GPUs:       {config.gpus}")
    logger.info(f"ACDC train: {config.acdc_train_size} (sampled from TRAIN split)")
    logger.info(f"Eval:       FULL TEST split (no sampling)")
    logger.info(f"Data:       {config.data_dir}")
    logger.info(f"Base met.:  {config.base_metrics_dir}")
    logger.info(f"Output:     {config.output_dir}")
    logger.info(f"Tasks:      {n_tasks}")
    logger.info(f"Thresholds:")
    for m, tau in thresholds.items():
        logger.info(f"  {m}: {tau}")
    logger.info("=" * 70)

    # ---- Analyze-only mode ----
    if args.analyze_only:
        summary = aggregate_results(config, output_dir)
        print_summary_table(summary, config)
        plot_summary(summary, config, output_dir)
        logger.info(f"\nSummary:  {output_dir / 'summary' / 'discovery_summary.json'}")
        return 0

    # ---- Validate data exists ----
    pool_dir = Path(config.pool_dir)
    data_dir = Path(config.data_dir)
    base_metrics_dir = Path(config.base_metrics_dir)
    missing = []

    # Check pool and dataset files
    for band in config.bands:
        if not (pool_dir / f"lsc_pool_{band}.json").exists():
            missing.append(f"pool: lsc_pool_{band}.json")
        for draw in config.draws:
            if not (
                data_dir / "datasets" / config.variant / draw / band / "train.json"
            ).exists():
                missing.append(
                    f"data: datasets/{config.variant}/{draw}/{band}/train.json"
                )
            if not (
                data_dir / "datasets" / config.variant / draw / band / "test.json"
            ).exists():
                missing.append(
                    f"data: datasets/{config.variant}/{draw}/{band}/test.json"
                )

    # Check pre-computed base metrics (required, draw-aware paths)
    for model in config.models:
        m_safe = model_safe_name(model)
        for draw in config.draws:
            for band in config.bands:
                metrics_file = base_metrics_dir / m_safe / draw / f"{band}.json"
                if not metrics_file.exists():
                    missing.append(f"base_metrics: {m_safe}/{draw}/{band}.json")

    if missing:
        for m in missing[:20]:  # show first 20
            logger.error(f"Missing: {m}")
        if len(missing) > 20:
            logger.error(f"... and {len(missing) - 20} more")
        logger.error(f"\nRun lsc_base_eval.py first to generate base metrics.")
        return 1

    # ---- Run tasks ----
    run_all_tasks(config, thresholds, output_dir, config.gpus, force=args.force)

    # ---- Aggregate ----
    logger.info("\n" + "=" * 70)
    logger.info("AGGREGATION & ANALYSIS")
    logger.info("=" * 70)

    summary = aggregate_results(config, output_dir)
    print_summary_table(summary, config)
    plot_summary(summary, config, output_dir)

    # ---- Final ----
    logger.info("\n" + "=" * 70)
    logger.info("CIRCUIT DISCOVERY COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Output:  {output_dir}")
    logger.info(f"Summary: {output_dir / 'summary' / 'discovery_summary.json'}")
    if config.cross_band:
        logger.info(f"Transfer: {output_dir / 'summary' / 'cross_band_transfer.json'}")
    logger.info(f"Plots:   {output_dir / 'summary' / 'plots'}")
    logger.info(f"\nDirectory structure:")
    logger.info(f"  {output_dir}/")
    logger.info(f"  ├── registry.json")
    logger.info(f"  ├── circuits/{{model}}/{{band}}/{{draw}}/")
    logger.info(f"  │   ├── prune_scores.pkl")
    logger.info(f"  │   └── metrics.json")
    logger.info(f"  └── summary/")
    logger.info(f"      ├── discovery_summary.json")
    logger.info(f"      ├── cross_band_transfer.json")
    logger.info(f"      └── plots/")

    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main() or 0)
