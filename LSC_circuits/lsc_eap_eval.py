#!/usr/bin/env python3
"""
LSC EAP/EAP-IG Evaluation (C2: Cross-Method Comparison)
=========================================================
Two-phase script:

Phase A: Thresholding
  Load continuous EAP/EAP-IG scores from lsc_eap_scoring.py output.
  For each (model, band, draw) select top-k edges at each circuit size in
  SIZE_MULTIPLIERS (relative to ACDC edge count). Save binary circuits in the
  same prune_scores.pkl format as ACDC. Compute Jaccard/Dice overlap.

Phase B: Cross-band evaluation
  Evaluate each binary EAP/EAP-IG circuit on all 5 test bands.
  Uses the same evaluation infrastructure as lsc_acdc_circuit.py.
  Saves eap_eval_results.csv and eap_overlap.csv.

Outputs (under EAP_methods/):
  eap_circuits/{model}/{band}/{draw}/prune_scores_{size}.pkl
  eap_ig_circuits/{model}/{band}/{draw}/prune_scores_{size}.pkl
  eap_overlap.csv           Jaccard/Dice vs ACDC per task x size
  eap_eval_results.csv      cross-band eval metrics per task x test_band x size
  eval_registry.json        completed evaluations

PARETO / SIZE SWEEP:
  Circuits are extracted at 10 sizes relative to the ACDC edge count:
    0.1x, 0.2x, 0.3x, 0.5x, 0.75x  - sparser than ACDC
    1.0x                              - size-matched to ACDC (primary comparison)
    1.5x, 2.0x, 3.0x, 5.0x          - denser than ACDC (capped at total edges)

  This produces a faithfulness-vs-size Pareto curve showing that phantom
  specialization (cross-band transfer) holds across the full range of circuit
  sizes, not just at the ACDC-matched point. The 1.0x result is the primary
  cross-method comparison; the full sweep turns size sensitivity from a
  potential weakness into a robustness result.

  Total eval tasks: 10 sizes x 2 methods x 75 (model,band,draw) x 5 test_bands
                  = 7,500 circuit evaluations.
  EAP scoring is cheap (single pass), making this sweep feasible.

Usage:
    python lsc_eap_eval.py                       # both methods, all tasks, all sizes
    python lsc_eap_eval.py --method eap          # EAP only
    python lsc_eap_eval.py --threshold-only      # Phase A only (no GPU eval)
    python lsc_eap_eval.py --models pythia-70m
    python lsc_eap_eval.py --gpus 0 1 2
    python lsc_eap_eval.py --force
    python lsc_eap_eval.py --summary-only
"""

import os
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import json
import pickle
import random
import argparse
import traceback
import fcntl
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

import numpy as np
import torch as t
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
ISC_ROOT = SCRIPT_DIR.parent
AUTOCIRCUIT_PATH = os.environ.get("AUTOCIRCUIT_PATH") or str(
    ISC_ROOT / "circuit_discovery" / "auto-circuit"
)
sys.path.insert(0, AUTOCIRCUIT_PATH)

ALL_BANDS = ["low", "medium", "high", "very_high", "control"]

DEFAULT_MODELS = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
]

MODEL_SIZE_ORDER = {
    "pythia-70m": 0,
    "pythia-160m": 1,
    "pythia-410m": 2,
    "pythia-1b": 3,
    "pythia-1.4b": 4,
}

# Eval batch sizes (same as ACDC)
MODEL_BATCH_SIZES = {
    "pythia-70m": 256,
    "pythia-160m": 256,
    "pythia-410m": 128,
    "pythia-1b": 96,
    "pythia-1.4b": 64,
}

N_SOURCE = 5
N_DISTRACT = 10
RAW_SEQ_LEN = N_SOURCE + 1 + N_DISTRACT + N_SOURCE  # 21
SEQ_LEN_WITH_BOS = RAW_SEQ_LEN + 1  # 22
DIVERGE_IDX = N_SOURCE + 1 + N_DISTRACT + 1  # 17 (with BOS)

EVAL_SEED = 123  # Same as lsc_acdc_circuit.py

# Size sweep relative to ACDC edge count.
# 1.0x is the primary size-matched comparison; the full range produces a
# faithfulness-vs-size Pareto curve used in the robustness analysis.
SIZE_MULTIPLIERS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
SIZE_MATCHED = 1.0  # multiplier used for the primary cross-method comparison


def size_to_key(mult: float) -> str:
    """Convert a size multiplier to a filesystem-safe string key.

    Strips trailing zeros after the decimal point, but always keeps at least
    one decimal place, so 1.0 -> '1.0x' and 0.75 -> '0.75x'.
    """
    s = f"{mult:.2f}".rstrip("0")
    if s.endswith("."):
        s += "0"
    return f"{s}x"


@dataclass
class EvalConfig:
    """Configuration for EAP/EAP-IG evaluation."""

    # Paths
    data_dir: str = field(default_factory=lambda: str(ISC_ROOT / "LSC_data"))
    pool_dir: str = field(
        default_factory=lambda: str(
            ISC_ROOT / "LSC_data" / "lsc_token_pools" / "matched"
        )
    )
    eap_methods_dir: str = field(
        default_factory=lambda: str(SCRIPT_DIR / "EAP_methods")
    )
    acdc_circuits_dir: str = field(
        default_factory=lambda: str(SCRIPT_DIR / "circuit_discovery" / "circuits")
    )
    base_metrics_dir: str = field(
        default_factory=lambda: str(SCRIPT_DIR / "base_metrics")
    )

    # Dataset
    variant: str = "matched"
    draws: List[str] = field(default_factory=lambda: ["draw_1", "draw_2", "draw_3"])

    # Experiment grid
    models: List[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    bands: List[str] = field(default_factory=lambda: list(ALL_BANDS))

    # Method
    method: str = "both"  # "eap" | "eap_ig" | "both"

    # Circuit settings
    factorized: bool = True
    separate_qkv: bool = False
    slice_output: str = "last_seq"

    # GPU
    gpus: List[int] = field(default_factory=list)


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


def model_safe_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def sort_models_by_size(models: List[str]) -> List[str]:
    return sorted(models, key=lambda m: MODEL_SIZE_ORDER.get(m, 999))


def get_batch_size(model_name: str) -> int:
    return MODEL_BATCH_SIZES.get(model_name, 32)


def setup_logging(output_dir: Path) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"eap_eval_{timestamp}.log"

    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  [%(process)d] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("eap_eval")


def load_pool(band: str, pool_dir: Path) -> dict:
    with open(pool_dir / f"lsc_pool_{band}.json") as f:
        return json.load(f)


def load_dataset(
    band: str, split: str, data_dir: Path, variant: str, draw: str
) -> dict:
    with open(data_dir / "datasets" / variant / draw / band / f"{split}.json") as f:
        return json.load(f)


def prepare_full_dataloader(
    dataset: dict,
    pool: dict,
    bos_token_id: int,
    batch_size: int,
    seed: int,
    device: str,
) -> Tuple[Any, List[int]]:
    """Build AutoCircuit PromptDataLoader using ALL examples (no sampling).
    Identical to lsc_acdc_circuit.py."""
    from auto_circuit.data import PromptDataset, PromptDataLoader

    examples = dataset["examples"]
    rng = random.Random(seed)

    indices = list(range(len(examples)))
    rng.shuffle(indices)

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


def compute_accuracy_metrics(logits: t.Tensor, answer_ids: List[int]) -> Dict[str, Any]:
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
        "mean_correct_prob": float(np.mean(correct_probs)) if correct_probs else 0.0,
        "n_samples": n,
    }


def compute_kl_divergence(circuit_logits: t.Tensor, base_logits: t.Tensor) -> float:
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


def load_base_metrics(
    model_name: str, band: str, split: str, base_metrics_dir: Path, draw: str = "draw_1"
) -> Dict[str, Any]:
    m_safe = model_safe_name(model_name)
    metrics_file = base_metrics_dir / m_safe / draw / f"{band}.json"
    if not metrics_file.exists():
        raise FileNotFoundError(f"Base metrics not found: {metrics_file}")
    with open(metrics_file) as f:
        data = json.load(f)
    splits = data.get("splits", {})
    if split not in splits:
        raise KeyError(f"Split '{split}' not in {metrics_file}")
    return splits[split]


def compute_base_logits(
    model,
    dataset: dict,
    pool: dict,
    bos_id: int,
    batch_size: int,
    eval_seed: int,
    device: str,
) -> Tuple[t.Tensor, List[int], List[int]]:
    loader, indices = prepare_full_dataloader(
        dataset, pool, bos_id, batch_size, eval_seed, device
    )
    logits_list = []
    aligned_answer_ids = []
    with t.no_grad():
        for batch in loader:
            logits = model(batch.clean)
            if len(logits.shape) == 3:
                logits = logits[:, -1, :]
            logits_list.append(logits.float())
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
) -> Tuple[t.Tensor, List[int]]:
    from auto_circuit.prune import run_circuits
    from auto_circuit.types import PatchType, AblationType

    set_all_seeds(eval_seed)
    loader1, _ = prepare_full_dataloader(
        dataset, pool, bos_id, batch_size, eval_seed, device
    )

    with t.no_grad():
        outputs = run_circuits(
            model=patchable,
            dataloader=loader1,
            test_edge_counts=[n_edges],
            prune_scores=prune_scores_dev,
            patch_type=PatchType.TREE_PATCH,
            ablation_type=AblationType.RESAMPLE,
        )

    set_all_seeds(eval_seed)
    loader2, _ = prepare_full_dataloader(
        dataset, pool, bos_id, batch_size, eval_seed, device
    )

    logits_list = []
    aligned_answer_ids = []
    for batch in loader2:
        logits = outputs[n_edges][batch.key]
        if len(logits.shape) == 3:
            logits = logits[:, -1, :]
        logits_list.append(logits.float())
        aligned_answer_ids.extend(batch.answers.squeeze(-1).tolist())

    return t.cat(logits_list, dim=0), aligned_answer_ids


# =============================================================================
# PHASE A: THRESHOLDING. Circuit extraction and overlap computation
# =============================================================================


def count_circuit_edges(prune_scores: Dict[str, t.Tensor]) -> int:
    return sum(t.isinf(s).sum().item() for s in prune_scores.values())


def get_total_edges(scores: Dict[str, t.Tensor]) -> int:
    return sum(s.numel() for s in scores.values())


def extract_top_k_edges(
    scores: Dict[str, t.Tensor],
    k: int,
) -> Dict[str, t.Tensor]:
    """Select top-k edges by absolute score. Returns binary prune_scores (inf=kept, 0=pruned)."""
    all_scores = []
    for name, tensor in scores.items():
        flat = tensor.abs().flatten()
        for idx in range(flat.numel()):
            all_scores.append((flat[idx].item(), name, idx))

    all_scores.sort(key=lambda x: x[0], reverse=True)

    top_k_set = set()
    for i in range(min(k, len(all_scores))):
        _, name, idx = all_scores[i]
        top_k_set.add((name, idx))

    circuit = {}
    for name, tensor in scores.items():
        binary = t.zeros_like(tensor)
        flat = binary.flatten()
        for idx in range(flat.numel()):
            if (name, idx) in top_k_set:
                flat[idx] = float("inf")
        circuit[name] = binary.reshape(tensor.shape)

    return circuit


def get_edge_set(prune_scores: Dict[str, t.Tensor]) -> set:
    edges = set()
    for name, scores in prune_scores.items():
        flat = scores.flatten()
        for idx in range(flat.numel()):
            if t.isinf(flat[idx]):
                edges.add((name, idx))
    return edges


def compute_overlap(
    circuit_a: Dict[str, t.Tensor],
    circuit_b: Dict[str, t.Tensor],
) -> Dict[str, float]:
    edges_a = get_edge_set(circuit_a)
    edges_b = get_edge_set(circuit_b)
    n_a = len(edges_a)
    n_b = len(edges_b)
    n_inter = len(edges_a & edges_b)
    n_union = len(edges_a | edges_b)
    return {
        "n_edges_method": n_a,
        "n_edges_acdc": n_b,
        "n_intersection": n_inter,
        "n_union": n_union,
        "jaccard": n_inter / n_union if n_union > 0 else 0.0,
        "recall_in_acdc": n_inter / n_b if n_b > 0 else 0.0,
        "recall_in_method": n_inter / n_a if n_a > 0 else 0.0,
        "dice": 2 * n_inter / (n_a + n_b) if (n_a + n_b) > 0 else 0.0,
    }


def run_thresholding(config: EvalConfig, eap_methods_dir: Path, force: bool = False):
    """
    Phase A: For each (method, model, band, draw) x SIZE_MULTIPLIERS:
    1. Load continuous EAP/EAP-IG scores
    2. Load ACDC circuit -> get edge count n_acdc
    3. For each size multiplier m: extract top round(n_acdc * m) EAP edges
    4. Compute Jaccard overlap vs ACDC (only meaningful at 1.0x, but recorded for all)
    5. Save binary circuits as prune_scores_{size_key}.pkl
    """
    logger = logging.getLogger("eap_eval")
    methods = ["eap", "eap_ig"] if config.method == "both" else [config.method]

    acdc_circuits_dir = Path(config.acdc_circuits_dir)
    overlap_rows = []
    skipped = 0
    processed = 0

    for method in methods:
        scores_dir = eap_methods_dir / f"{method}_scores"
        circuits_dir = eap_methods_dir / f"{method}_circuits"

        for draw in config.draws:
            for model in sort_models_by_size(config.models):
                m_safe = model_safe_name(model)
                for band in config.bands:
                    # Check if all size variants are already done
                    out_dir = circuits_dir / m_safe / band / draw
                    all_done = all(
                        (out_dir / f"prune_scores_{size_to_key(mult)}.pkl").exists()
                        for mult in SIZE_MULTIPLIERS
                    )
                    if all_done and not force:
                        skipped += 1
                        continue

                    # Load EAP scores
                    scores_path = scores_dir / m_safe / band / draw / "scores.pkl"
                    if not scores_path.exists():
                        logger.warning(f"Scores not found: {scores_path}")
                        continue

                    with open(scores_path, "rb") as f:
                        eap_scores = pickle.load(f)

                    # Load ACDC circuit to get reference edge count
                    acdc_path = (
                        acdc_circuits_dir / m_safe / band / draw / "prune_scores.pkl"
                    )
                    if not acdc_path.exists():
                        logger.warning(f"ACDC circuit not found: {acdc_path}")
                        continue

                    with open(acdc_path, "rb") as f:
                        acdc_circuit = pickle.load(f)

                    n_edges_acdc = count_circuit_edges(acdc_circuit)
                    total_edges = get_total_edges(eap_scores)

                    if n_edges_acdc == 0:
                        logger.warning(f"Empty ACDC circuit: {model}/{band}/{draw}")
                        continue

                    out_dir.mkdir(parents=True, exist_ok=True)

                    for mult in SIZE_MULTIPLIERS:
                        size_key = size_to_key(mult)
                        circuit_path = out_dir / f"prune_scores_{size_key}.pkl"

                        if circuit_path.exists() and not force:
                            continue

                        # Compute target edge count, capped at total
                        k = max(1, min(round(n_edges_acdc * mult), total_edges))
                        eap_circuit = extract_top_k_edges(eap_scores, k)
                        n_edges_method = count_circuit_edges(eap_circuit)

                        # Overlap vs ACDC (uses ACDC 1x circuit as reference)
                        overlap = compute_overlap(eap_circuit, acdc_circuit)

                        with open(circuit_path, "wb") as f:
                            pickle.dump(
                                {kk: v.cpu() for kk, v in eap_circuit.items()}, f
                            )

                        row = {
                            "method": method,
                            "model": model,
                            "band": band,
                            "draw": draw,
                            "size_multiplier": mult,
                            "n_edges": n_edges_method,
                            "n_edges_acdc_ref": n_edges_acdc,
                            "total_edges": total_edges,
                            "size_fraction": n_edges_method / total_edges
                            if total_edges
                            else 0.0,
                            **overlap,
                        }
                        overlap_rows.append(row)
                        processed += 1

                        logger.info(
                            f"[{method}] {model}/{band}/{draw} @{size_key}: "
                            f"{n_edges_method} edges, Jaccard={overlap['jaccard']:.3f}"
                        )

    # Save overlap CSV
    if overlap_rows:
        overlap_csv = eap_methods_dir / "eap_overlap.csv"
        fieldnames = list(overlap_rows[0].keys())
        write_mode = "a" if overlap_csv.exists() and not force else "w"
        with open(overlap_csv, write_mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_mode == "w":
                writer.writeheader()
            writer.writerows(overlap_rows)
        logger.info(f"Saved overlap CSV: {overlap_csv} ({len(overlap_rows)} rows)")

    logger.info(f"Thresholding done. Processed: {processed}, Skipped: {skipped}")
    return overlap_rows


# =============================================================================
# PHASE B: EVALUATION
# =============================================================================


class EvalRegistry:
    """Simple file-locked registry for completed evaluations."""

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
            return {"completed": []}
        finally:
            self._unlock(lf)

    def mark_done(self, task_id: str):
        lf = self._lock()
        try:
            reg = self.load_raw()
            if task_id not in reg["completed"]:
                reg["completed"].append(task_id)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(reg, f, indent=2)
            tmp.rename(self.path)
        finally:
            self._unlock(lf)

    def load_raw(self) -> dict:
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {"completed": []}

    def get_completed(self) -> Set[str]:
        return set(self.load().get("completed", []))

    @staticmethod
    def make_id(
        method: str,
        model: str,
        band: str,
        draw: str,
        test_band: str,
        size_key: str = "1.0x",
    ) -> str:
        m = model.replace("/", "_").replace("-", "_")
        return f"{method}__{m}__{band}__{draw}__{test_band}__{size_key}"


def run_eval_task(
    model_name: str,
    band: str,
    draw: str,
    method: str,
    size_key: str,
    config: EvalConfig,
    device: str,
    eap_methods_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Evaluate a single (method, model, circuit_band, draw, size_key) circuit on all 5 test bands.
    Loads model once, evaluates all test bands, returns list of result rows.
    """
    from auto_circuit.utils.graph_utils import patchable_model

    logger = logging.getLogger("eap_eval")

    m_safe = model_safe_name(model_name)
    circuits_dir = eap_methods_dir / f"{method}_circuits"
    circuit_path = circuits_dir / m_safe / band / draw / f"prune_scores_{size_key}.pkl"

    if not circuit_path.exists():
        logger.warning(f"Circuit not found: {circuit_path}")
        return []

    with open(circuit_path, "rb") as f:
        prune_scores_cpu = pickle.load(f)

    n_edges = count_circuit_edges(prune_scores_cpu)
    total_edges = get_total_edges(prune_scores_cpu)

    if n_edges == 0:
        logger.warning(f"Empty circuit: {method}/{model_name}/{band}/{draw}@{size_key}")
        return []

    batch_size = get_batch_size(model_name)
    pool_dir = Path(config.pool_dir)
    data_dir = Path(config.data_dir)
    base_metrics_dir = Path(config.base_metrics_dir)

    model = None
    patchable = None
    rows = []

    try:
        model = load_model(model_name, device)
        bos_id = model.tokenizer.bos_token_id

        patchable = patchable_model(
            model=model,
            factorized=config.factorized,
            slice_output=config.slice_output,
            seq_len=None,
            separate_qkv=config.separate_qkv,
            device=device,
        )

        prune_scores_dev = {
            k: v.to(device, non_blocking=True) for k, v in prune_scores_cpu.items()
        }

        for test_band in ALL_BANDS:
            logger.info(
                f"  [{method}@{size_key}] {model_name}/{band}/{draw} -> test:{test_band}"
            )

            try:
                pool = load_pool(test_band, pool_dir)
                test_data = load_dataset(
                    test_band, "test", data_dir, config.variant, draw
                )
                base_metrics = load_base_metrics(
                    model_name, test_band, "test", base_metrics_dir, draw=draw
                )
            except FileNotFoundError as e:
                logger.error(f"  Data not found: {e}")
                continue

            # Base logits (for KL div)
            base_logits, _, _ = compute_base_logits(
                model,
                test_data,
                pool,
                bos_id,
                batch_size,
                EVAL_SEED,
                device,
            )

            # Circuit logits
            circ_logits, circ_answers = run_circuit_and_collect(
                patchable,
                prune_scores_dev,
                n_edges,
                test_data,
                pool,
                bos_id,
                batch_size,
                EVAL_SEED,
                device,
            )
            circuit_metrics = compute_accuracy_metrics(circ_logits, circ_answers)
            kl_div = compute_kl_divergence(circ_logits, base_logits)

            rows.append(
                {
                    "method": method,
                    "size_multiplier": size_key,
                    "model": model_name,
                    "draw": draw,
                    "circuit_band": band,
                    "test_band": test_band,
                    "n_edges": n_edges,
                    "total_edges": total_edges,
                    "size_fraction": n_edges / total_edges if total_edges else 0.0,
                    "circuit_accuracy": circuit_metrics["accuracy"],
                    "circuit_top5_accuracy": circuit_metrics["top5_accuracy"],
                    "circuit_mean_prob": circuit_metrics["mean_correct_prob"],
                    "base_accuracy": base_metrics.get("accuracy", 0.0),
                    "kl_div": kl_div,
                    "n_samples": circuit_metrics["n_samples"],
                }
            )

            del base_logits, circ_logits
            cleanup_gpu()

        return rows

    except Exception as e:
        logger.error(
            f"Eval failed: {method}/{model_name}/{band}/{draw}@{size_key}: "
            f"{e}\n{traceback.format_exc()}"
        )
        return []
    finally:
        if patchable is not None:
            try:
                patchable.cpu()
            except Exception:
                pass
            del patchable
        if model is not None:
            try:
                model.cpu()
            except Exception:
                pass
            del model
        cleanup_gpu()


def eval_gpu_worker(
    gpu_id,
    task_queue,
    result_queue,
    config_dict,
    eap_methods_dir_str,
    progress_dict,
    worker_id,
    heartbeat_dict,
):
    config = EvalConfig(**config_dict)
    eap_methods_dir = Path(eap_methods_dir_str)
    device = f"cuda:{gpu_id}"
    logger = logging.getLogger("eap_eval")

    try:
        t.cuda.set_device(gpu_id)
    except Exception as e:
        logger.error(f"GPU {gpu_id} init failed: {e}")
        progress_dict[worker_id] = f"GPU {gpu_id}: FAILED"
        heartbeat_dict[worker_id] = -1
        return

    tasks_done = 0
    logger.info(f"[Worker {worker_id}] Started on GPU {gpu_id}")
    heartbeat_dict[worker_id] = time.time()

    while True:
        heartbeat_dict[worker_id] = time.time()
        try:
            task = task_queue.get(timeout=5)
        except queue_module.Empty:
            continue
        except Exception as e:
            logger.error(f"[Worker {worker_id}] Queue error: {e}")
            continue

        if task is None:
            logger.info(f"[Worker {worker_id}] Shutdown ({tasks_done} done)")
            break

        progress_dict[worker_id] = (
            f"GPU {gpu_id}: {task['method']}@{task['size_key']}/{task['model']}/{task['band']}/{task['draw']}"
        )
        heartbeat_dict[worker_id] = time.time()

        try:
            rows = run_eval_task(
                model_name=task["model"],
                band=task["band"],
                draw=task["draw"],
                method=task["method"],
                size_key=task["size_key"],
                config=config,
                device=device,
                eap_methods_dir=eap_methods_dir,
            )
            result_queue.put(("success", task, rows, None))
            tasks_done += 1
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"[Worker {worker_id}] FAILED: {task}: {error_msg[:400]}")
            result_queue.put(("error", task, [], error_msg))
            cleanup_gpu()

        progress_dict[worker_id] = f"GPU {gpu_id}: idle ({tasks_done} done)"
        heartbeat_dict[worker_id] = time.time()

    cleanup_gpu()


def run_all_evaluations(
    config: EvalConfig,
    eap_methods_dir: Path,
    gpus: List[int],
    force: bool = False,
):
    logger = logging.getLogger("eap_eval")
    methods = ["eap", "eap_ig"] if config.method == "both" else [config.method]

    registry = EvalRegistry(eap_methods_dir / "eval_registry.json")
    completed_ids = set() if force else registry.get_completed()

    # Build task list; one task = one (method, model, circuit_band, draw, size_key)
    # Each task evaluates on all 5 test bands internally
    all_tasks = []
    for method in methods:
        for draw in config.draws:
            for model in sort_models_by_size(config.models):
                for band in config.bands:
                    for mult in SIZE_MULTIPLIERS:
                        size_key = size_to_key(mult)
                        task_done = all(
                            EvalRegistry.make_id(
                                method, model, band, draw, tb, size_key
                            )
                            in completed_ids
                            for tb in ALL_BANDS
                        )
                        if not task_done:
                            all_tasks.append(
                                {
                                    "method": method,
                                    "model": model,
                                    "band": band,
                                    "draw": draw,
                                    "size_key": size_key,
                                }
                            )

    logger.info(f"Pending eval tasks: {len(all_tasks)}")
    if not all_tasks:
        logger.info("All evaluations already complete.")
        return

    # CSV output
    csv_path = eap_methods_dir / "eap_eval_results.csv"
    fieldnames = [
        "method",
        "size_multiplier",
        "model",
        "draw",
        "circuit_band",
        "test_band",
        "n_edges",
        "total_edges",
        "size_fraction",
        "circuit_accuracy",
        "circuit_top5_accuracy",
        "circuit_mean_prob",
        "base_accuracy",
        "kl_div",
        "n_samples",
    ]
    csv_mode = "w" if force else "a"
    csv_file = open(csv_path, csv_mode, newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if csv_mode == "w" or csv_path.stat().st_size == 0:
        writer.writeheader()
    write_lock = threading.Lock()

    def write_rows(rows):
        if not rows:
            return
        with write_lock:
            writer.writerows(rows)
            csv_file.flush()
            for row in rows:
                task_id = EvalRegistry.make_id(
                    row["method"],
                    row["model"],
                    row["circuit_band"],
                    row["draw"],
                    row["test_band"],
                    row["size_multiplier"],
                )
                registry.mark_done(task_id)

    if len(gpus) <= 1:
        device = f"cuda:{gpus[0]}" if gpus else "cpu"
        logger.info(f"Sequential execution on {device}")
        t_start = time.time()
        for i, task in enumerate(all_tasks):
            if i > 0:
                elapsed = time.time() - t_start
                avg = elapsed / i
                eta = timedelta(seconds=int(avg * (len(all_tasks) - i)))
                logger.info(f"Progress: {i}/{len(all_tasks)} ETA: {eta}")
            rows = run_eval_task(
                model_name=task["model"],
                band=task["band"],
                draw=task["draw"],
                method=task["method"],
                size_key=task["size_key"],
                config=config,
                device=device,
                eap_methods_dir=eap_methods_dir,
            )
            write_rows(rows)
        csv_file.close()
        return

    # Multi-GPU
    logger.info(f"Parallel execution on {len(gpus)} GPUs: {gpus}")
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    manager = ctx.Manager()
    progress_dict = manager.dict()
    heartbeat_dict = manager.dict()

    for task in all_tasks:
        task_queue.put(task)
    for _ in gpus:
        task_queue.put(None)

    config_dict = asdict(config)
    workers = []
    for i, gpu_id in enumerate(gpus):
        progress_dict[i] = f"GPU {gpu_id}: starting"
        heartbeat_dict[i] = time.time()
        p = ctx.Process(
            target=eval_gpu_worker,
            args=(
                gpu_id,
                task_queue,
                result_queue,
                config_dict,
                str(eap_methods_dir),
                progress_dict,
                i,
                heartbeat_dict,
            ),
        )
        p.start()
        workers.append(p)

    completed = 0
    failed = 0
    total = len(all_tasks)
    last_log = time.time()

    while completed + failed < total:
        try:
            status, task, rows, error = result_queue.get(timeout=10)
            if status == "success":
                write_rows(rows)
                completed += 1
                logger.info(
                    f"[{completed}/{total}] OK: {task['method']}@{task['size_key']}/{task['model']}/{task['band']}/{task['draw']} ({len(rows)} rows)"
                )
            else:
                failed += 1
                logger.error(
                    f"[failures:{failed}] FAILED: {task}: {error[:200] if error else ''}"
                )
        except Exception:
            pass

        now = time.time()
        if now - last_log > 60:
            status_lines = [f"  {v}" for v in progress_dict.values()]
            logger.info(
                f"Progress: {completed + failed}/{total}\n" + "\n".join(status_lines)
            )
            last_log = now

    for p in workers:
        p.join(timeout=60)
        if p.is_alive():
            logger.warning(f"Worker {p.pid} did not exit, terminating")
            p.terminate()

    csv_file.close()
    logger.info(
        f"Evaluation done. Completed: {completed}, Failed: {failed}, Total: {total}"
    )
    logger.info(f"Results: {csv_path}")


def print_summary(eap_methods_dir: Path, config: EvalConfig):
    methods = ["eap", "eap_ig"] if config.method == "both" else [config.method]
    expected_circuits = len(config.models) * len(config.bands) * len(config.draws)
    expected_eval_rows = expected_circuits * len(ALL_BANDS)

    for method in methods:
        circuits_dir = eap_methods_dir / f"{method}_circuits"
        n_circuits = (
            len(list(circuits_dir.glob("**/*.pkl"))) if circuits_dir.exists() else 0
        )
        print(f"\n{method.upper()} circuits: {n_circuits}/{expected_circuits}")

    csv_path = eap_methods_dir / "eap_eval_results.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            n_rows = sum(1 for _ in f) - 1  # exclude header
        print(f"\nEval CSV: {n_rows}/{expected_eval_rows * len(methods)} rows")
    else:
        print("\nEval CSV: not found")

    overlap_csv = eap_methods_dir / "eap_overlap.csv"
    if overlap_csv.exists():
        with open(overlap_csv) as f:
            n_rows = sum(1 for _ in f) - 1
        print(f"Overlap CSV: {n_rows} rows")


def main():
    parser = argparse.ArgumentParser(description="LSC EAP/EAP-IG Evaluation")
    parser.add_argument("--method", choices=["eap", "eap_ig", "both"], default="both")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--bands", nargs="+", default=None)
    parser.add_argument("--draws", nargs="+", default=None)
    parser.add_argument("--gpus", nargs="+", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--threshold-only",
        action="store_true",
        help="Only run Phase A (thresholding), skip GPU evaluation",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip Phase A, only run Phase B (GPU evaluation)",
    )
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    config = EvalConfig()
    if args.method:
        config.method = args.method
    if args.models:
        config.models = args.models
    if args.bands:
        config.bands = args.bands
    if args.draws:
        config.draws = args.draws

    if args.gpus is not None:
        gpus = args.gpus
    elif t.cuda.is_available():
        gpus = list(range(t.cuda.device_count()))
    else:
        gpus = []

    eap_methods_dir = Path(config.eap_methods_dir)
    eap_methods_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(eap_methods_dir)
    logger = logging.getLogger("eap_eval")

    if args.summary_only:
        print_summary(eap_methods_dir, config)
        return

    logger.info("=" * 60)
    logger.info("LSC EAP/EAP-IG EVALUATION")
    logger.info("=" * 60)
    logger.info(f"Method:  {config.method}")
    logger.info(f"Models:  {config.models}")
    logger.info(f"Bands:   {config.bands}")
    logger.info(f"Draws:   {config.draws}")
    logger.info(f"GPUs:    {gpus}")
    logger.info(f"Output:  {eap_methods_dir}")
    logger.info("=" * 60)

    if not args.eval_only:
        logger.info("\n--- Phase A: Thresholding ---")
        run_thresholding(config, eap_methods_dir, force=args.force)

    if not args.threshold_only:
        logger.info("\n--- Phase B: Cross-band Evaluation ---")
        run_all_evaluations(config, eap_methods_dir, gpus=gpus, force=args.force)

    print_summary(eap_methods_dir, config)


if __name__ == "__main__":
    main()
