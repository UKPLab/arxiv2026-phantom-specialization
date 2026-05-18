#!/usr/bin/env python3
"""
LSC EAP/EAP-IG Scoring (C2: Cross-Method Comparison)
=====================================================
Compute raw continuous edge importance scores for all (model, band, draw)
combinations using Edge Attribution Patching (EAP) or EAP with Integrated
Gradients (EAP-IG). Scores are post-hoc thresholded in lsc_eap_eval.py.

Uses the SAME data loading, model loading, and training split (256 examples,
seed=42) as lsc_acdc_circuit.py to ensure fair comparison.

Methods:
  EAP:    mask_gradient with mask_val=0.0  (single gradient at zero mask)
  EAP-IG: mask_gradient with integrated_grad_samples=10  (integrated gradients)

Outputs (under EAP_methods/):
  eap_scores/{model}/{band}/{draw}/scores.pkl     (Dict[str, Tensor])
  eap_ig_scores/{model}/{band}/{draw}/scores.pkl
  registry_eap.json
  registry_eap_ig.json

Usage:
    python lsc_eap_scoring.py                           # both methods, all tasks
    python lsc_eap_scoring.py --method eap              # EAP only
    python lsc_eap_scoring.py --method eap_ig           # EAP-IG only
    python lsc_eap_scoring.py --models pythia-70m pythia-160m
    python lsc_eap_scoring.py --bands low control
    python lsc_eap_scoring.py --ig-samples 5            # fewer IG steps (faster)
    python lsc_eap_scoring.py --gpus 0 1 2 3
    python lsc_eap_scoring.py --force                   # recompute all
    python lsc_eap_scoring.py --summary-only            # print status
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

# EAP batch sizes: EAP needs gradients, so lower batch sizes than ACDC inference
MODEL_BATCH_SIZES = {
    "pythia-70m": 128,
    "pythia-160m": 64,
    "pythia-410m": 32,
    "pythia-1b": 16,
    "pythia-1.4b": 8,
}

# EAP-IG batch sizes: smaller due to integrated gradients (multiple passes)
MODEL_BATCH_SIZES_IG = {
    "pythia-70m": 64,
    "pythia-160m": 32,
    "pythia-410m": 16,
    "pythia-1b": 8,
    "pythia-1.4b": 4,
}

N_SOURCE = 5
N_DISTRACT = 10
RAW_SEQ_LEN = N_SOURCE + 1 + N_DISTRACT + N_SOURCE  # 21
SEQ_LEN_WITH_BOS = RAW_SEQ_LEN + 1  # 22
DIVERGE_IDX = N_SOURCE + 1 + N_DISTRACT + 1  # 17 (with BOS)


@dataclass
class ScoringConfig:
    """Configuration for EAP/EAP-IG scoring."""

    # Paths
    data_dir: str = field(default_factory=lambda: str(ISC_ROOT / "LSC_data"))
    pool_dir: str = field(
        default_factory=lambda: str(
            ISC_ROOT / "LSC_data" / "lsc_token_pools" / "matched"
        )
    )
    output_dir: str = field(default_factory=lambda: str(SCRIPT_DIR / "EAP_methods"))
    # Dataset structure: datasets/{variant}/{draw}/{band}/{split}.json
    variant: str = "matched"
    draws: List[str] = field(default_factory=lambda: ["draw_1", "draw_2", "draw_3"])

    # Experiment grid
    models: List[str] = field(default_factory=lambda: list(DEFAULT_MODELS))
    bands: List[str] = field(default_factory=lambda: list(ALL_BANDS))

    # Method
    method: str = "both"  # "eap" | "eap_ig" | "both"
    ig_samples: int = 10  # number of integration steps for EAP-IG

    # Training settings (MUST match ACDC for comparability)
    train_size: int = 256
    acdc_seed: int = 42  # same seed as ACDC
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


def safe_delete_model(model):
    if model is not None:
        try:
            model.cpu()
        except Exception:
            pass
        del model
    cleanup_gpu()


def model_safe_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_")


def sort_models_by_size(models: List[str]) -> List[str]:
    return sorted(models, key=lambda m: MODEL_SIZE_ORDER.get(m, 999))


def setup_logging(output_dir: Path) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"eap_scoring_{timestamp}.log"

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
    logger = logging.getLogger("eap_scoring")
    logger.info(f"Log: {log_file}")
    return logger


def load_pool(band: str, pool_dir: Path) -> dict:
    with open(pool_dir / f"lsc_pool_{band}.json") as f:
        return json.load(f)


def load_dataset(
    band: str, split: str, data_dir: Path, variant: str, draw: str
) -> dict:
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
):
    """Build AutoCircuit PromptDataLoader from token-ID-based LSC data.
    Identical to lsc_acdc_circuit.py for comparability."""
    from auto_circuit.data import PromptDataset, PromptDataLoader

    examples = dataset["examples"]
    rng = random.Random(seed)

    indices = list(range(len(examples)))
    rng.shuffle(indices)
    n_actual = min(len(indices), n_samples)
    if len(indices) > n_actual:
        indices = indices[:n_actual]

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
    actual_batch_size = min(batch_size, len(indices))
    dataloader = PromptDataLoader(
        prompt_dataset=ds,
        seq_len=SEQ_LEN_WITH_BOS,
        diverge_idx=DIVERGE_IDX,
        batch_size=actual_batch_size,
    )
    return dataloader


def _patch_gpt_neox_config():
    """Compatibility patch for transformers >=4.48."""
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
    def make_task_id(model: str, band: str, draw: str, method: str) -> str:
        m = model_safe_name(model)
        return f"{method}__{m}__{band}__{draw}"


def run_scoring_task(
    model_name: str,
    band: str,
    draw: str,
    method: str,
    config: ScoringConfig,
    device: str,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Compute EAP or EAP-IG scores for a single (model, band, draw) combination.

    Uses TRAIN split with the same 256-example sample and seed=42 as ACDC.
    Saves scores.pkl to output_dir/{method}_scores/{model_safe}/{band}/{draw}/
    """
    from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
    from auto_circuit.utils.graph_utils import patchable_model
    from auto_circuit.types import AblationType

    logger = logging.getLogger("eap_scoring")

    assert method in ("eap", "eap_ig"), f"Unknown method: {method}"
    batch_size = (
        MODEL_BATCH_SIZES_IG if method == "eap_ig" else MODEL_BATCH_SIZES
    ).get(model_name, 16)

    logger.info(
        f"[{device}] {method} scoring: {model_name}/{band}/{draw} "
        f"(batch_size={batch_size})"
    )

    model = None
    patchable = None

    try:
        set_all_seeds(config.acdc_seed)

        model = load_model(model_name, device)
        bos_id = model.tokenizer.bos_token_id

        pool_dir = Path(config.pool_dir)
        data_dir = Path(config.data_dir)
        pool = load_pool(band, pool_dir)
        train_data = load_dataset(band, "train", data_dir, config.variant, draw)

        train_loader = prepare_dataloader(
            train_data,
            pool,
            bos_id,
            n_samples=config.train_size,
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
        logger.info(f"[{device}] {model_name}: {total_edges} total edges")

        t_start = time.time()

        if method == "eap":
            prune_scores = mask_gradient_prune_scores(
                model=patchable,
                dataloader=train_loader,
                official_edges=None,
                grad_function="logit",
                answer_function="avg_diff",
                mask_val=0.0,
                ablation_type=AblationType.RESAMPLE,
            )
        else:  # eap_ig
            prune_scores = mask_gradient_prune_scores(
                model=patchable,
                dataloader=train_loader,
                official_edges=None,
                grad_function="logit",
                answer_function="avg_diff",
                integrated_grad_samples=config.ig_samples,
                ablation_type=AblationType.RESAMPLE,
            )

        scoring_time = time.time() - t_start
        logger.info(f"[{device}] {method} done in {scoring_time:.1f}s")

        # Save to CPU
        scores_cpu = {k: v.cpu() for k, v in prune_scores.items()}
        del prune_scores
        cleanup_gpu()

        # Persist
        m_safe = model_safe_name(model_name)
        scores_dir = output_dir / f"{method}_scores" / m_safe / band / draw
        scores_dir.mkdir(parents=True, exist_ok=True)
        scores_path = scores_dir / "scores.pkl"

        def _save(path, data):
            with open(path, "wb") as f:
                pickle.dump(data, f)

        save_thread = threading.Thread(
            target=_save, args=(scores_path, scores_cpu), daemon=True
        )
        save_thread.start()
        save_thread.join(timeout=60)

        return {
            "status": "completed",
            "model": model_name,
            "band": band,
            "draw": draw,
            "method": method,
            "total_edges": total_edges,
            "scoring_time_s": round(scoring_time, 1),
            "scores_path": str(scores_path),
            "completed_at": datetime.now().isoformat(),
        }

    except Exception:
        raise
    finally:
        safe_delete_model(model)
        if patchable is not None:
            try:
                patchable.cpu()
            except Exception:
                pass
            del patchable
        cleanup_gpu()


def gpu_worker(
    gpu_id,
    task_queue,
    result_queue,
    config_dict,
    output_dir_str,
    progress_dict,
    worker_id,
    heartbeat_dict,
):
    config = ScoringConfig(**config_dict)
    output_dir = Path(output_dir_str)
    device = f"cuda:{gpu_id}"
    logger = logging.getLogger("eap_scoring")

    try:
        t.cuda.set_device(gpu_id)
    except Exception as e:
        logger.error(f"GPU {gpu_id} init failed: {e}")
        progress_dict[worker_id] = f"GPU {gpu_id}: FAILED (init)"
        heartbeat_dict[worker_id] = -1
        return

    registries = {
        "eap": Registry(output_dir / "registry_eap.json"),
        "eap_ig": Registry(output_dir / "registry_eap_ig.json"),
    }
    tasks_completed = 0
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
            logger.info(f"[Worker {worker_id}] Shutdown. Completed {tasks_completed}.")
            break

        progress_dict[worker_id] = (
            f"GPU {gpu_id}: {task['method']}/{task['model']}/{task['band']}/{task['draw']}"
        )
        heartbeat_dict[worker_id] = time.time()

        try:
            result = run_scoring_task(
                model_name=task["model"],
                band=task["band"],
                draw=task["draw"],
                method=task["method"],
                config=config,
                device=device,
                output_dir=output_dir,
            )
            registries[task["method"]].add_task(task["id"], result)
            result_queue.put(("success", task, result, None))
            tasks_completed += 1
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"[Worker {worker_id}] FAILED: {task}\n{error_msg[:500]}")
            err = {
                "status": "failed",
                "model": task["model"],
                "band": task["band"],
                "draw": task["draw"],
                "method": task["method"],
                "error": error_msg,
                "failed_at": datetime.now().isoformat(),
            }
            try:
                registries[task["method"]].add_task(task["id"], err)
            except Exception:
                pass
            result_queue.put(("error", task, None, error_msg))
            cleanup_gpu()

        progress_dict[worker_id] = f"GPU {gpu_id}: idle ({tasks_completed} done)"
        heartbeat_dict[worker_id] = time.time()

    cleanup_gpu()


def build_tasks(config: ScoringConfig, output_dir: Path, force: bool) -> List[dict]:
    """Generate all pending (method, model, band, draw) tasks."""
    methods = ["eap", "eap_ig"] if config.method == "both" else [config.method]

    reg_files = {
        "eap": output_dir / "registry_eap.json",
        "eap_ig": output_dir / "registry_eap_ig.json",
    }

    all_tasks = []
    for method in methods:
        reg = Registry(reg_files[method])
        completed_ids = set() if force else reg.get_completed_ids()

        for draw in config.draws:
            for model in sort_models_by_size(config.models):
                for band in config.bands:
                    task_id = Registry.make_task_id(model, band, draw, method)
                    if task_id not in completed_ids:
                        all_tasks.append(
                            {
                                "model": model,
                                "band": band,
                                "draw": draw,
                                "method": method,
                                "id": task_id,
                            }
                        )

    return all_tasks


def run_all_tasks(
    config: ScoringConfig,
    output_dir: Path,
    gpus: List[int],
    force: bool = False,
):
    logger = logging.getLogger("eap_scoring")
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(config, output_dir, force)
    logger.info(f"Pending tasks: {len(tasks)}")
    if not tasks:
        logger.info("All tasks already completed.")
        return

    if len(gpus) <= 1:
        device = f"cuda:{gpus[0]}" if gpus else "cpu"
        logger.info(f"Sequential execution on {device}")
        regs = {
            "eap": Registry(output_dir / "registry_eap.json"),
            "eap_ig": Registry(output_dir / "registry_eap_ig.json"),
        }
        t_start = time.time()
        for i, task in enumerate(tasks):
            if i > 0:
                elapsed = time.time() - t_start
                avg = elapsed / i
                eta = timedelta(seconds=int(avg * (len(tasks) - i)))
                logger.info(
                    f"Progress: {i}/{len(tasks)} ({100 * i / len(tasks):.0f}%) ETA: {eta}"
                )
            try:
                result = run_scoring_task(
                    model_name=task["model"],
                    band=task["band"],
                    draw=task["draw"],
                    method=task["method"],
                    config=config,
                    device=device,
                    output_dir=output_dir,
                )
                regs[task["method"]].add_task(task["id"], result)
                logger.info(
                    f"  Completed: {task['method']}/{task['model']}/{task['band']}/{task['draw']}"
                )
            except Exception as e:
                logger.error(f"  FAILED: {task}: {e}\n{traceback.format_exc()}")
        return

    # Multi-GPU
    logger.info(f"Parallel execution on {len(gpus)} GPUs: {gpus}")
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    manager = ctx.Manager()
    progress_dict = manager.dict()
    heartbeat_dict = manager.dict()

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
                str(output_dir),
                progress_dict,
                i,
                heartbeat_dict,
            ),
        )
        p.start()
        workers.append(p)

    completed = 0
    failed = 0
    total = len(tasks)
    last_log = time.time()
    HEARTBEAT_TIMEOUT = 7200  # 2 hours (EAP is fast)
    heartbeat_warned: Set[int] = set()

    while completed + failed < total:
        try:
            status, task, data, error = result_queue.get(timeout=10)
            if status == "success":
                completed += 1
                logger.info(
                    f"[{completed}/{total}] OK: {task['method']}/{task['model']}/{task['band']}/{task['draw']}"
                )
            else:
                failed += 1
                logger.error(
                    f"[failures:{failed}] FAILED: {task}: {error[:200] if error else ''}"
                )
        except Exception:
            pass

        # Check heartbeats
        now = time.time()
        for i, p in enumerate(workers):
            if not p.is_alive():
                continue
            last_hb = heartbeat_dict.get(i, now)
            if last_hb == -1:
                if i not in heartbeat_warned:
                    logger.error(f"Worker {i} reported fatal error")
                    heartbeat_warned.add(i)
            elif now - last_hb > HEARTBEAT_TIMEOUT and i not in heartbeat_warned:
                logger.warning(
                    f"Worker {i} heartbeat stale ({(now - last_hb) / 3600:.1f}h)"
                )
                heartbeat_warned.add(i)

        if now - last_log > 60:
            status_lines = [f"  {v}" for v in progress_dict.values()]
            logger.info(
                f"Progress: {completed + failed}/{total}\n" + "\n".join(status_lines)
            )
            last_log = now

    for p in workers:
        p.join(timeout=60)
        if p.is_alive():
            logger.warning(f"Worker {p.pid} did not exit cleanly, terminating")
            p.terminate()

    logger.info(f"All done. Completed: {completed}, Failed: {failed}, Total: {total}")


def print_summary(output_dir: Path, config: ScoringConfig):
    """Print current status of EAP/EAP-IG scoring."""
    methods = ["eap", "eap_ig"] if config.method == "both" else [config.method]
    expected = len(config.models) * len(config.bands) * len(config.draws)

    for method in methods:
        reg_path = output_dir / f"registry_{method}.json"
        if not reg_path.exists():
            print(f"\n{method.upper()}: No registry found.")
            continue
        reg = Registry(reg_path)
        data = reg.load()
        tasks = data.get("tasks", {})
        completed = sum(1 for d in tasks.values() if d.get("status") == "completed")
        failed = sum(1 for d in tasks.values() if d.get("status") == "failed")
        print(f"\n{method.upper()}: {completed}/{expected} completed, {failed} failed")

        if failed:
            failed_tasks = [
                (tid, d) for tid, d in tasks.items() if d.get("status") == "failed"
            ]
            print("  Failed tasks:")
            for tid, d in failed_tasks[:5]:
                print(f"    {tid}: {str(d.get('error', ''))[:80]}")


def main():
    parser = argparse.ArgumentParser(description="LSC EAP/EAP-IG Scoring")
    parser.add_argument("--method", choices=["eap", "eap_ig", "both"], default="both")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--bands", nargs="+", default=None)
    parser.add_argument("--draws", nargs="+", default=None)
    parser.add_argument("--ig-samples", type=int, default=10)
    parser.add_argument("--gpus", nargs="+", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    config = ScoringConfig()

    if args.method:
        config.method = args.method
    if args.models:
        config.models = args.models
    if args.bands:
        config.bands = args.bands
    if args.draws:
        config.draws = args.draws
    config.ig_samples = args.ig_samples

    # GPU detection
    if args.gpus is not None:
        gpus = args.gpus
    elif t.cuda.is_available():
        gpus = list(range(t.cuda.device_count()))
    else:
        gpus = []

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)
    logger = logging.getLogger("eap_scoring")

    if args.summary_only:
        print_summary(output_dir, config)
        return

    logger.info("=" * 60)
    logger.info("LSC EAP/EAP-IG SCORING")
    logger.info("=" * 60)
    logger.info(f"Method:  {config.method}")
    logger.info(f"Models:  {config.models}")
    logger.info(f"Bands:   {config.bands}")
    logger.info(f"Draws:   {config.draws}")
    logger.info(f"IG samples: {config.ig_samples}")
    logger.info(f"GPUs:    {gpus}")
    logger.info(f"Output:  {output_dir}")
    logger.info("=" * 60)

    run_all_tasks(config, output_dir, gpus=gpus, force=args.force)
    print_summary(output_dir, config)


if __name__ == "__main__":
    main()
