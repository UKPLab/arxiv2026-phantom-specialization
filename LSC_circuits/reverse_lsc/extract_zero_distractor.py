#!/usr/bin/env python3
"""
Zero-Distractor LSC: ACDC Circuit Extraction + 2x2 Transfer Matrix
===================================================================
Extracts a circuit for zero-distractor LSC on Pythia-160m and computes
the 2x2 cross-condition transfer matrix against the existing standard
LSC circuit.

Steps:
  1. Run ACDC on zero-distractor train data -> prune_scores
  2. Evaluate zero-distractor circuit on zero-distractor test -> diagonal
  3. Load existing standard LSC circuit (control/draw_1)
  4. Cross-evaluate: each circuit on the other condition's test data
  5. Compute Jaccard similarity between the two circuits
  6. Report 2x2 transfer matrix

difference from standard LSC ACDC:
  - seq_len = 12 (11 raw + BOS) instead of 22
  - diverge_idx = 7 (repeated prefix starts at raw pos 6, +1 for BOS)
  - Only 6 unique tokens per sequence (no distractors)
  - Corruption: replace positions 6-10 (raw) with random tokens from pool

IMPORTANT: The standard LSC circuit was extracted on 22-token sequences.
The patchable_model graph depends on seq_len, so we need TWO patchable
models: one for zero-distractor (seq_len=12) and one for standard
(seq_len=22). However, AutoCircuit's factorized edges are named by
layer/head/component; they do NOT depend on seq_len. So prune_scores
from one seq_len can be applied to another seq_len's patchable model
as long as the model architecture is the same. We verify this.
"""

import json
import sys
import os
import pickle
import random
import math
import gc
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np
import torch as t
import torch.nn.functional as F

ISC_ROOT = Path(__file__).resolve().parent.parent.parent

# Add AutoCircuit to path
AUTOCIRCUIT_PATH = os.environ.get("AUTOCIRCUIT_PATH") or str(
    ISC_ROOT / "circuit_discovery" / "auto-circuit"
)
sys.path.insert(0, AUTOCIRCUIT_PATH)

# Zero-distractor data
ZERO_DATA_DIR = ISC_ROOT / "LSC_data" / "reverse_lsc" / "zero_distractor"
ZERO_POOL_PATH = (
    ISC_ROOT / "LSC_data" / "lsc_token_pools" / "matched" / "lsc_pool_control.json"
)

# Standard LSC data and circuit
STD_DATA_DIR = ISC_ROOT / "LSC_data" / "datasets" / "matched" / "draw_1" / "control"
STD_POOL_PATH = ZERO_POOL_PATH  # Same pool
STD_CIRCUIT_PATH = (
    ISC_ROOT
    / "LSC_circuits"
    / "circuit_discovery"
    / "circuits"
    / "pythia_160m"
    / "control"
    / "draw_1"
    / "prune_scores.pkl"
)

# Output
OUTPUT_DIR = Path(__file__).resolve().parent / "circuit"

MODEL_NAME = "pythia-160m"
THRESHOLD = 6.31e-4  # τ* for Pythia-160m

# Zero-distractor sequence: [S1..S5][T][S1..S5] = 11 tokens, +BOS = 12
ZERO_RAW_LEN = 11
ZERO_SEQ_LEN = 12  # with BOS
ZERO_DIVERGE_IDX = 7  # repeated prefix starts at raw pos 6, +1 for BOS
ZERO_N_SOURCE = 5

# Standard LSC: [S1..S5][T][R1..R10][S1..S5] = 21 tokens, +BOS = 22
STD_RAW_LEN = 21
STD_SEQ_LEN = 22
STD_DIVERGE_IDX = 17
STD_N_SOURCE = 5

ACDC_SEED = 42
EVAL_SEED = 123
ACDC_TRAIN_SIZE = 256
BATCH_SIZE = 256  # Pythia-160m fits easily


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


def threshold_to_tao(threshold: float) -> Tuple[int, float]:
    exponent = math.floor(math.log10(threshold))
    base = round(threshold / (10**exponent), 6)
    return exponent, base


def load_model(device: str):
    import transformer_lens as tl

    _patch_gpt_neox_config()
    model = tl.HookedTransformer.from_pretrained(
        MODEL_NAME,
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


def load_pool(pool_path: Path) -> dict:
    with open(pool_path) as f:
        return json.load(f)


def load_dataset(data_path: Path, split: str) -> dict:
    with open(data_path / f"{split}.json") as f:
        return json.load(f)


def prepare_dataloader(
    dataset: dict,
    pool: dict,
    bos_id: int,
    n_samples: int,
    seed: int,
    device: str,
    seq_len: int,
    diverge_idx: int,
    n_source: int,
):
    """Build AutoCircuit PromptDataLoader."""
    from auto_circuit.data import PromptDataset, PromptDataLoader

    examples = dataset["examples"]
    rng = random.Random(seed)

    indices = list(range(len(examples)))
    rng.shuffle(indices)
    if n_samples and len(indices) > n_samples:
        indices = indices[:n_samples]

    pool_ids = [tok["token_id"] for tok in pool["tokens"]]

    clean_prompts, corrupt_prompts, answers, wrong_answers = [], [], [], []

    for idx in indices:
        ex = examples[idx]
        token_ids = ex["token_ids"]
        raw_len = len(token_ids)

        clean = [bos_id] + token_ids
        assert len(clean) == seq_len, f"clean len={len(clean)} != {seq_len}"

        # Corruption: replace the repeated prefix (last n_source positions)
        repeat_start = raw_len - n_source  # raw position where repetition starts
        used_set = set(token_ids)
        available = [tid for tid in pool_ids if tid not in used_set]
        if len(available) >= n_source:
            replacements = rng.sample(available, n_source)
        else:
            replacements = rng.sample(pool_ids, n_source)

        corrupt = [bos_id] + token_ids[:repeat_start] + replacements
        assert len(corrupt) == seq_len, f"corrupt len={len(corrupt)} != {seq_len}"

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
    actual_batch_size = min(BATCH_SIZE, n)
    while actual_batch_size > 1 and n % actual_batch_size != 0:
        actual_batch_size -= 1

    dataloader = PromptDataLoader(
        prompt_dataset=ds,
        seq_len=seq_len,
        diverge_idx=diverge_idx,
        batch_size=actual_batch_size,
    )
    return dataloader, indices


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
        "top10_accuracy": top10 / n if n else 0.0,
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


def run_circuit_eval(
    patchable,
    prune_scores_dev: dict,
    n_edges: int,
    dataset: dict,
    pool: dict,
    bos_id: int,
    eval_seed: int,
    device: str,
    seq_len: int,
    diverge_idx: int,
    n_source: int,
) -> Tuple[t.Tensor, List[int]]:
    """Run circuit on a dataset and return logits + aligned answer IDs."""
    from auto_circuit.prune import run_circuits
    from auto_circuit.types import PatchType, AblationType

    set_all_seeds(eval_seed)
    loader1, _ = prepare_dataloader(
        dataset,
        pool,
        bos_id,
        n_samples=0,
        seed=eval_seed,
        device=device,
        seq_len=seq_len,
        diverge_idx=diverge_idx,
        n_source=n_source,
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

    # Second loader to extract answer IDs
    set_all_seeds(eval_seed)
    loader2, _ = prepare_dataloader(
        dataset,
        pool,
        bos_id,
        n_samples=0,
        seed=eval_seed,
        device=device,
        seq_len=seq_len,
        diverge_idx=diverge_idx,
        n_source=n_source,
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


def compute_base_logits(
    model,
    dataset: dict,
    pool: dict,
    bos_id: int,
    eval_seed: int,
    device: str,
    seq_len: int,
    diverge_idx: int,
    n_source: int,
) -> Tuple[t.Tensor, List[int]]:
    """Run full model on dataset."""
    loader, _ = prepare_dataloader(
        dataset,
        pool,
        bos_id,
        n_samples=0,
        seed=eval_seed,
        device=device,
        seq_len=seq_len,
        diverge_idx=diverge_idx,
        n_source=n_source,
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
    return t.cat(logits_list, dim=0), aligned_answer_ids


def extract_edge_set(prune_scores: dict) -> set:
    """Extract set of edge identifiers from prune_scores (inf = in-circuit)."""
    edges = set()
    for name, scores in prune_scores.items():
        circuit_mask = t.isinf(scores) & (scores > 0)
        for idx in circuit_mask.nonzero(as_tuple=False):
            edge_id = f"{name}[{','.join(str(i.item()) for i in idx)}]"
            edges.add(edge_id)
    return edges


def compute_jaccard(set1: set, set2: set) -> float:
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def main():
    print("=" * 70)
    print("Zero-Distractor LSC: ACDC Extraction + 2x2 Transfer Matrix")
    print(f"Model: {MODEL_NAME}, Threshold: {THRESHOLD}")
    print("=" * 70)

    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load model ----
    print("\nLoading model...")
    model = load_model(device)
    bos_id = model.tokenizer.bos_token_id
    pool = load_pool(ZERO_POOL_PATH)

    # ---- Load data ----
    print("Loading datasets...")
    zero_train = load_dataset(ZERO_DATA_DIR, "train")
    zero_test = load_dataset(ZERO_DATA_DIR, "test")
    std_test = load_dataset(STD_DATA_DIR, "test")
    print(
        f"  Zero-distractor: {len(zero_train['examples'])} train, {len(zero_test['examples'])} test"
    )
    print(f"  Standard LSC:    {len(std_test['examples'])} test")

    # ================================================================
    # STEP 1: ACDC on zero-distractor
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 1: ACDC extraction on zero-distractor LSC")
    print("=" * 70)

    from auto_circuit.prune_algos.ACDC import acdc_prune_scores
    from auto_circuit.utils.graph_utils import patchable_model

    set_all_seeds(ACDC_SEED)

    # Create patchable model for zero-distractor seq_len
    patchable_zero = patchable_model(
        model=model,
        factorized=True,
        slice_output="last_seq",
        seq_len=None,
        separate_qkv=False,
        device=device,
    )
    total_edges_zero = len(patchable_zero.edges)
    print(f"  Total edges in graph: {total_edges_zero}")

    train_loader, train_indices = prepare_dataloader(
        zero_train,
        pool,
        bos_id,
        n_samples=ACDC_TRAIN_SIZE,
        seed=ACDC_SEED,
        device=device,
        seq_len=ZERO_SEQ_LEN,
        diverge_idx=ZERO_DIVERGE_IDX,
        n_source=ZERO_N_SOURCE,
    )

    tao_exp, tao_base = threshold_to_tao(THRESHOLD)
    print(f"  Running ACDC (τ={THRESHOLD}, {tao_base}x10^{tao_exp})...")
    t_start = time.time()

    zero_prune_scores = acdc_prune_scores(
        model=patchable_zero,
        dataloader=train_loader,
        official_edges=None,
        tao_exps=[tao_exp],
        tao_bases=[tao_base],
        faithfulness_target="kl_div",
        test_mode=False,
        show_graphs=False,
    )

    training_time = time.time() - t_start
    n_edges_zero = sum(t.isinf(s).sum().item() for s in zero_prune_scores.values())
    total_possible = sum(s.numel() for s in zero_prune_scores.values())
    print(
        f"  Done: {n_edges_zero}/{total_possible} edges ({n_edges_zero / total_possible:.1%})"
    )
    print(f"  Training time: {training_time:.1f}s")

    # Save zero-distractor circuit
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores_path = OUTPUT_DIR / "zero_distractor_prune_scores.pkl"
    zero_scores_cpu = {k: v.cpu() for k, v in zero_prune_scores.items()}
    with open(scores_path, "wb") as f:
        pickle.dump(zero_scores_cpu, f)
    print(f"  Saved: {scores_path}")

    # ================================================================
    # STEP 2: Load standard LSC circuit
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Load existing standard LSC circuit")
    print("=" * 70)

    if not STD_CIRCUIT_PATH.exists():
        print(f"  ERROR: Standard circuit not found at {STD_CIRCUIT_PATH}")
        sys.exit(1)

    with open(STD_CIRCUIT_PATH, "rb") as f:
        std_prune_scores_cpu = pickle.load(f)
    n_edges_std = sum(t.isinf(s).sum().item() for s in std_prune_scores_cpu.values())
    print(f"  Standard circuit: {n_edges_std} edges")
    print(f"  Zero-distractor circuit: {n_edges_zero} edges")

    # Verify edge name compatibility
    zero_keys = set(zero_scores_cpu.keys())
    std_keys = set(std_prune_scores_cpu.keys())
    if zero_keys != std_keys:
        print(f"  WARNING: Edge name mismatch!")
        print(f"    Zero-only: {zero_keys - std_keys}")
        print(f"    Std-only:  {std_keys - zero_keys}")
    else:
        print(f"  Edge names match ({len(zero_keys)} modules)")

    # ================================================================
    # STEP 3: Jaccard similarity
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Jaccard similarity")
    print("=" * 70)

    zero_edges = extract_edge_set(zero_scores_cpu)
    std_edges = extract_edge_set(std_prune_scores_cpu)
    jaccard = compute_jaccard(zero_edges, std_edges)
    intersection = len(zero_edges & std_edges)
    union = len(zero_edges | std_edges)

    print(f"  Zero-distractor edges: {len(zero_edges)}")
    print(f"  Standard LSC edges:    {len(std_edges)}")
    print(f"  Intersection:          {intersection}")
    print(f"  Union:                 {union}")
    print(f"  Jaccard similarity:    {jaccard:.4f}")
    print(f"  (Within-band Jaccard for Pythia-160m: ~0.59)")
    print(f"  (Between-band Jaccard for Pythia-160m: ~0.56)")

    # ================================================================
    # STEP 4: 2x2 Transfer Matrix
    # ================================================================
    print("\n" + "=" * 70)
    print("STEP 4: 2x2 Transfer Matrix")
    print("=" * 70)

    # We need patchable models for both seq_lens
    # For zero-distractor data: use patchable_zero (already created)
    # For standard data: need a new patchable model
    # But AutoCircuit patchable_model with seq_len=None should work for any seq_len
    # The key: prune_scores are keyed by module names which don't depend on seq_len

    std_prune_scores_dev = {k: v.to(device) for k, v in std_prune_scores_cpu.items()}
    zero_prune_scores_dev = {k: v.to(device) for k, v in zero_scores_cpu.items()}

    transfer_matrix = {}

    # Cell [zero, zero]: zero-distractor circuit on zero-distractor test
    print("\n  [1/4] Zero circuit -> Zero test data...")
    zz_logits, zz_answers = run_circuit_eval(
        patchable_zero,
        zero_prune_scores_dev,
        n_edges_zero,
        zero_test,
        pool,
        bos_id,
        EVAL_SEED,
        device,
        ZERO_SEQ_LEN,
        ZERO_DIVERGE_IDX,
        ZERO_N_SOURCE,
    )
    zz_metrics = compute_accuracy_metrics(zz_logits, zz_answers)
    transfer_matrix["zero_on_zero"] = zz_metrics
    print(f"     Accuracy: {zz_metrics['accuracy']:.1%}")

    # Cell [zero, std]: zero-distractor circuit on standard test
    print("  [2/4] Zero circuit -> Standard test data...")
    zs_logits, zs_answers = run_circuit_eval(
        patchable_zero,
        zero_prune_scores_dev,
        n_edges_zero,
        std_test,
        pool,
        bos_id,
        EVAL_SEED,
        device,
        STD_SEQ_LEN,
        STD_DIVERGE_IDX,
        STD_N_SOURCE,
    )
    zs_metrics = compute_accuracy_metrics(zs_logits, zs_answers)
    transfer_matrix["zero_on_std"] = zs_metrics
    print(f"     Accuracy: {zs_metrics['accuracy']:.1%}")

    # Cell [std, std]: standard circuit on standard test
    print("  [3/4] Standard circuit -> Standard test data...")
    ss_logits, ss_answers = run_circuit_eval(
        patchable_zero,
        std_prune_scores_dev,
        n_edges_std,
        std_test,
        pool,
        bos_id,
        EVAL_SEED,
        device,
        STD_SEQ_LEN,
        STD_DIVERGE_IDX,
        STD_N_SOURCE,
    )
    ss_metrics = compute_accuracy_metrics(ss_logits, ss_answers)
    transfer_matrix["std_on_std"] = ss_metrics
    print(f"     Accuracy: {ss_metrics['accuracy']:.1%}")

    # Cell [std, zero]: standard circuit on zero-distractor test
    print("  [4/4] Standard circuit -> Zero test data...")
    sz_logits, sz_answers = run_circuit_eval(
        patchable_zero,
        std_prune_scores_dev,
        n_edges_std,
        zero_test,
        pool,
        bos_id,
        EVAL_SEED,
        device,
        ZERO_SEQ_LEN,
        ZERO_DIVERGE_IDX,
        ZERO_N_SOURCE,
    )
    sz_metrics = compute_accuracy_metrics(sz_logits, sz_answers)
    transfer_matrix["std_on_zero"] = sz_metrics
    print(f"     Accuracy: {sz_metrics['accuracy']:.1%}")

    # Also compute base model accuracy for reference
    print("\n  Base model accuracy:")
    zero_base_logits, zero_base_answers = compute_base_logits(
        model,
        zero_test,
        pool,
        bos_id,
        EVAL_SEED,
        device,
        ZERO_SEQ_LEN,
        ZERO_DIVERGE_IDX,
        ZERO_N_SOURCE,
    )
    zero_base_metrics = compute_accuracy_metrics(zero_base_logits, zero_base_answers)
    print(f"    Zero-distractor: {zero_base_metrics['accuracy']:.1%}")

    std_base_logits, std_base_answers = compute_base_logits(
        model,
        std_test,
        pool,
        bos_id,
        EVAL_SEED,
        device,
        STD_SEQ_LEN,
        STD_DIVERGE_IDX,
        STD_N_SOURCE,
    )
    std_base_metrics = compute_accuracy_metrics(std_base_logits, std_base_answers)
    print(f"    Standard LSC:    {std_base_metrics['accuracy']:.1%}")

    # ================================================================
    # RESULTS
    # ================================================================
    print("\n" + "=" * 70)
    print("RESULTS: 2x2 Transfer Matrix")
    print("=" * 70)

    zz = transfer_matrix["zero_on_zero"]["accuracy"]
    zs = transfer_matrix["zero_on_std"]["accuracy"]
    sz = transfer_matrix["std_on_zero"]["accuracy"]
    ss = transfer_matrix["std_on_std"]["accuracy"]

    print(f"\n  {'':20s} | {'Zero test':>12s} | {'Std test':>12s}")
    print(f"  {'-' * 20}-+-{'-' * 12}-+-{'-' * 12}")
    print(f"  {'Zero circuit':20s} | {zz:12.1%} | {zs:12.1%}")
    print(f"  {'Standard circuit':20s} | {sz:12.1%} | {ss:12.1%}")

    print(f"\n  Jaccard(zero, std): {jaccard:.4f}")
    print(f"  Reference: within-band Jaccard ~0.59, between-band ~0.56")

    # Interpretation
    print(f"\n  Interpretation:")
    diag_mean = (zz + ss) / 2
    off_diag_mean = (zs + sz) / 2

    if off_diag_mean < diag_mean * 0.85:
        print(
            f"  -> BLOCK-DIAGONAL: off-diag ({off_diag_mean:.1%}) << diag ({diag_mean:.1%})"
        )
        print(f"  -> Pipeline DETECTS mechanistic difference -> POSITIVE CONTROL WORKS")
    elif off_diag_mean < diag_mean * 0.95:
        print(f"  -> MODERATE: off-diag ({off_diag_mean:.1%}) < diag ({diag_mean:.1%})")
        print(f"  -> Some evidence of different circuits, but transfer is substantial")
    else:
        print(f"  -> UNIFORM: off-diag ({off_diag_mean:.1%}) ≈ diag ({diag_mean:.1%})")
        print(
            f"  -> Same mechanism for both conditions -> positive control NOT available"
        )
        print(f"  -> This confirms LSC is a single-mechanism task on Pythia")

    # Compare to frequency-band transfer
    freq_transfer_eff = 0.926  # Pythia-160m from paper
    if off_diag_mean / diag_mean < freq_transfer_eff:
        print(
            f"\n  Cross-condition transfer ({off_diag_mean / diag_mean:.1%}) < "
            f"frequency-band transfer ({freq_transfer_eff:.1%})"
        )
        print(
            f"  -> Distractor-length difference produces MORE transfer degradation "
            f"than frequency-band difference"
        )
    else:
        print(
            f"\n  Cross-condition transfer ({off_diag_mean / diag_mean:.1%}) >= "
            f"frequency-band transfer ({freq_transfer_eff:.1%})"
        )
        print(
            f"  -> Distractor length does not differentiate circuits more than frequency bands"
        )

    # ================================================================
    # SAVE
    # ================================================================
    results = {
        "model": MODEL_NAME,
        "threshold": THRESHOLD,
        "zero_distractor": {
            "n_edges": n_edges_zero,
            "size_fraction": n_edges_zero / total_possible,
            "base_accuracy": zero_base_metrics["accuracy"],
            "training_time_seconds": training_time,
        },
        "standard_lsc": {
            "n_edges": n_edges_std,
            "base_accuracy": std_base_metrics["accuracy"],
        },
        "jaccard": {
            "similarity": jaccard,
            "intersection": intersection,
            "union": union,
            "zero_edges": len(zero_edges),
            "std_edges": len(std_edges),
        },
        "transfer_matrix": {
            "zero_on_zero": transfer_matrix["zero_on_zero"],
            "zero_on_std": transfer_matrix["zero_on_std"],
            "std_on_zero": transfer_matrix["std_on_zero"],
            "std_on_std": transfer_matrix["std_on_std"],
        },
        "base_metrics": {
            "zero_distractor": zero_base_metrics,
            "standard": std_base_metrics,
        },
    }

    results_path = OUTPUT_DIR / "positive_control_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
