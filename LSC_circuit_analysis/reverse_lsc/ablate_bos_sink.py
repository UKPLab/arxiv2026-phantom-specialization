#!/usr/bin/env python3
"""
Targeted BOS-Sink Ablation
===========================
Tests whether removing all BOS-sink edges from circuits
changes accuracy and whether the BOS-sink contribution is distinguishable
from noise.

For each model and band circuit:
1. Load full circuit prune_scores
2. Identify all edges whose destination head is classified as bos_sink
3. Remove those edges (set to 0 in prune_scores)
4. Evaluate accuracy of the modified circuit
5. Compare: full circuit vs circuit-minus-BOS-sink vs random-edge-removal control

Also tests: what happens if you remove only BOS-sink edges and keep
everything else ablated? (standalone BOS-sink contribution)
"""

import json
import sys
import os
import pickle
import random
import gc
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch as t
import torch.nn.functional as F
import pandas as pd

# Add AutoCircuit to path
ISC_ROOT = PROJECT_ROOT
sys.path.insert(0, str(ISC_ROOT / "circuit_discovery" / "auto-circuit"))

CIRCUIT_DIR = ISC_ROOT / "LSC_circuits" / "circuit_discovery" / "circuits"
DATA_DIR = ISC_ROOT / "LSC_data" / "datasets" / "matched"
POOL_DIR = ISC_ROOT / "LSC_data" / "lsc_token_pools" / "matched"
HEAD_ROLES_PATH = (
    ISC_ROOT
    / "LSC_circuit_analysis"
    / "03_Phase_Representational"
    / "outputs"
    / "attention"
    / "base"
    / "analysis"
    / "04_head_role_classification.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

MODELS = [
    ("pythia_160m", "pythia-160m", 12, 12),  # dir, name, n_layers, n_heads
    ("pythia_410m", "pythia-410m", 24, 16),
    ("pythia_1b", "pythia-1b", 16, 8),
    ("pythia_1.4b", "pythia-1.4b", 24, 16),
]
BANDS = ["low", "medium", "high", "very_high", "control"]

SEQ_LEN = 22  # 21 + BOS
DIVERGE_IDX = 17
N_SOURCE = 5
EVAL_SEED = 123
DEVICE = "cuda" if t.cuda.is_available() else "cpu"


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    if t.cuda.is_available():
        t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True


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


def load_model(model_name):
    import transformer_lens as tl

    _patch_gpt_neox_config()
    model = tl.HookedTransformer.from_pretrained(
        model_name,
        device=DEVICE,
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


# ============================================================================
# HEAD ROLE LOADING
# ============================================================================


def load_bos_sink_heads(model_name: str) -> set:
    """Load set of (layer, head) tuples classified as bos_sink."""
    df = pd.read_csv(HEAD_ROLES_PATH)
    bos = df[(df["model"] == model_name) & (df["role"] == "bos_sink")]
    return {(row["layer"], row["head"]) for _, row in bos.iterrows()}


# ============================================================================
# CIRCUIT MANIPULATION
# ============================================================================


def count_edges(prune_scores: dict) -> int:
    return sum(t.isinf(s).sum().item() for s in prune_scores.values())


def remove_bos_sink_edges(
    prune_scores: dict, bos_sink_heads: set, n_heads: int
) -> dict:
    """Remove all edges whose DESTINATION head is a BOS-sink head.

    In AutoCircuit's factorized representation, edge names contain
    destination head information. For attention input edges:
      blocks.{layer}.hook_attn_in[{head_idx},{src_idx}]
    We zero out edges where the destination head is a BOS-sink head.
    """
    modified = {}
    n_removed = 0

    for name, scores in prune_scores.items():
        new_scores = scores.clone()

        # Check if this is an attention input (destination is a specific head)
        if "hook_attn_in" in name:
            # Extract layer from name: blocks.{layer}.hook_attn_in
            layer = int(name.split(".")[1])

            # scores shape: (n_heads, n_sources) for factorized
            if scores.ndim == 2:
                for head in range(scores.shape[0]):
                    if (layer, head) in bos_sink_heads:
                        mask = t.isinf(scores[head]) & (scores[head] > 0)
                        n_removed += mask.sum().item()
                        new_scores[head][mask] = 0.0
            elif scores.ndim == 1:
                # Non-factorized or single-head: check if any head at this layer is BOS-sink
                for head in range(n_heads):
                    if (layer, head) in bos_sink_heads:
                        mask = t.isinf(scores) & (scores > 0)
                        n_removed += mask.sum().item()
                        new_scores[mask] = 0.0
                        break

        modified[name] = new_scores

    return modified, n_removed


def remove_random_edges(prune_scores: dict, n_to_remove: int, seed: int) -> dict:
    """Remove n_to_remove random circuit edges as a control."""
    rng = random.Random(seed)

    # Collect all circuit edge positions
    circuit_edges = []
    for name, scores in prune_scores.items():
        mask = t.isinf(scores) & (scores > 0)
        for idx in mask.nonzero(as_tuple=False):
            circuit_edges.append((name, tuple(i.item() for i in idx)))

    # Sample edges to remove
    if n_to_remove >= len(circuit_edges):
        n_to_remove = len(circuit_edges) - 1
    to_remove = set(rng.sample(range(len(circuit_edges)), n_to_remove))

    modified = {name: scores.clone() for name, scores in prune_scores.items()}
    for i in to_remove:
        name, idx = circuit_edges[i]
        modified[name][idx] = 0.0

    return modified


def evaluate_circuit(patchable, prune_scores_dev, n_edges, dataset, pool, bos_id):
    """Evaluate a circuit on a test dataset. Returns accuracy."""
    from auto_circuit.data import PromptDataset, PromptDataLoader
    from auto_circuit.prune import run_circuits
    from auto_circuit.types import PatchType, AblationType

    examples = dataset["examples"]
    pool_ids = [tok["token_id"] for tok in pool["tokens"]]
    rng = random.Random(EVAL_SEED)

    indices = list(range(len(examples)))
    rng.shuffle(indices)

    clean_prompts, corrupt_prompts, answers, wrong_answers = [], [], [], []

    for idx in indices:
        ex = examples[idx]
        token_ids = ex["token_ids"]
        clean = [bos_id] + token_ids

        used_set = set(token_ids)
        available = [tid for tid in pool_ids if tid not in used_set]
        if len(available) >= N_SOURCE:
            replacements = rng.sample(available, N_SOURCE)
        else:
            replacements = rng.sample(pool_ids, N_SOURCE)

        corrupt = [bos_id] + token_ids[:16] + replacements
        wrong_pool = [tid for tid in pool_ids if tid != ex["target_token_id"]]

        clean_prompts.append(t.tensor(clean, dtype=t.long, device=DEVICE))
        corrupt_prompts.append(t.tensor(corrupt, dtype=t.long, device=DEVICE))
        answers.append(t.tensor([ex["target_token_id"]], dtype=t.long, device=DEVICE))
        wrong_answers.append(
            t.tensor([rng.choice(wrong_pool)], dtype=t.long, device=DEVICE)
        )

    ds = PromptDataset(clean_prompts, corrupt_prompts, answers, wrong_answers)
    n = len(indices)
    bs = min(225, n)
    while bs > 1 and n % bs != 0:
        bs -= 1

    loader = PromptDataLoader(
        ds, seq_len=SEQ_LEN, diverge_idx=DIVERGE_IDX, batch_size=bs
    )

    if n_edges == 0:
        return 0.0

    set_all_seeds(EVAL_SEED)
    loader1 = PromptDataLoader(
        PromptDataset(clean_prompts, corrupt_prompts, answers, wrong_answers),
        seq_len=SEQ_LEN,
        diverge_idx=DIVERGE_IDX,
        batch_size=bs,
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

    set_all_seeds(EVAL_SEED)
    loader2 = PromptDataLoader(
        PromptDataset(clean_prompts, corrupt_prompts, answers, wrong_answers),
        seq_len=SEQ_LEN,
        diverge_idx=DIVERGE_IDX,
        batch_size=bs,
    )

    correct = 0
    total = 0
    for batch in loader2:
        logits = outputs[n_edges][batch.key]
        if len(logits.shape) == 3:
            logits = logits[:, -1, :]
        preds = logits.argmax(dim=-1)
        answer_ids = batch.answers.squeeze(-1)
        correct += (preds == answer_ids).sum().item()
        total += len(answer_ids)

    return correct / total if total > 0 else 0.0


def main():
    print("=" * 70)
    print("Targeted BOS-Sink Ablation")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for model_dir, model_name, n_layers, n_heads in MODELS:
        print(f"\n{'=' * 60}")
        print(f"  {model_name}")
        print(f"{'=' * 60}")

        # Load head roles
        bos_sink_heads = load_bos_sink_heads(model_name)
        print(f"  BOS-sink heads: {len(bos_sink_heads)} of {n_layers * n_heads}")
        for l, h in sorted(bos_sink_heads):
            print(f"    L{l}.H{h}", end="")
        print()

        # Load model
        from auto_circuit.utils.graph_utils import patchable_model

        model = load_model(model_name)
        bos_id = model.tokenizer.bos_token_id

        patchable = patchable_model(
            model=model,
            factorized=True,
            slice_output="last_seq",
            seq_len=None,
            separate_qkv=False,
            device=DEVICE,
        )

        # Load pool
        with open(POOL_DIR / "lsc_pool_control.json") as f:
            pool = json.load(f)

        for band in BANDS:
            # Load circuit
            scores_path = CIRCUIT_DIR / model_dir / band / "draw_1" / "prune_scores.pkl"
            if not scores_path.exists():
                # Try with dot notation
                scores_path = (
                    CIRCUIT_DIR
                    / model_dir.replace("_", ".")
                    / band
                    / "draw_1"
                    / "prune_scores.pkl"
                )
            if not scores_path.exists():
                print(f"  Skipping {band}: no circuit at {scores_path}")
                continue

            with open(scores_path, "rb") as f:
                prune_scores = pickle.load(f)

            n_full = count_edges(prune_scores)

            # Load test data
            with open(DATA_DIR / "draw_1" / band / "test.json") as f:
                test_data = json.load(f)

            # 1. Full circuit accuracy
            scores_dev = {k: v.to(DEVICE) for k, v in prune_scores.items()}
            full_acc = evaluate_circuit(
                patchable, scores_dev, n_full, test_data, pool, bos_id
            )

            # 2. Circuit minus BOS-sink edges
            modified_scores, n_removed = remove_bos_sink_edges(
                prune_scores, bos_sink_heads, n_heads
            )
            n_modified = count_edges(modified_scores)
            mod_dev = {k: v.to(DEVICE) for k, v in modified_scores.items()}
            no_bos_acc = evaluate_circuit(
                patchable, mod_dev, n_modified, test_data, pool, bos_id
            )

            # 3. Random control: remove same number of random edges
            if n_removed > 0:
                random_scores = remove_random_edges(prune_scores, n_removed, seed=42)
                n_random = count_edges(random_scores)
                rand_dev = {k: v.to(DEVICE) for k, v in random_scores.items()}
                random_acc = evaluate_circuit(
                    patchable, rand_dev, n_random, test_data, pool, bos_id
                )
            else:
                random_acc = full_acc

            acc_drop = full_acc - no_bos_acc
            random_drop = full_acc - random_acc

            print(
                f"  {band:12s}: full={full_acc:.1%} | -BOS={no_bos_acc:.1%} (Δ={acc_drop:+.1%}) | "
                f"-random={random_acc:.1%} (Δ={random_drop:+.1%}) | "
                f"removed={n_removed}/{n_full}"
            )

            results.append(
                {
                    "model": model_name,
                    "band": band,
                    "n_full_edges": n_full,
                    "n_bos_removed": n_removed,
                    "frac_bos_removed": n_removed / n_full if n_full else 0,
                    "full_accuracy": full_acc,
                    "no_bos_accuracy": no_bos_acc,
                    "random_control_accuracy": random_acc,
                    "bos_drop": acc_drop,
                    "random_drop": random_drop,
                }
            )

        # Cleanup
        del model, patchable
        gc.collect()
        t.cuda.empty_cache()

    # Summary
    print(f"\n\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    print(
        f"\n  {'Model':15s} | {'BOS edges':>10s} | {'Full acc':>8s} | {'-BOS acc':>8s} | {'Δ BOS':>8s} | {'Δ random':>8s}"
    )
    print(f"  {'-' * 15}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}-+-{'-' * 8}")

    for model_dir, model_name, _, _ in MODELS:
        model_results = [r for r in results if r["model"] == model_name]
        if not model_results:
            continue
        mean_full = np.mean([r["full_accuracy"] for r in model_results])
        mean_no_bos = np.mean([r["no_bos_accuracy"] for r in model_results])
        mean_random = np.mean([r["random_control_accuracy"] for r in model_results])
        mean_bos_removed = np.mean([r["n_bos_removed"] for r in model_results])
        mean_frac = np.mean([r["frac_bos_removed"] for r in model_results])
        mean_bos_drop = mean_full - mean_no_bos
        mean_rand_drop = mean_full - mean_random

        print(
            f"  {model_name:15s} | {mean_bos_removed:5.0f} ({mean_frac:4.1%}) | {mean_full:8.1%} | "
            f"{mean_no_bos:8.1%} | {mean_bos_drop:+8.1%} | {mean_rand_drop:+8.1%}"
        )

    # Save
    output_path = OUTPUT_DIR / "bos_sink_ablation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
