#!/usr/bin/env python3
"""
Reverse-Copy LSC: Base Model Feasibility Check
================================================
Tests whether Pythia-160m can solve the reverse-copy LSC task.

Runs the base model (no circuit intervention) on:
1. Reverse-copy test data (225 examples); the feasibility check
2. Standard LSC control-band test data (225 examples); sanity check

Reports:
- Top-1, top-5, top-10 accuracy
- What the model actually predicts (top-5 predictions per example)
- Whether the model uses standard induction (predicts R1, offset +1
  from first S5) vs some other mechanism

Decision criteria:
  >= 50%: proceed with ACDC extraction
  30-50%: proceed with caveats
  < 30%: reverse-copy not viable, acknowledge limitation
"""

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch as t

ISC_ROOT = Path(__file__).resolve().parent.parent.parent
REVERSE_DATA_DIR = ISC_ROOT / "LSC_data" / "reverse_lsc"
STANDARD_DATA_DIR = (
    ISC_ROOT / "LSC_data" / "datasets" / "matched" / "draw_1" / "control"
)
OUTPUT_DIR = Path(__file__).resolve().parent


# ============================================================================
# COMPATIBILITY PATCH
# ============================================================================


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


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate_base_model(model, dataset: dict, task_name: str, device: str) -> dict:
    """
    Run base model on a dataset, report accuracy and prediction analysis.
    """
    examples = dataset["examples"]
    bos_id = model.tokenizer.bos_token_id

    n_correct_top1 = 0
    n_correct_top5 = 0
    n_correct_top10 = 0
    correct_probs = []

    # Track what the model actually predicts
    prediction_analysis = []

    for i, ex in enumerate(examples):
        token_ids = ex["token_ids"]  # 21 tokens, no BOS
        target_id = ex["target_token_id"]

        # Prepend BOS -> 22 tokens
        input_ids = t.tensor([[bos_id] + token_ids], dtype=t.long, device=device)

        with t.no_grad():
            logits = model(input_ids)  # (1, seq_len, vocab_size)

        # Prediction at last position (position 21 with BOS = second S5)
        last_logits = logits[0, -1, :]  # (vocab_size,)
        probs = t.softmax(last_logits, dim=-1)

        top10_ids = t.topk(last_logits, k=10).indices.tolist()
        top10_probs = [probs[tid].item() for tid in top10_ids]

        if top10_ids[0] == target_id:
            n_correct_top1 += 1
        if target_id in top10_ids[:5]:
            n_correct_top5 += 1
        if target_id in top10_ids:
            n_correct_top10 += 1
        correct_probs.append(probs[target_id].item())

        # Decode predictions for analysis
        top5_tokens = [model.tokenizer.decode([tid]) for tid in top10_ids[:5]]

        # For reverse-copy: check if model predicts R1 (standard induction)
        # R1 is at position 6 (no BOS) = position 7 (with BOS)
        # In standard LSC: T is at position 5 (no BOS)
        # In reverse-copy: T is at position 0 (no BOS)
        r1_id = token_ids[6] if len(token_ids) > 6 else None  # R1 position
        t_after_s5_id = token_ids[6] if task_name == "reverse_lsc" else None

        prediction_analysis.append(
            {
                "example_id": i,
                "target_id": target_id,
                "target_token": ex["target_token_string"],
                "predicted_id": top10_ids[0],
                "predicted_token": top5_tokens[0],
                "correct": top10_ids[0] == target_id,
                "target_rank": (top10_ids.index(target_id) + 1)
                if target_id in top10_ids
                else ">10",
                "top5_tokens": top5_tokens,
                "top5_probs": top10_probs[:5],
                "correct_prob": probs[target_id].item(),
            }
        )

        # For reverse-copy, check if model predicts R1 (induction offset +1)
        if task_name == "reverse_lsc" and r1_id is not None:
            prediction_analysis[-1]["r1_id"] = r1_id
            prediction_analysis[-1]["predicts_r1"] = top10_ids[0] == r1_id

    n = len(examples)
    results = {
        "task": task_name,
        "n_examples": n,
        "top1_accuracy": n_correct_top1 / n,
        "top5_accuracy": n_correct_top5 / n,
        "top10_accuracy": n_correct_top10 / n,
        "mean_correct_prob": float(np.mean(correct_probs)),
        "prediction_analysis": prediction_analysis,
    }

    # Summary statistics for reverse-copy
    if task_name == "reverse_lsc":
        n_predicts_r1 = sum(
            1 for p in prediction_analysis if p.get("predicts_r1", False)
        )
        results["n_predicts_r1"] = n_predicts_r1
        results["frac_predicts_r1"] = n_predicts_r1 / n

    return results


def print_results(results: dict):
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 60}")
    print(f"  {results['task']}  ({results['n_examples']} examples)")
    print(f"{'=' * 60}")
    print(f"  Top-1 accuracy:  {results['top1_accuracy']:.1%}")
    print(f"  Top-5 accuracy:  {results['top5_accuracy']:.1%}")
    print(f"  Top-10 accuracy: {results['top10_accuracy']:.1%}")
    print(f"  Mean P(correct): {results['mean_correct_prob']:.4f}")

    if "frac_predicts_r1" in results:
        print(f"\n  Induction analysis (reverse-copy):")
        print(
            f"    Predicts R1 (std induction offset +1): {results['frac_predicts_r1']:.1%}"
        )
        print(
            f"    Predicts T (correct, reverse retrieval): {results['top1_accuracy']:.1%}"
        )

    # Show first 10 examples
    print(f"\n  First 10 predictions:")
    for p in results["prediction_analysis"][:10]:
        mark = "Y" if p["correct"] else "N"
        rank = p["target_rank"]
        print(
            f"    {mark} target={p['target_token']:15s} predicted={p['predicted_token']:15s} "
            f"rank={rank}  P(correct)={p['correct_prob']:.4f}"
        )

    # Prediction distribution
    pred_tokens = [p["predicted_token"] for p in results["prediction_analysis"]]
    counter = Counter(pred_tokens)
    print(f"\n  Most common predictions:")
    for tok, count in counter.most_common(10):
        print(f"    {tok:15s}: {count:4d} ({count / len(pred_tokens):.1%})")


def main():
    print("=" * 60)
    print("Reverse-Copy LSC: Base Model Feasibility Check")
    print("Model: Pythia-160m")
    print("=" * 60)

    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("\nLoading Pythia-160m...")
    import transformer_lens as tl

    _patch_gpt_neox_config()
    model = tl.HookedTransformer.from_pretrained(
        "pythia-160m",
        device=device,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Load datasets
    print("\nLoading datasets...")

    reverse_test_path = REVERSE_DATA_DIR / "test.json"
    if not reverse_test_path.exists():
        print(f"ERROR: Reverse-copy test data not found at {reverse_test_path}")
        sys.exit(1)
    with open(reverse_test_path) as f:
        reverse_test = json.load(f)
    print(f"  Reverse-copy test: {len(reverse_test['examples'])} examples")

    standard_test_path = STANDARD_DATA_DIR / "test.json"
    if not standard_test_path.exists():
        print(f"WARNING: Standard LSC test data not found at {standard_test_path}")
        standard_test = None
    else:
        with open(standard_test_path) as f:
            standard_test = json.load(f)
        print(f"  Standard LSC test: {len(standard_test['examples'])} examples")

    # Evaluate
    print("\nEvaluating on reverse-copy test data...")
    reverse_results = evaluate_base_model(model, reverse_test, "reverse_lsc", device)
    print_results(reverse_results)

    if standard_test is not None:
        print("\nEvaluating on standard LSC test data (sanity check)...")
        standard_results = evaluate_base_model(
            model, standard_test, "standard_lsc", device
        )
        print_results(standard_results)

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "base_eval_results.json"

    save_data = {
        "model": "pythia-160m",
        "device": device,
        "reverse_lsc": {
            "top1_accuracy": reverse_results["top1_accuracy"],
            "top5_accuracy": reverse_results["top5_accuracy"],
            "top10_accuracy": reverse_results["top10_accuracy"],
            "mean_correct_prob": reverse_results["mean_correct_prob"],
            "n_examples": reverse_results["n_examples"],
        },
    }
    if "frac_predicts_r1" in reverse_results:
        save_data["reverse_lsc"]["frac_predicts_r1"] = reverse_results[
            "frac_predicts_r1"
        ]

    if standard_test is not None:
        save_data["standard_lsc"] = {
            "top1_accuracy": standard_results["top1_accuracy"],
            "top5_accuracy": standard_results["top5_accuracy"],
            "top10_accuracy": standard_results["top10_accuracy"],
            "mean_correct_prob": standard_results["mean_correct_prob"],
            "n_examples": standard_results["n_examples"],
        }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Decision
    acc = reverse_results["top1_accuracy"]
    print(f"\n{'=' * 60}")
    print(f"DECISION:")
    if acc >= 0.50:
        print(f"  Top-1 accuracy = {acc:.1%} >= 50% -> PROCEED with ACDC extraction")
    elif acc >= 0.30:
        print(f"  Top-1 accuracy = {acc:.1%} (30-50%) -> PROCEED with caveats")
    else:
        print(f"  Top-1 accuracy = {acc:.1%} < 30% -> REVERSE-COPY NOT VIABLE")
        print(f"  Acknowledge limitation in paper; do not substitute IOI.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
