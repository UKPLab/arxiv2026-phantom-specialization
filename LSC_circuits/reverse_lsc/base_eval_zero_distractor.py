#!/usr/bin/env python3
"""
Zero-Distractor LSC: Base Model Feasibility Check
===================================================
Tests whether Pythia-160m solves zero-distractor LSC (it should; this is
easier than standard LSC) and checks what the model predicts, to confirm
the task is viable for a positive control.

Also runs logit-lens analysis to check if convergence depth differs between
standard and zero-distractor LSC, which would suggest different processing.
"""

import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch as t

ISC_ROOT = Path(__file__).resolve().parent.parent.parent
ZERO_DATA_DIR = ISC_ROOT / "LSC_data" / "reverse_lsc" / "zero_distractor"
STANDARD_DATA_DIR = (
    ISC_ROOT / "LSC_data" / "datasets" / "matched" / "draw_1" / "control"
)
OUTPUT_DIR = Path(__file__).resolve().parent


# ============================================================================
# COMPATIBILITY PATCH
# ============================================================================


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


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate_base_model(model, dataset: dict, task_name: str, device: str) -> dict:
    """Run base model on a dataset, report accuracy."""
    examples = dataset["examples"]
    bos_id = model.tokenizer.bos_token_id

    n_correct_top1 = 0
    n_correct_top5 = 0
    n_correct_top10 = 0
    correct_probs = []
    predictions = []

    for i, ex in enumerate(examples):
        token_ids = ex["token_ids"]
        target_id = ex["target_token_id"]

        input_ids = t.tensor([[bos_id] + token_ids], dtype=t.long, device=device)

        with t.no_grad():
            logits = model(input_ids)

        last_logits = logits[0, -1, :]
        probs = t.softmax(last_logits, dim=-1)

        top10_ids = t.topk(last_logits, k=10).indices.tolist()

        if top10_ids[0] == target_id:
            n_correct_top1 += 1
        if target_id in top10_ids[:5]:
            n_correct_top5 += 1
        if target_id in top10_ids:
            n_correct_top10 += 1
        correct_probs.append(probs[target_id].item())

        top5_tokens = [model.tokenizer.decode([tid]) for tid in top10_ids[:5]]
        predictions.append(
            {
                "example_id": i,
                "target_token": ex["target_token_string"],
                "predicted_token": top5_tokens[0],
                "correct": top10_ids[0] == target_id,
                "correct_prob": probs[target_id].item(),
            }
        )

    n = len(examples)
    return {
        "task": task_name,
        "n_examples": n,
        "top1_accuracy": n_correct_top1 / n,
        "top5_accuracy": n_correct_top5 / n,
        "top10_accuracy": n_correct_top10 / n,
        "mean_correct_prob": float(np.mean(correct_probs)),
        "predictions": predictions,
    }


def run_logit_lens(
    model, dataset: dict, task_name: str, device: str, n_examples: int = 50
) -> dict:
    """
    Run logit-lens analysis: at each layer, check P(correct) by applying
    the unembedding matrix to the residual stream at the prediction position.
    Returns per-layer P(correct) and convergence depth.
    """
    examples = dataset["examples"][:n_examples]
    bos_id = model.tokenizer.bos_token_id
    n_layers = model.cfg.n_layers

    layer_correct_probs = [[] for _ in range(n_layers + 1)]  # +1 for final

    for ex in examples:
        token_ids = ex["token_ids"]
        target_id = ex["target_token_id"]

        input_ids = t.tensor([[bos_id] + token_ids], dtype=t.long, device=device)

        with t.no_grad():
            _, cache = model.run_with_cache(input_ids)

        # For each layer, get residual stream at prediction position and unembed
        for layer in range(n_layers):
            resid = cache[f"blocks.{layer}.hook_resid_post"][0, -1, :]  # (d_model,)
            logits = model.unembed(resid.unsqueeze(0).unsqueeze(0))[0, 0, :]  # (vocab,)
            prob = t.softmax(logits, dim=-1)[target_id].item()
            layer_correct_probs[layer].append(prob)

        # Final output
        final_logits = model.unembed(cache[f"blocks.{n_layers - 1}.hook_resid_post"])[
            0, -1, :
        ]
        prob = t.softmax(final_logits, dim=-1)[target_id].item()
        layer_correct_probs[n_layers].append(prob)

    # Average across examples
    mean_probs = [float(np.mean(lp)) for lp in layer_correct_probs]

    # Find convergence: first layer where P(correct) exceeds 50% of final
    final_prob = mean_probs[-1]
    threshold = 0.5 * final_prob
    convergence_layer = n_layers  # default: last layer
    for layer, prob in enumerate(mean_probs[:-1]):
        if prob >= threshold:
            convergence_layer = layer
            break

    return {
        "task": task_name,
        "n_examples": n_examples,
        "n_layers": n_layers,
        "per_layer_mean_p_correct": mean_probs,
        "convergence_layer": convergence_layer,
        "convergence_frac_depth": convergence_layer / n_layers,
        "final_mean_p_correct": final_prob,
    }


def print_results(results: dict):
    print(f"\n{'=' * 60}")
    print(f"  {results['task']}  ({results['n_examples']} examples)")
    print(f"{'=' * 60}")
    print(f"  Top-1 accuracy:  {results['top1_accuracy']:.1%}")
    print(f"  Top-5 accuracy:  {results['top5_accuracy']:.1%}")
    print(f"  Top-10 accuracy: {results['top10_accuracy']:.1%}")
    print(f"  Mean P(correct): {results['mean_correct_prob']:.4f}")

    # First 10 examples
    print(f"\n  First 10 predictions:")
    for p in results["predictions"][:10]:
        mark = "Y" if p["correct"] else "N"
        print(
            f"    {mark} target={p['target_token']:15s} predicted={p['predicted_token']:15s} "
            f"P(correct)={p['correct_prob']:.4f}"
        )


def print_logit_lens(ll: dict):
    print(f"\n  Logit-lens ({ll['task']}, {ll['n_examples']} examples):")
    print(
        f"  Convergence layer: {ll['convergence_layer']}/{ll['n_layers']} "
        f"(frac depth: {ll['convergence_frac_depth']:.2f})"
    )
    print(f"  Final P(correct): {ll['final_mean_p_correct']:.4f}")
    print(f"  Per-layer P(correct):")
    for i, p in enumerate(ll["per_layer_mean_p_correct"]):
        bar = "█" * int(p * 40)
        label = "final" if i == ll["n_layers"] else f"L{i:2d}"
        print(f"    {label}: {p:.4f} {bar}")


def main():
    print("=" * 60)
    print("Zero-Distractor LSC: Feasibility Check + Logit Lens")
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

    zero_test_path = ZERO_DATA_DIR / "test.json"
    if not zero_test_path.exists():
        print(f"ERROR: Zero-distractor test data not found at {zero_test_path}")
        sys.exit(1)
    with open(zero_test_path) as f:
        zero_test = json.load(f)
    print(f"  Zero-distractor test: {len(zero_test['examples'])} examples")

    standard_test_path = STANDARD_DATA_DIR / "test.json"
    with open(standard_test_path) as f:
        standard_test = json.load(f)
    print(f"  Standard LSC test: {len(standard_test['examples'])} examples")

    # Base accuracy
    print("\n--- Base Model Accuracy ---")

    print("\nEvaluating zero-distractor LSC...")
    zero_results = evaluate_base_model(model, zero_test, "zero_distractor_lsc", device)
    print_results(zero_results)

    print("\nEvaluating standard LSC...")
    standard_results = evaluate_base_model(model, standard_test, "standard_lsc", device)
    print_results(standard_results)

    # Logit lens
    print("\n--- Logit Lens Analysis ---")

    print("\nRunning logit-lens on zero-distractor (50 examples)...")
    zero_ll = run_logit_lens(
        model, zero_test, "zero_distractor_lsc", device, n_examples=50
    )
    print_logit_lens(zero_ll)

    print("\nRunning logit-lens on standard LSC (50 examples)...")
    standard_ll = run_logit_lens(
        model, standard_test, "standard_lsc", device, n_examples=50
    )
    print_logit_lens(standard_ll)

    # Compare convergence
    print(f"\n--- Convergence Comparison ---")
    print(
        f"  Standard LSC:     layer {standard_ll['convergence_layer']}/{standard_ll['n_layers']} "
        f"(frac: {standard_ll['convergence_frac_depth']:.2f})"
    )
    print(
        f"  Zero-distractor:  layer {zero_ll['convergence_layer']}/{zero_ll['n_layers']} "
        f"(frac: {zero_ll['convergence_frac_depth']:.2f})"
    )
    diff = standard_ll["convergence_layer"] - zero_ll["convergence_layer"]
    if diff > 0:
        print(
            f"  -> Zero-distractor converges {diff} layers EARLIER (shallower processing)"
        )
    elif diff < 0:
        print(f"  -> Zero-distractor converges {-diff} layers LATER")
    else:
        print(f"  -> Same convergence depth")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "base_eval_zero_distractor.json"

    save_data = {
        "model": "pythia-160m",
        "zero_distractor_lsc": {
            "top1_accuracy": zero_results["top1_accuracy"],
            "top5_accuracy": zero_results["top5_accuracy"],
            "top10_accuracy": zero_results["top10_accuracy"],
            "mean_correct_prob": zero_results["mean_correct_prob"],
            "n_examples": zero_results["n_examples"],
        },
        "standard_lsc": {
            "top1_accuracy": standard_results["top1_accuracy"],
            "top5_accuracy": standard_results["top5_accuracy"],
            "top10_accuracy": standard_results["top10_accuracy"],
            "mean_correct_prob": standard_results["mean_correct_prob"],
            "n_examples": standard_results["n_examples"],
        },
        "logit_lens": {
            "zero_distractor": zero_ll,
            "standard": standard_ll,
        },
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Decision
    acc = zero_results["top1_accuracy"]
    print(f"\n{'=' * 60}")
    print(f"DECISION:")
    if acc >= 0.50:
        print(f"  Zero-distractor accuracy = {acc:.1%} >= 50% -> PROCEED with ACDC")
        if diff > 0:
            print(
                f"  Logit-lens confirms different processing depth -> strong positive control candidate"
            )
        else:
            print(
                f"  Logit-lens shows similar depth -> circuit may still differ structurally"
            )
    else:
        print(f"  Zero-distractor accuracy = {acc:.1%} < 50% -> NOT VIABLE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
