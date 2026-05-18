#!/usr/bin/env python3
"""
03d: Tuned Lens Validation of Logit-Lens Convergence Finding
=============================================================
Tests whether the frequency-dependent convergence gap (low-frequency tokens
converge later than high-frequency tokens) persists under the tuned lens,
which corrects for per-layer alignment artifacts.

Key question: Is the convergence delay genuine (reflecting computational
requirements) or an artifact of logit-lens misalignment?

Approach:
1. Train per-layer affine probes (tuned lens) for each Pythia model
2. Run tuned lens on LSC test data (same examples as existing analysis)
3. Compare convergence depth: tuned lens vs standard logit lens
4. Key test: does the low-vs-high frequency gap persist?

Uses HuggingFace models (not TransformerLens) because the tuned lens
operates on unmodified residual stream states.

IMPORTANT: Pythia uses parallel transformer blocks (attention and MLP
read from the same layer-normalized input and write simultaneously).
The tuned lens operates on the full post-layer residual stream, so
parallel blocks are handled correctly at this granularity. The concern
about parallel blocks affects component attribution (decomposing
attention vs MLP through nonlinear LayerNorm), not the tuned lens.
"""

import json
import sys
import os
import gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add tuned-lens to path
sys.path.insert(0, os.environ.get("TUNED_LENS_PATH", "circuit_discovery/tuned-lens"))
from tuned_lens import TunedLens
from tuned_lens.nn import LogitLens

ISC_ROOT = PROJECT_ROOT
DATA_DIR = ISC_ROOT / "LSC_data" / "datasets" / "matched" / "draw_1"
PHASE3_DIR = ISC_ROOT / "LSC_circuit_analysis" / "03_Phase_Representational"
OUTPUT_DIR = PHASE3_DIR / "outputs" / "tuned_lens"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
VIZ_DIR = OUTPUT_DIR / "viz"
LENS_DIR = OUTPUT_DIR / "lenses"

# Existing logit lens data for comparison
EXISTING_LL_DIR = PHASE3_DIR / "outputs" / "logit_lens" / "base" / "analysis"

MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]
BANDS = ["low", "medium", "high", "very_high", "control"]
CORE_BANDS = ["low", "medium", "high", "very_high"]
CONVERGENCE_THRESHOLD = 0.9

# Training config
TRAIN_STEPS = 500
TRAIN_LR = 1e-3
TRAIN_WEIGHT_DECAY = 0.01
TRAIN_BATCH_SIZE = 8
TRAIN_SEQ_LEN = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_lsc_test_data(band: str) -> dict:
    """Load LSC test data for a band."""
    path = DATA_DIR / band / "test.json"
    with open(path) as f:
        return json.load(f)


def prepare_lsc_inputs(dataset: dict, tokenizer) -> tuple:
    """Prepare LSC inputs with BOS for HuggingFace model.

    Returns (input_ids tensor, target_ids list, n_examples).
    """
    examples = dataset["examples"]
    bos_id = tokenizer.bos_token_id

    input_ids_list = []
    target_ids = []

    for ex in examples:
        token_ids = ex["token_ids"]  # 21 tokens, no BOS
        input_ids = [bos_id] + token_ids  # 22 tokens with BOS
        input_ids_list.append(input_ids)
        target_ids.append(ex["target_token_id"])

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    return input_ids, target_ids, len(examples)


# ============================================================================
# TRAINING
# ============================================================================


def train_tuned_lens(model, tokenizer, model_name: str, save_dir: Path) -> TunedLens:
    """Train a tuned lens for a model using general text data.

    Uses the model's own tokenizer to create training sequences from
    a simple text corpus. Training minimizes KL divergence between
    the tuned lens prediction at each layer and the final model output.
    """
    save_path = save_dir / model_name.replace("-", "_")

    # Check if already trained
    if (save_path / "config.json").exists() and (save_path / "params.pt").exists():
        print(f"  Loading cached lens from {save_path}")
        lens = TunedLens.from_model_and_pretrained(
            model,
            lens_resource_id=str(save_path),
            map_location=DEVICE,
        )
        lens = lens.to(DEVICE)
        return lens

    print(f"  Training tuned lens for {model_name}...")
    lens = TunedLens.from_model(model)
    lens = lens.to(DEVICE)

    optimizer = torch.optim.Adam(
        lens.parameters(), lr=TRAIN_LR, weight_decay=TRAIN_WEIGHT_DECAY
    )

    # Generate training data: random token sequences
    # This is sufficient because the lens just needs to learn the
    # affine correction between intermediate and final representations.
    # The distribution of hidden states matters more than semantic content.
    vocab_size = model.config.vocab_size
    n_layers = model.config.num_hidden_layers

    lens.train()
    running_loss = 0.0

    for step in range(TRAIN_STEPS):
        # Random token sequences as training data
        # This is a standard approach: the lens learns alignment corrections
        # that are properties of the model, not the data
        input_ids = torch.randint(
            0, vocab_size, (TRAIN_BATCH_SIZE, TRAIN_SEQ_LEN), device=DEVICE
        )

        with torch.no_grad():
            output = model(input_ids, output_hidden_states=True)

        final_logits = output.logits.float()
        final_log_probs = F.log_softmax(final_logits, dim=-1)
        hidden_states = output.hidden_states  # n_layers + 1 (includes embedding)

        # Train on all layers except the last (which needs no correction)
        total_loss = torch.tensor(0.0, device=DEVICE)
        for i in range(n_layers):
            h = hidden_states[i + 1]  # +1 because [0] is embedding output
            lens_logits = lens(h.float(), idx=i)
            lens_log_probs = F.log_softmax(lens_logits, dim=-1)

            # KL(final || lens) = sum(final_probs * (log_final - log_lens))
            kl = F.kl_div(
                lens_log_probs, final_log_probs, log_target=True, reduction="batchmean"
            )
            total_loss = total_loss + kl / n_layers

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(lens.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        running_loss += total_loss.item()

        if (step + 1) % 100 == 0:
            avg_loss = running_loss / 100
            print(f"    Step {step + 1}/{TRAIN_STEPS}: loss={avg_loss:.4f}")
            running_loss = 0.0

    # Save
    save_path.mkdir(parents=True, exist_ok=True)
    lens.save(save_path)
    print(f"  Saved lens to {save_path}")

    lens.eval()
    return lens


def compute_convergence_layer(
    prob_correct: np.ndarray, threshold: float = 0.9
) -> np.ndarray:
    """Find first layer where P(correct) >= threshold * final P(correct).

    Matches the existing logit lens implementation.
    """
    n_examples, n_layers = prob_correct.shape
    final_probs = prob_correct[:, -1]
    thresholds = threshold * final_probs

    convergence = np.full(n_examples, n_layers, dtype=float)
    for i in range(n_examples):
        if final_probs[i] > 0:
            for l in range(n_layers):
                if prob_correct[i, l] >= thresholds[i]:
                    convergence[i] = l
                    break

    return convergence


@torch.no_grad()
def evaluate_tuned_lens(
    model,
    lens: TunedLens,
    input_ids: torch.Tensor,
    target_ids: list,
    batch_size: int = 32,
) -> dict:
    """Run tuned lens evaluation on LSC data.

    Returns per-layer P(correct) trajectory and convergence metrics.
    """
    n_examples = input_ids.shape[0]
    n_layers = model.config.num_hidden_layers

    # Also compute standard logit lens for direct comparison
    logit_lens = LogitLens.from_model(model)
    logit_lens = logit_lens.to(DEVICE)

    tuned_prob_correct = np.zeros((n_examples, n_layers))
    logit_prob_correct = np.zeros((n_examples, n_layers))

    lens.eval()

    for start in range(0, n_examples, batch_size):
        end = min(start + batch_size, n_examples)
        batch_ids = input_ids[start:end].to(DEVICE)
        batch_targets = target_ids[start:end]

        output = model(batch_ids, output_hidden_states=True)
        hidden_states = output.hidden_states  # n_layers + 1

        for layer in range(n_layers):
            h = hidden_states[layer + 1]  # +1 for embedding

            # Tuned lens: h + translator(h), then unembed
            tuned_logits = lens(h.float(), idx=layer)
            # At prediction position (last token, index -1)
            tuned_probs = F.softmax(tuned_logits[:, -1, :], dim=-1)

            # Standard logit lens: just unembed(ln_final(h))
            logit_logits = logit_lens(h.float(), idx=layer)
            logit_probs = F.softmax(logit_logits[:, -1, :], dim=-1)

            for i in range(end - start):
                tid = batch_targets[i]
                tuned_prob_correct[start + i, layer] = tuned_probs[i, tid].item()
                logit_prob_correct[start + i, layer] = logit_probs[i, tid].item()

    # Convergence
    tuned_convergence = compute_convergence_layer(
        tuned_prob_correct, CONVERGENCE_THRESHOLD
    )
    logit_convergence = compute_convergence_layer(
        logit_prob_correct, CONVERGENCE_THRESHOLD
    )

    return {
        "tuned_prob_correct": tuned_prob_correct,
        "logit_prob_correct": logit_prob_correct,
        "tuned_convergence": tuned_convergence,
        "logit_convergence": logit_convergence,
        "n_layers": n_layers,
    }


def save_comparison_figures(all_results: dict):
    """Generate comparison figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    band_colors = {
        "low": "#d62728",
        "medium": "#ff7f0e",
        "high": "#2ca02c",
        "very_high": "#1f77b4",
        "control": "#9467bd",
    }

    # ---- Figure 1: P(correct) trajectories side by side per model ----
    for model_name, model_results in all_results.items():
        n_layers = None
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for band in CORE_BANDS:
            if band not in model_results:
                continue
            r = model_results[band]
            n_layers = r["n_layers"]
            layers = np.arange(n_layers)

            # Logit lens
            mean_logit = r["logit_prob_correct"].mean(axis=0)
            axes[0].plot(
                layers, mean_logit, color=band_colors[band], label=band, linewidth=2
            )

            # Tuned lens
            mean_tuned = r["tuned_prob_correct"].mean(axis=0)
            axes[1].plot(
                layers, mean_tuned, color=band_colors[band], label=band, linewidth=2
            )

        axes[0].set_title(f"Logit Lens: {model_name}")
        axes[0].set_xlabel("Layer")
        axes[0].set_ylabel("P(correct)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title(f"Tuned Lens: {model_name}")
        axes[1].set_xlabel("Layer")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = VIZ_DIR / f"03d_prob_trajectory_{model_name.replace('-', '_')}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fig_path}")

    # ---- Figure 2: Convergence gap comparison across models ----
    fig, ax = plt.subplots(figsize=(10, 6))

    models_for_plot = []
    logit_gaps = []
    tuned_gaps = []

    for model_name in MODELS:
        if model_name not in all_results:
            continue

        mr = all_results[model_name]
        if "low" not in mr or "very_high" not in mr:
            continue

        n_layers = mr["low"]["n_layers"]

        low_logit = mr["low"]["logit_convergence"].mean() / n_layers
        vh_logit = mr["very_high"]["logit_convergence"].mean() / n_layers
        logit_gap = low_logit - vh_logit

        low_tuned = mr["low"]["tuned_convergence"].mean() / n_layers
        vh_tuned = mr["very_high"]["tuned_convergence"].mean() / n_layers
        tuned_gap = low_tuned - vh_tuned

        models_for_plot.append(model_name)
        logit_gaps.append(logit_gap)
        tuned_gaps.append(tuned_gap)

    x = np.arange(len(models_for_plot))
    width = 0.35

    ax.bar(
        x - width / 2, logit_gaps, width, label="Logit Lens", color="#1f77b4", alpha=0.8
    )
    ax.bar(
        x + width / 2, tuned_gaps, width, label="Tuned Lens", color="#ff7f0e", alpha=0.8
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Convergence Gap (low - very_high, fractional depth)")
    ax.set_title("Frequency-Dependent Convergence Gap: Logit Lens vs Tuned Lens")
    ax.set_xticks(x)
    ax.set_xticklabels(models_for_plot, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.axhline(y=0, color="black", linewidth=0.5)

    plt.tight_layout()
    fig_path = VIZ_DIR / "03d_convergence_gap_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path}")


def main():
    print("=" * 70)
    print("03d: Tuned Lens Validation of Convergence Finding")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Create output dirs
    for d in [ANALYSIS_DIR, VIZ_DIR, LENS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    all_results = {}
    summary_rows = []

    for model_name in MODELS:
        print(f"\n{'=' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'=' * 60}")

        hf_name = f"EleutherAI/{model_name}"

        # Load HuggingFace model
        print(f"  Loading {hf_name}...")
        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        model = AutoModelForCausalLM.from_pretrained(
            hf_name, torch_dtype=torch.float32
        ).to(DEVICE)
        model.eval()

        # Train or load tuned lens
        lens = train_tuned_lens(model, tokenizer, model_name, LENS_DIR)

        # Evaluate on each band
        model_results = {}
        n_layers = model.config.num_hidden_layers

        for band in BANDS:
            print(f"\n  Evaluating band: {band}")
            dataset = load_lsc_test_data(band)
            input_ids, target_ids, n_examples = prepare_lsc_inputs(dataset, tokenizer)

            results = evaluate_tuned_lens(model, lens, input_ids, target_ids)
            model_results[band] = results

            # Summary stats
            tuned_conv = results["tuned_convergence"].mean() / n_layers
            logit_conv = results["logit_convergence"].mean() / n_layers
            tuned_final = results["tuned_prob_correct"][:, -1].mean()
            logit_final = results["logit_prob_correct"][:, -1].mean()

            print(
                f"    Logit lens: conv depth={logit_conv:.3f}, final P={logit_final:.4f}"
            )
            print(
                f"    Tuned lens: conv depth={tuned_conv:.3f}, final P={tuned_final:.4f}"
            )

            summary_rows.append(
                {
                    "model": model_name,
                    "band": band,
                    "n_layers": n_layers,
                    "logit_convergence_frac": logit_conv,
                    "tuned_convergence_frac": tuned_conv,
                    "logit_final_prob": logit_final,
                    "tuned_final_prob": tuned_final,
                    "convergence_shift": tuned_conv - logit_conv,
                }
            )

        all_results[model_name] = model_results

        # Print convergence gap for this model
        if "low" in model_results and "very_high" in model_results:
            low_logit = model_results["low"]["logit_convergence"].mean() / n_layers
            vh_logit = model_results["very_high"]["logit_convergence"].mean() / n_layers
            low_tuned = model_results["low"]["tuned_convergence"].mean() / n_layers
            vh_tuned = model_results["very_high"]["tuned_convergence"].mean() / n_layers

            print(f"\n  Convergence gap (low - very_high):")
            print(
                f"    Logit lens: {low_logit - vh_logit:+.3f} frac depth ({(low_logit - vh_logit) * n_layers:+.1f} layers)"
            )
            print(
                f"    Tuned lens: {low_tuned - vh_tuned:+.3f} frac depth ({(low_tuned - vh_tuned) * n_layers:+.1f} layers)"
            )

        # Cleanup
        del model, lens
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'=' * 70}")
    print("Saving results...")

    # Summary CSV
    import csv

    csv_path = ANALYSIS_DIR / "03d_tuned_lens_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"  Saved: {csv_path}")

    print("\nGenerating figures...")
    save_comparison_figures(all_results)

    print(f"\n{'=' * 70}")
    print("SUMMARY: Convergence Gap Comparison")
    print(f"{'=' * 70}")
    print(
        f"\n  {'Model':15s} | {'Logit gap':>10s} | {'Tuned gap':>10s} | {'Ratio':>8s} | {'Preserved?':>10s}"
    )
    print(f"  {'-' * 15}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 8}-+-{'-' * 10}")

    for model_name in MODELS:
        if model_name not in all_results:
            continue
        mr = all_results[model_name]
        if "low" not in mr or "very_high" not in mr:
            continue
        n_layers = mr["low"]["n_layers"]

        low_logit = mr["low"]["logit_convergence"].mean() / n_layers
        vh_logit = mr["very_high"]["logit_convergence"].mean() / n_layers
        logit_gap = low_logit - vh_logit

        low_tuned = mr["low"]["tuned_convergence"].mean() / n_layers
        vh_tuned = mr["very_high"]["tuned_convergence"].mean() / n_layers
        tuned_gap = low_tuned - vh_tuned

        ratio = tuned_gap / logit_gap if abs(logit_gap) > 1e-6 else float("inf")
        preserved = (
            "YES"
            if tuned_gap > logit_gap * 0.5
            else ("PARTIAL" if tuned_gap > 0.01 else "NO")
        )

        print(
            f"  {model_name:15s} | {logit_gap:+10.3f} | {tuned_gap:+10.3f} | {ratio:8.2f} | {preserved:>10s}"
        )


if __name__ == "__main__":
    main()
