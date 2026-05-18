#!/usr/bin/env python3
"""
Positive Control: Layer-Sweep Interchange Patching
===================================================
Demonstrates that the interchange patching framework has full dynamic range
by sweeping across ALL layers of each model.

Design:
  At each layer, patch the residual stream at position 21 (prediction position)
  from a source example with a DIFFERENT target token.

Expected results:
  - Early layers: source_iia ~ 0  (target info not yet computed)
  - Peak/late layers: source_iia ~ 1  (target info fully present)

This proves the framework can detect causally relevant properties (target
identity emergence across layers). The band-swap null result (high IIA
regardless of band at peak layers) is therefore meaningful: the framework
has the power to detect differences but finds none between bands.

Additional comparison at peak layer:
  - Within-band (different target): source_iia ~ X
  - Cross-band (different target): source_iia ~ X  (same)
  -> Band identity is not a causal factor.
"""

import os
import sys
import json
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch as t
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Paths ---
ISC_ROOT = PROJECT_ROOT
LSC_DIR = ISC_ROOT / "LSC_circuits"
AUTOCIRCUIT_PATH = os.environ.get("AUTOCIRCUIT_PATH") or str(
    ISC_ROOT / "circuit_discovery" / "auto-circuit"
)
sys.path.insert(0, AUTOCIRCUIT_PATH)
sys.path.insert(0, str(LSC_DIR))

from lsc_acdc_circuit import (
    load_model,
    model_safe_name,
    set_all_seeds,
    cleanup_gpu,
)

# --- Constants ---
MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]
N_LAYERS = {
    "pythia-70m": 6,
    "pythia-160m": 12,
    "pythia-410m": 24,
    "pythia-1b": 16,
    "pythia-1.4b": 24,
}
BANDS = ["low", "medium", "high", "very_high", "control"]
EVAL_SEED = 123
VARIANT = "matched"
DATA_DIR = ISC_ROOT / "LSC_data"
N_PAIRS = 100
PRED_POS = 21

# Output
PHASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = PHASE_DIR / "outputs" / "analysis"
VIZ_DIR = PHASE_DIR / "outputs" / "viz"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda:0"


# --- Load peak layers from NB13 ---
def load_peak_layers():
    path = ANALYSIS_DIR / "activation_patching_peak_layers.json"
    with open(path) as f:
        return json.load(f)


# --- Data ---
@dataclass
class InterchangePair:
    base_ids: t.Tensor
    source_ids: t.Tensor
    base_target_id: int
    source_target_id: int


def load_test_examples(band: str, draw: str) -> List[dict]:
    path = DATA_DIR / "datasets" / VARIANT / draw / band / "test.json"
    with open(path) as f:
        data = json.load(f)
    return data["examples"]


def create_different_target_pairs(
    band: str,
    draw: str,
    n_pairs: int = N_PAIRS,
    bos_id: int = 0,
    seed: int = EVAL_SEED,
) -> List[InterchangePair]:
    """Pair examples from the SAME band with DIFFERENT target tokens."""
    rng = np.random.default_rng(seed)
    examples = load_test_examples(band, draw)
    n = len(examples)
    indices = rng.permutation(n)

    pairs = []
    for i in range(0, min(2 * n_pairs, n - 1), 2):
        base_ex = examples[indices[i]]
        source_ex = examples[indices[i + 1]]
        assert base_ex["target_token_id"] != source_ex["target_token_id"]

        pairs.append(
            InterchangePair(
                base_ids=t.tensor([[bos_id] + base_ex["token_ids"]], dtype=t.long),
                source_ids=t.tensor([[bos_id] + source_ex["token_ids"]], dtype=t.long),
                base_target_id=base_ex["target_token_id"],
                source_target_id=source_ex["target_token_id"],
            )
        )

    return pairs[:n_pairs]


def create_cross_band_pairs(
    base_band: str,
    source_band: str,
    draw: str,
    n_pairs: int = N_PAIRS,
    bos_id: int = 0,
    seed: int = EVAL_SEED,
) -> List[InterchangePair]:
    """Pair examples from DIFFERENT bands."""
    rng = np.random.default_rng(seed)
    base_examples = load_test_examples(base_band, draw)
    source_examples = load_test_examples(source_band, draw)
    n_avail = min(len(base_examples), len(source_examples))
    indices = rng.permutation(n_avail)[:n_pairs]
    pairs = []
    for idx in indices:
        base_ex = base_examples[idx]
        source_ex = source_examples[idx]
        pairs.append(
            InterchangePair(
                base_ids=t.tensor([[bos_id] + base_ex["token_ids"]], dtype=t.long),
                source_ids=t.tensor([[bos_id] + source_ex["token_ids"]], dtype=t.long),
                base_target_id=base_ex["target_token_id"],
                source_target_id=source_ex["target_token_id"],
            )
        )
    return pairs


# --- Patching ---
def make_patch_hook(source_cache: t.Tensor, position: int):
    def hook_fn(activation, hook):
        patched = activation.clone()
        patched[:, position, :] = source_cache[:, position, :]
        return patched

    return hook_fn


@t.no_grad()
def compute_patching_metrics(
    model,
    pairs: List[InterchangePair],
    hook_name: str,
    position: int,
    batch_size: int = 50,
) -> Dict:
    """Compute base_faithfulness, source_iia, and unpatched_acc."""
    dev = next(model.parameters()).device

    base_ids = t.cat([p.base_ids for p in pairs], dim=0).to(dev)
    source_ids = t.cat([p.source_ids for p in pairs], dim=0).to(dev)
    base_tgts = t.tensor([p.base_target_id for p in pairs])
    source_tgts = t.tensor([p.source_target_id for p in pairs])

    n = len(pairs)
    unpatched_correct = []
    patched_base_correct = []
    patched_source_correct = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b_ids = base_ids[start:end]
        s_ids = source_ids[start:end]
        b_tgts = base_tgts[start:end]
        s_tgts = source_tgts[start:end]

        # Unpatched
        unpatched_logits = model(b_ids, prepend_bos=False)
        unpatched_preds = unpatched_logits[:, -1, :].argmax(dim=-1).cpu()
        unpatched_correct.append((unpatched_preds == b_tgts).float())
        del unpatched_logits

        # Cache source activations
        _, cache = model.run_with_cache(
            s_ids, prepend_bos=False, names_filter=[hook_name]
        )
        source_act = cache[hook_name]
        del cache

        # Patched
        hook_fn = make_patch_hook(source_act, position)
        patched_logits = model.run_with_hooks(
            b_ids, prepend_bos=False, fwd_hooks=[(hook_name, hook_fn)]
        )
        patched_preds = patched_logits[:, -1, :].argmax(dim=-1).cpu()

        patched_base_correct.append((patched_preds == b_tgts).float())
        patched_source_correct.append((patched_preds == s_tgts).float())

        del source_act, patched_logits

    return {
        "unpatched_acc": t.cat(unpatched_correct).mean().item(),
        "base_faithfulness": t.cat(patched_base_correct).mean().item(),
        "source_iia": t.cat(patched_source_correct).mean().item(),
        "n_pairs": n,
    }


# --- Main ---
def main():
    peak_layers = load_peak_layers()

    sweep_results = []  # layer sweep
    compare_results = []  # within-band vs cross-band at peak layer

    for model_name in MODELS:
        n_layers = N_LAYERS[model_name]
        peak_layer = peak_layers[model_name][0]

        print(f"\n{'=' * 60}")
        print(f"{model_name}: {n_layers} layers, peak={peak_layer}")
        print(f"{'=' * 60}")

        model = load_model(model_name, device)
        bos_id = model.tokenizer.bos_token_id

        # Use 'low' band pairs for the layer sweep (representative)
        pairs = create_different_target_pairs(
            "low",
            "draw_1",
            n_pairs=N_PAIRS,
            bos_id=bos_id,
        )

        # -- Part 1: Layer sweep --
        print(f"\n  Layer sweep (within-band different-target, low band):")
        for layer in range(n_layers):
            hook_name = f"blocks.{layer}.hook_resid_post"
            metrics = compute_patching_metrics(
                model,
                pairs,
                hook_name,
                PRED_POS,
            )
            print(
                f"    Layer {layer:2d}: source_iia={metrics['source_iia']:.3f}  "
                f"base_faith={metrics['base_faithfulness']:.3f}  "
                f"unpatched={metrics['unpatched_acc']:.3f}"
            )

            sweep_results.append(
                {
                    "model": model_name,
                    "layer": layer,
                    "n_layers": n_layers,
                    "peak_layer": peak_layer,
                    **metrics,
                }
            )

        # -- Part 2: Within-band vs cross-band at peak layer --
        print(f"\n  Peak-layer comparison (layer {peak_layer}):")
        peak_hook = f"blocks.{peak_layer}.hook_resid_post"

        for band in BANDS:
            # Within-band, different target
            wb_pairs = create_different_target_pairs(
                band,
                "draw_1",
                n_pairs=N_PAIRS,
                bos_id=bos_id,
            )
            wb_metrics = compute_patching_metrics(
                model,
                wb_pairs,
                peak_hook,
                PRED_POS,
            )
            print(
                f"    Within-band {band:>9s}: source_iia={wb_metrics['source_iia']:.3f}"
            )
            compare_results.append(
                {
                    "model": model_name,
                    "condition": "within_band",
                    "band": band,
                    "peak_layer": peak_layer,
                    **wb_metrics,
                }
            )

        # Cross-band pairs
        cross_pairs_list = [
            ("low", "high"),
            ("high", "low"),
            ("low", "very_high"),
            ("very_high", "low"),
            ("medium", "control"),
            ("control", "medium"),
        ]
        for base_band, source_band in cross_pairs_list:
            cb_pairs = create_cross_band_pairs(
                base_band,
                source_band,
                "draw_1",
                n_pairs=N_PAIRS,
                bos_id=bos_id,
            )
            cb_metrics = compute_patching_metrics(
                model,
                cb_pairs,
                peak_hook,
                PRED_POS,
            )
            label = f"{base_band}->{source_band}"
            print(
                f"    Cross-band  {label:>16s}: source_iia={cb_metrics['source_iia']:.3f}"
            )
            compare_results.append(
                {
                    "model": model_name,
                    "condition": "cross_band",
                    "band": label,
                    "peak_layer": peak_layer,
                    **cb_metrics,
                }
            )

        # Cleanup
        try:
            model.cpu()
        except Exception:
            pass
        del model
        cleanup_gpu()

    # -- Save results --
    df_sweep = pd.DataFrame(sweep_results)
    df_compare = pd.DataFrame(compare_results)

    sweep_csv = ANALYSIS_DIR / "positive_control_layer_sweep.csv"
    compare_csv = ANALYSIS_DIR / "positive_control_band_comparison.csv"
    df_sweep.to_csv(sweep_csv, index=False)
    df_compare.to_csv(compare_csv, index=False)
    print(f"\nSaved layer sweep to {sweep_csv}")
    print(f"Saved band comparison to {compare_csv}")

    # -- Summary --
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for model_name in MODELS:
        ds = df_sweep[df_sweep["model"] == model_name]
        dc = df_compare[df_compare["model"] == model_name]
        peak = ds["peak_layer"].iloc[0]

        iia_layer0 = ds[ds["layer"] == 0]["source_iia"].iloc[0]
        iia_peak = ds[ds["layer"] == peak]["source_iia"].iloc[0]
        iia_last = ds[ds["layer"] == ds["layer"].max()]["source_iia"].iloc[0]

        wb = dc[dc["condition"] == "within_band"]["source_iia"].mean()
        cb = dc[dc["condition"] == "cross_band"]["source_iia"].mean()

        print(f"\n{model_name} (peak={peak}):")
        print(
            f"  Layer sweep:  layer_0={iia_layer0:.3f}  peak={iia_peak:.3f}  last={iia_last:.3f}"
        )
        print(
            f"  At peak:      within_band={wb:.3f}  cross_band={cb:.3f}  Δ={wb - cb:+.3f}"
        )

    # -- Visualization: Layer Sweep --
    fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=True)

    for col, model_name in enumerate(MODELS):
        ax = axes[col]
        ds = df_sweep[df_sweep["model"] == model_name]
        peak = ds["peak_layer"].iloc[0]

        ax.plot(
            ds["layer"],
            ds["source_iia"],
            "o-",
            color="#E24A33",
            markersize=4,
            linewidth=1.5,
            label="Source IIA",
        )
        ax.plot(
            ds["layer"],
            ds["base_faithfulness"],
            "s-",
            color="#348ABD",
            markersize=3,
            linewidth=1,
            alpha=0.7,
            label="Base faithfulness",
        )
        ax.axvline(
            x=peak, color="gray", linestyle="--", alpha=0.5, label=f"Peak (L{peak})"
        )

        ax.set_xlabel("Layer")
        if col == 0:
            ax.set_ylabel("Rate")
        ax.set_title(model_name.replace("pythia-", ""), fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        if col == 4:
            ax.legend(fontsize=7, loc="center right")

    plt.suptitle(
        "Positive Control: Source IIA Across Layers\n"
        "(within-band, different target: low band, draw 1)",
        fontsize=11,
    )
    plt.tight_layout()
    fig_path = VIZ_DIR / "positive_control_layer_sweep.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved layer sweep figure to {fig_path}")

    # -- Visualization: Within-band vs Cross-band at peak --
    fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=True)

    for col, model_name in enumerate(MODELS):
        ax = axes[col]
        dc = df_compare[df_compare["model"] == model_name]

        wb = dc[dc["condition"] == "within_band"]
        cb = dc[dc["condition"] == "cross_band"]

        x = np.arange(2)
        vals = [wb["source_iia"].mean(), cb["source_iia"].mean()]
        errs = [wb["source_iia"].std(), cb["source_iia"].std()]

        bars = ax.bar(
            x,
            vals,
            yerr=errs,
            capsize=5,
            width=0.5,
            color=["#E24A33", "#348ABD"],
            alpha=0.85,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(["Within-band", "Cross-band"], fontsize=9)
        ax.set_title(model_name.replace("pythia-", ""), fontsize=11)
        if col == 0:
            ax.set_ylabel("Source IIA at peak layer")
        ax.set_ylim(0, 1.1)

        # Annotate difference
        diff = vals[0] - vals[1]
        ax.text(
            0.5,
            max(vals) + max(errs) + 0.05,
            f"Δ={diff:+.3f}",
            ha="center",
            fontsize=8,
            color="gray",
        )

    plt.suptitle(
        "Within-Band vs Cross-Band IIA at Peak Layer\n"
        "(both use different-target pairs)",
        fontsize=11,
    )
    plt.tight_layout()
    fig_path = VIZ_DIR / "positive_control_band_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved band comparison figure to {fig_path}")


if __name__ == "__main__":
    main()
