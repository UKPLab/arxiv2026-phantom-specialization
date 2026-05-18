"""Cross-band evaluation for pythia-1.4b on cuda:3.

Appends results to cross_band_eval_results.csv and recomputes ablation_comparison_table.csv.
"""

import os
import sys
import pickle
import gc

import numpy as np
import pandas as pd
import torch as t
from pathlib import Path
from scipy.stats import wilcoxon

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# -- Paths --------------------------------------------------------------------
ANALYSIS_ROOT = Path("LSC_circuit_analysis")
ISC_ROOT = PROJECT_ROOT
LSC_DIR = ISC_ROOT / "LSC_circuits"
CIRCUITS_DIR = LSC_DIR / "circuit_discovery" / "circuits"
DATA_DIR = ISC_ROOT / "LSC_data"
POOL_DIR = DATA_DIR / "lsc_token_pools" / "matched"

PHASE5_DIR = ANALYSIS_ROOT / "05_Phase_Targeted"
ANALYSIS_DIR = PHASE5_DIR / "outputs" / "analysis"

AUTOCIRCUIT_PATH = os.environ.get("AUTOCIRCUIT_PATH") or str(
    ISC_ROOT / "circuit_discovery" / "auto-circuit"
)
sys.path.insert(0, AUTOCIRCUIT_PATH)
sys.path.insert(0, str(LSC_DIR))

from lsc_acdc_circuit import (
    load_model,
    load_pool,
    load_dataset,
    run_circuit_and_collect,
    compute_accuracy_metrics,
    model_safe_name,
    get_batch_size,
    set_all_seeds,
    cleanup_gpu,
    safe_delete_model,
)
from auto_circuit.utils.graph_utils import patchable_model

# -- Constants ----------------------------------------------------------------
MODEL_NAME = "pythia-1.4b"
ALL_MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]
BANDS = ["low", "medium", "high", "very_high", "control"]
DRAWS = ["draw_1", "draw_2", "draw_3"]
EVAL_SEED = 123
VARIANT = "matched"
K_RANDOM = 5
RANDOM_BASE_SEED = 42
DEVICE = "cuda:3"

EVAL_CSV = ANALYSIS_DIR / "cross_band_eval_results.csv"
ABLATION_CSV = ANALYSIS_DIR / "ablation_comparison_table.csv"
NB01_CSV = ANALYSIS_DIR / "universal_core_comparison.csv"


# -- Helper from notebook -----------------------------------------------------
def generate_universal_plus_random(universal_scores, n_extra, seed):
    rng = np.random.RandomState(seed)
    non_universal_positions = []
    for name, scores in universal_scores.items():
        flat = scores.view(-1)
        for pos in range(flat.numel()):
            if not t.isinf(flat[pos]):
                non_universal_positions.append((name, pos))

    n_available = len(non_universal_positions)
    n_sample = min(n_extra, n_available)
    selected = set(rng.choice(n_available, size=n_sample, replace=False).tolist())

    module_positions = {}
    for flat_idx in selected:
        name, pos = non_universal_positions[flat_idx]
        if name not in module_positions:
            module_positions[name] = set()
        module_positions[name].add(pos)

    result = {}
    for name, scores in universal_scores.items():
        new_scores = scores.clone()
        if name in module_positions:
            flat = new_scores.view(-1)
            for pos in module_positions[name]:
                flat[pos] = float("inf")
            new_scores = flat.view(scores.shape)
        result[name] = new_scores
    return result


# -- Step 1: Build caches for 1.4b -------------------------------------------
print(f"Building prune-score caches for {MODEL_NAME} ...")
m_safe = model_safe_name(MODEL_NAME)

band_scores_cache = {}  # (model, band, draw) -> scores (CPU)
universal_scores_cache = {}  # (model, draw) -> scores (CPU)
mean_n_specific = {}  # (model, draw) -> int

for draw in DRAWS:
    all_band_scores = {}
    band_edge_counts = {}
    missing = False

    for band in BANDS:
        path = CIRCUITS_DIR / m_safe / band / draw / "prune_scores.pkl"
        if not path.exists():
            print(f"  WARNING: missing {path}")
            missing = True
            break
        with open(path, "rb") as f:
            scores = pickle.load(f)
        all_band_scores[band] = scores
        band_edge_counts[band] = sum(t.isinf(s).sum().item() for s in scores.values())
        band_scores_cache[(MODEL_NAME, band, draw)] = scores

    if missing:
        continue

    # Universal core = AND of all bands
    universal = {}
    for module_name in all_band_scores[BANDS[0]]:
        mask = t.ones_like(all_band_scores[BANDS[0]][module_name], dtype=t.bool)
        for band in BANDS:
            mask &= t.isinf(all_band_scores[band][module_name])
        tensor = t.zeros_like(all_band_scores[BANDS[0]][module_name])
        tensor[mask] = float("inf")
        universal[module_name] = tensor

    n_universal = sum(t.isinf(s).sum().item() for s in universal.values())
    universal_scores_cache[(MODEL_NAME, draw)] = universal

    specific_counts = [band_edge_counts[b] - n_universal for b in BANDS]
    mean_n_specific[(MODEL_NAME, draw)] = int(np.mean(specific_counts))
    print(
        f"  {draw}: universal={n_universal}, mean_specific={mean_n_specific[(MODEL_NAME, draw)]}"
    )

print("Caches built.")


# -- Step 2: Evaluation -------------------------------------------------------
total_evals = len(DRAWS) * (len(BANDS) * len(BANDS) + len(BANDS) * K_RANDOM)
print(f"\nRunning {total_evals} evaluations for {MODEL_NAME} on {DEVICE} ...")

batch_size = get_batch_size(MODEL_NAME)
model = load_model(MODEL_NAME, DEVICE)
bos_id = model.tokenizer.bos_token_id
patchable = patchable_model(
    model=model,
    factorized=True,
    slice_output="last_seq",
    seq_len=None,
    separate_qkv=False,
    device=DEVICE,
)

results = []
eval_idx = 0

for draw in DRAWS:
    band_scores_gpu = {}
    band_n_edges = {}

    for band in BANDS:
        scores_cpu = band_scores_cache[(MODEL_NAME, band, draw)]
        band_scores_gpu[band] = {k: v.to(DEVICE) for k, v in scores_cpu.items()}
        band_n_edges[band] = sum(
            t.isinf(v).sum().item() for v in band_scores_gpu[band].values()
        )

    universal_cpu = universal_scores_cache[(MODEL_NAME, draw)]
    universal_gpu = {k: v.to(DEVICE) for k, v in universal_cpu.items()}
    n_extra = mean_n_specific[(MODEL_NAME, draw)]

    # Cache test data
    test_cache = {}
    for test_band in BANDS:
        pool = load_pool(test_band, POOL_DIR)
        test_data = load_dataset(test_band, "test", DATA_DIR, VARIANT, draw)
        test_cache[test_band] = (test_data, pool)

    # A) Cross-band: 25 evals
    for source_band in BANDS:
        source_scores = band_scores_gpu[source_band]
        n_edges = band_n_edges[source_band]

        for test_band in BANDS:
            eval_idx += 1
            test_data, pool = test_cache[test_band]

            set_all_seeds(EVAL_SEED)
            logits, answer_ids = run_circuit_and_collect(
                patchable,
                source_scores,
                n_edges,
                test_data,
                pool,
                bos_id,
                batch_size,
                EVAL_SEED,
                DEVICE,
            )
            metrics = compute_accuracy_metrics(logits, answer_ids)

            results.append(
                {
                    "model": MODEL_NAME,
                    "draw": draw,
                    "source_band": source_band,
                    "test_band": test_band,
                    "circuit_type": "cross_band",
                    "k_sample": 0,
                    "n_edges": n_edges,
                    "accuracy": metrics["accuracy"],
                    "top5_accuracy": metrics["top5_accuracy"],
                }
            )
            tag = "DIAG" if source_band == test_band else "    "
            print(
                f"  [{eval_idx:3d}/{total_evals}] {tag} {source_band:>9s} -> {test_band:<9s}: "
                f"acc={metrics['accuracy']:.4f}"
            )

    # B) Random control: K_RANDOM x 5 test bands
    for k in range(K_RANDOM):
        random_seed = RANDOM_BASE_SEED + k * 1000 + hash(f"{MODEL_NAME}_{draw}") % 10000
        random_cpu = generate_universal_plus_random(
            universal_cpu, n_extra, seed=random_seed
        )
        random_gpu = {name: s.to(DEVICE) for name, s in random_cpu.items()}
        n_random_total = sum(t.isinf(v).sum().item() for v in random_gpu.values())
        del random_cpu

        for test_band in BANDS:
            eval_idx += 1
            test_data, pool = test_cache[test_band]

            set_all_seeds(EVAL_SEED)
            logits, answer_ids = run_circuit_and_collect(
                patchable,
                random_gpu,
                n_random_total,
                test_data,
                pool,
                bos_id,
                batch_size,
                EVAL_SEED,
                DEVICE,
            )
            metrics = compute_accuracy_metrics(logits, answer_ids)

            results.append(
                {
                    "model": MODEL_NAME,
                    "draw": draw,
                    "source_band": "random",
                    "test_band": test_band,
                    "circuit_type": "random",
                    "k_sample": k + 1,
                    "n_edges": n_random_total,
                    "accuracy": metrics["accuracy"],
                    "top5_accuracy": metrics["top5_accuracy"],
                }
            )
            print(
                f"  [{eval_idx:3d}/{total_evals}] RAND(k={k + 1}) -> {test_band:<9s}: "
                f"acc={metrics['accuracy']:.4f} ({n_random_total} edges)"
            )

        del random_gpu
        cleanup_gpu()

    del band_scores_gpu, universal_gpu, test_cache
    cleanup_gpu()

del patchable
safe_delete_model(model)
print(f"\n{MODEL_NAME} done, GPU freed.")


# -- Step 3: Append to existing CSV -------------------------------------------
df_new = pd.DataFrame(results)
print(f"\nNew rows: {len(df_new)}")

if EVAL_CSV.exists():
    df_existing = pd.read_csv(EVAL_CSV)
    # Remove any stale 1.4b rows (in case of partial runs)
    df_existing = df_existing[df_existing["model"] != MODEL_NAME]
    df_full = pd.concat([df_existing, df_new], ignore_index=True)
else:
    df_full = df_new

df_full.to_csv(EVAL_CSV, index=False)
print(f"Saved: {EVAL_CSV} ({len(df_full)} total rows)")


# -- Step 4: Recompute ablation_comparison_table.csv -------------------------
print("\nRecomputing ablation_comparison_table.csv ...")

df_nb01 = pd.read_csv(NB01_CSV)
univ_ref = df_nb01[["model", "test_band", "universal_acc"]].copy()

df_cross = df_full[df_full["circuit_type"] == "cross_band"].copy()
df_cross = df_cross.merge(univ_ref, on=["model", "test_band"], how="left")
df_cross["boost"] = df_cross["accuracy"] - df_cross["universal_acc"]
df_cross["is_diagonal"] = df_cross["source_band"] == df_cross["test_band"]

df_rand = df_full[df_full["circuit_type"] == "random"].copy()
df_rand = df_rand.merge(univ_ref, on=["model", "test_band"], how="left")
df_rand["boost"] = df_rand["accuracy"] - df_rand["universal_acc"]

df_rand_per_draw = (
    df_rand.groupby(["model", "draw", "test_band"])
    .agg(
        mean_acc=("accuracy", "mean"),
        mean_boost=("boost", "mean"),
        mean_n_edges=("n_edges", "mean"),
    )
    .reset_index()
)

ablation_rows = []

for model_name in ALL_MODELS:
    same_boost_vals = []
    cross_boost_vals = []
    random_boost_vals = []

    for draw in DRAWS:
        for test_band in BANDS:
            same = df_cross[
                (df_cross["model"] == model_name)
                & (df_cross["draw"] == draw)
                & (df_cross["source_band"] == test_band)
                & (df_cross["test_band"] == test_band)
            ]
            if len(same) > 0:
                same_boost_vals.append(same["boost"].values[0])

            cross = df_cross[
                (df_cross["model"] == model_name)
                & (df_cross["draw"] == draw)
                & (df_cross["source_band"] != test_band)
                & (df_cross["test_band"] == test_band)
            ]
            if len(cross) > 0:
                cross_boost_vals.append(cross["boost"].mean())

            rand = df_rand_per_draw[
                (df_rand_per_draw["model"] == model_name)
                & (df_rand_per_draw["draw"] == draw)
                & (df_rand_per_draw["test_band"] == test_band)
            ]
            if len(rand) > 0:
                random_boost_vals.append(rand["mean_boost"].values[0])

    if not same_boost_vals:
        print(f"  SKIP {model_name}: no data")
        continue

    same_arr = np.array(same_boost_vals)
    cross_arr = np.array(cross_boost_vals)
    random_arr = np.array(random_boost_vals)

    diff_sc = same_arr - cross_arr
    if len(diff_sc) > 1 and not np.all(diff_sc == 0):
        _, p_sc = wilcoxon(diff_sc, alternative="greater")
    else:
        p_sc = np.nan

    n_min = min(len(same_arr), len(random_arr))
    if n_min > 1:
        diff_sr = same_arr[:n_min] - random_arr[:n_min]
        if not np.all(diff_sr == 0):
            _, p_sr = wilcoxon(diff_sr, alternative="greater")
        else:
            p_sr = np.nan
    else:
        p_sr = np.nan

    pooled_std = np.sqrt((np.var(same_arr) + np.var(cross_arr)) / 2)
    cohens_d = (
        (same_arr.mean() - cross_arr.mean()) / pooled_std if pooled_std > 0 else 0
    )
    transfer_eff = cross_arr.mean() / same_arr.mean() if same_arr.mean() > 0 else np.nan
    verdict = "GENERIC" if p_sc >= 0.05 else "BAND-SPECIFIC"

    print(
        f"  {model_name}: same={same_arr.mean():.4f}, cross={cross_arr.mean():.4f}, "
        f"transfer_eff={transfer_eff:.1%}, p={p_sc:.4f} -> {verdict}"
    )

    ablation_rows.append(
        {
            "model": model_name,
            "same_band_boost_mean": same_arr.mean(),
            "same_band_boost_std": same_arr.std(),
            "cross_band_boost_mean": cross_arr.mean(),
            "cross_band_boost_std": cross_arr.std(),
            "random_boost_mean": random_arr.mean(),
            "random_boost_std": random_arr.std(),
            "transfer_efficiency": transfer_eff,
            "cohens_d_same_vs_cross": cohens_d,
            "p_same_gt_cross": p_sc,
            "p_same_gt_random": p_sr,
            "n_pairs": len(same_arr),
            "verdict": verdict,
        }
    )

df_ablation = pd.DataFrame(ablation_rows)
df_ablation.to_csv(ABLATION_CSV, index=False)
print(f"\nSaved: {ABLATION_CSV} ({len(df_ablation)} rows)")

# Print the 1.4b row
row_1p4b = df_ablation[df_ablation["model"] == MODEL_NAME]
if len(row_1p4b) > 0:
    r = row_1p4b.iloc[0]
    print(f"\n=== pythia-1.4b transfer_efficiency = {r['transfer_efficiency']:.1%} ===")
    print(f"    same_band_boost_mean  = {r['same_band_boost_mean']:.4f}")
    print(f"    cross_band_boost_mean = {r['cross_band_boost_mean']:.4f}")
    print(f"    p_same_gt_cross       = {r['p_same_gt_cross']:.4f}")
    print(f"    cohens_d              = {r['cohens_d_same_vs_cross']:.3f}")
    print(f"    verdict               = {r['verdict']}")

print("\nDone.")
