"""Edge-level cross-band transfer under genuine mean ablation.

Replicates NB12 (12_zero_ablation_robustness.ipynb) Parts A + C with the
ablation type swapped from ZERO to a dataset-mean ablation. The existing
"mean" rows in ablation_method_comparison.csv are the resample results
relabeled (NB12 cell 16 stamped ablation='mean' on ablation_comparison_table
.csv), so this computes the real thing.

Protocol per (model, draw):
  Part A: each band circuit evaluated on all 5 test bands (25 evals)
  Part C: universal core (AND of the 5 band masks) on all 5 test bands
Aggregation (--aggregate): boost = circuit_acc - universal_acc per
(draw, test_band); same-band vs cross-band boosts -> transfer efficiency,
Cohen's d, Wilcoxon p. Matches NB12 cell 16 exactly.

Default ablation: TOKENWISE_MEAN_CORRUPT (mean at each token position over
the corrupt input activations for the test band's dataloader) - the natural
mean analog of the paper's resample-from-corrupt ablation.

Outputs (under --out-dir):
  edge_cross_band_{model}.csv, edge_universal_{model}.csv  (incremental)
  mean_ablation_summary.csv, ablation_method_comparison.csv (--aggregate)
"""

import argparse
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch as t
from scipy import stats

# --- Paths ---
ISC_ROOT = Path(__file__).resolve().parents[2]
LSC_DIR = ISC_ROOT / "LSC_circuits"
sys.path.insert(0, str(ISC_ROOT / "circuit_discovery" / "auto-circuit"))
sys.path.insert(0, str(LSC_DIR))

from lsc_acdc_circuit import (  # noqa: E402
    cleanup_gpu,
    compute_accuracy_metrics,
    get_batch_size,
    load_dataset,
    load_model,
    load_pool,
    model_safe_name,
    prepare_full_dataloader,
    safe_delete_model,
    set_all_seeds,
)

ALL_MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]
ALL_BANDS = ["low", "medium", "high", "very_high", "control"]
ALL_DRAWS = ["draw_1", "draw_2", "draw_3"]
EVAL_SEED = 123
VARIANT = "matched"

DATA_DIR = ISC_ROOT / "LSC_data"
POOL_DIR = DATA_DIR / "lsc_token_pools" / "matched"
CIRCUITS_DIR = LSC_DIR / "circuit_discovery" / "circuits"
# Source of the paper's resample rows (Table 2), for the combined comparison
RESAMPLE_TABLE = (
    ISC_ROOT
    / "LSC_circuit_analysis/05_Phase_Targeted/outputs/analysis/ablation_comparison_table.csv"
)
ZERO_METHOD_TABLE = (
    ISC_ROOT
    / "LSC_circuit_analysis/05_Phase_Targeted/outputs/analysis/ablation_method_comparison.csv"
)

ABLATION_CHOICES = {
    "tokenwise_mean_corrupt": "TOKENWISE_MEAN_CORRUPT",
    "tokenwise_mean_clean": "TOKENWISE_MEAN_CLEAN",
    "tokenwise_mean_clean_and_corrupt": "TOKENWISE_MEAN_CLEAN_AND_CORRUPT",
    "batch_tokenwise_mean": "BATCH_TOKENWISE_MEAN",
    "zero": "ZERO",  # parity check against NB12
    "resample": "RESAMPLE",  # parity check against NB02 / Table 2
}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_circuit_ablated(patchable, prune_scores_dev, n_edges, dataset, pool,
                        bos_id, batch_size, eval_seed, device, ablation_type):
    """Edge-level eval with the given ablation on non-circuit edges.

    Identical to NB12's run_circuit_zero_ablation except ablation_type is a
    parameter. For mean_over_dataset types run_circuits builds the mean patch
    from this dataloader internally (auto_circuit ablation_activations.py).
    """
    from auto_circuit.prune import run_circuits
    from auto_circuit.types import PatchType

    set_all_seeds(eval_seed)
    loader1, _ = prepare_full_dataloader(dataset, pool, bos_id, batch_size, eval_seed, device)

    with t.no_grad():
        outputs = run_circuits(
            model=patchable, dataloader=loader1,
            test_edge_counts=[n_edges], prune_scores=prune_scores_dev,
            patch_type=PatchType.TREE_PATCH, ablation_type=ablation_type,
        )

    # Second loader (same seed) to extract answer IDs in batch order
    set_all_seeds(eval_seed)
    loader2, _ = prepare_full_dataloader(dataset, pool, bos_id, batch_size, eval_seed, device)

    logits_list, aligned_answer_ids = [], []
    for batch in loader2:
        logits = outputs[n_edges][batch.key]
        if len(logits.shape) == 3:
            logits = logits[:, -1, :]
        logits_list.append(logits.float())
        aligned_answer_ids.extend(batch.answers.squeeze(-1).tolist())

    return t.cat(logits_list, dim=0), aligned_answer_ids


def build_universal(all_band_scores, bands):
    """Universal core = AND of the bands' inf masks (NB12 cell 5)."""
    first = bands[0]
    universal = {}
    for module_name in all_band_scores[first]:
        mask = t.ones_like(all_band_scores[first][module_name], dtype=t.bool)
        for band in bands:
            mask &= t.isinf(all_band_scores[band][module_name])
        tensor = t.zeros_like(all_band_scores[first][module_name])
        tensor[mask] = float("inf")
        universal[module_name] = tensor
    return universal


def load_done_keys(csv_path, key_cols):
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path)
        return set(map(tuple, df[key_cols].astype(str).values.tolist()))
    except Exception:
        return set()


def append_row(csv_path, row):
    df = pd.DataFrame([row])
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def run_model(model_name, bands, draws, device, out_dir, ablation_type, abl_name):
    m_safe = model_safe_name(model_name)
    batch_size = get_batch_size(model_name)
    cross_csv = out_dir / f"edge_cross_band_{m_safe}.csv"
    univ_csv = out_dir / f"edge_universal_{m_safe}.csv"
    done_cross = load_done_keys(cross_csv, ["draw", "source_band", "test_band"])
    done_univ = load_done_keys(univ_csv, ["draw", "test_band"])

    n_todo_cross = len(draws) * len(bands) * len(bands) - len(done_cross)
    n_todo_univ = len(draws) * len(bands) - len(done_univ)
    log(f"{model_name}: batch={batch_size}, todo cross={n_todo_cross}, univ={n_todo_univ}")
    if n_todo_cross <= 0 and n_todo_univ <= 0:
        log(f"{model_name}: already complete, skipping")
        return

    # --- Load prune scores + universal core per draw (NB12 cell 5) ---
    band_scores = {}
    universal_scores = {}
    for draw in draws:
        per_band = {}
        for band in bands:
            with open(CIRCUITS_DIR / m_safe / band / draw / "prune_scores.pkl", "rb") as f:
                per_band[band] = pickle.load(f)
        band_scores[draw] = per_band
        universal_scores[draw] = build_universal(per_band, bands)
        n_univ = sum(t.isinf(s).sum().item() for s in universal_scores[draw].values())
        log(f"  {draw}: universal={n_univ} edges, "
            f"per-band={{{', '.join(f'{b}:{sum(t.isinf(v).sum().item() for v in per_band[b].values())}' for b in bands)}}}")

    from auto_circuit.utils.graph_utils import patchable_model

    model = load_model(model_name, device)
    bos_id = model.tokenizer.bos_token_id
    patchable = patchable_model(
        model=model, factorized=True, slice_output="last_seq",
        seq_len=None, separate_qkv=False, device=device,
    )

    try:
        for draw in draws:
            t0 = time.time()
            band_scores_gpu = {
                b: {k: v.to(device) for k, v in band_scores[draw][b].items()} for b in bands
            }
            band_n_edges = {
                b: sum(t.isinf(v).sum().item() for v in band_scores_gpu[b].values()) for b in bands
            }
            test_cache = {}
            for band in bands:
                pool = load_pool(band, POOL_DIR)
                test_data = load_dataset(band, "test", DATA_DIR, VARIANT, draw)
                test_cache[band] = (test_data, pool)

            # Part A: cross-band
            for source_band in bands:
                for test_band in bands:
                    if (draw, source_band, test_band) in done_cross:
                        continue
                    test_data, pool = test_cache[test_band]
                    set_all_seeds(EVAL_SEED)
                    logits, answer_ids = run_circuit_ablated(
                        patchable, band_scores_gpu[source_band], band_n_edges[source_band],
                        test_data, pool, bos_id, batch_size, EVAL_SEED, device, ablation_type,
                    )
                    metrics = compute_accuracy_metrics(logits, answer_ids)
                    append_row(cross_csv, {
                        "model": model_name, "draw": draw,
                        "source_band": source_band, "test_band": test_band,
                        "circuit_type": "cross_band", "ablation": abl_name,
                        "n_edges": band_n_edges[source_band],
                        "accuracy": metrics["accuracy"],
                        "top5_accuracy": metrics["top5_accuracy"],
                    })
                    del logits
                log(f"  {model_name} {draw} source={source_band}: done")

            # Part C: universal core
            universal_gpu = {k: v.to(device) for k, v in universal_scores[draw].items()}
            n_univ = sum(t.isinf(v).sum().item() for v in universal_gpu.values())
            for test_band in bands:
                if (draw, test_band) in done_univ:
                    continue
                test_data, pool = test_cache[test_band]
                set_all_seeds(EVAL_SEED)
                logits, answer_ids = run_circuit_ablated(
                    patchable, universal_gpu, n_univ,
                    test_data, pool, bos_id, batch_size, EVAL_SEED, device, ablation_type,
                )
                metrics = compute_accuracy_metrics(logits, answer_ids)
                append_row(univ_csv, {
                    "model": model_name, "draw": draw, "test_band": test_band,
                    "circuit_type": "universal_edge", "ablation": abl_name,
                    "n_edges": n_univ,
                    "accuracy": metrics["accuracy"],
                    "top5_accuracy": metrics["top5_accuracy"],
                })
                del logits
            log(f"  {model_name} {draw} universal ({n_univ} edges): done "
                f"[draw took {(time.time() - t0) / 60:.1f} min]")

            del band_scores_gpu, universal_gpu
            cleanup_gpu()
    finally:
        del patchable
        safe_delete_model(model)
        cleanup_gpu()
        gc.collect()

    log(f"{model_name}: COMPLETE")


def aggregate(out_dir, bands, draws, abl_name):
    """NB12 cell 16: boosts, transfer efficiency, Cohen's d, Wilcoxon p."""
    cross_files = sorted(out_dir.glob("edge_cross_band_*.csv"))
    univ_files = sorted(out_dir.glob("edge_universal_*.csv"))
    if not cross_files:
        log("aggregate: no result files found")
        return
    df_edge = pd.concat([pd.read_csv(f) for f in cross_files], ignore_index=True)
    df_univ = pd.concat([pd.read_csv(f) for f in univ_files], ignore_index=True)

    univ_acc = df_univ.set_index(["model", "draw", "test_band"])["accuracy"].to_dict()
    df_edge["universal_acc"] = df_edge.apply(
        lambda r: univ_acc.get((r["model"], r["draw"], r["test_band"]), np.nan), axis=1
    )
    df_edge["boost"] = df_edge["accuracy"] - df_edge["universal_acc"]

    rows = []
    for model_name in [m for m in ALL_MODELS if (df_edge["model"] == m).any()]:
        df_m = df_edge[df_edge["model"] == model_name]
        same_vals, cross_vals = [], []
        for draw in draws:
            for test_band in bands:
                same = df_m[(df_m["draw"] == draw) & (df_m["source_band"] == test_band)
                            & (df_m["test_band"] == test_band)]
                if len(same) > 0:
                    same_vals.append(same["boost"].values[0])
                cross = df_m[(df_m["draw"] == draw) & (df_m["source_band"] != test_band)
                             & (df_m["test_band"] == test_band)]
                if len(cross) > 0:
                    cross_vals.append(cross["boost"].mean())
        same_arr, cross_arr = np.array(same_vals), np.array(cross_vals)
        te = np.mean(cross_arr) / np.mean(same_arr) if np.mean(same_arr) > 0 else np.nan
        pooled = np.sqrt((np.var(same_arr) + np.var(cross_arr)) / 2)
        d = (np.mean(same_arr) - np.mean(cross_arr)) / pooled if pooled > 0 else 0.0
        diff = same_arr - cross_arr
        p = stats.wilcoxon(diff, alternative="greater")[1] if np.any(diff != 0) else 1.0
        # mean absolute accuracy of full band circuits on their own band
        own = df_m[df_m["source_band"] == df_m["test_band"]]["accuracy"].mean()
        rows.append({
            "model": model_name, "ablation": abl_name,
            "same_band_boost_mean": np.mean(same_arr),
            "cross_band_boost_mean": np.mean(cross_arr),
            "transfer_efficiency": te, "cohens_d": d, "p_value": p,
            "n_pairs": len(same_arr), "own_band_accuracy_mean": own,
        })
        log(f"{model_name}: own-band acc={own:.3f}, same={np.mean(same_arr):+.3f}, "
            f"cross={np.mean(cross_arr):+.3f}, TE={te if not np.isnan(te) else float('nan'):.3f}, "
            f"d={d:.3f}, p={p:.4f}")

    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(out_dir / "mean_ablation_summary.csv", index=False)

    # Combined table: resample (Table 2 source) + zero (NB12) + genuine mean
    combined = [df_sum[["model", "ablation", "same_band_boost_mean", "cross_band_boost_mean",
                        "transfer_efficiency", "cohens_d"]]]
    if RESAMPLE_TABLE.exists():
        rs = pd.read_csv(RESAMPLE_TABLE)
        rs = rs.rename(columns={"cohens_d_same_vs_cross": "cohens_d"})
        rs["ablation"] = "resample"
        combined.append(rs[["model", "ablation", "same_band_boost_mean", "cross_band_boost_mean",
                            "transfer_efficiency", "cohens_d"]])
    if ZERO_METHOD_TABLE.exists():
        z = pd.read_csv(ZERO_METHOD_TABLE)
        z = z[z["ablation"] == "zero"]
        combined.append(z[["model", "ablation", "same_band_boost_mean", "cross_band_boost_mean",
                           "transfer_efficiency", "cohens_d"]])
    df_comb = pd.concat(combined, ignore_index=True).sort_values(["model", "ablation"])
    df_comb.to_csv(out_dir / "ablation_method_comparison.csv", index=False)
    log(f"Saved mean_ablation_summary.csv + ablation_method_comparison.csv to {out_dir}")
    print(df_comb.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", choices=ALL_MODELS, default=[])
    ap.add_argument("--ablation", choices=list(ABLATION_CHOICES), default="tokenwise_mean_corrupt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=str(ISC_ROOT / "supplementary_experiments/results/mean_ablation"))
    ap.add_argument("--draws", nargs="+", default=ALL_DRAWS)
    ap.add_argument("--bands", nargs="+", default=ALL_BANDS)
    ap.add_argument("--smoke", action="store_true",
                    help="reduced run: draw_1 only, bands low+control")
    ap.add_argument("--aggregate", action="store_true", help="aggregate existing results and exit")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bands, draws = args.bands, args.draws
    if args.smoke:
        bands, draws = ["low", "control"], ["draw_1"]
        out_dir = out_dir / "smoke"
        out_dir.mkdir(parents=True, exist_ok=True)

    from auto_circuit.types import AblationType
    ablation_type = getattr(AblationType, ABLATION_CHOICES[args.ablation])

    log(f"ablation={args.ablation}, device={args.device}, models={args.models}, "
        f"bands={bands}, draws={draws}, out={out_dir}")
    log(f"CUDA available: {t.cuda.is_available()}"
        + (f" ({t.cuda.get_device_name(0)})" if t.cuda.is_available() else ""))

    if args.aggregate:
        aggregate(out_dir, bands, draws, args.ablation)
        return

    for model_name in args.models:
        run_model(model_name, bands, draws, args.device, out_dir, ablation_type, args.ablation)

    log("ALL MODELS DONE")


if __name__ == "__main__":
    main()
