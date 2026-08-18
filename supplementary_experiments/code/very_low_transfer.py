"""very_low <-> core-band cross-band transfer.

Extends the paper's Part A/C protocol to 6 bands: very_low (unmatched
variant, circuits from supplementary_experiments/results/very_low_circuits/) plus
the 5 canonical bands (matched variant, canonical circuit dir). Eval =
edge-level TREE_PATCH RESAMPLE via run_circuit_ablated (float-exact vs
Table 2 for the resample setting).

Per (model, draw): 6x6 circuit-on-band evals + universal cores
(univ6 = AND over all 6 bands; univ5 = AND over the 5 core bands, for
comparability with existing tables). Incremental/resumable; skips draws
whose very_low circuit is missing.

--aggregate: cell-16 stats over 6 bands vs univ6 + focused very_low rows.
"""

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t
from scipy import stats

ISC_ROOT = Path(__file__).resolve().parents[2]
RE = ISC_ROOT / "supplementary_experiments"
sys.path.insert(0, str(RE / "code"))
sys.path.insert(0, str(ISC_ROOT / "LSC_circuits"))
sys.path.insert(0, str(ISC_ROOT / "circuit_discovery" / "auto-circuit"))

import mean_ablation_cross_band as mab  # noqa: E402
from lsc_acdc_circuit import (  # noqa: E402
    cleanup_gpu, get_batch_size, load_dataset, load_model, load_pool,
    model_safe_name, safe_delete_model, set_all_seeds,
)
from mean_ablation_cross_band import (  # noqa: E402
    append_row, build_universal, load_done_keys, log, run_circuit_ablated,
)
from lsc_acdc_circuit import compute_accuracy_metrics  # noqa: E402

BANDS6 = ["very_low", "low", "medium", "high", "very_high", "control"]
CORE5 = BANDS6[1:]
DRAWS = ["draw_1", "draw_2", "draw_3"]
EVAL_SEED = 123
DATA_DIR = mab.DATA_DIR
POOL_MATCHED = DATA_DIR / "lsc_token_pools" / "matched"
POOL_UNMATCHED = DATA_DIR / "lsc_token_pools" / "unmatched"
CORE_CIRCUITS = mab.CIRCUITS_DIR
VL_CIRCUITS = RE / "results" / "very_low_circuits"
OUT = RE / "results" / "very_low_transfer"


def band_paths(band, m_safe, draw):
    if band == "very_low":
        return (VL_CIRCUITS / draw / "circuits" / m_safe / band / draw
                / "prune_scores.pkl", "unmatched", POOL_UNMATCHED)
    return (CORE_CIRCUITS / m_safe / band / draw / "prune_scores.pkl",
            "matched", POOL_MATCHED)


def run_model(model_name, device):
    from auto_circuit.utils.graph_utils import patchable_model
    from auto_circuit.types import AblationType
    m_safe = model_safe_name(model_name)
    bsz = get_batch_size(model_name)
    cross_csv = OUT / f"cross_band_{m_safe}.csv"
    univ_csv = OUT / f"universal_{m_safe}.csv"
    done_cross = load_done_keys(cross_csv, ["draw", "source_band", "test_band"])
    done_univ = load_done_keys(univ_csv, ["draw", "universal_kind", "test_band"])

    draws = [d for d in DRAWS
             if band_paths("very_low", m_safe, d)[0].exists()]
    if not draws:
        log(f"{model_name}: no very_low circuits yet, skipping")
        return
    log(f"{model_name}: draws with very_low circuits: {draws}")

    model = load_model(model_name, device)
    bos = model.tokenizer.bos_token_id
    patchable = patchable_model(model=model, factorized=True,
                                slice_output="last_seq", seq_len=None,
                                separate_qkv=False, device=device)
    try:
        for draw in draws:
            t0 = time.time()
            scores, n_edges, test_cache = {}, {}, {}
            for band in BANDS6:
                pkl, variant, pool_dir = band_paths(band, m_safe, draw)
                with open(pkl, "rb") as f:
                    scores[band] = {k: v.to(device)
                                    for k, v in pickle.load(f).items()}
                n_edges[band] = sum(int(t.isinf(v).sum())
                                    for v in scores[band].values())
                pool = load_pool(band, pool_dir)
                data = load_dataset(band, "test", DATA_DIR, variant, draw)
                test_cache[band] = (data, pool)

            for sb in BANDS6:                       # Part A: 6x6
                for tb in BANDS6:
                    if (draw, sb, tb) in done_cross:
                        continue
                    data, pool = test_cache[tb]
                    set_all_seeds(EVAL_SEED)
                    logits, ans = run_circuit_ablated(
                        patchable, scores[sb], n_edges[sb], data, pool, bos,
                        bsz, EVAL_SEED, device, AblationType.RESAMPLE)
                    m = compute_accuracy_metrics(logits, ans)
                    append_row(cross_csv, {
                        "model": model_name, "draw": draw, "source_band": sb,
                        "test_band": tb, "n_edges": n_edges[sb],
                        "accuracy": m["accuracy"],
                        "top5_accuracy": m["top5_accuracy"]})
                    del logits
                log(f"  {model_name} {draw} source={sb} done")

            for kind, bands in [("univ6", BANDS6), ("univ5", CORE5)]:
                univ = build_universal(
                    {b: {k: v.cpu() for k, v in scores[b].items()}
                     for b in bands}, bands)
                univ = {k: v.to(device) for k, v in univ.items()}
                nu = sum(int(t.isinf(v).sum()) for v in univ.values())
                for tb in BANDS6:
                    if (draw, kind, tb) in done_univ:
                        continue
                    data, pool = test_cache[tb]
                    set_all_seeds(EVAL_SEED)
                    logits, ans = run_circuit_ablated(
                        patchable, univ, nu, data, pool, bos, bsz,
                        EVAL_SEED, device, AblationType.RESAMPLE)
                    m = compute_accuracy_metrics(logits, ans)
                    append_row(univ_csv, {
                        "model": model_name, "draw": draw,
                        "universal_kind": kind, "test_band": tb,
                        "n_edges": nu, "accuracy": m["accuracy"],
                        "top5_accuracy": m["top5_accuracy"]})
                    del logits
                del univ
            log(f"  {model_name} {draw} complete "
                f"[{(time.time() - t0) / 60:.1f} min]")
            del scores
            cleanup_gpu()
    finally:
        del patchable
        safe_delete_model(model)
        cleanup_gpu()
        gc.collect()
    log(f"{model_name}: COMPLETE")


def aggregate():
    """Exact count-lattice Wilcoxon (accuracies are
    k/225; float noise must not break ties), n_pairs=18 (3 draws x 6 bands),
    draw-level clustering caveat (cells cluster in 3 draws; draw-mean sign
    test has min p=0.125 at n=3), same-test retention comparators, and
    explicit confound metadata columns."""
    cross_files = sorted(OUT.glob("cross_band_*.csv"))
    cross = pd.concat([pd.read_csv(f) for f in cross_files], ignore_index=True)
    # in-place variant metadata on per-cell files (idempotent)
    if "source_variant" not in cross.columns:
        for f in cross_files:
            df = pd.read_csv(f)
            df["source_variant"] = np.where(df.source_band == "very_low",
                                            "unmatched", "matched")
            df["test_variant"] = np.where(df.test_band == "very_low",
                                          "unmatched", "matched")
            df.to_csv(f, index=False)
        cross = pd.concat([pd.read_csv(f) for f in cross_files],
                          ignore_index=True)
    univ = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("universal_*.csv"))],
                     ignore_index=True)
    u6 = univ[univ.universal_kind == "univ6"].set_index(
        ["model", "draw", "test_band"]).accuracy.to_dict()
    N = 225  # full test split; accuracies live on the k/225 lattice

    def k(acc):
        return int(round(acc * N))

    rows = []
    for mn in cross.model.unique():
        dm = cross[cross.model == mn]
        draws = sorted(dm.draw.unique())
        same_i, cross_i, draw_means = [], [], []
        boosts_same, boosts_cross = [], []
        for draw in draws:
            d_diffs = []
            for tb in BANDS6:
                ku = k(u6[(mn, draw, tb)])
                ks = k(dm[(dm.draw == draw) & (dm.source_band == tb)
                          & (dm.test_band == tb)].accuracy.values[0])
                kc = [k(a) for a in dm[(dm.draw == draw)
                                       & (dm.source_band != tb)
                                       & (dm.test_band == tb)].accuracy]
                # exact integers: same*len(kc) vs sum(cross), common scale
                s_i = len(kc) * (ks - ku)
                c_i = sum(kk - ku for kk in kc)
                same_i.append(s_i)
                cross_i.append(c_i)
                d_diffs.append(s_i - c_i)
                boosts_same.append((ks - ku) / N)
                boosts_cross.append((sum(kc) / len(kc) - ku) / N)
            draw_means.append(np.mean(d_diffs))
        diff = np.array(same_i) - np.array(cross_i)      # exact int lattice
        sa, ca = np.array(boosts_same), np.array(boosts_cross)
        te = ca.mean() / sa.mean() if sa.mean() > 0 else np.nan
        pooled = np.sqrt((sa.var() + ca.var()) / 2)
        d = (sa.mean() - ca.mean()) / pooled if pooled > 0 else 0.0
        p = (stats.wilcoxon(diff, alternative="greater")[1]
             if np.any(diff != 0) else 1.0)
        draws_positive = int(sum(1 for x in draw_means if x > 0))

        # same-test retention comparators (retention must be compared on the
        # same test set, not across sets)
        core = CORE5
        vl_ret, rev_ret = [], []
        for draw in draws:
            for tb in core:
                own = dm[(dm.draw == draw) & (dm.source_band == tb)
                         & (dm.test_band == tb)].accuracy.values[0]
                vl = dm[(dm.draw == draw) & (dm.source_band == "very_low")
                        & (dm.test_band == tb)].accuracy.values[0]
                if own > 0:
                    vl_ret.append(vl / own)
            vl_own = dm[(dm.draw == draw) & (dm.source_band == "very_low")
                        & (dm.test_band == "very_low")].accuracy.values[0]
            for sb in core:
                cv = dm[(dm.draw == draw) & (dm.source_band == sb)
                        & (dm.test_band == "very_low")].accuracy.values[0]
                if vl_own > 0:
                    rev_ret.append(cv / vl_own)
        rows.append({
            "model": mn, "n_draws": len(draws), "n_pairs": len(diff),
            "same_band_boost_mean": sa.mean(),
            "cross_band_boost_mean": ca.mean(),
            "transfer_efficiency": te, "cohens_d": d,
            "wilcoxon_p_cell_level_exact_lattice": p,
            "draws_with_positive_mean_diff": f"{draws_positive}/{len(draws)}",
            "draw_level_sign_test_min_p": 0.5 ** len(draws),
            "vl_on_core_retention_mean": np.mean(vl_ret),
            "vl_on_core_retention_min": np.min(vl_ret),
            "vl_on_core_retention_max": np.max(vl_ret),
            "core_on_vl_retention_mean": np.mean(rev_ret),
            "core_on_vl_retention_min": np.min(rev_ret),
            "core_on_vl_retention_max": np.max(rev_ret),
            "contains_unmatched_very_low": True,
            "very_low_length_matched": False,
            "very_low_pool_n_tokens": 97,
        })
        pfmt = "<0.0001" if p < 1e-4 else f"{p:.4g}"
        log(f"{mn}: TE6={te:.3f} d={d:.3f} p(cell,lattice)={pfmt} "
            f"draws+={draws_positive}/{len(draws)} | "
            f"vl->core retention {np.mean(vl_ret):.3f} "
            f"[{np.min(vl_ret):.3f}-{np.max(vl_ret):.3f}] | "
            f"core->vl retention {np.mean(rev_ret):.3f} "
            f"[{np.min(rev_ret):.3f}-{np.max(rev_ret):.3f}]")
    pd.DataFrame(rows).to_csv(OUT / "very_low_transfer_summary.csv", index=False)
    log(f"saved {OUT / 'very_low_transfer_summary.csv'} "
        f"(n_pairs=18/model; cell-level protocol-matched exploratory stats; "
        f"draw-level n=3 floor p=0.125)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate()
        return
    for mn in args.models:
        run_model(mn, args.device)
    log("ALL DONE")


if __name__ == "__main__":
    main()
