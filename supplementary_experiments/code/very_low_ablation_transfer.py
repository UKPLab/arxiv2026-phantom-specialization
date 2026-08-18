"""very_low 6x6 cross-band transfer under zero and mean ablation.

Extends the resample-ablation very_low grid (very_low_transfer.py) to the
two alternative interventions used by the paper's robustness protocols:
ZERO (NB12) and TOKENWISE_MEAN_CORRUPT (mean_ablation_cross_band.py).
Everything else is identical: same circuits (very_low from
results/very_low_circuits/, core bands canonical), same univ6/univ5
construction, same eval seam (run_circuit_ablated), same test splits and
seeds. Only ablation_type and the output directory differ.

Outputs (under results/very_low_ablation/{ablation}/):
  cross_band_{model}.csv, universal_{model}.csv   (incremental/resumable)
  very_low_ablation_summary.csv                   (--aggregate, per ablation)

The aggregation mirrors very_low_transfer.aggregate exactly (count-lattice
Wilcoxon on k/225, n_pairs=18, TE vs univ6, same-test retention both
directions, confound metadata) and adds the per-model mean same-band
circuit accuracy under the ablation, so accuracy compression under zero
ablation is visible next to any TE readout.
"""

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import torch as t

ISC_ROOT = Path(__file__).resolve().parents[2]
RE = ISC_ROOT / "supplementary_experiments"
sys.path.insert(0, str(RE / "code"))
sys.path.insert(0, str(ISC_ROOT / "LSC_circuits"))
sys.path.insert(0, str(ISC_ROOT / "circuit_discovery" / "auto-circuit"))

from lsc_acdc_circuit import (  # noqa: E402
    cleanup_gpu, compute_accuracy_metrics, get_batch_size, load_dataset,
    load_model, load_pool, model_safe_name, safe_delete_model, set_all_seeds,
)
from mean_ablation_cross_band import (  # noqa: E402
    append_row, build_universal, load_done_keys, log, run_circuit_ablated,
)
from very_low_transfer import (  # noqa: E402
    BANDS6, CORE5, DATA_DIR, DRAWS, EVAL_SEED, band_paths,
)

ABLATIONS = {"zero": "ZERO",
             "tokenwise_mean_corrupt": "TOKENWISE_MEAN_CORRUPT"}
OUT_BASE = RE / "results" / "very_low_ablation"


def run_model(model_name, device, ablation, out_dir):
    from auto_circuit.utils.graph_utils import patchable_model
    from auto_circuit.types import AblationType
    ablation_type = getattr(AblationType, ABLATIONS[ablation])
    m_safe = model_safe_name(model_name)
    bsz = get_batch_size(model_name)
    cross_csv = out_dir / f"cross_band_{m_safe}.csv"
    univ_csv = out_dir / f"universal_{m_safe}.csv"
    done_cross = load_done_keys(cross_csv, ["draw", "source_band", "test_band"])
    done_univ = load_done_keys(univ_csv, ["draw", "universal_kind", "test_band"])

    draws = [d for d in DRAWS
             if band_paths("very_low", m_safe, d)[0].exists()]
    if not draws:
        log(f"{model_name}: no very_low circuits, skipping")
        return
    log(f"{model_name} [{ablation}]: draws {draws}")

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
                        bsz, EVAL_SEED, device, ablation_type)
                    m = compute_accuracy_metrics(logits, ans)
                    append_row(cross_csv, {
                        "model": model_name, "draw": draw, "source_band": sb,
                        "test_band": tb, "n_edges": n_edges[sb],
                        "ablation": ablation,
                        "accuracy": m["accuracy"],
                        "top5_accuracy": m["top5_accuracy"]})
                    del logits
                log(f"  {model_name} [{ablation}] {draw} source={sb} done")

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
                        EVAL_SEED, device, ablation_type)
                    m = compute_accuracy_metrics(logits, ans)
                    append_row(univ_csv, {
                        "model": model_name, "draw": draw,
                        "universal_kind": kind, "test_band": tb,
                        "n_edges": nu, "ablation": ablation,
                        "accuracy": m["accuracy"],
                        "top5_accuracy": m["top5_accuracy"]})
                    del logits
                del univ
            log(f"  {model_name} [{ablation}] {draw} complete "
                f"[{(time.time() - t0) / 60:.1f} min]")
            del scores
            cleanup_gpu()
    finally:
        del patchable
        safe_delete_model(model)
        cleanup_gpu()
        gc.collect()
    log(f"{model_name} [{ablation}]: COMPLETE")


def aggregate(ablation, out_dir):
    """Identical protocol to very_low_transfer.aggregate (count-lattice
    Wilcoxon n_pairs=18, TE vs univ6, same-test retention, confound
    metadata), plus mean same-band circuit accuracy under this ablation."""
    cross_files = sorted(out_dir.glob("cross_band_*.csv"))
    cross = pd.concat([pd.read_csv(f) for f in cross_files], ignore_index=True)
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
    univ = pd.concat(
        [pd.read_csv(f) for f in sorted(out_dir.glob("universal_*.csv"))],
        ignore_index=True)
    u6 = univ[univ.universal_kind == "univ6"].set_index(
        ["model", "draw", "test_band"]).accuracy.to_dict()
    N = 225

    def k(acc):
        return int(round(acc * N))

    rows = []
    for mn in cross.model.unique():
        dm = cross[cross.model == mn]
        draws = sorted(dm.draw.unique())
        same_i, cross_i, draw_means = [], [], []
        boosts_same, boosts_cross, same_accs = [], [], []
        for draw in draws:
            d_diffs = []
            for tb in BANDS6:
                ku = k(u6[(mn, draw, tb)])
                ks = k(dm[(dm.draw == draw) & (dm.source_band == tb)
                          & (dm.test_band == tb)].accuracy.values[0])
                kc = [k(a) for a in dm[(dm.draw == draw)
                                       & (dm.source_band != tb)
                                       & (dm.test_band == tb)].accuracy]
                s_i = len(kc) * (ks - ku)
                c_i = sum(kk - ku for kk in kc)
                same_i.append(s_i)
                cross_i.append(c_i)
                d_diffs.append(s_i - c_i)
                boosts_same.append((ks - ku) / N)
                boosts_cross.append((sum(kc) / len(kc) - ku) / N)
                same_accs.append(ks / N)
            draw_means.append(np.mean(d_diffs))
        diff = np.array(same_i) - np.array(cross_i)
        sa, ca = np.array(boosts_same), np.array(boosts_cross)
        te = ca.mean() / sa.mean() if sa.mean() > 0 else np.nan
        pooled = np.sqrt((sa.var() + ca.var()) / 2)
        d = (sa.mean() - ca.mean()) / pooled if pooled > 0 else 0.0
        p = (stats.wilcoxon(diff, alternative="greater")[1]
             if np.any(diff != 0) else 1.0)
        draws_positive = int(sum(1 for x in draw_means if x > 0))

        vl_ret, rev_ret = [], []
        for draw in draws:
            for tb in CORE5:
                own = dm[(dm.draw == draw) & (dm.source_band == tb)
                         & (dm.test_band == tb)].accuracy.values[0]
                vl = dm[(dm.draw == draw) & (dm.source_band == "very_low")
                        & (dm.test_band == tb)].accuracy.values[0]
                if own > 0:
                    vl_ret.append(vl / own)
            vl_own = dm[(dm.draw == draw) & (dm.source_band == "very_low")
                        & (dm.test_band == "very_low")].accuracy.values[0]
            for sb in CORE5:
                cv = dm[(dm.draw == draw) & (dm.source_band == sb)
                        & (dm.test_band == "very_low")].accuracy.values[0]
                if vl_own > 0:
                    rev_ret.append(cv / vl_own)
        rows.append({
            "model": mn, "ablation": ablation,
            "n_draws": len(draws), "n_pairs": len(diff),
            "same_band_boost_mean": sa.mean(),
            "cross_band_boost_mean": ca.mean(),
            "transfer_efficiency": te, "cohens_d": d,
            "wilcoxon_p_cell_level_exact_lattice": p,
            "draws_with_positive_mean_diff": f"{draws_positive}/{len(draws)}",
            "draw_level_sign_test_min_p": 0.5 ** len(draws),
            "mean_same_band_circuit_acc": float(np.mean(same_accs)),
            "vl_on_core_retention_mean": np.mean(vl_ret) if vl_ret else np.nan,
            "vl_on_core_retention_min": np.min(vl_ret) if vl_ret else np.nan,
            "vl_on_core_retention_max": np.max(vl_ret) if vl_ret else np.nan,
            "core_on_vl_retention_mean": (np.mean(rev_ret) if rev_ret
                                          else np.nan),
            "core_on_vl_retention_min": (np.min(rev_ret) if rev_ret
                                         else np.nan),
            "core_on_vl_retention_max": (np.max(rev_ret) if rev_ret
                                         else np.nan),
            "n_retention_pairs_vl_on_core": len(vl_ret),
            "n_retention_pairs_core_on_vl": len(rev_ret),
            "contains_unmatched_very_low": True,
            "very_low_length_matched": False,
            "very_low_pool_n_tokens": 97,
        })
        pfmt = "<0.0001" if p < 1e-4 else f"{p:.4g}"
        te_fmt = "nan" if np.isnan(te) else f"{te:.3f}"
        vlr = f"{np.mean(vl_ret):.3f}" if vl_ret else "n/a"
        cvr = f"{np.mean(rev_ret):.3f}" if rev_ret else "n/a"
        log(f"{mn} [{ablation}]: TE6={te_fmt} d={d:.3f} p={pfmt} "
            f"draws+={draws_positive}/{len(draws)} "
            f"same_acc={np.mean(same_accs):.3f} | "
            f"vl->core {vlr} ({len(vl_ret)} pairs) | "
            f"core->vl {cvr} ({len(rev_ret)} pairs)")
    out_csv = out_dir / "very_low_ablation_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log(f"saved {out_csv} (protocol-matched exploratory stats; "
        f"draw-level n=3 floor p=0.125)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", required=True, choices=sorted(ABLATIONS))
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    out_dir = OUT_BASE / args.ablation
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate(args.ablation, out_dir)
        return
    for mn in args.models:
        run_model(mn, args.device, args.ablation, out_dir)
    log(f"ALL DONE [{args.ablation}]")


if __name__ == "__main__":
    main()
