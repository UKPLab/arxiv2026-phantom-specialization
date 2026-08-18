"""EAP-IG on the very_low band + 6-band completion of the cross-method grid.

Extends the paper's cross-method comparison (lsc_eap_scoring.py +
lsc_eap_eval.py, 75 conditions, method eap_ig) to the confound-relaxed
very_low band. The published EAP_methods/ artifacts are READ-ONLY; all new
outputs live under supplementary_experiments/results/very_low_eapig/.

Stages (--stage):
  score      EAP-IG scores for very_low x 5 models x 3 draws, settings
             identical to the canonical run (ig_samples 10, train 256,
             acdc_seed 42) except variant/pool = unmatched, since
             very_low has no length-matched pool.
  threshold  Top-k circuits at the 10 canonical size multipliers relative
             to the ACDC very_low edge count; Jaccard/Dice vs the ACDC
             very_low circuits (canonical overlap schema).
  eval       RESAMPLE tree-patch evals for the 6x6 COMPLETION cells only:
             very_low EAP-IG circuits on all 6 test bands + canonical
             core-band eap_ig circuits on the very_low test, per size.
             Canonical eap_eval_results.csv schema; the published 5x5
             cells are reused unchanged at aggregation.
  aggregate  Same-band advantage per (model, size) over the merged 6-band
             grid; overlap range; focused very_low rows.

Base accuracy / KL: base logits are computed in-eval per (model, draw,
test_band) and reused across sizes; where a stored base-metrics file
exists the computed accuracy is cross-checked against it and any
difference is logged (eval-path verification).
"""

import argparse
import gc
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t

ISC_ROOT = Path(__file__).resolve().parents[2]
RE = ISC_ROOT / "supplementary_experiments"
sys.path.insert(0, str(RE / "code"))
sys.path.insert(0, str(ISC_ROOT / "LSC_circuits"))
sys.path.insert(0, str(ISC_ROOT / "circuit_discovery" / "auto-circuit"))

import lsc_eap_eval as ev  # noqa: E402
import lsc_eap_scoring as sc  # noqa: E402
from mean_ablation_cross_band import append_row, load_done_keys, log  # noqa: E402
from very_low_transfer import BANDS6, CORE5, DRAWS, band_paths  # noqa: E402

MODELS = list(sc.DEFAULT_MODELS)
SIZES = list(ev.SIZE_MULTIPLIERS)
EVAL_SEED = 123
OUT = RE / "results" / "very_low_eapig"
EAP_METHODS = Path(sc.SCRIPT_DIR) / "EAP_methods"          # read-only
POOL_UNMATCHED = ISC_ROOT / "LSC_data" / "lsc_token_pools" / "unmatched"
POOL_MATCHED = ISC_ROOT / "LSC_data" / "lsc_token_pools" / "matched"
DATA_DIR = ISC_ROOT / "LSC_data"
VL_CIRCUITS = RE / "results" / "very_low_circuits"


def stage_score(models, draws, gpus):
    config = sc.ScoringConfig(
        pool_dir=str(POOL_UNMATCHED),
        output_dir=str(OUT),
        variant="unmatched",
        models=models, bands=["very_low"], draws=draws,
        method="eap_ig",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    sc.setup_logging(OUT)
    sc.run_all_tasks(config, OUT, gpus=gpus, force=False)
    sc.print_summary(OUT, config)


def stage_threshold(models, draws):
    """CPU. Top-k at the canonical multipliers vs the ACDC very_low size."""
    overlap_csv = OUT / "vl_eap_overlap.csv"
    done = load_done_keys(overlap_csv, ["model", "draw", "size_multiplier"])
    for mn in models:
        m_safe = ev.model_safe_name(mn)
        for draw in draws:
            scores_pkl = OUT / "eap_ig_scores" / m_safe / "very_low" / draw \
                / "scores.pkl"
            if not scores_pkl.exists():
                log(f"threshold: missing scores {scores_pkl}, skip")
                continue
            with open(scores_pkl, "rb") as f:
                raw_scores = pickle.load(f)
            acdc_pkl = band_paths("very_low", m_safe, draw)[0]
            with open(acdc_pkl, "rb") as f:
                acdc_circuit = pickle.load(f)
            n_acdc = ev.count_circuit_edges(acdc_circuit)
            total = ev.get_total_edges(raw_scores)
            cdir = OUT / "eap_ig_circuits" / m_safe / "very_low" / draw
            cdir.mkdir(parents=True, exist_ok=True)
            for mult in SIZES:
                key = ev.size_to_key(mult)
                if (mn, draw, str(mult)) in done:
                    continue
                k = min(int(round(n_acdc * mult)), total)
                circuit = ev.extract_top_k_edges(raw_scores, k)
                with open(cdir / f"prune_scores_{key}.pkl", "wb") as f:
                    pickle.dump(circuit, f)
                ov = ev.compute_overlap(circuit, acdc_circuit)
                append_row(overlap_csv, {
                    "method": "eap_ig", "model": mn, "band": "very_low",
                    "draw": draw, "size_multiplier": mult,
                    "n_edges": ev.count_circuit_edges(circuit),
                    "n_edges_acdc_ref": n_acdc, "total_edges": total,
                    "size_fraction": (ev.count_circuit_edges(circuit) / total
                                      if total else 0.0),
                    **ov})
            log(f"threshold: {mn} {draw} done (acdc ref {n_acdc} edges)")


def _band_data(band, draw):
    if band == "very_low":
        variant, pool_dir = "unmatched", POOL_UNMATCHED
    else:
        variant, pool_dir = "matched", POOL_MATCHED
    pool = ev.load_pool(band, pool_dir)
    data = ev.load_dataset(band, "test", DATA_DIR, variant, draw)
    return data, pool


def _stored_base_acc(mn, band, draw):
    """Stored base accuracy for cross-checking, if a file exists."""
    m_safe = ev.model_safe_name(mn)
    if band == "very_low":
        path = VL_CIRCUITS / draw / "base_metrics" / m_safe / draw \
            / f"{band}.json"
    else:
        path = Path(ev.SCRIPT_DIR) / "base_metrics" / m_safe / draw \
            / f"{band}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)["splits"]["test"].get("accuracy")


def stage_eval(models, draws, device):
    from auto_circuit.utils.graph_utils import patchable_model
    results_csv = OUT / "vl_eap_eval_results.csv"
    done = load_done_keys(
        results_csv,
        ["model", "size_multiplier", "circuit_band", "draw", "test_band"])

    for mn in models:
        m_safe = ev.model_safe_name(mn)
        bsz = ev.get_batch_size(mn)
        # (circuit_band, size_key, pkl_path) completion tasks for this model
        model = None
        patchable = None
        try:
            for draw in draws:
                tasks = []
                for mult in SIZES:
                    key = ev.size_to_key(mult)
                    vl_pkl = OUT / "eap_ig_circuits" / m_safe / "very_low" \
                        / draw / f"prune_scores_{key}.pkl"
                    tasks.append(("very_low", key, vl_pkl, BANDS6))
                    for band in CORE5:
                        core_pkl = EAP_METHODS / "eap_ig_circuits" / m_safe \
                            / band / draw / f"prune_scores_{key}.pkl"
                        tasks.append((band, key, core_pkl, ["very_low"]))
                todo = [(cb, key, pkl, tbs) for cb, key, pkl, tbs in tasks
                        if any((mn, key, cb, draw, tb) not in done
                               for tb in tbs)]
                if not todo:
                    continue
                if model is None:
                    model = ev.load_model(mn, device)
                    bos = model.tokenizer.bos_token_id
                    patchable = patchable_model(
                        model=model, factorized=True, slice_output="last_seq",
                        seq_len=None, separate_qkv=False, device=device)
                t0 = time.time()
                cache, base_logits = {}, {}
                for band in BANDS6:
                    cache[band] = _band_data(band, draw)
                for cb, key, pkl, test_bands in todo:
                    if not pkl.exists():
                        log(f"eval: missing circuit {pkl}, skip")
                        continue
                    with open(pkl, "rb") as f:
                        circuit_cpu = pickle.load(f)
                    n_edges = ev.count_circuit_edges(circuit_cpu)
                    total = ev.get_total_edges(circuit_cpu)
                    if n_edges == 0:
                        log(f"eval: empty circuit {pkl}, skip")
                        continue
                    circuit = {k2: v.to(device)
                               for k2, v in circuit_cpu.items()}
                    for tb in test_bands:
                        if (mn, key, cb, draw, tb) in done:
                            continue
                        data, pool = cache[tb]
                        if tb not in base_logits:
                            ev.set_all_seeds(EVAL_SEED)
                            bl, ba, _ = ev.compute_base_logits(
                                model, data, pool, bos, bsz, EVAL_SEED,
                                device)
                            bacc = ev.compute_accuracy_metrics(bl, ba)
                            stored = _stored_base_acc(mn, tb, draw)
                            if stored is not None and \
                                    abs(stored - bacc["accuracy"]) > 1e-9:
                                log(f"  BASE-ACC MISMATCH {mn} {draw} {tb}: "
                                    f"computed {bacc['accuracy']:.6f} vs "
                                    f"stored {stored:.6f}")
                            base_logits[tb] = (bl, bacc["accuracy"])
                        bl, bacc = base_logits[tb]
                        circ_logits, circ_ans = ev.run_circuit_and_collect(
                            patchable, circuit, n_edges, data, pool, bos,
                            bsz, EVAL_SEED, device)
                        cm = ev.compute_accuracy_metrics(circ_logits,
                                                         circ_ans)
                        kl = ev.compute_kl_divergence(circ_logits, bl)
                        append_row(results_csv, {
                            "method": "eap_ig", "size_multiplier": key,
                            "model": mn, "draw": draw, "circuit_band": cb,
                            "test_band": tb, "n_edges": n_edges,
                            "total_edges": total,
                            "size_fraction": (n_edges / total if total
                                              else 0.0),
                            "circuit_accuracy": cm["accuracy"],
                            "circuit_top5_accuracy": cm["top5_accuracy"],
                            "circuit_mean_prob": cm["mean_correct_prob"],
                            "base_accuracy": bacc, "kl_div": kl,
                            "n_samples": cm["n_samples"]})
                        del circ_logits
                    del circuit
                    ev.cleanup_gpu()
                log(f"eval: {mn} {draw} complete "
                    f"[{(time.time() - t0) / 60:.1f} min]")
                del cache, base_logits
                ev.cleanup_gpu()
        finally:
            if patchable is not None:
                del patchable
            if model is not None:
                del model
            ev.cleanup_gpu()
            gc.collect()
        log(f"eval: {mn} COMPLETE")


def stage_aggregate():
    """Merge the published 5x5 eap_ig cells (read-only) with the completion
    cells; same-band advantage per (model, size) over the 6-band grid."""
    pub = pd.read_csv(EAP_METHODS / "eap_eval_results.csv")
    pub = pub[pub.method == "eap_ig"].copy()
    new = pd.read_csv(OUT / "vl_eap_eval_results.csv")
    full = pd.concat([pub, new], ignore_index=True)
    full = full.drop_duplicates(
        subset=["size_multiplier", "model", "draw", "circuit_band",
                "test_band"], keep="first")

    rows = []
    for mn in MODELS:
        for key in [ev.size_to_key(m) for m in SIZES]:
            g = full[(full.model == mn) & (full.size_multiplier == key)]
            n_cells = len(g)
            same = g[g.circuit_band == g.test_band]
            cross = g[g.circuit_band != g.test_band]
            if same.empty or cross.empty:
                continue
            # per (draw, test_band): own acc minus mean cross acc
            gaps = []
            for (draw, tb), sub in cross.groupby(["draw", "test_band"]):
                own = same[(same.draw == draw) & (same.test_band == tb)]
                if own.empty:
                    continue
                gaps.append(own.circuit_accuracy.values[0]
                            - sub.circuit_accuracy.mean())
            rows.append({
                "model": mn, "size_multiplier": key, "n_cells": n_cells,
                "n_gap_cells": len(gaps),
                "same_band_advantage": float(np.mean(gaps)),
                "same_band_acc_mean": float(same.circuit_accuracy.mean()),
                "cross_band_acc_mean": float(cross.circuit_accuracy.mean()),
            })
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "vl_eapig_same_band_advantage.csv", index=False)

    ov = pd.read_csv(OUT / "vl_eap_overlap.csv")
    ov1 = ov[ov.size_multiplier == 1.0]
    for mn in MODELS:
        sm = summ[summ.model == mn]
        if sm.empty:
            log(f"{mn}: no aggregated cells yet")
            continue
        worst = sm.loc[sm.same_band_advantage.idxmax()]
        j = ov1[ov1.model == mn].jaccard
        jfmt = f"{j.min():.2f}-{j.max():.2f}" if len(j) else "n/a"
        log(f"{mn}: max same-band advantage over sizes "
            f"{worst.same_band_advantage:+.4f} (at {worst.size_multiplier}, "
            f"{int(worst.n_gap_cells)} gap cells) | "
            f"1.0x jaccard vs ACDC(very_low) {jfmt}")
    log(f"saved {OUT / 'vl_eapig_same_band_advantage.csv'} "
        f"(6-band grid incl. unmatched very_low; published 5x5 cells "
        f"reused read-only; exploratory)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["score", "threshold", "eval", "aggregate"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--draws", nargs="+", default=DRAWS)
    ap.add_argument("--gpus", nargs="+", type=int, default=[0])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "score":
        stage_score(args.models, args.draws, args.gpus)
    elif args.stage == "threshold":
        stage_threshold(args.models, args.draws)
    elif args.stage == "eval":
        stage_eval(args.models, args.draws, args.device)
    else:
        stage_aggregate()


if __name__ == "__main__":
    main()
