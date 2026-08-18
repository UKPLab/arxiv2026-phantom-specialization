"""Does position aggregation hide band-specific positional structure?

ACDC (and the saved EAP-IG scores) use a position-aggregated graph
(patchable_model(seq_len=None)): one keep/prune decision per edge across all
sequence positions. Because clean and corrupt prompts share positions 0-16
(DIVERGE_IDX=17) and attention is causal, patching at those positions is a
no-op; the nonzero resample-patch delta is confined to the five
repeated-prefix positions (17-21). This script measures whether bands differ
in HOW shared edges are used across those five positions.

Stage "score" (GPU): EAP-IG on a POSITIONAL graph (seq_len=22), same data,
seeds, and hyperparameters as the saved aggregated scores
(lsc_eap_scoring.py: train split, 256 examples, seed 42, ig_samples=10,
grad_function=logit, answer_function=avg_diff, RESAMPLE).

Stage "analyze" (CPU), per model:
  - sanity: prefix-mass fraction at positions 0-16 (expected exactly 0) and
    parity of the signed positional sum against the saved shared-mask EAP-IG
    scores (implementation check);
  - cancellation ratio sum|sum_p s| / sum sum_p|s| (Haklay-style
    cross-position cancellation) and per-position mass over 17-21;
  - similarity metrics per run pair, distinguishing actual aggregated
    scores |sum_p s| from positional marginal mass sum_p|s|;
  - PRIMARY: per-edge normalized positional profiles on shared
    ACDC-selected edges (intersection of the two runs' kept-masks),
    cosine and total-variation, unweighted and attribution-mass-weighted,
    plus per-position edge-vector cosines;
  - exact blocked relabeling test (band labels permuted within draw; the
    statistic is invariant to a global permutation, so the null has 14,400
    distinct relabelings, enumerated exhaustively; directional and
    doubled-tail two-sided p reported) for the between-band vs within-band
    contrast; the readout is the size of the difference, not a claim of
    equivalence.
Caveat recorded in the appendix text: EAP-IG here uses all 256 train
examples, while AutoCircuit ACDC consumed only the first batch
(256/256/128/96/64 for 70m/160m/410m/1b/1.4b), so the instrument is
settings-matched but not data-identical for the three larger models.

Outputs: supplementary_experiments/results/positional/
  scores/{model}/{band}/{draw}.pkl
  positional_pairs.csv, positional_summary.csv
"""

import argparse
import pickle
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t

ISC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ISC / "LSC_circuits"))

from lsc_eap_scoring import (  # noqa: E402
    ALL_BANDS, MODEL_BATCH_SIZES_IG, SEQ_LEN_WITH_BOS, DIVERGE_IDX,
    ScoringConfig, set_all_seeds, cleanup_gpu,
    load_pool, load_dataset, prepare_dataloader, load_model,
)

OUT = ISC / "supplementary_experiments/results/positional"
DRAWS = ["draw_1", "draw_2", "draw_3"]


def score_one(model_name, band, draw, config, device):
    from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
    from auto_circuit.utils.graph_utils import patchable_model
    from auto_circuit.types import AblationType

    out_path = OUT / "scores" / model_name / band / f"{draw}.pkl"
    if out_path.exists():
        print(f"skip (exists): {model_name}/{band}/{draw}", flush=True)
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    set_all_seeds(config.acdc_seed)
    model = load_model(model_name, device)
    bos_id = model.tokenizer.bos_token_id
    pool = load_pool(band, Path(config.pool_dir))
    train_data = load_dataset(band, "train", Path(config.data_dir),
                              config.variant, draw)
    batch_size = MODEL_BATCH_SIZES_IG.get(model_name, 8)
    loader = prepare_dataloader(train_data, pool, bos_id,
                                n_samples=config.train_size,
                                batch_size=batch_size,
                                seed=config.acdc_seed, device=device)
    patchable = patchable_model(
        model=model, factorized=config.factorized,
        slice_output=config.slice_output, seq_len=SEQ_LEN_WITH_BOS,
        separate_qkv=config.separate_qkv, device=device)

    t0 = time.time()
    scores = mask_gradient_prune_scores(
        model=patchable, dataloader=loader, official_edges=None,
        grad_function="logit", answer_function="avg_diff",
        integrated_grad_samples=config.ig_samples,
        ablation_type=AblationType.RESAMPLE)
    dt = time.time() - t0

    cpu_scores = {k: v.detach().cpu() for k, v in scores.items()}
    with open(out_path, "wb") as f:
        pickle.dump(cpu_scores, f)

    shapes = {k: tuple(v.shape) for k, v in list(cpu_scores.items())[:3]}
    total = sum(v.abs().sum().item() for v in cpu_scores.values())
    prefix = sum(v.abs()[:DIVERGE_IDX].sum().item() for v in cpu_scores.values()
                 if v.ndim >= 2 and v.shape[0] == SEQ_LEN_WITH_BOS)
    print(f"{model_name}/{band}/{draw}: {dt:.0f}s, sample shapes {shapes}, "
          f"prefix-mass fraction {prefix / max(total, 1e-30):.2e}", flush=True)

    del scores, cpu_scores, patchable, model
    cleanup_gpu()


CIRCUITS_DIR = ISC / "LSC_circuits/circuit_discovery/circuits"
EAP_IG_DIR = ISC / "LSC_circuits/EAP_methods/eap_ig_scores"
N_TAIL = SEQ_LEN_WITH_BOS - DIVERGE_IDX  # 5 corrupted tail positions


def load_run(model_name, band, draw):
    """Load one run: signed positional tail scores, ACDC mask, parity info.

    Returns dict with:
      pos:   [5, E] signed positional scores (positions 17-21, edges flattened
             over sorted module keys)
      mask:  [E] bool ACDC kept-mask (retained == inf convention)
      parity_cos: cosine(sum_p signed positional, saved aggregated EAP-IG)
      prefix_frac: |score| mass fraction at positions 0-16 (expected 0)
    """
    with open(OUT / "scores" / model_name / band / f"{draw}.pkl", "rb") as f:
        d = pickle.load(f)
    with open(CIRCUITS_DIR / model_name.replace("-", "_") / band / draw
              / "prune_scores.pkl", "rb") as f:
        acdc = pickle.load(f)
    with open(EAP_IG_DIR / model_name.replace("-", "_") / band / draw
              / "scores.pkl", "rb") as f:
        agg_saved = pickle.load(f)

    modules = sorted(d.keys())
    assert modules == sorted(acdc.keys()) == sorted(agg_saved.keys())
    pos_parts, mask_parts, saved_parts = [], [], []
    prefix_mass = total_mass = 0.0
    for k in modules:
        v = d[k].float()
        assert v.ndim >= 2 and v.shape[0] == SEQ_LEN_WITH_BOS, \
            f"unexpected shape {tuple(v.shape)} for {k}"
        total_mass += v.abs().sum().item()
        prefix_mass += v.abs()[:DIVERGE_IDX].sum().item()
        tail = v[DIVERGE_IDX:].flatten(1)            # [5, E_k], signed
        assert tuple(tail.sum(0).shape) == tuple(acdc[k].flatten().shape)
        pos_parts.append(tail)
        mask_parts.append(t.isinf(acdc[k]).flatten())
        saved_parts.append(agg_saved[k].flatten().float())
    pos = t.cat(pos_parts, dim=1).numpy()            # [5, E]
    signed_sum = pos.sum(axis=0)
    saved = t.cat(saved_parts).numpy()
    parity = float(np.dot(signed_sum, saved)
                   / (np.linalg.norm(signed_sum) * np.linalg.norm(saved)))
    return {"pos": pos, "mask": t.cat(mask_parts).numpy(),
            "parity_cos": parity,
            "prefix_frac": prefix_mass / max(total_mass, 1e-30)}


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else np.nan


def pair_metrics(ra, rb):
    """All pairwise similarity metrics for two runs."""
    pa, pb = np.abs(ra["pos"]), np.abs(rb["pos"])
    out = {"cos_positional_global": cos(pa.ravel(), pb.ravel()),
           "cos_actual_aggregated": cos(np.abs(ra["pos"].sum(0)),
                                        np.abs(rb["pos"].sum(0))),
           "cos_marginal_mass": cos(pa.sum(0), pb.sum(0))}
    for p in range(N_TAIL):
        out[f"cos_pos_{DIVERGE_IDX + p}"] = cos(pa[p], pb[p])
    # primary: per-edge positional profiles on shared ACDC-selected edges
    shared = ra["mask"] & rb["mask"]
    out["n_shared_acdc_edges"] = int(shared.sum())
    if shared.sum() > 0:
        qa, qb = pa[:, shared], pb[:, shared]        # [5, S]
        ma, mb = qa.sum(0), qb.sum(0)
        ok = (ma > 0) & (mb > 0)
        qa, qb, ma, mb = qa[:, ok], qb[:, ok], ma[ok], mb[ok]
        prof_a, prof_b = qa / ma, qb / mb            # normalized profiles
        per_edge_cos = (prof_a * prof_b).sum(0) / (
            np.linalg.norm(prof_a, axis=0) * np.linalg.norm(prof_b, axis=0))
        per_edge_tv = 0.5 * np.abs(prof_a - prof_b).sum(0)
        # normalize each run's edge masses before taking the min so the
        # weight is invariant to globally rescaling either run
        w = np.minimum(ma / ma.sum(), mb / mb.sum()); w = w / w.sum()
        out["shared_profile_cos_unweighted"] = float(per_edge_cos.mean())
        out["shared_profile_cos_weighted"] = float((per_edge_cos * w).sum())
        out["shared_profile_tv_unweighted"] = float(per_edge_tv.mean())
        out["shared_profile_tv_weighted"] = float((per_edge_tv * w).sum())
    return out


def blocked_permutation_p(dist, labels):
    """Exact blocked relabeling test for the between-vs-within contrast.

    dist: {(i, j): distance} over run indices; labels: list of (band, draw).
    Band labels are permuted independently within each draw; the statistic is
    mean(between-band same-draw distance) - mean(within-band cross-draw
    distance). The statistic is invariant to applying one permutation to all
    draws jointly, so the exact null has (5!)^3 / 5! = 14,400 distinct
    relabelings, enumerated by fixing draw 1 and permuting draws 2 and 3.
    Returns (obs, p_upper directional, p_two doubled-tail).
    """
    from itertools import permutations

    draws = sorted({d for _, d in labels})
    bands = sorted({b for b, _ in labels})
    assert len(draws) == 3 and len(bands) == 5

    def contrast(lab):
        within, between = [], []
        for (i, j), v in dist.items():
            (b1, d1), (b2, d2) = lab[i], lab[j]
            if b1 == b2 and d1 != d2:
                within.append(v)
            elif d1 == d2 and b1 != b2:
                between.append(v)
        return np.mean(between) - np.mean(within)

    obs = contrast(labels)
    idx_by_draw = {d: [i for i, (_, dd) in enumerate(labels) if dd == d]
                   for d in draws}
    ts = []
    for p2 in permutations(bands):
        for p3 in permutations(bands):
            lab = list(labels)
            for i, b in zip(idx_by_draw[draws[1]], p2):
                lab[i] = (b, draws[1])
            for i, b in zip(idx_by_draw[draws[2]], p3):
                lab[i] = (b, draws[2])
            ts.append(contrast(lab))
    ts = np.array(ts)
    p_upper = float((ts >= obs - 1e-15).mean())
    p_lower = float((ts <= obs + 1e-15).mean())
    p_two = min(1.0, 2 * min(p_upper, p_lower))
    return obs, p_upper, p_two


def analyze(models):
    rows, srows = [], []
    for m in models:
        keys = [(b, d) for b in ALL_BANDS for d in DRAWS]
        runs = {k: load_run(m, *k) for k in keys}

        cancel = []
        per_pos_mass = np.zeros(N_TAIL)
        for r in runs.values():
            marg = np.abs(r["pos"]).sum()
            cancel.append(np.abs(r["pos"].sum(0)).sum() / marg)
            per_pos_mass += np.abs(r["pos"]).sum(1) / marg
        per_pos_mass /= len(runs)

        pair_rows = {}
        for k1, k2 in combinations(keys, 2):
            met = pair_metrics(runs[k1], runs[k2])
            pair_rows[(k1, k2)] = met
            same_band, same_draw = k1[0] == k2[0], k1[1] == k2[1]
            kind = ("within" if same_band and not same_draw else
                    "between" if same_draw and not same_band else "other")
            rows.append({"model": m, "kind": kind,
                         "run_a": f"{k1[0]}/{k1[1]}",
                         "run_b": f"{k2[0]}/{k2[1]}", **met})

        dm = pd.DataFrame([r for r in rows if r["model"] == m])
        s = {"model": m,
             "max_prefix_mass_fraction": max(r["prefix_frac"]
                                             for r in runs.values()),
             "min_parity_cos": min(r["parity_cos"] for r in runs.values()),
             "mean_cancellation_ratio": float(np.mean(cancel))}
        for i, frac in enumerate(per_pos_mass):
            s[f"mass_pos_{DIVERGE_IDX + i}"] = frac
        metrics = ["cos_positional_global", "cos_actual_aggregated",
                   "cos_marginal_mass", "shared_profile_cos_weighted",
                   "shared_profile_cos_unweighted",
                   "shared_profile_tv_weighted"]
        for met in metrics:
            for kind in ("within", "between"):
                s[f"{kind}_{met}"] = dm[dm.kind == kind][met].mean()
        # blocked permutation tests on the two key metrics (as distances)
        idx = {k: i for i, k in enumerate(keys)}
        for met, to_dist in [("cos_positional_global", lambda v: 1 - v),
                             ("shared_profile_cos_weighted", lambda v: 1 - v),
                             ("shared_profile_tv_weighted", lambda v: v)]:
            dist = {(idx[k1], idx[k2]): to_dist(v[met])
                    for (k1, k2), v in pair_rows.items()}
            obs, p_dir, p_two = blocked_permutation_p(dist, keys)
            s[f"perm_contrast_{met}"] = obs
            s[f"perm_p_directional_{met}"] = p_dir
            s[f"perm_p_{met}"] = p_two
        srows.append(s)
        print(f"{m}: prefix {s['max_prefix_mass_fraction']:.1e} | parity "
              f"{s['min_parity_cos']:.8f} | cancel {s['mean_cancellation_ratio']:.3f} | "
              f"global cos w/b {s['within_cos_positional_global']:.4f}/"
              f"{s['between_cos_positional_global']:.4f} "
              f"(perm p {s['perm_p_cos_positional_global']:.4f}) | "
              f"shared-edge wcos w/b {s['within_shared_profile_cos_weighted']:.4f}/"
              f"{s['between_shared_profile_cos_weighted']:.4f} "
              f"(perm p {s['perm_p_shared_profile_cos_weighted']:.4f})", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "positional_pairs.csv", index=False)
    pd.DataFrame(srows).to_csv(OUT / "positional_summary.csv", index=False)
    print("saved:", OUT / "positional_summary.csv", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["score", "analyze"], required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--bands", nargs="+", default=ALL_BANDS)
    ap.add_argument("--draws", nargs="+", default=DRAWS)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    if args.stage == "score":
        config = ScoringConfig()
        for m in args.models:
            for b in args.bands:
                for d in args.draws:
                    score_one(m, b, d, config, args.device)
    else:
        analyze(args.models)


if __name__ == "__main__":
    main()
