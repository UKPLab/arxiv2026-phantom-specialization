"""SVA verification controls (GPU) + aggregation cross-check.

Model selected via SVA_CONTROLS_MODEL (default pythia-160m); all paths,
edge counts, and the paired LSC circuits follow that selection.

1. SENSITIVITY (evaluation-only): recompute the full transfer matrix +
   universal boosts with brow/disc-contaminated TEST examples excluded
   (circuits unchanged; discovery used the frozen datasets and retained
   the training-set contamination).
2. RANDOM CONTROL: 10 random circuits size-matched to the model's
   control/draw_1 circuit, on the control test set.
3. CROSS-TASK SANITY CHECK (task specificity, not a positive control
   for within-task specialization):
   a) each LSC band circuit for this model (draw_1) evaluated on the
      SVA control test;
   b) the SVA control/draw_1 circuit evaluated on the LSC control test
      via the canonical (Table-2-validated) eval machinery.
   Expected: cross-task transfer collapses while within-task cross-band
   transfer stays high.
4. DETERMINISM: re-run ACDC for low/draw_1, compare to saved scores.
5. CELL-16 CROSS-CHECK (CPU): run the NB12-validated aggregate() from
   mean_ablation_cross_band.py on the SVA numbers; compare TE/d/p to
   sva_aggregate.py output.
"""

import json
import pickle
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "supplementary_experiments" / "code"))

from sva_discovery import (  # noqa: E402
    BANDS, DRAWS, get_patchable, load_split, metrics_from_logits, run_acdc,
    run_circuit,
)
from sva_aggregate import build_universal  # noqa: E402
from sva_stats_util import exact_wilcoxon_greater  # noqa: E402

import os
BAD_IDS = {6479, 22931, 1262, 28217}  # Ġbrow, Ġbrows, Ġdisc, Ġdiscs (STEM tokens)
MODEL = os.environ.get("SVA_CONTROLS_MODEL", "pythia-160m")
CIRCUITS = BASE / "circuits" / MODEL
RESULTS = BASE / "results"
DEVICE = "cuda:0"
LSC_CIRCUITS = (Path(__file__).resolve().parents[3] / "LSC_circuits"
                    / "circuit_discovery" / "circuits") / MODEL.replace("-", "_")
# Determinism re-run threshold: the model's own selected tau*, read from
# the saved low/draw_1 circuit (thresholds differ per model: 1e-2 at 70m,
# 1.58e-3 at 160m/410m/1b, 6.31e-4 at 1.4b); env var overrides.
_env_tau = os.environ.get("SVA_CONTROLS_TAU")
TAU_STAR = (float(_env_tau) if _env_tau else float(json.load(
    open(BASE / "circuits" / MODEL / "low" / "draw_1" / "metrics.json"))
    ["threshold"]))


def load_scores(band, draw):
    with open(CIRCUITS / band / draw / "prune_scores.pkl", "rb") as f:
        return pickle.load(f)


def filt(examples):
    return [e for e in examples
            if e["token_ids"][1] not in BAD_IDS and e["token_ids"][4] not in BAD_IDS]


def main():
    from lsc_acdc_circuit import load_model, safe_delete_model, cleanup_gpu
    scores = {(b, d): load_scores(b, d) for b in BANDS for d in DRAWS}
    n_edges = {k: sum(int(t.isinf(v).sum()) for v in vv.values())
               for k, vv in scores.items()}

    model = load_model(MODEL, DEVICE)
    bos = model.tokenizer.bos_token_id
    patchable = get_patchable(model, DEVICE)

    # ---------- 1. sensitivity: filtered test sets ----------
    print("== 1. brow/disc sensitivity (filtered test sets) ==", flush=True)
    test_f = {(b, d): filt(load_split(b, d, "test")) for b in BANDS for d in DRAWS}
    circ_acc, univ_acc = {}, {}
    for d in DRAWS:
        univ = build_universal({b: scores[(b, d)] for b in BANDS})
        nu = sum(int(t.isinf(v).sum()) for v in univ.values())
        univ_dev = {k: v.to(DEVICE) for k, v in univ.items()}
        for tb in BANDS:
            L, T, W = run_circuit(patchable, univ_dev, nu, test_f[(tb, d)], bos, DEVICE)
            univ_acc[(d, tb)] = metrics_from_logits(L, T, W)["acc_2way"]
        del univ_dev
        for sb in BANDS:
            dev = {k: v.to(DEVICE) for k, v in scores[(sb, d)].items()}
            for tb in BANDS:
                L, T, W = run_circuit(patchable, dev, n_edges[(sb, d)],
                                      test_f[(tb, d)], bos, DEVICE)
                circ_acc[(d, sb, tb)] = metrics_from_logits(L, T, W)["acc_2way"]
            del dev
        cleanup_gpu()
        print(f"  {d} done", flush=True)
    same = [circ_acc[(d, b, b)] - univ_acc[(d, b)] for d in DRAWS for b in BANDS]
    cross = [np.mean([circ_acc[(d, s, b)] - univ_acc[(d, b)]
                      for s in BANDS if s != b]) for d in DRAWS for b in BANDS]
    sa, ca = np.array(same), np.array(cross)
    te = ca.mean() / sa.mean()
    pooled = np.sqrt((sa.var() + ca.var()) / 2)
    dd = (sa.mean() - ca.mean()) / pooled if pooled > 0 else 0.0
    # Exact test on Fraction diffs: filtered test-set sizes vary per
    # (band, draw), so cell diffs live on different denominators; the
    # universal-core term cancels in same-minus-cross, and
    # d_cell = (5*own_cnt - colsum_cnt) / (4*n_cell) is exact.
    frac_diffs = []
    for d in DRAWS:
        for b in BANDS:
            n_cell = len(test_f[(b, d)])
            cnts = {s: round(circ_acc[(d, s, b)] * n_cell) for s in BANDS}
            assert all(abs(circ_acc[(d, s, b)] * n_cell - cnts[s]) < 1e-3
                       for s in BANDS)
            frac_diffs.append(Fraction(5 * cnts[b] - sum(cnts.values()),
                                       4 * n_cell))
    p = exact_wilcoxon_greater(frac_diffs)
    filt_summary = {"same_boost": sa.mean(), "cross_boost": ca.mean(),
                    "te": te, "d": dd, "p": p}
    print(f"  FILTERED: same={sa.mean():.3f} cross={ca.mean():.3f} "
          f"TE={te:.3f} d={dd:.3f} p={p:.4f}", flush=True)
    pd.DataFrame([{"draw": d, "source_band": s, "test_band": b,
                   "acc_2way": circ_acc[(d, s, b)]}
                  for d in DRAWS for s in BANDS for b in BANDS]).to_csv(
        RESULTS / f"sva_transfer_filtered_{MODEL}.csv", index=False)

    # ---------- 2. random size-matched circuits ----------
    print(f"== 2. random {n_edges[('control', 'draw_1')]}-edge circuits "
          "on control/draw_1 ==", flush=True)
    template = scores[("control", "draw_1")]
    shapes = [(k, v.shape, v.numel()) for k, v in template.items()]
    total = sum(n for _, _, n in shapes)
    test_ctrl = load_split("control", "draw_1", "test")
    rand_accs = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        n_rand = n_edges[("control", "draw_1")]
        pick = set(rng.choice(total, size=n_rand, replace=False).tolist())
        rand_scores, off = {}, 0
        for k, shape, n in shapes:
            v = t.zeros(shape)
            sel = [i - off for i in pick if off <= i < off + n]
            if sel:
                v.view(-1)[t.tensor(sel)] = float("inf")
            rand_scores[k] = v
            off += n
        dev = {k: v.to(DEVICE) for k, v in rand_scores.items()}
        L, T, W = run_circuit(patchable, dev, n_rand, test_ctrl, bos, DEVICE)
        rand_accs.append(metrics_from_logits(L, T, W)["acc_2way"])
        del dev
    cleanup_gpu()
    print(f"  random circuits 2way: mean={np.mean(rand_accs):.3f} "
          f"min={min(rand_accs):.3f} max={max(rand_accs):.3f}", flush=True)

    # ---------- 3a. LSC circuits on SVA ----------
    print(f"== 3a. LSC {MODEL} band circuits (draw_1) on SVA control test ==",
          flush=True)
    lsc_on_sva = {}
    for band in BANDS:
        with open(LSC_CIRCUITS / band / "draw_1" / "prune_scores.pkl", "rb") as f:
            lsc_scores = pickle.load(f)
        ne = sum(int(t.isinf(v).sum()) for v in lsc_scores.values())
        dev = {k: v.to(DEVICE) for k, v in lsc_scores.items()}
        L, T, W = run_circuit(patchable, dev, ne, test_ctrl, bos, DEVICE)
        m = metrics_from_logits(L, T, W)
        lsc_on_sva[band] = (ne, m["acc_2way"])
        print(f"  LSC {band} ({ne} edges) -> SVA control 2way={m['acc_2way']:.3f}",
              flush=True)
        del dev
    cleanup_gpu()

    # ---------- 4. determinism: re-run ACDC low/draw_1 ----------
    print("== 4. ACDC determinism re-run (low/draw_1) ==", flush=True)
    train = load_split("low", "draw_1", "train")
    re_scores, re_n, dt = run_acdc(patchable, train, bos, TAU_STAR, DEVICE)
    saved = scores[("low", "draw_1")]
    same_mask = all(t.equal(t.isinf(re_scores[k]).cpu(), t.isinf(saved[k]).cpu())
                    for k in saved)
    bitexact = all(t.equal(re_scores[k].cpu(), saved[k].cpu()) for k in saved)
    print(f"  edges {re_n} vs saved {n_edges[('low', 'draw_1')]}; "
          f"mask identical={same_mask}; tensors bit-exact={bitexact} [{dt:.0f}s]",
          flush=True)

    del patchable
    safe_delete_model(model)
    cleanup_gpu()

    # ---------- 3b. SVA circuit on LSC (canonical eval machinery) ----------
    print("== 3b. SVA control/draw_1 circuit on LSC control test ==", flush=True)
    import mean_ablation_cross_band as mab
    from auto_circuit.types import AblationType
    from auto_circuit.utils.graph_utils import patchable_model
    from lsc_acdc_circuit import (compute_accuracy_metrics, get_batch_size,
                                  load_dataset, load_pool, set_all_seeds)
    model = load_model(MODEL, DEVICE)
    bos = model.tokenizer.bos_token_id
    patchable = patchable_model(model=model, factorized=True,
                                slice_output="last_seq", seq_len=None,
                                separate_qkv=False, device=DEVICE)
    pool = load_pool("control", mab.POOL_DIR)
    lsc_test = load_dataset("control", "test", mab.DATA_DIR, mab.VARIANT, "draw_1")
    bsz = get_batch_size(MODEL)
    sva_dev = {k: v.to(DEVICE) for k, v in scores[("control", "draw_1")].items()}
    set_all_seeds(123)
    logits, ans = mab.run_circuit_ablated(
        patchable, sva_dev, n_edges[("control", "draw_1")], lsc_test, pool,
        bos, bsz, 123, DEVICE, AblationType.RESAMPLE)
    m_sva_on_lsc = compute_accuracy_metrics(logits, ans)
    # reference: the LSC control circuit itself on the same data
    with open(LSC_CIRCUITS / "control" / "draw_1" / "prune_scores.pkl", "rb") as f:
        lsc_ctrl = pickle.load(f)
    ne_l = sum(int(t.isinf(v).sum()) for v in lsc_ctrl.values())
    lsc_dev = {k: v.to(DEVICE) for k, v in lsc_ctrl.items()}
    set_all_seeds(123)
    logits, ans = mab.run_circuit_ablated(
        patchable, lsc_dev, ne_l, lsc_test, pool, bos, bsz, 123, DEVICE,
        AblationType.RESAMPLE)
    m_lsc_on_lsc = compute_accuracy_metrics(logits, ans)
    print(f"  SVA circuit ({n_edges[('control','draw_1')]} e) on LSC: "
          f"acc={m_sva_on_lsc['accuracy']:.3f} | LSC circuit ({ne_l} e) on LSC: "
          f"acc={m_lsc_on_lsc['accuracy']:.3f}", flush=True)
    del sva_dev, lsc_dev, patchable
    safe_delete_model(model)
    cleanup_gpu()

    # ---------- 5. cell-16 cross-check on ORIGINAL (unfiltered) numbers ----------
    print("== 5. aggregate() cross-check (NB12-validated implementation) ==",
          flush=True)
    tmp = BASE / "results" / "cell16_xcheck"
    tmp.mkdir(exist_ok=True)
    for f in tmp.glob("*.csv"):
        f.unlink()
    rows = []
    acc_unfilt = {}
    for d in DRAWS:
        for sb in BANDS:
            mj = json.load(open(CIRCUITS / sb / d / "metrics.json"))
            for tb in BANDS:
                acc_unfilt[(d, sb, tb)] = mj["transfer"][tb]["acc_2way"]
                rows.append({"model": MODEL, "draw": d, "source_band": sb,
                             "test_band": tb, "circuit_type": "cross_band",
                             "ablation": "resample", "n_edges": mj["n_edges"],
                             "accuracy": mj["transfer"][tb]["acc_2way"],
                             "top5_accuracy": np.nan})
    pd.DataFrame(rows).to_csv(
        tmp / f"edge_cross_band_{MODEL.replace('-', '_')}.csv", index=False)
    u = pd.read_csv(RESULTS / f"sva_universal_{MODEL}.csv")
    u = u.rename(columns={"universal_2way": "accuracy"})
    u["model"] = MODEL
    u["circuit_type"] = "universal_edge"
    u["ablation"] = "resample"
    u["top5_accuracy"] = np.nan
    u.to_csv(tmp / f"edge_universal_{MODEL.replace('-', '_')}.csv", index=False)
    mab.RESAMPLE_TABLE = Path("/nonexistent")  # keep the check pure-SVA
    mab.ZERO_METHOD_TABLE = Path("/nonexistent")
    mab.aggregate(tmp, BANDS, DRAWS, "resample")
    ref = pd.read_csv(tmp / "mean_ablation_summary.csv").iloc[0]
    mine = pd.read_csv(RESULTS / f"sva_phantom_summary_{MODEL}.csv").iloc[0]
    for a, b in [("transfer_efficiency", "transfer_efficiency"),
                 ("cohens_d", "cohens_d"),
                 ("same_band_boost_mean", "same_band_boost_mean"),
                 ("cross_band_boost_mean", "cross_band_boost_mean")]:
        match = np.isclose(ref[a], mine[b], atol=1e-12)
        print(f"  {b}: mine={mine[b]:.6f} validated-impl={ref[a]:.6f} "
              f"{'MATCH' if match else 'MISMATCH'}", flush=True)
    # p-value compared informationally, not as an exact-match check: the
    # NB12 reference path uses scipy's wilcoxon (method resolution does
    # not midrank ties); the canonical value is the exact sign-flip
    # midrank enumeration (sva_stats_util).
    int_diffs = []
    for d in DRAWS:
        for b in BANDS:
            cnts = {s: round(acc_unfilt[(d, s, b)] * 225) for s in BANDS}
            int_diffs.append(5 * cnts[b] - sum(cnts.values()))
    p_exact = exact_wilcoxon_greater(int_diffs)
    print(f"  wilcoxon_p: canonical-exact={p_exact:.6f} "
          f"(saved-summary={mine['wilcoxon_p']:.6f}, "
          f"scipy-reference={ref['p_value']:.6f}; scipy paths do not "
          "midrank ties -- informational only)", flush=True)

    # ---------- save everything ----------
    json.dump({
        "filtered_phantom": {k: float(v) for k, v in filt_summary.items()},
        "random_circuit_2way": rand_accs,
        "lsc_circuits_on_sva_control": {b: {"n_edges": v[0], "acc_2way": v[1]}
                                        for b, v in lsc_on_sva.items()},
        "sva_circuit_on_lsc_control": m_sva_on_lsc,
        "lsc_circuit_on_lsc_control_ref": m_lsc_on_lsc,
        "acdc_determinism": {"mask_identical": bool(same_mask),
                             "bit_exact": bool(bitexact)},
        "bad_ids": sorted(BAD_IDS),
    }, open(RESULTS / f"sva_controls_{MODEL}.json", "w"), indent=1)
    print(f"\nsaved results/sva_controls_{MODEL}.json", flush=True)


if __name__ == "__main__":
    main()
