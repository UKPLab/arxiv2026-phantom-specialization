"""Do band-specific edges sit near the selection boundary?

ACDC's per-edge patching scores are not recoverable from the saved
artifacts (prune_scores.pkl is binary by AutoCircuit design: retained=inf,
pruned=threshold sentinel), so we use the saved EAP-IG attribution
scores (EAP_methods/eap_ig_scores/, graded over the full graph for all 75
conditions) as an independent per-edge importance measure, matching the
pipeline's own |score| top-k convention (lsc_eap_eval.extract_top_k_edges).

Test: classify each ACDC-selected edge by per-draw sharing count kappa
(1 = band-specific, 5 = universal). For each (model, band, draw), rank all
graph edges by |EAP-IG score| descending; the ACDC-size-matched boundary is
rank k = |ACDC circuit|. For each selected edge report
  rho  = rank / k          (1 = at boundary, <<1 = far above, >1 = below)
  ratio = |score| / cutoff (cutoff = k-th largest |score|)
Hypothesis (paper Sec 6.1): kappa=1 edges cluster near the boundary
(rho ~ 1), kappa=5 edges sit well above (rho << 1).

Stats: per-edge Mann-Whitney U (descriptive; edges not independent) plus a
conservative per-condition Wilcoxon on median-rho differences (n = 15
band x draw conditions per model).

Outputs under supplementary_experiments/results/edge_marginality/:
  per_edge_rows.csv.gz, summary_by_model.csv, fig_rho_distributions.{png,pdf}
The grouped-bar paper figure is produced by edge_marginality_figure.py.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t
from scipy import stats

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ISC_ROOT = Path(__file__).resolve().parents[2]
CIRCUITS_DIR = ISC_ROOT / "LSC_circuits/circuit_discovery/circuits"
EAP_IG_DIR = ISC_ROOT / "LSC_circuits/EAP_methods/eap_ig_scores"
OUT_DIR = ISC_ROOT / "supplementary_experiments/results/edge_marginality"

MODELS = ["pythia_70m", "pythia_160m", "pythia_410m", "pythia_1b", "pythia_1.4b"]
BANDS = ["low", "medium", "high", "very_high", "control"]
DRAWS = ["draw_1", "draw_2", "draw_3"]


def load_masks(model, draw):
    """Per-band boolean kept-masks from ACDC prune_scores (retained == inf)."""
    masks = {}
    for band in BANDS:
        with open(CIRCUITS_DIR / model / band / draw / "prune_scores.pkl", "rb") as f:
            scores = pickle.load(f)
        masks[band] = {m: t.isinf(v) for m, v in scores.items()}
    return masks


def flat_concat(dict_of_tensors, modules):
    return t.cat([dict_of_tensors[m].flatten() for m in modules])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for model in MODELS:
        for draw in DRAWS:
            masks = load_masks(model, draw)
            modules = sorted(masks[BANDS[0]].keys())
            band_flat = {b: flat_concat(masks[b], modules) for b in BANDS}
            kappa = sum(band_flat[b].long() for b in BANDS)  # 0..5 per edge

            for band in BANDS:
                with open(EAP_IG_DIR / model / band / draw / "scores.pkl", "rb") as f:
                    eap = pickle.load(f)
                s = t.cat([eap[m].flatten().float() for m in modules]).abs()
                n_total = s.numel()
                sel = band_flat[band]
                k = int(sel.sum().item())
                if k == 0:
                    continue
                # rank of every edge by |score| descending (1 = largest)
                order = t.argsort(s, descending=True)
                rank = t.empty(n_total, dtype=t.long)
                rank[order] = t.arange(1, n_total + 1)
                cutoff = s[order[k - 1]].item()  # k-th largest |score|

                sel_idx = t.nonzero(sel, as_tuple=True)[0]
                for i in sel_idx.tolist():
                    rows.append({
                        "model": model, "band": band, "draw": draw,
                        "kappa": int(kappa[i]), "circuit_size": k,
                        "rho": rank[i].item() / k,
                        "ratio": (s[i].item() / cutoff) if cutoff > 0 else np.nan,
                    })
        print(f"{model}: done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_edge_rows.csv.gz", index=False, compression="gzip")

    # --- Summaries + stats per model, kappa=1 vs kappa=5 ---
    summary = []
    for model in MODELS:
        dm = df[df.model == model]
        k1, k5 = dm[dm.kappa == 1], dm[dm.kappa == 5]
        # descriptive per-edge Mann-Whitney (one-sided: k1 rho > k5 rho)
        mw_p = stats.mannwhitneyu(k1.rho, k5.rho, alternative="greater")[1] \
            if len(k1) and len(k5) else np.nan
        u = stats.mannwhitneyu(k1.rho, k5.rho, alternative="greater")[0] \
            if len(k1) and len(k5) else np.nan
        rank_biserial = 1 - 2 * u / (len(k1) * len(k5)) if len(k1) and len(k5) else np.nan
        # conservative: per-(band,draw) median rho difference, Wilcoxon across 15 conds
        cond_diffs = []
        for (band, draw), g in dm.groupby(["band", "draw"]):
            g1, g5 = g[g.kappa == 1], g[g.kappa == 5]
            if len(g1) and len(g5):
                cond_diffs.append(g1.rho.median() - g5.rho.median())
        w_p = stats.wilcoxon(cond_diffs, alternative="greater")[1] \
            if len(cond_diffs) >= 6 else np.nan
        for name, sub in [("kappa=1 (band-specific)", k1), ("kappa=5 (universal)", k5)]:
            summary.append({
                "model": model, "class": name, "n_edges": len(sub),
                "median_rho": sub.rho.median(), "iqr_rho_lo": sub.rho.quantile(0.25),
                "iqr_rho_hi": sub.rho.quantile(0.75),
                "frac_below_boundary": (sub.rho > 1).mean(),
                "median_ratio": sub.ratio.median(),
                "mw_p_one_sided": mw_p, "rank_biserial": rank_biserial,
                "wilcoxon_p_conditions": w_p, "n_conditions": len(cond_diffs),
            })
        print(f"{model}: k1 median rho={k1.rho.median():.2f} (n={len(k1)}), "
              f"k5 median rho={k5.rho.median():.2f} (n={len(k5)}), "
              f"MW p={mw_p:.2e}, cond-Wilcoxon p={w_p:.4f}", flush=True)

    pd.DataFrame(summary).to_csv(OUT_DIR / "summary_by_model.csv", index=False)

    # --- Figure: log10(rho) distributions, kappa=1 vs kappa=5, per model ---
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4 * len(MODELS), 4), sharey=True)
    for ax, model in zip(axes, MODELS):
        dm = df[df.model == model]
        data = [np.log10(dm[dm.kappa == 5].rho.clip(lower=1e-6)),
                np.log10(dm[dm.kappa == 1].rho.clip(lower=1e-6))]
        parts = ax.violinplot(data, positions=[0, 1], showmedians=True, widths=0.8)
        for pc, col in zip(parts["bodies"], ["#4878d0", "#d65f5f"]):
            pc.set_facecolor(col)
            pc.set_alpha(0.6)
        ax.axhline(0, color="gray", ls="--", lw=1, label="selection boundary")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["universal\n($\\kappa$=5)", "band-specific\n($\\kappa$=1)"])
        ax.set_title(model.replace("pythia_", "Pythia-"))
        ax.grid(False)
    axes[0].set_ylabel(r"$\log_{10}(\rho)$  [rank / ACDC-matched cutoff]")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("EAP-IG importance rank of ACDC-selected edges relative to the "
                 "size-matched selection boundary", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"fig_rho_distributions.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved outputs to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
