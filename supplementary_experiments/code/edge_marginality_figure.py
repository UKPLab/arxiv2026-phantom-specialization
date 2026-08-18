"""Paper figure: fraction of edges beyond the selection boundary.

Grouped bars per model, band-specific (kappa=1) vs universal (kappa=5),
from summary_by_model.csv (frac_below_boundary = fraction with rho > 1,
i.e. EAP-IG rank beyond the ACDC-size-matched cutoff). Chosen over the
violin/ECDF variants: the appendix table already carries medians/IQRs,
so the figure only needs the two headline fractions per model.
Okabe-Ito colorblind-safe pair.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results/edge_marginality"
MODELS = ["pythia_70m", "pythia_160m", "pythia_410m", "pythia_1b", "pythia_1.4b"]
C_UNIV, C_SPEC = "#0072B2", "#D55E00"


def main():
    s = pd.read_csv(OUT / "summary_by_model.csv")
    names = [m.replace("pythia_", "Pythia-") for m in MODELS]
    frac = lambda m, pat: s[(s.model == m) & s["class"].str.contains(pat)] \
        .frac_below_boundary.iloc[0] * 100
    spec = [frac(m, "band") for m in MODELS]
    univ = [frac(m, "universal") for m in MODELS]

    x = np.arange(len(MODELS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = [ax.bar(x - w / 2, spec, w, color=C_SPEC,
                   label=r"band-specific edges ($\kappa$=1)"),
            ax.bar(x + w / 2, univ, w, color=C_UNIV,
                   label=r"universal edges ($\kappa$=5)")]
    for bs in bars:
        for r in bs:
            ax.annotate(f"{r.get_height():.0f}%",
                        (r.get_x() + r.get_width() / 2, r.get_height()),
                        ha="center", va="bottom", fontsize=9)
    ax.axhline(50, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("edges ranked beyond the selection\n"
                  "boundary by EAP-IG importance (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_rho_bars.{ext}", dpi=150, bbox_inches="tight")
    plt.close()
    print("saved fig_rho_bars.png/pdf")


if __name__ == "__main__":
    main()
