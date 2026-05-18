#!/usr/bin/env python3
"""Power analysis for Jaccard gap detection.

Computes:
1. Cohen's d for the within-vs-between Jaccard gap per model
2. Required number of Jaccard pairs to detect the gap at 80% power (alpha=0.05)
3. Maps pair counts back to number of ACDC draws (given 5 bands)
4. CLES: probability that a random within-band pair > random between-band pair
5. Outputs CSV and LaTeX table
"""

import csv
import math
from pathlib import Path
from scipy import stats
import numpy as np

BASE = Path(__file__).resolve().parent
INPUT_CSV = BASE / "outputs" / "analysis" / "jaccard_summary.csv"
OUTPUT_CSV = BASE / "outputs" / "analysis" / "jaccard_power_analysis.csv"
OUTPUT_TEX = BASE / "outputs" / "analysis" / "jaccard_power_table.tex"

ALPHA = 0.05
POWER = 0.80
N_BANDS = 5  # number of experimental conditions


def pairs_from_draws(n_draws: int, n_bands: int = N_BANDS) -> tuple[int, int]:
    """Compute number of within-band and between-band Jaccard pairs
    from a given number of draws per condition.

    Within-band: for each band, C(d,2) pairs; total = n_bands * C(d,2)
    Between-band: for each pair of bands, d^2 pairs; total = C(n_bands,2) * d^2
    """
    within = n_bands * (n_draws * (n_draws - 1) // 2)
    between = (n_bands * (n_bands - 1) // 2) * n_draws * n_draws
    return within, between


def required_n_welch(
    d: float, alpha: float = ALPHA, power: float = POWER, ratio: float = 3.0
) -> tuple[int, int]:
    """Required sample sizes for a two-sample Welch t-test.

    ratio = n2/n1 (we use ratio=3 since n_between ~ 3 * n_within with 5 bands).
    Returns (n1, n2) where n1 is the smaller group.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    # For equal variance assumption with unequal n:
    # n1 = (1 + 1/ratio) * ((z_alpha + z_beta) / d)^2
    n1 = math.ceil((1 + 1 / ratio) * ((z_alpha + z_beta) / d) ** 2)
    n2 = math.ceil(ratio * n1)
    return n1, n2


def min_draws_for_power(
    d: float, alpha: float = ALPHA, power: float = POWER, n_bands: int = N_BANDS
) -> int:
    """Find minimum number of draws per condition to achieve required power.

    Searches draws from 2 upward until the number of within-band pairs
    meets the required n1 from the Welch t-test.
    """
    n1_required, _ = required_n_welch(d, alpha, power, ratio=3.0)
    for n_draws in range(2, 100):
        n_within, n_between = pairs_from_draws(n_draws, n_bands)
        if n_within >= n1_required:
            return n_draws
    return 100  # fallback


def cles(mean1: float, sd1: float, mean2: float, sd2: float) -> float:
    """Common Language Effect Size: P(X1 > X2) where X1 ~ N(mean1, sd1^2)
    and X2 ~ N(mean2, sd2^2).

    This is the probability that a randomly chosen within-band Jaccard
    exceeds a randomly chosen between-band Jaccard.
    """
    diff_mean = mean1 - mean2
    diff_sd = math.sqrt(sd1**2 + sd2**2)
    if diff_sd == 0:
        return 0.5
    return stats.norm.cdf(diff_mean / diff_sd)


def main():
    rows = []
    with open(INPUT_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    results = []
    for row in rows:
        model = row["model"]
        w_mean = float(row["within_mean"])
        w_sd = float(row["within_std"])
        b_mean = float(row["between_mean"])
        b_sd = float(row["between_std"])
        n_within = int(row["n_within"])
        n_between = int(row["n_between"])

        gap = w_mean - b_mean
        pooled_sd = math.sqrt((w_sd**2 + b_sd**2) / 2)
        cohens_d = gap / pooled_sd if pooled_sd > 0 else 0.0

        # Required pairs for 80% power
        n1_req, n2_req = required_n_welch(cohens_d)

        # Minimum draws
        min_draws = min_draws_for_power(cohens_d)

        # Pairs available at our 3 draws
        n_w_3, n_b_3 = pairs_from_draws(3)

        # Pairs at the minimum draws
        n_w_min, n_b_min = pairs_from_draws(min_draws)

        # CLES
        p_within_gt = cles(w_mean, w_sd, b_mean, b_sd)

        # Achieved power at 3 draws (post-hoc)
        # Using Welch t-test: t = gap / sqrt(sd1^2/n1 + sd2^2/n2)
        se = math.sqrt(w_sd**2 / n_within + b_sd**2 / n_between)
        t_obs = gap / se if se > 0 else 0
        # Welch df
        num = (w_sd**2 / n_within + b_sd**2 / n_between) ** 2
        den = (w_sd**2 / n_within) ** 2 / (n_within - 1) + (
            b_sd**2 / n_between
        ) ** 2 / (n_between - 1)
        df = num / den if den > 0 else 1
        t_crit = stats.t.ppf(1 - ALPHA / 2, df)
        # Non-centrality parameter
        ncp = gap / se
        # Power = P(T > t_crit | ncp) + P(T < -t_crit | ncp)
        achieved_power = (
            1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
        )

        results.append(
            {
                "model": model,
                "gap": round(gap, 4),
                "pooled_sd": round(pooled_sd, 4),
                "cohens_d": round(cohens_d, 3),
                "cles": round(p_within_gt, 3),
                "n1_required": n1_req,
                "n2_required": n2_req,
                "min_draws": min_draws,
                "pairs_at_min_draws_within": n_w_min,
                "pairs_at_min_draws_between": n_b_min,
                "pairs_at_3_draws_within": n_w_3,
                "pairs_at_3_draws_between": n_b_3,
                "achieved_power_3_draws": round(achieved_power, 3),
            }
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved: {OUTPUT_CSV}")

    print(
        f"\n{'Model':<15} {'Gap':>6} {'Pool SD':>8} {'d':>5} {'CLES':>5} "
        f"{'n1_req':>7} {'Min draws':>10} {'Power@3':>8}"
    )
    print("-" * 75)
    for r in results:
        print(
            f"{r['model']:<15} {r['gap']:>6.4f} {r['pooled_sd']:>8.4f} "
            f"{r['cohens_d']:>5.2f} {r['cles']:>5.3f} "
            f"{r['n1_required']:>7d} {r['min_draws']:>10d} "
            f"{r['achieved_power_3_draws']:>8.3f}"
        )

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Power analysis for detecting the within-band vs.\ between-band Jaccard gap.",
        r"Cohen's $d$ and CLES (probability that a random within-band pair exceeds a",
        r"random between-band pair) characterise the effect size.",
        r"$n_{\min}$ is the minimum number of ACDC draws per condition (with 5~bands)",
        r"needed to achieve 80\% power at $\alpha{=}0.05$.",
        r"Achieved power reports the post-hoc power of our 3-draw design.}",
        r"\label{tab:power_analysis}",
        r"\small",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Gap & Pooled SD & $d$ & CLES & $n_{\min}$ draws & Power (3 draws) \\",
        r"\midrule",
    ]
    for r in results:
        tex_lines.append(
            f"Pythia-{r['model'].replace('pythia-', '')} & "
            f"{r['gap']:.3f} & {r['pooled_sd']:.3f} & "
            f"{r['cohens_d']:.2f} & {r['cles']:.2f} & "
            f"{r['min_draws']} & {r['achieved_power_3_draws']:.2f} \\\\"
        )
    tex_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    with open(OUTPUT_TEX, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"\nSaved: {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
