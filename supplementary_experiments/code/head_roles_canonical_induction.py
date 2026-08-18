"""C4 correction analysis: reclassify head roles with the canonical
induction position.

The pipeline's induction_score measures attention from the prediction
position to the PREVIOUS OCCURRENCE of the current token (source S5,
INDUCTION_CUE_POS=4): the duplicate-token / prefix-match pattern. The
canonical induction pattern (Olsson et al.) attends to the token
FOLLOWING the previous occurrence, which in LSC is the answer position T;
that quantity was saved as target_fraction but never used in
classification.

This script reruns the published classifier verbatim with a single
substitution (induction statistic := target_fraction) and reports label
agreement, per-model label counts, and the Fisher enrichment of each
role among universal heads. Published tables remain as computed; this is
the correction analysis reported alongside them.

Outputs: results/head_roles_canonical_induction/canonical_induction.csv
"""

from pathlib import Path

import pandas as pd
from scipy import stats

ISC = Path(__file__).resolve().parents[2]
SRC = (ISC / "LSC_circuit_analysis/05_Phase_Targeted"
       / "outputs/analysis/head_role_universality.csv")
OUT = ISC / "supplementary_experiments/results/head_roles_canonical_induction"

THR = {"induction": 0.15, "bos_sink": 0.3, "previous_token": 0.2,
       "entropy_diffuse": 3.0, "dominance": 1.5}
MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]


def classify(row, induction_stat):
    scores = {"induction": induction_stat / THR["induction"],
              "bos_sink": row.bos_fraction / THR["bos_sink"],
              "previous_token": row.prev_fraction / THR["previous_token"]}
    passes = {k: v >= 1 for k, v in scores.items()}
    (top, ts), (_, rs) = sorted(scores.items(), key=lambda kv: -kv[1])[:2]
    if passes[top] and (rs == 0 or ts / max(rs, 1e-10) >= THR["dominance"]):
        return top
    if row.entropy >= THR["entropy_diffuse"]:
        return "diffuse"
    return top if passes[top] else "diffuse"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    df["role_repro"] = df.apply(lambda r: classify(r, r.induction_score), axis=1)
    assert (df.role_repro == df.role).all(), "published labels not reproduced"
    df["role_canon"] = df.apply(lambda r: classify(r, r.target_fraction), axis=1)
    print(f"published labels reproduced exactly; canonical-position agreement "
          f"{(df.role_canon == df.role).mean():.1%}")
    print("canonical label counts:", df.role_canon.value_counts().to_dict())

    rows = []
    for m in MODELS:
        s = df[df.model == m]
        for role in ["previous_token", "bos_sink", "induction"]:
            a = ((s.role_canon == role) & s.is_universal).sum()
            b = ((s.role_canon == role) & ~s.is_universal).sum()
            c = ((s.role_canon != role) & s.is_universal).sum()
            d = ((s.role_canon != role) & ~s.is_universal).sum()
            OR, p = stats.fisher_exact([[a, b], [c, d]])
            rows.append({"model": m, "role": role, "n_role": a + b,
                         "n_universal_role": a, "odds_ratio": OR, "p": p,
                         "label_agreement": (s.role_canon == s.role).mean()})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "canonical_induction.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
