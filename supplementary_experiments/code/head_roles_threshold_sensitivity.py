"""C4 defensibility check: are head-role labels and the role-enrichment
result robust to the classification thresholds?

Reclassifies all 1,088 heads (5 models) from the saved per-head attention
statistics at scaled versions of HEAD_ROLE_THRESHOLDS (all three role
thresholds multiplied by m in {0.5, 0.75, 1.0, 1.5, 2.0}; dominance ratio
and entropy cutoff unchanged), then re-runs the Fisher exact test for
previous_token enrichment and bos_sink depletion among universal heads.

Inputs (saved artifacts, no model needed):
  LSC_circuit_analysis/05_Phase_Targeted/outputs/
    analysis/head_role_universality.csv (per-head stats + is_universal)

Outputs: results/head_roles_threshold_sensitivity/sensitivity.csv + stdout table.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ISC = Path(__file__).resolve().parents[2]
SRC = (ISC / "LSC_circuit_analysis/05_Phase_Targeted"
       / "outputs/analysis/head_role_universality.csv")
OUT = ISC / "supplementary_experiments/results/head_roles_threshold_sensitivity"

BASE = {"induction_min": 0.15, "bos_sink_min": 0.3, "prev_token_min": 0.2,
        "entropy_diffuse_min": 3.0, "dominance_ratio": 1.5}
MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]


def classify(row, thr):
    """Mirror of utils/attention.py classify_head_role."""
    scores = {
        "induction": row.induction_score / max(thr["induction_min"], 1e-10),
        "bos_sink": row.bos_fraction / max(thr["bos_sink_min"], 1e-10),
        "previous_token": row.prev_fraction / max(thr["prev_token_min"], 1e-10),
    }
    passes = {
        "induction": row.induction_score >= thr["induction_min"],
        "bos_sink": row.bos_fraction >= thr["bos_sink_min"],
        "previous_token": row.prev_fraction >= thr["prev_token_min"],
    }
    (top, ts), (_, rs) = sorted(scores.items(), key=lambda kv: -kv[1])[:2]
    if passes[top] and (rs == 0 or ts / max(rs, 1e-10) >= thr["dominance_ratio"]):
        return top
    if row.entropy >= thr["entropy_diffuse_min"]:
        return "diffuse"
    if passes[top]:
        return top
    return "diffuse"


def fisher(sub, role):
    a = ((sub.role_new == role) & sub.is_universal).sum()
    b = ((sub.role_new == role) & ~sub.is_universal).sum()
    c = ((sub.role_new != role) & sub.is_universal).sum()
    d = ((sub.role_new != role) & ~sub.is_universal).sum()
    return stats.fisher_exact([[a, b], [c, d]])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SRC)
    # sanity: reproduce the published labels at m = 1.0
    df["role_new"] = df.apply(lambda r: classify(r, BASE), axis=1)
    agree = (df.role_new == df.role).mean()
    print(f"label reproduction at baseline thresholds: {agree:.1%}")

    rows = []
    for m in [0.5, 0.75, 1.0, 1.5, 2.0]:
        thr = dict(BASE)
        for k in ("induction_min", "bos_sink_min", "prev_token_min"):
            thr[k] = BASE[k] * m
        df["role_new"] = df.apply(lambda r: classify(r, thr), axis=1)
        overall_agree = (df.role_new == df.role).mean()
        for model in MODELS:
            sub = df[df.model == model]
            or_pt, p_pt = fisher(sub, "previous_token")
            or_bs, p_bs = fisher(sub, "bos_sink")
            rows.append({
                "multiplier": m, "model": model,
                "label_agreement": (sub.role_new == sub.role).mean(),
                "prev_token_OR": or_pt, "prev_token_p": p_pt,
                "bos_sink_OR": or_bs, "bos_sink_p": p_bs,
            })
        print(f"m={m}: overall label agreement {overall_agree:.1%}")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "sensitivity.csv", index=False)
    pt_ok = ((res.prev_token_OR > 1) & (res.prev_token_p < 0.05))
    bs_ok = ((res.bos_sink_OR < 1) & (res.bos_sink_p < 0.05))
    print(f"\nprev_token enrichment (OR>1, p<0.05): {pt_ok.sum()}/{len(res)} cells")
    print(f"bos_sink depletion   (OR<1, p<0.05): {bs_ok.sum()}/{len(res)} cells")
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
