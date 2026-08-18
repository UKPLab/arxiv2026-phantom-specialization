"""Gated multi-seed runner + across-seed aggregation.

Per seed: train -> accuracy gate -> double-dissociation certificate (pre-locked
component groups, held-out val). Discovery/sweep/aggregate run ONLY if both
gates pass. Failures are recorded and the seed is excluded from the control.

Across seeds: descriptive statistics only (no per-seed inferential tests;
the per-cell Wilcoxon is not model-level evidence). Writes
results/summary.json.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
CODE = BASE / "code"


def run(script, *args):
    r = subprocess.run([sys.executable, "-u", str(CODE / script), *args],
                       cwd=BASE)
    if r.returncode != 0:
        raise RuntimeError(f"{script} {args} failed rc={r.returncode}")


def stage_seed(seed):
    s = str(seed)
    gate_f = BASE / f"models/gate_s{seed}.json"
    if not gate_f.exists():
        run("tm_train.py", "--seed", s)
    gate = json.load(open(gate_f))
    if not gate["pass"]:
        print(f"seed {seed}: accuracy gate FAIL -> excluded, no discovery",
              flush=True)
        return {"seed": seed, "stage": "gate", "status": "FAIL"}
    cert_f = BASE / f"results/certificate_s{seed}.json"
    if not cert_f.exists():
        run("tm_certificate.py", "--seed", s)
    cert2_f = BASE / f"results/certificate_s{seed}_draw_2.json"
    if not cert2_f.exists():
        run("tm_certificate.py", "--seed", s, "--draw", "draw_2")
    cert = json.load(open(cert_f))
    if cert["verdict"] != "PASS":
        print(f"seed {seed}: certificate FAIL -> excluded, no discovery",
              flush=True)
        return {"seed": seed, "stage": "certificate", "status": "FAIL"}
    if not (BASE / f"sweep/sweep_toy_s{seed}.csv").exists():
        run("tm_discovery.py", "--mode", "sweep", "--seed", s)
    run("tm_discovery.py", "--mode", "discover", "--seed", s)
    verdict_f = BASE / f"results/pipeline_verdict_s{seed}.json"
    if not verdict_f.exists():
        run("tm_discovery.py", "--mode", "aggregate", "--seed", s)
    return {"seed": seed, "stage": "complete", "status": "PASS"}


def summarize(seeds):
    rows, qualified = [], []
    for seed in seeds:
        gate_f = BASE / f"models/gate_s{seed}.json"
        cert_f = BASE / f"results/certificate_s{seed}.json"
        cert2_f = BASE / f"results/certificate_s{seed}_draw_2.json"
        v_f = BASE / f"results/pipeline_verdict_s{seed}.json"
        gate = json.load(open(gate_f)) if gate_f.exists() else None
        cert = json.load(open(cert_f)) if cert_f.exists() else None
        cert2 = json.load(open(cert2_f)) if cert2_f.exists() else None
        v = json.load(open(v_f)) if v_f.exists() else None
        gate_ok = bool(gate and gate["pass"])
        cert_ok = bool(cert and cert["verdict"] == "PASS")
        ok = gate_ok and cert_ok
        det = None
        if ok and v:
            det = {
                "te_AB_lt_0.5": v["universal_te"]["univ_AB"]["te"] < 0.5,
                "te_ABmix_lt_0.5": v["universal_te"]["univ_ABmix"]["te"] < 0.5,
                "within_cross_draw_ge_0.8":
                    v["within_cond_cross_draw_min"] >= 0.8,
                "jaccard_gap_ge_0.15":
                    v["jaccard_within"] - v["jaccard_cross"] >= 0.15,
            }
            det["detection_pass"] = all(det.values())
        rows.append({"seed": seed, "gate_pass": gate_ok,
                     "route_separation_pass": cert_ok,
                     "route_separation_pass_disjoint_draw2":
                         (cert2["verdict"] == "PASS") if cert2 else None,
                     "route_qualified": ok,
                     "detection": det})
        if ok and v:
            qualified.append(v)

    def stat(key, sub=None):
        vals = [(c[key][sub]["te"] if sub else c[key]) for c in qualified]
        return {"mean": float(np.mean(vals)),
                "sample_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": [float(x) for x in vals]}

    n_gate = sum(r["gate_pass"] for r in rows)
    n_qual = len(qualified)
    n_det = sum(1 for r in rows if r["detection"] and
                r["detection"]["detection_pass"])
    summary = {
        "seeds_attempted": list(seeds),
        "per_seed": rows,
        "decomposition": {
            "accuracy_gate": f"{n_gate}/{len(seeds)}",
            "route_separation_among_task_solvers": f"{n_qual}/{n_gate}",
            "route_qualified_overall": f"{n_qual}/{len(seeds)}",
            "detection_among_route_qualified": f"{n_det}/{n_qual}",
        },
        "note": ("Descriptive statistics over ROUTE-QUALIFIED models only; "
                 "per-seed Wilcoxon values are within-model and not used "
                 "inferentially. Certificate = pre-locked component groups "
                 "(all attn L1-3 vs all MLP L1-3), held-out val; "
                 "component-type evidence of learned route separation."),
    }
    if qualified:
        summary.update({
            "te_univ_AB": stat("universal_te", "univ_AB"),
            "te_univ_ABmix": stat("universal_te", "univ_ABmix"),
            "own_acc": stat("own_acc"),
            "within_cond_cross_draw_acc": stat("within_cond_cross_draw_acc"),
            "cross_cond_acc": stat("cross_cond_acc"),
            "jaccard_within": stat("jaccard_within"),
            "jaccard_cross": stat("jaccard_cross"),
            "jaccard_size_only_baseline": stat("jaccard_size_matched_null"),
            "containment_smaller_in_larger": stat("containment_smaller_in_larger"),
        })
    json.dump(summary, open(BASE / "results/summary.json", "w"), indent=1)
    print(json.dumps(summary["decomposition"], indent=1))
    for r in rows:
        d = r["detection"]
        print(f"s{r['seed']}: gate={'P' if r['gate_pass'] else 'F'} "
              f"route={'P' if r['route_separation_pass'] else 'F'} "
              f"disjoint2={r['route_separation_pass_disjoint_draw2']} "
              f"detection={'PASS' if d and d['detection_pass'] else '-'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if not args.summarize_only:
        for seed in args.seeds:
            print(f"===== seed {seed} =====", flush=True)
            stage_seed(seed)
    summarize(args.seeds)


if __name__ == "__main__":
    main()
