"""SVA checks after data generation (run on GPU).

1. Prompt-NLL screen (pythia-1b): per (draw, band, split) keep the lowest-
   prompt-NLL examples, trimming surplus 1875-split sizes to LSC-exact
   1050/225/225. Within-band percentile trimming -> no differential filtering
   across bands. Writes data_final/.
2. Plausibility sample: decodes 12 random kept + 6 dropped sentences per band
   for human inspection (results/plausibility_samples.txt).
3. Confound profile of final data: subject length, sg/pl balance, prompt NLL
   distribution per band (results/confound_profile.csv).
4. Base-model gate: all 5 Pythia models on final val+test per band x draw:
   2-way forced-choice accuracy (is vs are), target-top1, target-in-top5.
   Gate: 2-way >= 0.90 in every band, spread <= 5 pts (results/base_gate.csv).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t

BASE = Path(__file__).resolve().parent.parent
GEN, FINAL, RESULTS = BASE / "data_generated", BASE / "data_final", BASE / "results"
FINAL.mkdir(exist_ok=True)
DEVICE = "cuda:0"
BANDS = ["low", "medium", "high", "very_high", "control"]
DRAWS = ["draw_1", "draw_2", "draw_3"]
TARGET = {"train": 1050, "val": 225, "test": 225}
MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "LSC_circuits"))
from lsc_acdc_circuit import load_model, safe_delete_model, cleanup_gpu  # noqa: E402


def batched_prompt_nll(model, examples, bs=256):
    """Mean NLL of prompt tokens (positions 1..4 given prefix), BOS prepended."""
    bos = model.tokenizer.bos_token_id
    nlls = []
    with t.no_grad():
        for i in range(0, len(examples), bs):
            chunk = examples[i:i + bs]
            ids = t.tensor([[bos] + e["token_ids"] for e in chunk], device=DEVICE)
            logp = t.log_softmax(model(ids), dim=-1)
            tgt = ids[:, 1:]
            token_lp = logp[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            nlls.extend((-token_lp.mean(dim=1)).tolist())
    return nlls


def eval_gate(model, examples, bs=256):
    bos = model.tokenizer.bos_token_id
    n2, ntop1, ntop5 = 0, 0, 0
    with t.no_grad():
        for i in range(0, len(examples), bs):
            chunk = examples[i:i + bs]
            ids = t.tensor([[bos] + e["token_ids"] for e in chunk], device=DEVICE)
            last = model(ids)[:, -1, :]
            tgt = t.tensor([e["target_token_id"] for e in chunk], device=DEVICE)
            wrg = t.tensor([e["wrong_token_id"] for e in chunk], device=DEVICE)
            n2 += int((last.gather(1, tgt[:, None]) > last.gather(1, wrg[:, None])).sum())
            top5 = last.topk(5, dim=-1).indices
            ntop1 += int((top5[:, 0] == tgt).sum())
            ntop5 += int((top5 == tgt[:, None]).any(dim=1).sum())
    n = len(examples)
    return n2 / n, ntop1 / n, ntop5 / n


def main():
    # ---------- 1. prompt-NLL screen with pythia-1b ----------
    model = load_model("pythia-1b", DEVICE)
    tok = model.tokenizer
    sample_lines = []
    for draw in DRAWS:
        for band in BANDS:
            for split in ["train", "val", "test"]:
                f = GEN / draw / band / f"{split}.json"
                d = json.load(open(f))
                ex = d["examples"]
                nll = batched_prompt_nll(model, ex)
                order = np.argsort(nll)
                keep_n = TARGET[split]
                keep = [ex[i] for i in order[:keep_n]]
                for j, e in enumerate(keep):
                    e["prompt_nll"] = float(nll[order[j]])
                    e["example_id"] = j
                d["examples"] = keep
                d["n_examples"] = keep_n
                d["ppl_screen"] = {"model": "pythia-1b", "surplus": len(ex),
                                   "kept": keep_n}
                od = FINAL / draw / band
                od.mkdir(parents=True, exist_ok=True)
                json.dump(d, open(od / f"{split}.json", "w"))
                if draw == "draw_1" and split == "test":
                    dropped = [ex[i] for i in order[keep_n:]]
                    rng = np.random.default_rng(0)
                    sample_lines.append(f"\n===== {band} (draw_1/test) =====")
                    for tag, src, k in [("KEPT", keep, 12), ("DROPPED", dropped, 6)]:
                        for e in rng.choice(src, size=min(k, len(src)), replace=False):
                            txt = tok.decode(e["token_ids"])
                            ans = tok.decode([e["target_token_id"]])
                            sample_lines.append(f"[{tag}] {txt} ->{ans}")
            print(f"{draw}: screened", flush=True)
    (RESULTS / "plausibility_samples.txt").write_text("\n".join(sample_lines))
    safe_delete_model(model)
    cleanup_gpu()

    # ---------- 2. confound profile ----------
    rows = []
    for band in BANDS:
        ex = json.load(open(FINAL / "draw_1" / band / "test.json"))["examples"]
        subj_len = [len(e["subject"]) for e in ex]
        rows.append({
            "band": band,
            "n": len(ex),
            "sg_fraction": np.mean([e["number"] == "sg" for e in ex]),
            "subj_len_mean": np.mean(subj_len), "subj_len_std": np.std(subj_len),
            "prompt_nll_median": np.median([e["prompt_nll"] for e in ex]),
            "unique_subjects": len({e["subject"] for e in ex}),
        })
    prof = pd.DataFrame(rows)
    prof.to_csv(RESULTS / "confound_profile.csv", index=False)
    print("\n=== confound profile (draw_1/test) ===")
    print(prof.round(3).to_string(index=False))

    # ---------- 3. base-model gate on final data ----------
    gate_rows = []
    for mname in MODELS:
        model = load_model(mname, DEVICE)
        for band in BANDS:
            accs2, acc1, acc5 = [], [], []
            for draw in DRAWS:
                ex = (json.load(open(FINAL / draw / band / "val.json"))["examples"]
                      + json.load(open(FINAL / draw / band / "test.json"))["examples"])
                a2, a1, a5 = eval_gate(model, ex)
                accs2.append(a2); acc1.append(a1); acc5.append(a5)
            gate_rows.append({"model": mname, "band": band,
                              "acc_2way": np.mean(accs2),
                              "acc_top1": np.mean(acc1),
                              "acc_top5": np.mean(acc5)})
        sub = pd.DataFrame([r for r in gate_rows if r["model"] == mname])
        print(f"\n{mname}: 2way " +
              " ".join(f"{r.band}={r.acc_2way:.3f}" for r in sub.itertuples()))
        safe_delete_model(model)
        cleanup_gpu()

    gate = pd.DataFrame(gate_rows)
    gate.to_csv(RESULTS / "base_gate.csv", index=False)
    print("\n=== GATE (2-way >=0.90 all bands, spread <=0.05) ===")
    for mname in MODELS:
        g = gate[gate.model == mname]
        ok = (g.acc_2way.min() >= 0.90) and (g.acc_2way.max() - g.acc_2way.min() <= 0.05)
        print(f"{mname:12s} min={g.acc_2way.min():.3f} "
              f"spread={g.acc_2way.max()-g.acc_2way.min():.3f} -> "
              f"{'PASS' if ok else 'REVIEW'}")


if __name__ == "__main__":
    main()
