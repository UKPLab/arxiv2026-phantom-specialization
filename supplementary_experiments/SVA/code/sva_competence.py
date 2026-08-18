"""SVA feasibility, stage 2: base-model competence per band.

Question: is Pythia's subject-verb agreement frequency-INVARIANT across the
canonical bands? (The design requirement from paper Sec 4.1 - the gate that
IOI failed via output priors and that SVA should pass because the answers
" is"/" are" are constant function words.)

Metric: 2-way forced choice, correct iff logit of the number-matching verb
exceeds the mismatching one at the final position. Two templates:
  bare:      "The {N}"                     -> is/are
  attractor: "The {N} near the {M}"        -> is/are   (M = opposite number)

Reads ../results/sva_noun_pairs.csv; writes ../results/competence.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as t

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / "results"
MODELS = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1b"]
BANDS = ["low", "medium", "high", "very_high"]
N_PER_BAND = 60          # nouns sampled per band (very_high has 61)
SEED = 42
DEVICE = "cuda:0"

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "LSC_circuits"))


def build_prompts(pairs, rng):
    """Per noun: 2 numbers x 2 templates. Attractor = opposite-number noun
    drawn from the same band (frequency-matched context)."""
    prompts = []
    for _, r in pairs.iterrows():
        others = pairs[pairs.singular != r.singular]
        attr = others.iloc[rng.integers(len(others))]
        for number, subj in [("sg", r.singular), ("pl", r.plural)]:
            attr_form = attr.plural if number == "sg" else attr.singular
            for tmpl, text in [
                ("bare", f"The {subj}"),
                ("attractor", f"The {subj} near the {attr_form}"),
            ]:
                prompts.append({
                    "band": r.sg_band, "noun": r.singular, "number": number,
                    "template": tmpl, "text": text,
                    "correct": " is" if number == "sg" else " are",
                    "wrong": " are" if number == "sg" else " is",
                })
    return prompts


def main():
    pairs = pd.read_csv(RESULTS / "sva_noun_pairs.csv")
    rng = np.random.default_rng(SEED)

    band_prompts = []
    for band in BANDS:
        b = pairs[pairs.sg_band == band]
        take = b.sample(n=min(N_PER_BAND, len(b)), random_state=SEED)
        band_prompts += build_prompts(take, rng)
    df = pd.DataFrame(band_prompts)
    print(f"prompts: {len(df)} ({df.band.value_counts().to_dict()})")

    from lsc_acdc_circuit import load_model  # applies _patch_gpt_neox_config
    rows = []
    for model_name in MODELS:
        model = load_model(model_name, DEVICE)
        tok = model.tokenizer
        # TL sets add_bos_token=True; without add_special_tokens=False the
        # first id is BOS, not the answer token
        ids_is = tok.encode(" is", add_special_tokens=False)
        ids_are = tok.encode(" are", add_special_tokens=False)
        assert len(ids_is) == 1 and len(ids_are) == 1, (ids_is, ids_are)
        id_is, id_are = ids_is[0], ids_are[0]

        correct_flags = []
        with t.no_grad():
            for i in range(0, len(df), 64):
                chunk = df.iloc[i:i + 64]
                toks = [tok.encode(s) for s in chunk.text]
                maxlen = max(len(x) for x in toks)
                batch = t.full((len(toks), maxlen), tok.eos_token_id,
                               dtype=t.long, device=DEVICE)
                lens = []
                for j, x in enumerate(toks):
                    batch[j, :len(x)] = t.tensor(x, device=DEVICE)
                    lens.append(len(x) - 1)
                logits = model(batch)
                last = logits[t.arange(len(toks)), t.tensor(lens, device=DEVICE)]
                pref_is = last[:, id_is] > last[:, id_are]
                want_is = (chunk.number == "sg").values
                correct_flags.extend((pref_is.cpu().numpy() == want_is).tolist())
        df[f"correct_{model_name}"] = correct_flags

        for band in BANDS:
            for tmpl in ["bare", "attractor"]:
                sub = df[(df.band == band) & (df.template == tmpl)]
                acc = sub[f"correct_{model_name}"].mean()
                rows.append({"model": model_name, "band": band,
                             "template": tmpl, "n": len(sub), "accuracy": acc})
        del model
        t.cuda.empty_cache()
        piv = pd.DataFrame(rows)
        print(f"\n{model_name}:")
        print(piv[piv.model == model_name]
              .pivot(index="template", columns="band", values="accuracy")
              .reindex(columns=BANDS).round(3).to_string())

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "competence.csv", index=False)
    df.to_csv(RESULTS / "competence_per_prompt.csv.gz", index=False,
              compression="gzip")
    print(f"\nsaved {RESULTS/'competence.csv'}")


if __name__ == "__main__":
    main()
