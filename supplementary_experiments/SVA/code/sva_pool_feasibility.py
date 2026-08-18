"""SVA feasibility, stage 1: per-band pools of (singular, plural) noun pairs.

The SVA design needs subject nouns whose singular AND plural forms are both
single Pythia tokens (word_en convention: leading space, lowercase Latin),
banded by the paper's canonical log-frequency band design. The answer tokens
(" is"/" are"/" was"/" were") are constant high-frequency function words, so
the frequency manipulation is input-side only (avoids the IOI output-prior
confound).

Inputs (copied into ../data/):
  valid_tokens_with_bands.csv   LLM-classified Pythia tokens + log_frequency
                                (from the IOI attempt-1 classification)
  final_bands.json              canonical LSC band design (log-freq ranges)
  merged_token_frequencies.csv  raw Pile counts for all 50,064 tokens

Outputs (../results/):
  sva_noun_pairs.csv            surviving pairs w/ ids, freqs, band labels
  pool_feasibility_summary.csv  per-band counts (lenient/strict)
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA, RESULTS = BASE / "data", BASE / "results"
RESULTS.mkdir(exist_ok=True)

NOUN_CATEGORIES = {"OBJECT_NOUN", "PLACE_NOUN"}

IRREGULAR = {
    "man": "men", "woman": "women", "child": "children", "person": "people",
    "foot": "feet", "tooth": "teeth", "goose": "geese", "mouse": "mice",
    "ox": "oxen", "louse": "lice", "die": "dice", "penny": "pence",
    "leaf": "leaves", "loaf": "loaves", "knife": "knives", "wife": "wives",
    "life": "lives", "shelf": "shelves", "wolf": "wolves", "thief": "thieves",
    "half": "halves", "calf": "calves", "elf": "elves", "scarf": "scarves",
}
# no distinct plural form -> unusable for the number manipulation
INVARIANT = {"sheep", "deer", "fish", "species", "series", "aircraft", "means"}


def pluralize(w: str):
    if w in INVARIANT:
        return None
    if w in IRREGULAR:
        return IRREGULAR[w]
    if re.search(r"(s|x|z|ch|sh)$", w):
        return w + "es"
    if re.search(r"[^aeiou]y$", w):
        return w[:-1] + "ies"
    if w.endswith("fe"):
        return w[:-2] + "ves"
    return w + "s"


def main():
    df = pd.read_csv(DATA / "valid_tokens_with_bands.csv")
    nouns = df[
        df["category"].isin(NOUN_CATEGORIES)
        & (df["is_valid_english_word"] == True)  # noqa: E712
        & df["cleaned_token"].astype(str).str.fullmatch(r"[a-z]+")
        & df["token_string"].astype(str).str.startswith("Ġ")
    ].copy()
    print(f"candidate singular nouns (lowercase word_en): {len(nouns)}")

    # scaling constant: freq_per_million = raw_count / total * 1e6
    ref = df.dropna(subset=["raw_count", "freq_per_million"]).iloc[0]
    total_tokens = ref["raw_count"] / ref["freq_per_million"] * 1e6
    print(f"corpus total tokens (derived): {total_tokens:.3e}")

    # full vocab: token string -> (id, count)
    freq = pd.read_csv(DATA / "merged_token_frequencies.csv")
    tok2row = {s: (int(i), int(c)) for i, s, c in
               zip(freq["token_id"], freq["token_string"], freq["count"])}

    # canonical bands
    bands_cfg = json.load(open(DATA / "final_bands.json"))["bands"]
    band_ranges = {name: cfg["log_freq_range"] for name, cfg in bands_cfg.items()}
    core_order = ["very_low", "low", "medium", "high", "very_high"]

    def band_of(logf):
        for name in core_order + [b for b in band_ranges if b not in core_order]:
            lo, hi = band_ranges[name]
            if lo <= logf < hi:
                return name
        return None

    # tokenizer for the single-token plural check
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    for ans in [" is", " are", " was", " were"]:
        ids = tk.encode(ans)
        assert len(ids) == 1, f"answer token {ans!r} not single: {ids}"
    print("answer tokens ' is'/' are'/' was'/' were': all single tokens OK")

    rows = []
    for _, r in nouns.iterrows():
        sg = r["cleaned_token"]
        pl = pluralize(sg)
        if pl is None or pl == sg:
            continue
        pl_tok = "Ġ" + pl
        if pl_tok not in tok2row:
            continue
        ids = tk.encode(" " + pl)
        if len(ids) != 1:
            continue
        pl_id, pl_count = tok2row[pl_tok]
        pl_logf = np.log10(pl_count / total_tokens * 1e6) if pl_count > 0 else -np.inf
        sg_band, pl_band = band_of(r["log_frequency"]), band_of(pl_logf)
        rows.append({
            "singular": sg, "plural": pl,
            "sg_token_id": int(r["token_id"]), "pl_token_id": pl_id,
            "category": r["category"],
            "sg_log_freq": r["log_frequency"], "pl_log_freq": pl_logf,
            "sg_band": sg_band, "pl_band": pl_band,
            "same_band": sg_band == pl_band,
            "sg_len": len(sg),
        })

    pairs = pd.DataFrame(rows)
    pairs.to_csv(RESULTS / "sva_noun_pairs.csv", index=False)
    print(f"\npairs with single-token plural: {len(pairs)} "
          f"(of {len(nouns)} candidates)")

    summary = []
    for band in core_order:
        b = pairs[pairs.sg_band == band]
        summary.append({
            "band": band,
            "pairs_lenient": len(b),                      # banded by singular
            "pairs_strict": int(b.same_band.sum()),       # plural in same band
            "pairs_pl_in_core": int(b.pl_band.isin(core_order).sum()),
            "object_nouns": int((b.category == "OBJECT_NOUN").sum()),
            "place_nouns": int((b.category == "PLACE_NOUN").sum()),
            "median_sg_len": b.sg_len.median() if len(b) else np.nan,
        })
    s = pd.DataFrame(summary)
    s.to_csv(RESULTS / "pool_feasibility_summary.csv", index=False)
    print("\n=== per-band pairs (canonical LSC bands) ===")
    print(s.to_string(index=False))
    print("\nbenchmarks: LSC matched pool = 703/band; band-design floor = 500;"
          "\n            LSC very_low (unmatched, as now running) = 97")


if __name__ == "__main__":
    main()
