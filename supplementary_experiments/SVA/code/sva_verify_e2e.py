"""End-to-end verification of SVA data against CANONICAL LSC sources.

Every check re-derives the property from the canonical artifact (Pile counts,
final_bands.json, pos_classification_validated.csv, HF tokenizer) rather than
trusting any intermediate SVA file. Prints PASS/FAIL per check.

KNOWN EXCEPTIONS: the frozen datasets carry a documented pool-labeling
error. brow and disc (with their plurals) passed the pool filter although
the canonical classifier labels the singular forms STEM, not NOUN_COMMON;
cap/caps was affected too but never sampled into prompts. The affected
prompts are valid English agreement items; the violation is of the formal
pool-selection rule. Scope: 195/22,500 total examples, 21/1,920 examples
in ACDC's effective first batches. Evaluation-only sensitivity (circuits
unchanged, contaminated test prompts excluded) leaves the transfer result
unchanged; discovery used the frozen datasets and retained the
contamination. A check that fails ONLY because of these tokens is
reported as KNOWN-EXCEPTION and does not fail the run; any other
violation still fails.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
ANON = Path(__file__).resolve().parents[3]
BANDS = ["low", "medium", "high", "very_high", "control"]
MATCHED = ["low", "medium", "high"]
DRAWS = ["draw_1", "draw_2", "draw_3"]

# Documented contamination (see module docstring): singular forms are STEM
# in the canonical classifier; plurals listed for prompt-level filtering.
KNOWN_BAD_WORDS = {"brow", "disc", "cap"}
KNOWN_BAD_IDS = {6479, 22931, 1262, 28217}  # Ġbrow, Ġbrows, Ġdisc, Ġdiscs

FAILURES = []
KNOWN_EXCEPTIONS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def check_known(name, violations, detail_fn=str):
    """PASS if no violations; KNOWN-EXCEPTION if every violation is covered
    by the documented brow/disc/cap contamination; FAIL otherwise."""
    unexpected = [v for v in violations if v not in KNOWN_BAD_WORDS]
    if not violations:
        check(name, True)
    elif not unexpected:
        print(f"[KNOWN-EXCEPTION] {name} - {len(violations)} violations, "
              f"all documented brow/disc contamination: "
              f"{detail_fn(violations)}")
        KNOWN_EXCEPTIONS.append(name)
    else:
        check(name, False, f"{len(unexpected)} UNDOCUMENTED violations "
              f"{detail_fn(unexpected)}")


# ---------- load canonical sources ----------
pos = pd.read_csv(ANON / "pythia_data/pos_classification/pos_classification_validated.csv")
pos_by_id = pos.set_index("token_id").final_category.to_dict()
freq = pd.read_csv(BASE / "data/merged_token_frequencies.csv")
count_by_id = freq.set_index("token_id")["count"].to_dict()
fb = json.load(open(BASE / "data/final_bands.json"))
band_ranges = {b: v["log_freq_range"] for b, v in fb["bands"].items()}

pools = {b: json.load(open(BASE / f"pools/sva_pool_{b}.json"))
         for b in MATCHED + ["very_high"]}
all_pairs = [p for b in MATCHED + ["very_high"] for p in pools[b]["pairs"]]

from transformers import AutoTokenizer  # noqa: E402
tk = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

# ---------- A. pair level ----------
# A1: POS - both forms NOUN_COMMON in the canonical file (by token_id)
bad = [p["singular"] for p in all_pairs
       if pos_by_id.get(p["sg_token_id"]) != "NOUN_COMMON"
       or pos_by_id.get(p["pl_token_id"]) != "NOUN_COMMON"]
check_known("A1 POS: both forms NOUN_COMMON in canonical classifier output",
            bad, lambda v: str(sorted(set(v))[:5]))

# A2: tokenization - ' '+word is exactly the stored single token id
bad = []
for p in all_pairs:
    for w, tid in [(p["singular"], p["sg_token_id"]), (p["plural"], p["pl_token_id"])]:
        ids = tk.encode(" " + w, add_special_tokens=False)
        if ids != [tid]:
            bad.append((w, tid, ids))
check("A2 tokenization: ' '+word == [stored id], single token",
      not bad, f"{len(bad)} violations {bad[:3]}")

# A3: frequency + band - recompute log_freq from raw Pile counts; infer the
# normalization constant from the data, require it be a single constant, then
# re-derive the band from final_bands.json ranges
c = np.array([count_by_id.get(p["sg_token_id"], np.nan) for p in all_pairs], dtype=float)
lf = np.array([p["sg_log_freq"] for p in all_pairs])
consts = np.log10(c) - lf
check("A3a log_freq: single normalization constant vs raw Pile counts",
      np.nanstd(consts) < 1e-6, f"std={np.nanstd(consts):.2e}, C={np.nanmedian(consts):.6f}")
bad = []
for p in all_pairs:
    lo, hi = band_ranges[p["sg_band"]]
    if not (lo <= p["sg_log_freq"] <= hi):
        bad.append((p["singular"], p["sg_band"], p["sg_log_freq"]))
check("A3b band assignment: sg_log_freq inside canonical band range",
      not bad, f"{len(bad)} violations {bad[:3]}")

# A4: sg_len is the character length used for matching
bad = [p for p in all_pairs if p["sg_len"] != len(p["singular"])]
check("A4 sg_len == len(singular)", not bad, f"{len(bad)} violations")

# ---------- B. pool level ----------
lens = {b: sorted(p["sg_len"] for p in pools[b]["pairs"]) for b in MATCHED}
check("B1 exact length matching: identical sg_len multiset across low/medium/high",
      lens["low"] == lens["medium"] == lens["high"],
      f"sizes={[len(lens[b]) for b in MATCHED]}")
for b in MATCHED + ["very_high"]:
    lemmas = [p["singular"] for p in pools[b]["pairs"]]
    check(f"B2 no duplicate lemmas in pool {b}", len(lemmas) == len(set(lemmas)))
vh_expected = pools["very_high"]["n_pairs"]
check("B3 very_high flagged unmatched", pools["very_high"]["matched"] is False,
      f"n={vh_expected}")

# ---------- C. dataset level ----------
pool_ids = {b: {p["sg_token_id"] for p in pools[b]["pairs"]}
            | {p["pl_token_id"] for p in pools[b]["pairs"]} for b in MATCHED + ["very_high"]}
pool_ids["control"] = set.union(*pool_ids.values())
pair_by_sg = {p["sg_token_id"]: p for p in all_pairs}
pair_by_pl = {p["pl_token_id"]: p for p in all_pairs}
T = {s: tk.encode(s, add_special_tokens=False)[0]
     for s in ["The", " near", " the", " is", " are"]}

n_checked = 0
tmpl_bad = corrupt_bad = ans_bad = attr_bad = band_bad = 0
for draw in DRAWS:
    for band in BANDS:
        gen_keys, fin_keys = {}, {}
        for split in ["train", "val", "test"]:
            g = json.load(open(BASE / f"data_generated/{draw}/{band}/{split}.json"))["examples"]
            f = json.load(open(BASE / f"data_final/{draw}/{band}/{split}.json"))["examples"]
            gen_keys[split] = {(e["token_ids"][1], e["token_ids"][4]) for e in g}
            fin_keys[split] = {(e["token_ids"][1], e["token_ids"][4]) for e in f}
            for e in f:
                n_checked += 1
                ti, ci = e["token_ids"], e["corrupt_token_ids"]
                if not (ti[0] == T["The"] and ti[2] == T[" near"] and ti[3] == T[" the"]):
                    tmpl_bad += 1
                # corrupt differs ONLY at subject; swap is the pair's other form
                pair = pair_by_sg.get(ti[1]) or pair_by_pl.get(ti[1])
                expected_swap = (pair["pl_token_id"] if ti[1] == pair["sg_token_id"]
                                 else pair["sg_token_id"])
                if ci != [ti[0], expected_swap, ti[2], ti[3], ti[4]]:
                    corrupt_bad += 1
                # answers match subject number
                is_sg = ti[1] == pair["sg_token_id"]
                tgt, wrg = (T[" is"], T[" are"]) if is_sg else (T[" are"], T[" is"])
                if e["target_token_id"] != tgt or e["wrong_token_id"] != wrg:
                    ans_bad += 1
                # attractor is opposite number
                apair = pair_by_sg.get(ti[4]) or pair_by_pl.get(ti[4])
                attr_is_sg = ti[4] == apair["sg_token_id"]
                if attr_is_sg == is_sg:
                    attr_bad += 1
                # band membership of BOTH banded nouns
                if ti[1] not in pool_ids[band] or ti[4] not in pool_ids[band]:
                    band_bad += 1
        # split integrity
        u = [len(gen_keys[s]) for s in gen_keys]
        inter = (gen_keys["train"] & gen_keys["val"]) | (gen_keys["train"] & gen_keys["test"]) \
            | (gen_keys["val"] & gen_keys["test"])
        if inter:
            check(f"C5 splits disjoint {draw}/{band}", False, f"{len(inter)} overlaps")
        for s, n_want in [("train", 1050), ("val", 225), ("test", 225)]:
            if len(fin_keys[s]) != n_want or not fin_keys[s] <= gen_keys[s]:
                check(f"C6 final split {draw}/{band}/{s}", False,
                      f"n={len(fin_keys[s])}, subset={fin_keys[s] <= gen_keys[s]}")

check("C1 template ids at fixed positions", tmpl_bad == 0, f"{tmpl_bad}/{n_checked}")
check("C2 corruption: subject-only swap to the pair's other form",
      corrupt_bad == 0, f"{corrupt_bad}/{n_checked}")
check("C3 answer/wrong ids match subject number", ans_bad == 0, f"{ans_bad}/{n_checked}")
check("C4 attractor is opposite number", attr_bad == 0, f"{attr_bad}/{n_checked}")
check("C7 subject+attractor from the condition's pool", band_bad == 0,
      f"{band_bad}/{n_checked}")
check("C5/C6 split integrity (disjoint, 1050/225/225, final within generated)",
      all("C5" not in f and "C6" not in f for f in FAILURES))

# sg/pl balance on final test sets
for band in BANDS:
    fr = np.mean([e["number"] == "sg"
                  for e in json.load(open(BASE / f"data_final/draw_1/{band}/test.json"))["examples"]])
    check(f"C8 sg/pl balance {band} (0.40-0.60)", 0.40 <= fr <= 0.60, f"sg={fr:.3f}")

print(f"\n{'='*60}")
if FAILURES:
    verdict = f"{len(FAILURES)} FAILURES: {FAILURES}"
elif KNOWN_EXCEPTIONS:
    verdict = (f"PASS WITH {len(KNOWN_EXCEPTIONS)} KNOWN EXCEPTION(S) "
               f"(documented brow/disc contamination): {KNOWN_EXCEPTIONS}")
else:
    verdict = "ALL CHECKS PASS"
print(f"TOTAL: {n_checked} final examples checked; {verdict}")
sys.exit(1 if FAILURES else 0)
