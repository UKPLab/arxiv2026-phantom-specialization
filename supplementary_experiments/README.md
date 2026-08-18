# Supplementary Experiments

Code and artifacts for the experiments beyond the main LSC study, grouped by topic. Scripts read existing artifacts from `../LSC_circuits/` and `../LSC_circuit_analysis/` and write only under `results/` (or a subfolder's own `results/`).

## Layout

- `code/` - standalone analysis and extraction scripts (table below).
- `results/` - their outputs, in per-item subfolders (CSV, figures).
- `SVA/` - second-task replication (subject-verb agreement).
- `two_mechanism/` - two-route positive control.
- `fewshot/` - few-shot behavioral variant (scoped as future work).

## `code/` scripts

| Topic | Script | What it does |
| :-- | :-- | :-- |
| Edge marginality | `edge_marginality.py` | Marginality of ACDC-selected edges under independent EAP-IG importance: band-specific (k=1) vs universal-core (k=5) edge ranks relative to the size-matched selection boundary, across all 75 conditions. |
| Edge marginality | `edge_marginality_figure.py` | Figure: fraction of edges beyond the selection boundary. |
| very_low band | `very_low_transfer.py` | very_low <-> core-band cross-band transfer (6x6 grid, resample ablation). |
| very_low band | `very_low_eapig.py` | EAP-IG on the very_low band plus the 6-band completion of the cross-method grid. |
| very_low band | `very_low_ablation_transfer.py` | very_low 6x6 cross-band transfer under zero and mean ablation. |
| Position aggregation | `positional_eapig.py` | Whether position aggregation hides band-specific positional structure (EAP-IG attribution). |
| Head roles | `head_roles_canonical_induction.py` | Head-role reclassification by the canonical following-token position; induction-head enrichment in the universal core. |
| Head roles | `head_roles_kcomposition.py` | Weight-level K-composition check (induction heads read from previous-token heads). |
| Head roles | `head_roles_threshold_sensitivity.py` | Robustness of the head-role labels and role-enrichment to the classification thresholds. |
| Mean ablation | `mean_ablation_cross_band.py` | Edge-level cross-band transfer under genuine tokenwise-mean ablation; `--aggregate` reproduces the NB12 cell-16 summary and writes the resample/zero/mean comparison table. |

## Subfolders

- **`SVA/`** - subject-verb agreement on the same frequency bands: data build, competence check, circuit discovery, matched and majority-core evaluation, controls, and statistics.
- **`two_mechanism/`** - a trained two-route task used as a positive control: data generation, training, a held-out double-dissociation certificate, and the discovery/transfer pipeline. Full run via `two_mechanism/code/tm_run_all.py`.
- **`fewshot/`** - k=0 vs k=2 in-context demonstration variant across all models and bands; a behavioral variant, scoped as future work.
