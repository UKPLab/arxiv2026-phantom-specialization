# Setup

## 1. Python environment

Tested with Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pinned versions are in `requirements.txt`.

## 2. PROJECT_ROOT

Most scripts and notebooks resolve paths relative to a `PROJECT_ROOT`
environment variable. From the repo root:

```bash
export PROJECT_ROOT="$(pwd)"
```

If unset, scripts fall back to walking up the file tree from `__file__`
to find `src/config.py`. Notebooks fall back to the current working
directory, so launching `jupyter` from the repo root is recommended.

## 3. External repositories

Two upstream repositories are imported via `sys.path` and must be cloned
separately. The defaults below are picked up automatically; override with
`AUTOCIRCUIT_PATH` / `TUNED_LENS_PATH` if you cloned elsewhere.

```bash
mkdir -p circuit_discovery
git clone https://github.com/UFO-101/auto-circuit.git    circuit_discovery/auto-circuit
git clone https://github.com/AlignmentResearch/tuned-lens.git circuit_discovery/tuned-lens

export AUTOCIRCUIT_PATH="$PROJECT_ROOT/circuit_discovery/auto-circuit"
export TUNED_LENS_PATH="$PROJECT_ROOT/circuit_discovery/tuned-lens"
```

| Repo         | Purpose                                         | Upstream                                                |
|--------------|-------------------------------------------------|---------------------------------------------------------|
| auto-circuit | ACDC / EAP circuit-discovery and patching utils | <https://github.com/UFO-101/auto-circuit>               |
| tuned-lens   | Tuned-lens training and inference               | <https://github.com/AlignmentResearch/tuned-lens>       |

## 4. Data and models

- **Pythia models** (`EleutherAI/pythia-70m`, `pythia-160m`, `pythia-410m`,
  `pythia-1b`, `pythia-1.4b`) are downloaded from the HuggingFace Hub on first
  use. Not vendored here.
- **The Pile** (deduplicated subset): downloaded separately. See
  `pythia_data/00_download_pile.py`. Not redistributed.
- **FineWeb sample**: used for context extraction. See
  `pythia_data/11_download_fineweb_sample.py`. Not redistributed.
- **Token-frequency artifacts** and **matched token pools** ship under
  `pythia_data/` and `LSC_data/`.
- **Activation caches** (`*.npz`) and **tuned-lens weights** (`*.pt`) are
  excluded; regenerate with the scripts in
  `LSC_circuit_analysis/03_Phase_Representational/`.

## 5. Verification

```bash
python -c "from src.config import PROJECT_ROOT; print(PROJECT_ROOT)"
```

For an end-to-end test, pick a small notebook in
`LSC_circuit_analysis/01_Phase_Functional/`.

## 6. Licensing

Code in this repository is released under the Apache License 2.0 (see `LICENSE`).
External dependencies retain their own licenses; check each upstream repository.
