"""Central path configuration.

Resolves PROJECT_ROOT (the repository root) from $PROJECT_ROOT if set,
otherwise inferred from this file's location.

Other modules should import PROJECT_ROOT from here and build paths via
PROJECT_ROOT / "subdir" / ... rather than hard-coding absolute paths.

Optional env vars used elsewhere in the codebase:
- AUTOCIRCUIT_PATH : path to a local clone of the auto-circuit repo
- TUNED_LENS_PATH  : path to a local clone of the tuned-lens repo
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])
).resolve()

PYTHIA_DATA = PROJECT_ROOT / "pythia_data"
LSC_DATA = PROJECT_ROOT / "LSC_data"
LSC_CIRCUITS = PROJECT_ROOT / "LSC_circuits"
LSC_CIRCUIT_ANALYSIS = PROJECT_ROOT / "LSC_circuit_analysis"


def external_path(env_var: str, hint: str) -> Path:
    """Return a Path from env_var or raise with a helpful message."""
    val = os.environ.get(env_var)
    if not val:
        raise EnvironmentError(f"{env_var} is not set. {hint} See SETUP.md.")
    return Path(val).resolve()
