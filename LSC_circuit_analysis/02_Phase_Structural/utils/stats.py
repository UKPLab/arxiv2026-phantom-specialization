"""
Re-export Phase 1 statistical utilities for Phase 2 notebooks.

Sets up a synthetic package context so Phase 1's relative imports
(from .constants import ...) resolve correctly.
"""

import sys
import types
import importlib
import importlib.util
from pathlib import Path


import os as _os
from pathlib import Path as _Path


def _find_project_root() -> _Path:
    env = _os.environ.get("PROJECT_ROOT")
    if env:
        return _Path(env).resolve()
    for p in _Path(__file__).resolve().parents:
        if (p / "src" / "config.py").exists():
            return p
    return _Path(__file__).resolve().parents[1]


PROJECT_ROOT = _find_project_root()


_PHASE1_UTILS = PROJECT_ROOT / "LSC_circuit_analysis/01_Phase_Functional/utils"
_PKG_NAME = "_phase1_utils"

# 1. Load Phase 1's constants module
_const_spec = importlib.util.spec_from_file_location(
    f"{_PKG_NAME}.constants", str(_PHASE1_UTILS / "constants.py")
)
_const_mod = importlib.util.module_from_spec(_const_spec)
_const_spec.loader.exec_module(_const_mod)

# 2. Create synthetic parent package so relative imports work
_pkg = types.ModuleType(_PKG_NAME)
_pkg.__path__ = [str(_PHASE1_UTILS)]
_pkg.__package__ = _PKG_NAME
sys.modules[_PKG_NAME] = _pkg
sys.modules[f"{_PKG_NAME}.constants"] = _const_mod

# 3. Load Phase 1's stats module with package context
_stats_spec = importlib.util.spec_from_file_location(
    f"{_PKG_NAME}.stats", str(_PHASE1_UTILS / "stats.py")
)
_stats_mod = importlib.util.module_from_spec(_stats_spec)
_stats_mod.__package__ = _PKG_NAME
sys.modules[f"{_PKG_NAME}.stats"] = _stats_mod
_stats_spec.loader.exec_module(_stats_mod)

# Re-export everything
# Effect sizes
cohens_d = _stats_mod.cohens_d
cohens_d_paired = _stats_mod.cohens_d_paired
rank_biserial = _stats_mod.rank_biserial
eta_squared = _stats_mod.eta_squared
cliff_delta = _stats_mod.cliff_delta
interpret_d = _stats_mod.interpret_d
interpret_r = _stats_mod.interpret_r
interpret_eta2 = _stats_mod.interpret_eta2

# Bootstrap
bootstrap_ci = _stats_mod.bootstrap_ci
bootstrap_ci_diff = _stats_mod.bootstrap_ci_diff

# Safe test wrappers
safe_kruskal = _stats_mod.safe_kruskal
safe_mannwhitneyu = _stats_mod.safe_mannwhitneyu
safe_wilcoxon = _stats_mod.safe_wilcoxon
safe_spearmanr = _stats_mod.safe_spearmanr
jonckheere_terpstra = _stats_mod.jonckheere_terpstra

# Test accumulator
TestAccumulator = _stats_mod.TestAccumulator
