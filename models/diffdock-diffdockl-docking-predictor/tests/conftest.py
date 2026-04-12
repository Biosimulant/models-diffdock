from __future__ import annotations

import sys
from pathlib import Path

import pytest


_MODEL_DIR = Path(__file__).resolve().parents[1]
_MONOREPO_ROOT = _MODEL_DIR.parents[3]


@pytest.fixture(scope="session", autouse=True)
def _paths():
    if str(_MODEL_DIR) not in sys.path:
        sys.path.insert(0, str(_MODEL_DIR))
    biosim_src = _MONOREPO_ROOT / "biosim" / "src"
    if biosim_src.exists() and str(biosim_src) not in sys.path:
        sys.path.insert(0, str(biosim_src))


@pytest.fixture(scope="session")
def biosim(_paths):
    import biosim as _bsim

    return _bsim

