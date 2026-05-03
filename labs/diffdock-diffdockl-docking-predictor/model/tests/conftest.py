from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MODEL_DIR = Path(__file__).resolve().parents[1]


def _find_bsim_src(start: Path) -> Path | None:
    for parent in [start, *start.parents]:
        for candidate in (parent / "biosim" / "src", parent / "bsim-active" / "biosim" / "src"):
            if (candidate / "biosim").is_dir():
                return candidate
    return None


def _ensure_test_paths() -> None:
    model_dir = str(_MODEL_DIR)
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    bsim_src = _find_bsim_src(_MODEL_DIR)
    if bsim_src is not None and str(bsim_src) not in sys.path:
        sys.path.insert(0, str(bsim_src))


_ensure_test_paths()


@pytest.fixture(scope="session", autouse=True)
def _paths():
    _ensure_test_paths()


@pytest.fixture(scope="session")
def biosim(_paths):
    import biosim as _bsim

    return _bsim
