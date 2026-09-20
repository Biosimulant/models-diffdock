from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

@pytest.fixture(scope="session")
def biosim():
    import biosim
    return biosim
