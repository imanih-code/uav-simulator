import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from uavsim.experiments.harness import DT, Sim  # noqa: E402,F401


@pytest.fixture
def sim() -> Sim:
    return Sim()
