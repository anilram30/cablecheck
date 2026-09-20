from pathlib import Path

import numpy as np
import pytest

REF = Path(__file__).parent / "reference"


@pytest.fixture(scope="session")
def f_grid():
    return np.linspace(1e6, 600e6, 400)


@pytest.fixture(scope="session")
def ref_dir():
    return REF
