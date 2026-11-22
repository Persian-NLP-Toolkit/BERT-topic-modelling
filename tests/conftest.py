from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def sample_csv() -> Path:
    return ROOT / "examples" / "sample.csv"
