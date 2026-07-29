from pathlib import Path

import pytest

from pelgnn.data import LandscapeSample


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def held_out_sample(repository_root: Path) -> LandscapeSample:
    return LandscapeSample.load(
        repository_root / "data/example.npz"
    )
