import pytest

from closure_risk.features import build_features
from closure_risk.simulate import simulate


@pytest.fixture(scope="session")
def raw():
    return simulate(seed=42)


@pytest.fixture(scope="session")
def feats(raw):
    return build_features(raw)
