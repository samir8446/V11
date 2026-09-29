import pytest
from battwin.data.synthetic import make_synthetic
from battwin.data import io, cycles


@pytest.fixture(scope="session")
def synth():
    m, i, truth = make_synthetic(seed=3)
    m, i = io.load(m, i)
    cd = cycles.build(m, i, eol_ah=1.5)
    return cd, truth


@pytest.fixture(scope="session")
def cells(synth):
    cd, truth = synth
    return {k: c for c, k in truth.groupby("Cell_ID").kind.first().items()}
