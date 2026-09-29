import numpy as np
from battwin.prognostics import metrics as M
from battwin.prognostics.rul import crossing
from battwin.prognostics.update_policy import update_frequency, ecm_check_interval
from battwin.prognostics.informativeness import rank_measurements


def test_r2_not_clipped():
    y = np.array([1.0, 0.9, 0.8, 0.7])
    assert M.r2(y, y[::-1] - 1) < -5


def test_coverage_and_wilson():
    assert M.coverage([1, 2, 3], [0, 0, 0], [1.5, 1.5, 1.5]) == 1 / 3
    lo, hi = M.wilson(9, 10)
    assert lo < 0.9 < hi


def test_crossing():
    hs = np.arange(0, 100, 10)
    assert abs(crossing(hs, 1 - 0.01 * hs, 0.7) - 30) < 1e-9
    assert np.isnan(crossing(hs, np.ones(10), 0.7))


def test_update_frequency_degrades_gracefully(synth, cells):
    cd, _ = synth
    d = update_frequency(cd, [cells["reference"]], intervals=(1, 10), models=["Trend KF"])
    r = d.set_index("interval").rmse
    assert r.loc[1] <= r.loc[10] * 1.5 and np.isfinite(r).all()


def test_ecm_check_interval(synth, cells):
    cd, _ = synth
    d = ecm_check_interval(cd, cells["reference"], intervals=(10, 10_000))
    assert len(d) == 2 and d.rmse.notna().all()


def test_measurement_informativeness(synth):
    cd, _ = synth
    r = rank_measurements(cd.table)
    assert len(r["single"]) >= 3 and r["single"].rmse.notna().all()
