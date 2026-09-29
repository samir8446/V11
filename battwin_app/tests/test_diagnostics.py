import dataclasses
import numpy as np
from battwin.diagnostics import indicators, pca, stress, icadva, halfcell


def test_indicator_ranking(synth):
    cd, _ = synth
    r = indicators.rank_indicators(cd.table)
    assert {"monotonicity", "trendability", "prognosability", "loco_rmse"} <= set(r.columns)
    cap = r[r.column == "cap_norm"].iloc[0]
    assert cap.monotonicity > 0.3 and cap.trendability > 0.8
    assert r.score.between(0, 1).all()


def test_one_or_several(synth):
    cd, _ = synth
    res = indicators.one_or_several(cd.table)
    assert res["verdict"] in ("one", "several") and res["set_rmse"] <= res["single_rmse"] + 1e-12


def test_pca(synth):
    cd, _ = synth
    p = pca.hi_pca(cd.table)
    assert p["ok"] and abs(p["explained"].sum() - 1) < 1e-9 and abs(p["pc1_soh_corr"]) > 0.5


def test_stress_regression_reports_confounding(synth):
    cd, _ = synth
    rates = stress.fade_rates(cd)
    res = stress.regress(rates, ["ambient", "I_mean"])
    assert res["power"] in ("not estimable", "very low power", "low power", "adequate")
    if res["ok"]:
        assert "vif" in res["coef"] and np.isfinite(res["r2"])   # R² unclipped, may be negative


def test_ica_dva(synth, cells):
    cd, _ = synth
    cur = cd.curves(cells["reference"], n=200)
    c = cur[min(cur)]
    v, d = icadva.ica(c)
    q, dv = icadva.dva(c)
    assert v.size and np.nanmax(d) > 0 and q.size
    assert np.isfinite(icadva.ica_peak(c)).all()


def test_halfcell_recovers_mechanisms():
    e = dataclasses.replace(halfcell.BOL, Li=halfcell.BOL.Li * 0.9, Qp=halfcell.BOL.Qp * 0.97)
    q, v, _ = halfcell.discharge(e, 0.5, 2.7, n=150)
    rng = np.random.default_rng(0)
    fit = halfcell.fit_curve(q, v + rng.normal(0, 0.002, v.size), 0.5)
    mech = halfcell.mechanisms(fit.electrodes)
    assert fit.success and fit.rmse_mV < 5
    assert abs(mech["LLI"] - 0.10) < 0.04


def test_halfcell_capacity_is_rate_dependent():
    c1 = halfcell.discharge(halfcell.BOL, 1.0, 2.7)[2]
    c4 = halfcell.discharge(halfcell.BOL, 4.0, 2.7)[2]
    assert 0.03 < 1 - c4 / c1 < 0.15
