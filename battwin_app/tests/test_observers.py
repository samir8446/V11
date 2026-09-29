import numpy as np
from battwin.observers.stream import run_cell, make_observations, default_members, Stream, ENSEMBLE
from battwin.observers.conformal import Conformal
from battwin.observers.ensemble import EnsembleWeights
from battwin.observers.trend_kf import TrendKF
from battwin.observers.base import CycleObs
from battwin.prognostics.evaluate import forecast_accuracy


def _obs(k, soh):
    return CycleObs("X", k, soh, soh * 1.9, 2.0, 24.0, 24.0, 1.9)


def test_trend_kf_tracks_and_gates_outlier():
    kf = TrendKF()
    for k in range(1, 80):
        s = 1 - 0.002 * k
        if k == 50:
            s -= 0.2
        e = kf.update(_obs(k, s))
    assert abs(e.soh - (1 - 0.002 * 79)) < 0.01 and kf.rejects >= 1


def test_conformal_radius_quantile():
    c = Conformal(alpha=0.1)
    for r in np.linspace(0, 1, 101):
        c.add(("m", 1), r)
    assert 0.88 < c.radius(("m", 1)) < 0.95


def test_ensemble_excludes_failing_member():
    w = EnsembleWeights(["a", "b", "c"], warmup=2)
    for _ in range(10):
        w.observe("a", 0.01); w.observe("b", 0.012); w.observe("c", 0.3)
    ws = w.weights({"a": True, "b": True, "c": True}, {"a": True, "b": True, "c": True})
    assert "c" not in ws and "c" in w.excluded and abs(sum(ws.values()) - 1) < 1e-9
    ws = w.weights({"a": False, "b": True, "c": True}, {"a": True, "b": True, "c": True})
    assert "a" not in ws


def test_all_models_run_and_are_accurate(synth, cells):
    cd, _ = synth
    r = run_cell(cd, cells["reference"])
    assert len(r["failures"]) == 0
    acc = forecast_accuracy(r["forecasts"]).set_index(["model", "h"])
    for m in ["ECM twin", "Mechanistic PF", "Power-law PF", "Trend KF", "Hierarchical Bayes", ENSEMBLE]:
        assert acc.loc[(m, 1), "rmse"] < 0.01, m
    assert acc.loc[(ENSEMBLE, 25), "rmse"] < 0.03
    assert acc.loc[(ENSEMBLE, 25), "coverage"] > 0.6


def test_ecm_invalid_on_pulsed(synth, cells):
    cd, _ = synth
    r = run_cell(cd, cells["pulsed"], ["ECM twin", "Trend KF"])
    assert r["status"]["ECM twin"][0] is False
    w = r["weights"]
    assert (w[w.model == "ECM twin"].weight == 0).all()


def test_knee_and_cold_and_mixed(synth, cells):
    cd, _ = synth
    for kind in ("knee", "cold", "mixed"):
        r = run_cell(cd, cells[kind])
        acc = forecast_accuracy(r["forecasts"]).set_index(["model", "h"])
        assert acc.loc[(ENSEMBLE, 10), "rmse"] < 0.03, kind
        assert np.isfinite(acc.loc[("Mechanistic PF", 25), "r2"])       # reported, even if negative


def test_mechanism_shares_cold_vs_reference(synth, cells):
    cd, _ = synth
    sh = {}
    for kind in ("reference", "cold"):
        e = run_cell(cd, cells[kind], ["Mechanistic PF"], rul_every=0)["estimates"]
        last = e.iloc[-1]
        assert abs(last.share_sei + last.share_plating + last.share_lam - 1) < 1e-6
        sh[kind] = last.share_plating
    assert sh["cold"] > sh["reference"]


def test_stream_is_incremental(synth, cells):
    cd, _ = synth
    cell = cells["reference"]
    obs = make_observations(cd, cell, with_curves=False)
    s = Stream(obs, default_members(cd, cell, ["Trend KF", "Hierarchical Bayes"]), 0.7)
    s.run(10)
    assert s.i == 10 and not s.done
    s.run()
    assert s.done and len(s.frames()["estimates"]) > 0


def test_fleet_prior_excludes_own_cell(synth, cells):
    from battwin.observers.powerlaw_pf import fleet_prior
    cd, _ = synth
    c = cells["knee"]
    p_all = fleet_prior(cd.table)
    p_ex = fleet_prior(cd.table, exclude=c)
    assert p_ex["n_cells"] == p_all["n_cells"] - 1
