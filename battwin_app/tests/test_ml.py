import numpy as np
from battwin.ml import features, models, validation


def test_only_four_models():
    assert models.MODELS == ["Gaussian Process", "Extra Trees", "Hist. Gradient Boosting", "Bayesian Ridge"]


def test_soh_estimation_grouped_cv(synth):
    cd, _ = synth
    X, y, g, _ = features.soh_dataset(cd.table)
    assert "capacity" not in X and "cap_norm" not in X and "energy_Wh" not in X
    for name in ["Bayesian Ridge", "Extra Trees"]:
        r = validation.grouped_cv(X, y, g, name, n_splits=4, tune=False)
        assert np.isfinite(r["rmse"]) and r["rmse"] < 0.08, name
        assert r["per_cell"].shape[0] == len(np.unique(g))


def test_fade_dataset_and_backtest(synth, cells):
    cd, _ = synth
    f = features.fade_dataset(cd, horizon=25, step=10)
    assert len(f) > 30 and f.target.notna().all()
    assert f[f.cell == cells["pulsed"]].dq_logvar.isna().all()      # no ΔQ(V) on pulsed load
    assert f[f.cell == cells["reference"]].dq_logvar.notna().any()
    bt = validation.backtest_cell(f, cells["reference"], "Hist. Gradient Boosting", features.FADE_FEATURES, tune=False)
    assert len(bt) and cells["reference"] not in set(f[f.cell != cells["reference"]].cell)
    assert np.sqrt(np.mean((bt.soh_pred - bt.soh_future) ** 2)) < 0.06


def test_gp_tuned_nested_cv_runs(synth):
    cd, _ = synth
    f = features.fade_dataset(cd, horizon=25, step=20, with_dq=False)
    r = validation.grouped_cv(f[features.FADE_FEATURES], f.target, f.cell, "Gaussian Process", n_splits=3, tune=True)
    assert np.isfinite(r["rmse"]) and np.isfinite(r["coverage90"])
