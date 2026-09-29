"""Caching. Raw data and CycleData are resources keyed by a content hash; every derived result is
cached with the engine version in its key, so a new engine never serves stale results."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import streamlit as st
import battwin
from battwin.data import io, cycles
from battwin.data.synthetic import make_synthetic

V = battwin.__version__
SYNTH_KEY = f"synthetic-demo-{V}"


# ---------------------------------------------------------------- raw data / CycleData
@st.cache_resource(max_entries=3, show_spinner="Building the synthetic demo fleet…")
def _synthetic(key: str):
    m, i, truth = make_synthetic(replicas=2, seed=0)
    m, i = io.load(m, i)
    return m, i, truth


@st.cache_resource(max_entries=2, show_spinner="Reading parquet files…")
def _files(key: str, master_path: str, imp_path: str | None):
    return (*io.load(master_path, imp_path if imp_path and os.path.exists(imp_path) else None), None)


@st.cache_resource(max_entries=2, show_spinner="Reading uploaded files…")
def _uploads(key: str, _master: bytes, _imp: bytes | None):
    return (*io.load(_master, _imp), None)


@st.cache_data(show_spinner=False)
def file_key(path: str, mtime: float, size: int) -> str:
    return io.data_hash(path)


def raw(source: dict):
    k = source["key"]
    if source["kind"] == "synthetic":
        return _synthetic(k)
    if source["kind"] == "files":
        return _files(k, source["master"], source.get("imp"))
    return _uploads(k, source["master_bytes"], source.get("imp_bytes"))


@st.cache_resource(max_entries=6, show_spinner="Building the cycle table…")
def _cycle_data(key: str, version: str, eol: float, _source: dict):
    m, i, _ = raw(_source)
    return cycles.build(m, i, eol)


def cycle_data(source: dict, eol: float):
    return _cycle_data(source["key"], V, float(eol), source)


def truth(source: dict):
    return raw(source)[2]


# ---------------------------------------------------------------- derived results
def _cd(key, eol):
    return _cycle_data(key, V, float(eol), st.session_state["_source"])


@st.cache_data(show_spinner="Ranking health indicators…")
def hi_ranking(key, version, eol, cells: tuple):
    from battwin.diagnostics import indicators
    cd = _cd(key, eol)
    return indicators.rank_indicators(cd.table, list(cells)), indicators.one_or_several(cd.table, list(cells))


@st.cache_data(show_spinner="PCA…")
def pca(key, version, eol, cells: tuple):
    from battwin.diagnostics import pca as P
    return P.hi_pca(_cd(key, eol).table, list(cells))


@st.cache_data(show_spinner="Stress regression…")
def stress(key, version, eol, cells: tuple, factors: tuple, max_k):
    from battwin.diagnostics import stress as S
    cd = _cd(key, eol)
    rates = S.fade_rates(cd, list(cells), max_k)
    return rates, S.regress(rates, list(factors))


@st.cache_data(show_spinner="Fitting half-cell model…")
def halfcell(key, version, eol, cell: str, n_fits: int):
    from battwin.diagnostics import halfcell as H
    cd = _cd(key, eol)
    g = cd.cell(cell)
    g = g[~g.outlier]
    pick = g.iloc[np.unique(np.linspace(0, len(g) - 1, n_fits).astype(int))]
    cur = cd.curves(cell, pick.Cycle_Index, n=120)
    series = [(int(r.k), cur[int(r.Cycle_Index)]["q"], cur[int(r.Cycle_Index)]["v"], float(r.I_mean))
              for r in pick.itertuples() if int(r.Cycle_Index) in cur]
    return pd.DataFrame(H.fit_series(series))


@st.cache_data(show_spinner="Streaming the cell through the twin…")
def stream_cell(key, version, eol, cell, models: tuple, horizons: tuple, alpha, every):
    from battwin.observers.stream import run_cell
    return run_cell(_cd(key, eol), cell, list(models), horizons, alpha, every)


@st.cache_data(show_spinner="Update-frequency study…")
def update_freq(key, version, eol, cells: tuple, intervals: tuple, h, models: tuple):
    from battwin.prognostics.update_policy import update_frequency
    return update_frequency(_cd(key, eol), list(cells), intervals, h, list(models))


@st.cache_data(show_spinner="ECM capacity-check study…")
def ecm_check(key, version, eol, cell, h):
    from battwin.prognostics.update_policy import ecm_check_interval
    return ecm_check_interval(_cd(key, eol), cell, h=h)


@st.cache_data(show_spinner="Ranking measurements…")
def informativeness(key, version, eol, cells: tuple):
    from battwin.prognostics.informativeness import rank_measurements
    return rank_measurements(_cd(key, eol).table, list(cells))


@st.cache_data(show_spinner="Grouped cross-validation (whole cells held out)…")
def ml_soh(key, version, eol, cells: tuple, model, tune):
    from battwin.ml import features, validation
    X, y, g, t = features.soh_dataset(_cd(key, eol).table, list(cells))
    r = validation.grouped_cv(X, y, g, model, tune=tune)
    r["frame"] = pd.DataFrame(dict(cell=g, k=t.k.to_numpy(), y=y, pred=r["pred"]))
    r["importance"] = validation.feature_importance(X, y, g, model)
    return r


@st.cache_data(show_spinner="Building fade-rate features (ΔQ(V))…")
def fade_data(key, version, eol, cells: tuple, horizon):
    from battwin.ml import features
    return features.fade_dataset(_cd(key, eol), list(cells), horizon)


@st.cache_data(show_spinner="Fade-rate cross-validation…")
def ml_fade(key, version, eol, cells: tuple, horizon, model, tune):
    from battwin.ml import features, validation
    f = fade_data(key, version, eol, cells, horizon)
    r = validation.grouped_cv(f[features.FADE_FEATURES], f.target, f.cell.to_numpy(), model, tune=tune)
    r["frame"] = f.assign(pred=r["pred"])
    r["importance"] = validation.feature_importance(f[features.FADE_FEATURES], f.target, f.cell.to_numpy(), model)
    return r


@st.cache_data(show_spinner="Backtesting…")
def ml_backtest(key, version, eol, cells: tuple, horizon, cell, models: tuple):
    from battwin.ml import features, validation
    f = fade_data(key, version, eol, cells, horizon)
    return pd.concat([validation.backtest_cell(f, cell, m, features.FADE_FEATURES) for m in models],
                     ignore_index=True)


@st.cache_data(show_spinner="Solving the dynamic programme…")
def operations(key, version, eol, cell, calibrated: bool, eco: tuple, years, s0, base_I):
    from battwin.operations.plant import calibrate, Plant
    from battwin.operations.economics import Economics
    from battwin.operations.scenarios import compare
    cd = _cd(key, eol)
    plant = calibrate(cd, cell) if calibrated else Plant(sources={k: "default" for k in
                                                                  ("bol_ah", "r_ref", "kappa", "alpha", "resistance")})
    economics = Economics(**dict(eco))
    return plant, economics, compare(plant, economics, s0, years, 1, (base_I, None))


def fingerprint(eol) -> tuple:
    """(data key, engine version, eol) — the prefix of every derived cache key."""
    return st.session_state["_source"]["key"], V, float(eol)
