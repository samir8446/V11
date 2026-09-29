"""M2: which measurements are most informative about SOH?

SOH is estimated from each measurement group alone, with a Bayesian ridge model evaluated by
grouped cross-validation (whole cells held out). Groups are then added greedily."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import GroupKFold
from .metrics import rmse

GROUPS = {
    "Resistance R0 (discharge step)": ["R0"],
    "EIS (Re, Rct)": ["Re_ohm", "Rct_ohm"],
    "Temperature rise": ["T_rise"],
    "Charge times (CC, CV)": ["cc_time_s", "cv_time_s"],
    "Discharge voltage": ["V_mean"],
    "Cycle count": ["k"],
}


def _relative(t, cols):
    t = t.copy()
    for c in cols:
        if c == "k":
            continue
        base = t.groupby("Cell_ID")[c].transform(lambda s: s.dropna().iloc[:3].mean() if s.notna().any() else np.nan)
        t[c] = t[c] / base
    return t


def _cv_rmse(t, cols, n_splits=5):
    d = t[["Cell_ID", "soh"] + cols].dropna()
    cells = d.Cell_ID.unique()
    if len(cells) < 3:
        return np.nan, 0
    gkf = GroupKFold(n_splits=min(n_splits, len(cells)))
    pred = np.full(len(d), np.nan)
    X, y, g = d[cols].to_numpy(float), d.soh.to_numpy(float), d.Cell_ID.to_numpy()
    for tr, te in gkf.split(X, y, g):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        m = BayesianRidge().fit((X[tr] - mu) / sd, y[tr])
        pred[te] = m.predict((X[te] - mu) / sd)
    return rmse(y, pred), len(cells)


def rank_measurements(table, cells=None) -> dict:
    t = table[~table.outlier]
    if cells is not None:
        t = t[t.Cell_ID.isin(cells)]
    allc = sorted({c for v in GROUPS.values() for c in v})
    t = _relative(t, [c for c in allc if c in t])
    base = float(np.sqrt(np.mean((t.soh - t.soh.mean()) ** 2)))
    single = []
    for name, cols in GROUPS.items():
        cols = [c for c in cols if c in t and t[c].notna().mean() > 0.3]
        if not cols:
            continue
        r, n = _cv_rmse(t, cols)
        single.append(dict(measurement=name, rmse=r, n_cells=n, gain_vs_mean=1 - r / base if np.isfinite(r) else np.nan))
    single = pd.DataFrame(single).sort_values("rmse").reset_index(drop=True)
    chosen, path, best = [], [], np.inf
    avail = list(single.measurement)
    while avail:
        trial = {m: _cv_rmse(t, sum((GROUPS[x] for x in chosen + [m]), []))[0] for m in avail}
        trial = {m: v for m, v in trial.items() if np.isfinite(v)}
        if not trial:
            break
        m, v = min(trial.items(), key=lambda kv: kv[1])
        if v > best * 0.97:
            break
        chosen.append(m); avail.remove(m); best = v
        path.append(dict(step=len(chosen), added=m, rmse=v))
    return dict(single=single, path=pd.DataFrame(path), baseline_rmse=base)
