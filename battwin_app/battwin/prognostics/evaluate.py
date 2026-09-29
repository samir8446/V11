"""Accuracy summaries of a streamed cell (per model and horizon)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import metrics as M


def forecast_accuracy(fc: pd.DataFrame) -> pd.DataFrame:
    if fc is None or fc.empty:
        return pd.DataFrame()
    rows = []
    for (m, h), g in fc.groupby(["model", "h"]):
        y = g["actual"].to_numpy(float)
        n = int(np.isfinite(y).sum())
        rows.append(dict(model=m, h=h, n=n, rmse=M.rmse(y, g.pred), mae=M.mae(y, g.pred), r2=M.r2(y, g.pred),
                         coverage=M.coverage(y, g.lo, g.hi), coverage_model=M.coverage(y, g.lo_model, g.hi_model),
                         fade_skill=M.fade_skill(g.soh_now, y, g.pred)))
    return pd.DataFrame(rows)


def estimation_accuracy(est: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m, g in est.groupby("model"):
        rows.append(dict(model=m, rmse=M.rmse(g.measured, g.soh), mae=M.mae(g.measured, g.soh),
                         r2=M.r2(g.measured, g.soh), n=int(np.isfinite(g.measured).sum())))
    return pd.DataFrame(rows)


def rul_accuracy(rul: pd.DataFrame, true_eol) -> pd.DataFrame:
    if rul is None or rul.empty:
        return pd.DataFrame()
    rows = []
    for m, g in rul.groupby("model"):
        if true_eol is None or not np.isfinite(true_eol):
            rows.append(dict(model=m, rul_mae=np.nan, status="EOL not reachable in data")); continue
        g = g[g.k0 < true_eol]
        err = (g.eol_pred - true_eol).abs()
        rows.append(dict(model=m, rul_mae=float(err.mean()) if err.notna().any() else np.nan,
                         status="ok" if err.notna().any() else "no EOL predicted",
                         frac_predicted=float(err.notna().mean()) if len(err) else np.nan))
    return pd.DataFrame(rows)
