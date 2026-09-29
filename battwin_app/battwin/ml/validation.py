"""Leakage-free evaluation: nested grouped CV (whole cells held out; tuning only inside the
training cells) and per-cell backtests trained on other cells only."""
from __future__ import annotations
import warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GridSearchCV
from ..prognostics import metrics as M
from .models import make, GRIDS, GP_MAX_TRAIN, predict_std


def _fit(name, X, y, groups, tune=True, seed=0):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    if name == "Gaussian Process" and len(X) > GP_MAX_TRAIN:
        idx = np.random.default_rng(seed).choice(len(X), GP_MAX_TRAIN, replace=False)
        X, y, groups = X.iloc[idx], y[idx], groups[idx]
    model = make(name, seed)
    n_groups = len(np.unique(groups))
    if tune and n_groups >= 3:
        gs = GridSearchCV(model, GRIDS[name], cv=GroupKFold(n_splits=min(3, n_groups)),
                          scoring="neg_root_mean_squared_error")
        gs.fit(X, y, groups=groups)
        return gs.best_estimator_, gs.best_params_
    return model.fit(X, y), {}


def grouped_cv(X: pd.DataFrame, y, groups, name: str, n_splits: int = 5, tune: bool = True, seed: int = 0):
    y, groups = np.asarray(y, float), np.asarray(groups)
    cells = np.unique(groups)
    if len(cells) < 3:
        raise ValueError("grouped CV needs at least 3 cells")
    outer = GroupKFold(n_splits=min(n_splits, len(cells)))
    pred, std = np.full(len(y), np.nan), np.full(len(y), np.nan)
    params = []
    for tr, te in outer.split(X, y, groups):
        m, p = _fit(name, X.iloc[tr], y[tr], groups[tr], tune, seed)
        pred[te] = m.predict(X.iloc[te])
        s = predict_std(m, X.iloc[te])
        if s is not None:
            std[te] = s
        params.append(p)
    per_cell = pd.DataFrame(dict(cell=groups, y=y, pred=pred)).groupby("cell").apply(
        lambda d: pd.Series(dict(rmse=M.rmse(d.y, d.pred), n=len(d))), include_groups=False).reset_index()
    return dict(pred=pred, std=std, rmse=M.rmse(y, pred), mae=M.mae(y, pred), r2=M.r2(y, pred),
                per_cell=per_cell, params=params,
                coverage90=M.coverage(y, pred - 1.645 * std, pred + 1.645 * std) if np.isfinite(std).any() else np.nan)


def feature_importance(X, y, groups, name, seed=0) -> pd.Series:
    """Permutation importance on held-out cells (one grouped split)."""
    from sklearn.inspection import permutation_importance
    g = np.asarray(groups)
    tr, te = next(GroupKFold(n_splits=min(4, len(np.unique(g)))).split(X, y, g))
    m, _ = _fit(name, X.iloc[tr], np.asarray(y)[tr], g[tr], tune=False, seed=seed)
    r = permutation_importance(m, X.iloc[te], np.asarray(y)[te], n_repeats=5, random_state=seed,
                               scoring="neg_root_mean_squared_error")
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def backtest_cell(fade_df: pd.DataFrame, cell: str, name: str, features, tune=True, seed=0) -> pd.DataFrame:
    """Train on all other cells; forecast the target cell's SOH at k0 + horizon from each origin."""
    tr, te = fade_df[fade_df.cell != cell], fade_df[fade_df.cell == cell]
    if te.empty or tr.cell.nunique() < 3:
        return pd.DataFrame()
    m, _ = _fit(name, tr[features], tr.target.to_numpy(), tr.cell.to_numpy(), tune, seed)
    rate = m.predict(te[features])
    s = predict_std(m, te[features])
    out = te[["cell", "k0", "soh_now", "soh_future", "target", "horizon"]].copy()
    out["rate_pred"] = rate
    out["soh_pred"] = out.soh_now - rate * out.horizon / 100
    out["soh_std"] = (s * out.horizon / 100) if s is not None else np.nan
    out["model"] = name
    return out
