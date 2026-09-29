"""Health-indicator (HI) ranking: monotonicity, trendability, prognosability and
leave-one-cell-out (LOCO) SOH-estimation error, for single HIs and the best HI set."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HI_COLUMNS = {
    "Capacity (rate-normalised)": "cap_norm",
    "Internal resistance R0": "R0",
    "Electrolyte resistance Re": "Re_ohm",
    "Charge-transfer resistance Rct": "Rct_ohm",
    "Discharge mean voltage": "V_mean",
    "Discharge energy": "energy_Wh",
    "CC charge time": "cc_time_s",
    "CV charge time": "cv_time_s",
    "Temperature rise": "T_rise",
}
# HIs that are (near) direct measurements of capacity; excluded from the "non-capacity" answer
CAPACITY_LIKE = {"Capacity (rate-normalised)", "Discharge energy"}


def _smooth(x, w=5):
    return pd.Series(x).rolling(w, center=True, min_periods=1).median().to_numpy()


def monotonicity(x) -> float:
    x = _smooth(np.asarray(x, float)[np.isfinite(x)])
    if x.size < 3:
        return np.nan
    d = np.diff(x)
    return abs((d > 0).sum() - (d < 0).sum()) / (x.size - 1)


def rank_indicators(table: pd.DataFrame, cells=None) -> pd.DataFrame:
    t = table[~table["outlier"]]
    if cells is not None:
        t = t[t["Cell_ID"].isin(cells)]
    rows = []
    for name, col in HI_COLUMNS.items():
        if col not in t or t[col].notna().sum() < 10:
            continue
        mono, trend, starts, ends = [], [], [], []
        for _, g in t.groupby("Cell_ID"):
            x = g[col].to_numpy(float)
            ok = np.isfinite(x)
            if ok.sum() < 8 or np.nanstd(x) == 0:
                continue
            mono.append(monotonicity(x))
            rho = spearmanr(g["k"][ok], x[ok]).statistic
            trend.append(abs(rho) if np.isfinite(rho) else np.nan)
            xs = _smooth(x[ok])
            starts.append(xs[:3].mean()); ends.append(xs[-3:].mean())
        if len(mono) < 2:
            continue
        starts, ends = np.array(starts), np.array(ends)
        prog = float(np.exp(-np.std(ends) / (np.mean(np.abs(ends - starts)) + 1e-12)))
        rows.append(dict(indicator=name, column=col, n_cells=len(mono),
                         monotonicity=float(np.nanmean(mono)), trendability=float(np.nanmean(trend)),
                         prognosability=prog))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["score"] = df[["monotonicity", "trendability", "prognosability"]].mean(axis=1)
    loco = loco_errors(table, [r for r in df["column"]], cells)
    df = df.merge(loco, on="column", how="left")
    return df.sort_values("score", ascending=False).reset_index(drop=True)


def _fit_predict(Xtr, ytr, Xte, alpha=1e-3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    A = (Xtr - mu) / sd
    A1 = np.column_stack([np.ones(len(A)), A])
    beta = np.linalg.solve(A1.T @ A1 + alpha * np.eye(A1.shape[1]), A1.T @ ytr)
    return np.column_stack([np.ones(len(Xte)), (Xte - mu) / sd]) @ beta


def loco_errors(table, columns, cells=None) -> pd.DataFrame:
    """RMSE of SOH estimated from each HI with a model trained on the other cells.
    HIs are expressed relative to each cell's first valid value (removes cell-to-cell offsets)."""
    t = table[~table["outlier"]].copy()
    if cells is not None:
        t = t[t["Cell_ID"].isin(cells)]
    rows = []
    for col in columns:
        d = t[["Cell_ID", "soh", col]].dropna()
        d = d.assign(x=d[col] / d.groupby("Cell_ID")[col].transform(lambda s: s.iloc[:3].mean()))
        cells_ = d["Cell_ID"].unique()
        if len(cells_) < 3:
            rows.append(dict(column=col, loco_rmse=np.nan)); continue
        err = []
        for c in cells_:
            tr, te = d[d.Cell_ID != c], d[d.Cell_ID == c]
            p = _fit_predict(tr[["x"]].to_numpy(), tr["soh"].to_numpy(), te[["x"]].to_numpy())
            err.append(np.sqrt(np.mean((p - te["soh"].to_numpy()) ** 2)))
        rows.append(dict(column=col, loco_rmse=float(np.mean(err))))
    return pd.DataFrame(rows)


def one_or_several(table, cells=None, exclude_capacity=True) -> dict:
    """Greedy forward selection of HIs under LOCO: does a set beat the best single HI?"""
    t = table[~table["outlier"]].copy()
    if cells is not None:
        t = t[t["Cell_ID"].isin(cells)]
    cand = [c for n, c in HI_COLUMNS.items() if not (exclude_capacity and n in CAPACITY_LIKE)
            and c in t and t[c].notna().mean() > 0.5]
    for c in cand:
        t[c] = t[c] / t.groupby("Cell_ID")[c].transform(lambda s: s.dropna().iloc[:3].mean()
                                                        if s.notna().any() else np.nan)

    def score(cols):
        d = t[["Cell_ID", "soh"] + cols].dropna()
        cs = d.Cell_ID.unique()
        if len(cs) < 3:
            return np.nan
        e = []
        for c in cs:
            tr, te = d[d.Cell_ID != c], d[d.Cell_ID == c]
            p = _fit_predict(tr[cols].to_numpy(), tr.soh.to_numpy(), te[cols].to_numpy())
            e.append(np.sqrt(np.mean((p - te.soh.to_numpy()) ** 2)))
        return float(np.mean(e))

    chosen, path, best = [], [], np.inf
    while cand:
        s = {c: score(chosen + [c]) for c in cand}
        s = {c: v for c, v in s.items() if np.isfinite(v)}
        if not s:
            break
        c, v = min(s.items(), key=lambda kv: kv[1])
        if v > best * 0.97:          # require a >3 % improvement to add an HI
            break
        chosen.append(c); cand.remove(c); best = v
        path.append(dict(step=len(chosen), added=c, loco_rmse=v))
    single = path[0]["loco_rmse"] if path else np.nan
    return dict(selected=chosen, path=pd.DataFrame(path), single_rmse=single, set_rmse=best,
                verdict=("several" if len(chosen) > 1 else "one") if path else "undetermined")
