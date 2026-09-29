"""ML features: cycle-level SOH estimation features and origin-level fade-rate features with
early-life ΔQ(V) statistics. Capacity-derived quantities are never used as inputs for SOH."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from ..data.cycles import qv

SOH_FEATURES = ["k", "R0", "Re_ohm", "Rct_ohm", "T_rise", "T_mean", "cc_time_s", "cv_time_s", "V_mean",
                "I_mean", "ambient"]
FADE_FEATURES = ["k0", "soh_now", "slope20", "curv40", "I_mean", "ambient", "cutoff_V",
                 "dq_logvar", "dq_min", "dq_mean", "R0_rel", "Rct_rel"]


def soh_dataset(table: pd.DataFrame, cells=None):
    t = table[~table.outlier & table.soh.notna()]
    if cells is not None:
        t = t[t.Cell_ID.isin(cells)]
    t = t.copy()
    for c in ["R0", "Re_ohm", "Rct_ohm", "cc_time_s", "cv_time_s", "V_mean"]:
        base = t.groupby("Cell_ID")[c].transform(lambda s: s.dropna().iloc[:3].mean() if s.notna().any() else np.nan)
        t[c] = t[c] / base          # relative to the cell's own early values
    return t[SOH_FEATURES].astype(float), t["soh"].to_numpy(float), t["Cell_ID"].to_numpy(), t


def _dq_stats(cd, cell, k_ref_ci, k0_ci, vgrid):
    cur = cd.curves(cell, [k_ref_ci, k0_ci], n=200)
    if k_ref_ci not in cur or k0_ci not in cur:
        return np.nan, np.nan, np.nan
    dq = qv(cur[k0_ci], vgrid) - qv(cur[k_ref_ci], vgrid)
    dq = dq[np.isfinite(dq)]
    if dq.size < 10:
        return np.nan, np.nan, np.nan
    return float(np.log10(np.var(dq) + 1e-12)), float(dq.min()), float(dq.mean())


def fade_dataset(cd, cells=None, horizon: int = 25, step: int = 10, first: int = 20, with_dq=True):
    """One row per (cell, origin k0): features known at k0, target = fade per 100 cycles over the
    next `horizon` cycles (robust: medians of 3-cycle windows)."""
    rows = []
    vgrid = np.linspace(3.3, 3.9, 60)
    summ = cd.summary.set_index("cell")
    for cell in (cells or cd.cells):
        g = cd.cell(cell)
        g = g[~g.outlier & g.soh.notna()].reset_index(drop=True)
        if len(g) < first + horizon:
            continue
        ref_ci = int(g.Cycle_Index.iloc[min(2, len(g) - 1)])
        pulsed = bool(summ.loc[cell, "pulsed"])
        for k0 in range(first, int(g.k.max()) - horizon + 1, step):
            past = g[g.k <= k0]
            fut = g[(g.k > k0 + horizon - 2) & (g.k <= k0 + horizon + 1)]
            if len(past) < 10 or fut.empty:
                continue
            s_now = past.soh.iloc[-3:].median()
            last20 = past[past.k > k0 - 20]
            last40 = past[past.k > k0 - 40]
            slope = stats.theilslopes(last20.soh, last20.k).slope * 100 if len(last20) > 4 else np.nan
            curv = np.polyfit(last40.k, last40.soh, 2)[0] * 1e4 if len(last40) > 8 else np.nan
            dq = (_dq_stats(cd, cell, ref_ci, int(past.Cycle_Index.iloc[-1]), vgrid)
                  if with_dq and not pulsed and cd.master is not None else (np.nan,) * 3)
            r0 = past.R0.dropna(); rct = past.Rct_ohm.dropna()
            rows.append(dict(cell=cell, group=summ.loc[cell, "group"], k0=k0, soh_now=s_now,
                             slope20=slope, curv40=curv, I_mean=past.I_mean.iloc[-10:].mean(),
                             ambient=past.ambient.iloc[-10:].mean(),
                             cutoff_V=float(summ.loc[cell, "cutoff_V"]) if summ.loc[cell, "cutoff_V"] is not None else np.nan,
                             dq_logvar=dq[0], dq_min=dq[1], dq_mean=dq[2],
                             R0_rel=r0.iloc[-1] / r0.iloc[:3].mean() if len(r0) > 3 else np.nan,
                             Rct_rel=rct.iloc[-1] / rct.iloc[:3].mean() if len(rct) > 3 else np.nan,
                             target=(s_now - fut.soh.median()) / horizon * 100,
                             soh_future=fut.soh.median(), horizon=horizon))
    return pd.DataFrame(rows)
