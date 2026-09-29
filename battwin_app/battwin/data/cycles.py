"""Canonical CycleTable: one row per (cell, discharge cycle). Every downstream module uses it.

Raw samples are only re-read for voltage-curve work (ICA/DVA, half-cell fitting, ECM, ΔQ(V)).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from .. import config as C
from . import registry
from .quality import hampel, robust_bol, detect_eol
from .soh import rate_normalise


def _trapz(y, x):
    return float(np.trapezoid(y, x)) if len(x) > 1 else 0.0


def _cumq(t, i):
    """Cumulative discharged charge (Ah) from |I|."""
    a = np.abs(i)
    dq = np.diff(t) * (a[1:] + a[:-1]) / 2 / 3600
    return np.concatenate([[0.0], np.cumsum(np.clip(dq, 0, None))])


def _groups(df):
    keys = df[["Cell_ID", "Cycle_Index"]].to_numpy()
    if len(df) == 0:
        return
    change = np.nonzero((keys[1:] != keys[:-1]).any(axis=1))[0] + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(df)]])
    for s, e in zip(starts, ends):
        yield keys[s][0], int(keys[s][1]), s, e


def _discharge_features(d: pd.DataFrame) -> pd.DataFrame:
    t_all, v_all, i_all = d["Time_s"].to_numpy(), d["Voltage_V"].to_numpy(), d["Current_A"].to_numpy()
    T_all, c_all = d["Temp_C"].to_numpy(), d["Capacity_Ah"].to_numpy()
    amb_all = d["Ambient_C"].to_numpy() if "Ambient_C" in d else None
    rows = []
    for cell, cyc, s, e in _groups(d):
        t, v, i, T = t_all[s:e], v_all[s:e], i_all[s:e], T_all[s:e]
        a = np.abs(i)
        loaded = a > 0.1
        if loaded.sum() < 3:
            continue
        cap_rec = np.nanmax(c_all[s:e]) if np.isfinite(c_all[s:e]).any() else np.nan
        cap_int = _cumq(t[loaded], i[loaded])[-1]
        first = int(np.argmax(loaded))
        li = np.nonzero(loaded)[0]
        span = li[-1] - li[0] + 1
        idle_frac = 1 - loaded[li[0]:li[-1] + 1].sum() / span
        r0 = np.nan
        if first > 0 and a[first] > 0.5:
            r0 = (np.nanmax(v[:first]) - v[first]) / a[first]
            r0 = r0 if 0 < r0 < 1.0 else np.nan
        rows.append(dict(
            Cell_ID=cell, Cycle_Index=cyc,
            capacity=cap_rec if np.isfinite(cap_rec) and cap_rec > 0 else cap_int,
            I_mean=float(a[loaded].mean()), I_rms=float(np.sqrt((i[li[0]:li[-1] + 1] ** 2).mean())),
            pulsed=bool(idle_frac > 0.25),
            T_start=float(T[0]), T_mean=float(np.nanmean(T)), T_max=float(np.nanmax(T)),
            T_rise=float(np.nanmax(T) - T[0]),
            V_mean=float(np.nanmean(v[loaded])), V_min=float(np.nanmin(v[loaded])),
            duration_s=float(t[li[-1]] - t[li[0]]),
            energy_Wh=_trapz(np.abs(v * i)[loaded], t[loaded]) / 3600, R0=r0,
            ambient_obs=float(np.nanmean(amb_all[s:e])) if amb_all is not None else np.nan,
        ))
    return pd.DataFrame(rows)


def _charge_features(d: pd.DataFrame) -> pd.DataFrame:
    t_all, v_all = d["Time_s"].to_numpy(), d["Voltage_V"].to_numpy()
    rows = []
    for cell, cyc, s, e in _groups(d):
        t, v = t_all[s:e], v_all[s:e]
        if len(t) < 3:
            continue
        hit = np.nonzero(v >= 4.195)[0]
        cc = float(t[hit[0]] - t[0]) if hit.size else np.nan
        rows.append(dict(Cell_ID=cell, Cycle_Index=cyc, cc_time_s=cc,
                         cv_time_s=float(t[-1] - t[hit[0]]) if hit.size else np.nan))
    return pd.DataFrame(rows)


def _asof(left, right, cols):
    if right is None or len(right) == 0:
        for c in cols:
            left[c] = np.nan
        return left
    out = []
    for cell, g in left.groupby("Cell_ID", sort=False):
        r = right[right["Cell_ID"] == cell].sort_values("Cycle_Index")
        g = g.sort_values("Cycle_Index")
        if len(r) == 0:
            g = g.assign(**{c: np.nan for c in cols})
        else:
            g = pd.merge_asof(g, r[["Cycle_Index"] + cols], on="Cycle_Index", direction="backward")
            f = pd.merge_asof(g[["Cycle_Index"]], r[["Cycle_Index"] + cols], on="Cycle_Index",
                              direction="forward")
            for c in cols:   # before the first measurement: use the first one
                g[c] = g[c].fillna(f[c])
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _ambient(cell, obs, t_start):
    if np.isfinite(obs):
        return obs
    amb = registry.info(cell)["ambient"]
    if isinstance(amb, list):
        return float(min(amb, key=lambda a: abs(a - t_start)))
    return float(amb) if amb is not None else np.nan


@dataclass
class CycleData:
    table: pd.DataFrame          # one row per discharge cycle
    summary: pd.DataFrame        # one row per cell
    eol_ah: float
    master: pd.DataFrame = field(repr=False, default=None)

    @property
    def cells(self) -> list[str]:
        return list(self.summary["cell"])

    def cell(self, cid: str) -> pd.DataFrame:
        return self.table[self.table["Cell_ID"] == cid].reset_index(drop=True)

    def curves(self, cid: str, cycle_indices=None, n: int = C.CURVE_POINTS) -> dict:
        return get_curves(self.master, cid, cycle_indices, n)


def build(master: pd.DataFrame, imp: pd.DataFrame | None, eol_ah: float = C.EOL_AH) -> CycleData:
    dis = master[master["Cycle_Type"] == "discharge"]
    ch = master[master["Cycle_Type"] == "charge"]
    tab = _discharge_features(dis)
    if tab.empty:
        raise ValueError("no discharge cycles found")
    chf = _charge_features(ch)
    tab = _asof(tab, chf, ["cc_time_s", "cv_time_s"])
    tab = _asof(tab, imp, ["Re_ohm", "Rct_ohm"])
    tab = tab.sort_values(["Cell_ID", "Cycle_Index"]).reset_index(drop=True)
    tab["k"] = tab.groupby("Cell_ID").cumcount() + 1
    tab["ambient"] = [_ambient(c, o, ts) for c, o, ts in zip(tab.Cell_ID, tab.ambient_obs, tab.T_start)]
    tab["group"] = tab["Cell_ID"].map(registry.group_of)
    parts, summ = [], []
    for cell, g in tab.groupby("Cell_ID", sort=True):
        g = g.copy()
        lo, hi = C.PLAUSIBLE_AH
        implaus = ~g["capacity"].between(lo, hi)
        out, _ = hampel(g["capacity"].where(~implaus))
        g["outlier"] = out | implaus.to_numpy()
        cap_n, rinfo = rate_normalise(g.k, g.capacity, g.I_mean, g.ambient, ~g.outlier)
        g["cap_norm"] = cap_n
        bol, crash = robust_bol(g["cap_norm"], g["outlier"])
        if crash:   # early bogus values: flag everything before the plausible level is reached
            early = (g.k <= 20) & ((g.cap_norm > bol * 1.08) | (g.cap_norm < bol * 0.85))
            g.loc[early, "outlier"] = True
        g["bol_ah"] = bol
        g["soh"] = g["cap_norm"] / bol
        g["soh_clean"] = g["soh"].where(~g["outlier"])
        eol_k, status = detect_eol(g.k, g.cap_norm, eol_ah, outlier=g.outlier)
        parts.append(g)
        info = registry.info(cell)
        summ.append(dict(cell=cell, group=info["group"], n_cycles=len(g), bol_ah=bol, crash_flag=crash,
                         eol_k=eol_k, eol_status=status, last_cap=float(g.cap_norm[~g.outlier].iloc[-1])
                         if (~g.outlier).any() else np.nan,
                         n_outliers=int(g.outlier.sum()), pulsed=bool(g.pulsed.mean() > 0.5)
                         or registry.is_pulsed(cell), cutoff_V=info["cutoff"],
                         ambient=float(g.ambient.median()), I_mean=float(g.I_mean.median()),
                         I_levels=int(np.unique(np.round(g.I_mean * 2) / 2).size),
                         **{k: v for k, v in rinfo.items() if k != "offsets"},
                         offsets=str({k: round(v, 4) for k, v in rinfo["offsets"].items()})))
    table = pd.concat(parts, ignore_index=True)
    table["eol_soh"] = eol_ah / table["bol_ah"]
    return CycleData(table, pd.DataFrame(summ), eol_ah, master)


def get_curves(master, cid, cycle_indices=None, n: int = C.CURVE_POINTS) -> dict:
    """{Cycle_Index: dict(t, v, i, q)} for discharge cycles, resampled to n points in time."""
    d = master[(master["Cell_ID"] == cid) & (master["Cycle_Type"] == "discharge")]
    if cycle_indices is not None:
        d = d[d["Cycle_Index"].isin(set(int(c) for c in cycle_indices))]
    out = {}
    for cyc, g in d.groupby("Cycle_Index", sort=True):
        t, v, i = g.Time_s.to_numpy(), g.Voltage_V.to_numpy(), g.Current_A.to_numpy()
        m = np.abs(i) > 0.1
        if m.sum() < 5:
            continue
        idx = np.nonzero(m)[0]
        t, v, i = t[idx[0]:idx[-1] + 1], v[idx[0]:idx[-1] + 1], i[idx[0]:idx[-1] + 1]
        q = _cumq(t, i)
        if n and len(t) > n:
            tg = np.linspace(t[0], t[-1], n)
            v, i, q, t = np.interp(tg, t, v), np.interp(tg, t, i), np.interp(tg, t, q), tg
        out[int(cyc)] = dict(t=t - t[0], v=v, i=np.abs(i), q=q)
    return out


def qv(curve: dict, vgrid: np.ndarray) -> np.ndarray:
    """Discharged charge as a function of voltage on a common grid (monotone part only)."""
    v, q = curve["v"], curve["q"]
    vm = np.minimum.accumulate(v)          # enforce monotone decrease
    order = np.argsort(vm)
    return np.interp(vgrid, vm[order], q[order], left=np.nan, right=np.nan)
