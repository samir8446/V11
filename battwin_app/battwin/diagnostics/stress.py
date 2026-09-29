"""Stress-factor regression of per-cell fade rate with confounding checks.

fade_rate (SOH per 100 cycles, Theil-Sen) ~ ambient + current + cut-off V + pulsed + temperature rise.
Reports standardised coefficients, SE, p-values, unclipped R², VIFs and flagged confounding.
With ~30 cells the power is limited; results carry an explicit power label.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

FACTORS = {"ambient": "Ambient T (°C)", "I_mean": "Discharge current (A)", "cutoff_V": "Cut-off V",
           "pulsed": "Pulsed load", "T_rise": "Temperature rise (K)"}


def fade_rates(cd, cells=None, max_k: int | None = None) -> pd.DataFrame:
    rows = []
    for c in (cells or cd.cells):
        g = cd.cell(c)
        g = g[~g.outlier]
        if max_k:
            g = g[g.k <= max_k]
        if len(g) < 10:
            continue
        slope = stats.theilslopes(g.soh, g.k).slope * 100
        s = cd.summary.set_index("cell").loc[c]
        rows.append(dict(cell=c, group=s.group, fade_per_100=-slope, ambient=g.ambient.median(),
                         I_mean=g.I_mean.median(), cutoff_V=float(s.cutoff_V) if s.cutoff_V is not None
                         else np.nan, pulsed=float(s.pulsed), T_rise=g.T_rise.median(), n=len(g)))
    return pd.DataFrame(rows)


def vif(X: np.ndarray) -> np.ndarray:
    out = []
    for j in range(X.shape[1]):
        y, A = X[:, j], np.delete(X, j, axis=1)
        A1 = np.column_stack([np.ones(len(A)), A])
        r = y - A1 @ np.linalg.lstsq(A1, y, rcond=None)[0]
        r2 = 1 - r @ r / (((y - y.mean()) ** 2).sum() + 1e-12)
        out.append(1 / max(1 - r2, 1e-9))
    return np.array(out)


def power_label(n: int, p: int) -> str:
    dof = n - p - 1
    if dof < 3:
        return "not estimable"
    if dof < 10:
        return "very low power"
    if dof < 25:
        return "low power"
    return "adequate"


def regress(rates: pd.DataFrame, factors=None) -> dict:
    factors = [f for f in (factors or list(FACTORS)) if f in rates and rates[f].std() > 0]
    d = rates.dropna(subset=factors + ["fade_per_100"])
    n, p = len(d), len(factors)
    res = dict(n=n, factors=factors, power=power_label(n, p))
    if n - p - 1 < 2 or p == 0:
        res.update(ok=False, reason=f"n={n} cells is too few for {p} factors")
        return res
    X = d[factors].to_numpy(float)
    Z = (X - X.mean(0)) / X.std(0)
    y = d["fade_per_100"].to_numpy(float)
    A = np.column_stack([np.ones(n), Z])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ beta
    dof = n - p - 1
    s2 = r @ r / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    tval = beta / se
    pval = 2 * stats.t.sf(np.abs(tval), dof)
    r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    vifs = vif(Z)
    corr = pd.DataFrame(np.corrcoef(Z.T), index=factors, columns=factors)
    flags = [f"{a} ~ {b} (r={corr.loc[a, b]:+.2f})" for i, a in enumerate(factors)
             for b in factors[i + 1:] if abs(corr.loc[a, b]) > 0.7]
    coef = pd.DataFrame(dict(factor=[FACTORS[f] for f in factors], std_coef=beta[1:], se=se[1:],
                             p_value=pval[1:], vif=vifs))
    coef["confounded"] = coef["vif"] > 5
    res.update(ok=True, coef=coef, r2=float(r2), adj_r2=float(1 - (1 - r2) * (n - 1) / dof),
               corr=corr, confounding=flags, residual_sd=float(np.sqrt(s2)))
    return res
