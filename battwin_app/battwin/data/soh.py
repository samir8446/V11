"""Rate-normalised capacity and SOH.

Capacity delivered to the cut-off depends on the discharge current (and ambient). Within a cell,
    ln C = s(k) + b_I (I - I_ref) + b_T (T_amb - T_ref) + e
is fitted robustly (hinge basis for s(k), Huber IRLS). The normalised capacity
C_norm = C exp(-b_I (I-I_ref) - b_T (T-T_ref)) is capacity referred to the cell's reference
current (2 A if the cell ever ran at 2 A, else its modal current). Constant-condition cells are
left unchanged, so the correction only acts where loads/ambients vary within a cell.
"""
from __future__ import annotations
import numpy as np


def _hinge_basis(k, n_knots=4):
    k = np.asarray(k, float)
    z = (k - k.min()) / max(np.ptp(k), 1.0)
    knots = np.quantile(z, np.linspace(0, 1, n_knots + 2)[1:-1])
    return np.column_stack([np.ones_like(z), z] + [np.maximum(z - t, 0) for t in knots])


def _huber_irls(X, y, iters=20, c=1.345):
    w = np.ones(len(y))
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        r = y - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        u = np.abs(r) / (c * s)
        w = np.where(u <= 1, 1.0, 1.0 / u)
        sw = np.sqrt(w)
        beta = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
    return beta


def _levels(x, step):
    return np.round(np.asarray(x, float) / step) * step


def rate_normalise(k, cap, current, ambient, valid):
    """Return (cap_norm, info). Level offsets (dummies) per current / ambient level are fitted
    jointly with a flexible ageing trend, so the correction need not be linear in I."""
    k, cap = np.asarray(k, float), np.asarray(cap, float)
    I = _levels(current, 0.5)
    T = np.round(np.asarray(ambient, float))
    good = np.asarray(valid) & np.isfinite(cap) & (cap > 0) & np.isfinite(I)
    vals, counts = np.unique(I[good], return_counts=True)
    i_ref = 2.0 if np.any(np.isclose(vals, 2.0)) else (float(vals[np.argmax(counts)]) if vals.size else 2.0)
    tv, tc = np.unique(T[good & np.isfinite(T)], return_counts=True)
    t_ref = float(tv[np.argmax(tc)]) if tv.size else 24.0
    i_lv = [v for v, c in zip(vals, counts) if c >= 5 and not np.isclose(v, i_ref)]
    t_lv = [v for v, c in zip(tv, tc) if c >= 5 and v != t_ref]
    info = dict(i_ref=i_ref, t_ref=t_ref, offsets={}, normalised=bool(i_lv or t_lv), rate_coef=0.0)
    if not info["normalised"] or good.sum() < 10:
        info["normalised"] = False
        return cap.copy(), info
    cols = [_hinge_basis(k[good])]
    names = []
    for v in i_lv:
        cols.append((I[good] == v).astype(float)[:, None]); names.append(("I", v))
    for v in t_lv:
        cols.append((T[good] == v).astype(float)[:, None]); names.append(("T", v))
    beta = _huber_irls(np.hstack(cols), np.log(cap[good]))
    off = beta[cols[0].shape[1]:]
    corr = np.zeros(len(cap))
    for (kind, v), b in zip(names, off):
        corr -= b * ((I == v) if kind == "I" else (T == v))
        info["offsets"][f"{kind}={v:g}"] = float(b)
    # summary slope d ln C / dI across identified current levels (for reporting only)
    pts = [(v, info["offsets"][f"I={v:g}"]) for v in i_lv]
    if pts:
        x = np.array([p[0] - i_ref for p in pts]); y = np.array([p[1] for p in pts])
        info["rate_coef"] = float((x @ y) / (x @ x))
    return cap * np.exp(corr), info
