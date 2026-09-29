"""Incremental capacity (dQ/dV) and differential voltage (dV/dQ) analysis."""
from __future__ import annotations
import numpy as np
from scipy.signal import savgol_filter


def _sg(y, w=9, o=2):
    w = min(w, len(y) - (1 - len(y) % 2))
    return savgol_filter(y, w, o) if w > o + 2 else y


def ica(curve: dict, n: int = 120):
    """Return (V grid, dQ/dV) from a discharge curve (constant-current part)."""
    v, q = np.minimum.accumulate(curve["v"]), curve["q"]
    lo, hi = np.nanmin(v) + 0.02, np.nanmax(v) - 0.02
    if hi - lo < 0.2:
        return np.array([]), np.array([])
    vg = np.linspace(lo, hi, n)
    order = np.argsort(v)
    qg = np.interp(vg, v[order], q[order])
    return vg, -_sg(np.gradient(qg, vg))


def dva(curve: dict, n: int = 120):
    """Return (Q grid, dV/dQ)."""
    v, q = curve["v"], curve["q"]
    if q[-1] <= 0.1:
        return np.array([]), np.array([])
    qg = np.linspace(q[0] + 0.02 * q[-1], q[-1] * 0.98, n)
    vg = np.interp(qg, q, v)
    return qg, _sg(np.gradient(vg, qg))


def ica_peak(curve: dict) -> tuple[float, float]:
    vg, d = ica(curve)
    if d.size == 0:
        return np.nan, np.nan
    j = int(np.nanargmax(d))
    return float(vg[j]), float(d[j])
