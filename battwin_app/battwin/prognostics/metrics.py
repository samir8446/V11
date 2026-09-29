"""Honest accuracy metrics. R² is never clipped; failures are reported, not dropped."""
from __future__ import annotations
import numpy as np


def rmse(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2))) if m.any() else np.nan


def mae(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(np.abs(y[m] - p[m]))) if m.any() else np.nan


def r2(y, p):
    """Unclipped coefficient of determination (can be very negative)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3:
        return np.nan
    ss = ((y[m] - y[m].mean()) ** 2).sum()
    return float(1 - ((y[m] - p[m]) ** 2).sum() / ss) if ss > 0 else np.nan


def coverage(y, lo, hi):
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    m = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m]))) if m.any() else np.nan


def fade_skill(y_now, y_true, y_pred):
    """1 - MSE(forecast) / MSE(persistence). >0 means better than 'no further fade'."""
    a, b = rmse(y_true, y_pred), rmse(y_true, y_now)
    return float(1 - a**2 / b**2) if np.isfinite(a) and b and np.isfinite(b) else np.nan


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h
