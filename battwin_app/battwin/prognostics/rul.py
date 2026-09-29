"""End-of-life / remaining-useful-life from forecast paths."""
from __future__ import annotations
import numpy as np

RUL_GRID = np.arange(5, 605, 5)


def crossing(hs, path, threshold):
    """First horizon where path < threshold (linear interpolation); NaN if never within grid."""
    path = np.asarray(path, float)
    below = np.nonzero(path < threshold)[0]
    if below.size == 0:
        return np.nan
    j = below[0]
    if j == 0:
        return float(hs[0])
    x0, x1, y0, y1 = hs[j - 1], hs[j], path[j - 1], path[j]
    return float(x0 + (threshold - y0) * (x1 - x0) / (y1 - y0 + 1e-12))


def predict_eol(observer_or_forecast, k_now, threshold, alpha=0.1):
    """Return (eol_mean, eol_early, eol_late) cycles; early uses the lower band."""
    m, lo, hi = observer_or_forecast(RUL_GRID, alpha)
    return (k_now + crossing(RUL_GRID, m, threshold), k_now + crossing(RUL_GRID, lo, threshold),
            k_now + crossing(RUL_GRID, hi, threshold))
