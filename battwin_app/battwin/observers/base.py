"""Common interface for all live (cycle-by-cycle) models."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import norm


@dataclass
class CycleObs:
    cell: str
    k: int
    soh: float            # rate-normalised SOH, NaN if outlier / not measured this cycle
    cap_norm: float
    current: float
    temperature: float    # measured mean cell temperature (falls back to ambient)
    ambient: float
    bol_ah: float
    R0: float = np.nan
    Re: float = np.nan
    Rct: float = np.nan
    curve: dict | None = None
    outlier: bool = False


@dataclass
class Estimate:
    soh: float
    std: float
    extras: dict = field(default_factory=dict)


class Observer:
    name = "base"
    uses_voltage = False

    def __init__(self):
        self.valid, self.reason = True, ""

    def update(self, obs: CycleObs) -> Estimate:   # pragma: no cover - interface
        raise NotImplementedError

    def forecast(self, hs, alpha: float = 0.1):
        """(mean, lo, hi) SOH at k_now + hs. Default: Gaussian from forecast_moments."""
        m, s = self.forecast_moments(np.asarray(hs, float))
        z = norm.ppf(1 - alpha / 2)
        return m, m - z * s, m + z * s

    def forecast_moments(self, hs):  # pragma: no cover - interface
        raise NotImplementedError


def weighted_quantile(x, w, q):
    """Column-wise weighted quantiles; x: (N, H), w: (N,)."""
    idx = np.argsort(x, axis=0)
    xs = np.take_along_axis(x, idx, axis=0)
    ws = w[idx]
    cw = np.cumsum(ws, axis=0)
    cw /= cw[-1]
    out = []
    for qq in np.atleast_1d(q):
        j = np.argmax(cw >= qq, axis=0)
        out.append(xs[j, np.arange(x.shape[1])])
    return out


def systematic_resample(w, rng):
    n = len(w)
    pos = (rng.random() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w), pos), n - 1)
