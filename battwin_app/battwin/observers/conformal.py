"""Online split-conformal calibration of forecast bands, per model and horizon."""
from __future__ import annotations
from collections import defaultdict
import numpy as np

MIN_RESIDUALS = 8


class Conformal:
    def __init__(self, alpha=0.1, window=200):
        self.alpha, self.window = alpha, window
        self.res = defaultdict(list)

    def add(self, key, residual):
        if np.isfinite(residual):
            r = self.res[key]; r.append(abs(residual))
            if len(r) > self.window:
                del r[0]

    def radius(self, key):
        r = self.res.get(key, [])
        n = len(r)
        if n < MIN_RESIDUALS:
            return np.nan
        level = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        return float(np.quantile(r, level))

    def band(self, key, mean, lo, hi):
        """Conformal band if calibrated, else the model's own band."""
        q = self.radius(key)
        return (mean - q, mean + q, True) if np.isfinite(q) else (lo, hi, False)
