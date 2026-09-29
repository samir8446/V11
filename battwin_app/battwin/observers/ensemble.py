"""Live ensemble: members weighted by recent one-step error; failing members excluded."""
from __future__ import annotations
import numpy as np


class EnsembleWeights:
    def __init__(self, names, lam=0.85, fail_factor=3.0, warmup=5):
        self.err = {n: np.nan for n in names}
        self.n = {n: 0 for n in names}
        self.lam, self.fail_factor, self.warmup = lam, fail_factor, warmup
        self.excluded: dict[str, str] = {}

    def observe(self, name, abs_err):
        if not np.isfinite(abs_err):
            return
        e = self.err[name]
        self.err[name] = abs_err**2 if not np.isfinite(e) else self.lam * e + (1 - self.lam) * abs_err**2
        self.n[name] += 1

    def weights(self, valid: dict[str, bool], finite: dict[str, bool]) -> dict[str, float]:
        self.excluded = {}
        ready = {n: v for n, v in self.err.items() if np.isfinite(v) and self.n[n] >= self.warmup}
        med = np.median(list(ready.values())) if ready else np.nan
        w = {}
        for n in self.err:
            if not valid.get(n, True):
                self.excluded[n] = "invalid model"
            elif not finite.get(n, True):
                self.excluded[n] = "no estimate"
            elif n in ready and np.isfinite(med) and ready[n] > self.fail_factor**2 * med:
                self.excluded[n] = "error > 3x median"
            else:
                w[n] = 1.0 / (self.err[n] + 1e-6) if n in ready else 1.0 / (np.nanmedian(list(ready.values()))
                                                                            + 1e-6 if ready else 1.0)
        tot = sum(w.values())
        return {n: v / tot for n, v in w.items()} if tot > 0 else {}
