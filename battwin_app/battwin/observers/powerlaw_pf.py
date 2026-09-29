"""Power-law particle filter with a fleet prior: SOH = s0 - L100 (k/100)^b.

The prior on (log L100, b) is fitted on other cells only (leakage-free). b > 1 captures knees.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from .base import Observer, Estimate, weighted_quantile, systematic_resample

DEFAULT_PRIOR = dict(logL_mu=np.log(0.12), logL_sd=0.9, b_mu=1.2, b_sd=0.6)


def fit_power_law(k, soh):
    k, soh = np.asarray(k, float), np.asarray(soh, float)
    m = np.isfinite(soh)
    k, soh = k[m], soh[m]
    if len(k) < 10:
        return None
    r = least_squares(lambda p: p[0] - np.exp(p[1]) * (k / 100) ** p[2] - soh, [1.0, np.log(0.1), 1.0],
                      bounds=([0.9, -8, 0.2], [1.1, 1, 5]))
    return r.x


def fleet_prior(table, exclude=None) -> dict:
    ps = []
    for c, g in table.groupby("Cell_ID"):
        if c == exclude:
            continue
        p = fit_power_law(g.k, g.soh_clean)
        if p is not None:
            ps.append(p)
    if len(ps) < 3:
        return dict(DEFAULT_PRIOR, n_cells=len(ps))
    ps = np.array(ps)
    return dict(logL_mu=float(ps[:, 1].mean()), logL_sd=float(max(ps[:, 1].std() * 1.5, 0.3)),
                b_mu=float(ps[:, 2].mean()), b_sd=float(max(ps[:, 2].std() * 1.5, 0.2)), n_cells=len(ps))


class PowerLawPF(Observer):
    name = "Power-law PF"

    def __init__(self, prior=None, n=600, sigma=0.008, seed=0):
        super().__init__()
        p = prior or DEFAULT_PRIOR
        self.rng = np.random.default_rng(seed)
        self.s0 = self.rng.normal(1.0, 0.01, n)
        self.logL = self.rng.normal(p["logL_mu"], p["logL_sd"], n)
        self.b = np.clip(self.rng.normal(p["b_mu"], p["b_sd"], n), 0.2, 5)
        self.w = np.full(n, 1.0 / n)
        self.sigma, self.k = sigma, 0

    def _pred(self, k):
        k = np.atleast_1d(np.asarray(k, float))
        return self.s0[:, None] - np.exp(self.logL)[:, None] * (k[None, :] / 100) ** self.b[:, None]

    def update(self, obs):
        self.k = obs.k
        if np.isfinite(obs.soh):
            r = (obs.soh - self._pred(obs.k)[:, 0]) / self.sigma
            lw = np.log(self.w + 1e-300) - 0.5 * np.minimum(r**2, 50)   # clipped: robust to outliers
            w = np.exp(lw - lw.max()); self.w = w / w.sum()
            if 1 / (self.w**2).sum() < len(self.w) / 2:
                i = systematic_resample(self.w, self.rng)
                n = len(i)
                self.s0 = self.s0[i] + self.rng.normal(0, 0.001, n)
                self.logL = self.logL[i] + self.rng.normal(0, 0.03, n)
                self.b = np.clip(self.b[i] + self.rng.normal(0, 0.02, n), 0.2, 5)
                self.w = np.full(n, 1.0 / n)
        p = self._pred(obs.k)[:, 0]
        m = float(self.w @ p)
        return Estimate(m, float(np.sqrt(self.w @ (p - m) ** 2)),
                        dict(b=float(self.w @ self.b), ess=float(1 / (self.w**2).sum())))

    def forecast(self, hs, alpha=0.1):
        P = self._pred(self.k + np.asarray(hs, float))
        m = self.w @ P
        lo, hi = weighted_quantile(P, self.w, [alpha / 2, 1 - alpha / 2])
        pad = 1.64 * self.sigma * 0.5
        return m, lo - pad, hi + pad

    def forecast_moments(self, hs):
        P = self._pred(self.k + np.asarray(hs, float))
        m = self.w @ P
        return m, np.sqrt(self.w @ (P - m) ** 2)
