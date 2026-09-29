"""Hierarchical (empirical) Bayes: SOH = b0 + b1 z + b2 z^2, z = k/100.
Population prior N(mu, Sigma) from other cells; exact conjugate update per cycle."""
from __future__ import annotations
import numpy as np
from .base import Observer, Estimate

DEFAULT = dict(mu=np.array([1.0, -0.12, -0.02]), cov=np.diag([0.01**2, 0.08**2, 0.08**2]))


def _x(k):
    z = np.asarray(k, float) / 100
    return np.column_stack([np.ones_like(z), z, z**2])


def population_prior(table, exclude=None) -> dict:
    bs = []
    for c, g in table.groupby("Cell_ID"):
        if c == exclude:
            continue
        g = g[np.isfinite(g.soh_clean)]
        if len(g) < 10:
            continue
        bs.append(np.linalg.lstsq(_x(g.k), g.soh_clean.to_numpy(), rcond=None)[0])
    if len(bs) < 3:
        return dict(DEFAULT, n_cells=len(bs))
    bs = np.array(bs)
    return dict(mu=bs.mean(0), cov=np.cov(bs.T) * 1.5 + np.diag([1e-5, 1e-4, 1e-4]), n_cells=len(bs))


class HierBayes(Observer):
    name = "Hierarchical Bayes"

    def __init__(self, prior=None, sigma=0.008):
        super().__init__()
        p = prior or DEFAULT
        self.P = np.linalg.inv(p["cov"])          # precision
        self.h = self.P @ p["mu"]
        self.s2, self.k = sigma**2, 0

    def _post(self):
        cov = np.linalg.inv(self.P)
        return cov @ self.h, cov

    def update(self, obs):
        self.k = obs.k
        if np.isfinite(obs.soh):
            x = _x(obs.k)[0]
            mu, cov = self._post()
            if abs(obs.soh - x @ mu) < 5 * np.sqrt(x @ cov @ x + self.s2):   # outlier gate
                self.P = self.P + np.outer(x, x) / self.s2
                self.h = self.h + x * obs.soh / self.s2
        mu, cov = self._post()
        x = _x(obs.k)[0]
        return Estimate(float(x @ mu), float(np.sqrt(x @ cov @ x)))

    def forecast_moments(self, hs):
        mu, cov = self._post()
        X = _x(self.k + np.asarray(hs, float))
        return X @ mu, np.sqrt(np.einsum("ij,jk,ik->i", X, cov, X) + self.s2 * 0.25)
