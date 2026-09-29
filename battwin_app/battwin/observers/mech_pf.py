"""Mechanistic particle filter: SEI, lithium plating and LAM kinetics driven by measured T and I.

  dL_sei = a  Arr(T) / (2 sqrt k)                                  (LLI, diffusion-limited SEI)
  dL_pl  = b (I/2) exp((15 - T)/7) min(1 + ((L_lam - 0.05)+/0.015)^3, 8)  (LLI, plating; LAM lowers N/P)
  dL_lam = c (I/2)^1.5                                             (LAM)
  SOH    = s0 - (L_sei + L_pl) - W_LAM L_lam
Mechanism shares are posterior means of each term's contribution to capacity loss. They are
identifiable only through their different dependence on k, T and I; read them as model-based
attributions, to be cross-checked with half-cell fitting.
"""
from __future__ import annotations
import numpy as np
from .base import Observer, Estimate, weighted_quantile, systematic_resample

W_LAM = 0.3
AMP_MAX = 8.0   # cap on the LAM->plating coupling (keeps forecasts bounded)


def _arr(T, Ea=30e3):
    return np.exp(Ea / 8.314 * (1 / 297.15 - 1 / (np.asarray(T) + 273.15)))


class MechanisticPF(Observer):
    name = "Mechanistic PF"

    def __init__(self, n=800, sigma=0.008, seed=1):
        super().__init__()
        r = self.rng = np.random.default_rng(seed)
        self.a = np.exp(r.normal(np.log(0.012), 0.8, n))
        self.b = np.exp(r.normal(np.log(1e-4), 1.3, n))
        self.c = np.exp(r.normal(np.log(3e-4), 1.0, n))
        self.s0 = r.normal(1.0, 0.01, n)
        self.sei, self.pl, self.lam = np.zeros(n), np.zeros(n), np.zeros(n)
        self.w = np.full(n, 1.0 / n)
        self.sigma, self.k = sigma, 0
        self.hist_IT: list[tuple[float, float]] = []

    @staticmethod
    def _step(k, I, T, a, b, c, sei, pl, lam):
        i2 = max(I, 0.05) / 2
        sei = sei + a * _arr(T) / (2 * np.sqrt(max(k, 1)))
        amp = np.minimum(1 + (np.maximum(lam - 0.05, 0) / 0.015) ** 3, AMP_MAX)
        pl = pl + b * i2 * np.exp((15 - T) / 7) * amp
        lam = lam + c * i2**1.5
        return sei, pl, lam

    def _soh(self, sei, pl, lam):
        return np.clip(self.s0 - sei - pl - W_LAM * lam, 0.0, 1.1)

    def update(self, obs):
        T = obs.temperature if np.isfinite(obs.temperature) else obs.ambient
        I = obs.current if np.isfinite(obs.current) else 2.0
        self.hist_IT.append((I, T))
        for kk in range(self.k + 1, obs.k + 1):      # also covers skipped cycles
            self.sei, self.pl, self.lam = self._step(kk, I, T, self.a, self.b, self.c,
                                                     self.sei, self.pl, self.lam)
        self.k = obs.k
        pred = self._soh(self.sei, self.pl, self.lam)
        if np.isfinite(obs.soh):
            r = (obs.soh - pred) / self.sigma
            lw = np.log(self.w + 1e-300) - 0.5 * np.minimum(r**2, 50)
            w = np.exp(lw - lw.max()); self.w = w / w.sum()
            if 1 / (self.w**2).sum() < len(self.w) / 2:
                i = systematic_resample(self.w, self.rng); n = len(i)
                j = lambda x, s: x[i] * np.exp(self.rng.normal(0, s, n))
                self.a, self.b, self.c = j(self.a, 0.04), j(self.b, 0.06), j(self.c, 0.04)
                self.s0 = self.s0[i] + self.rng.normal(0, 0.001, n)
                self.sei, self.pl, self.lam = self.sei[i], self.pl[i], self.lam[i]
                self.w = np.full(n, 1.0 / n)
            pred = self._soh(self.sei, self.pl, self.lam)
        m = float(self.w @ pred)
        return Estimate(m, float(np.sqrt(self.w @ (pred - m) ** 2)), self.shares())

    def shares(self) -> dict:
        tot = self.sei + self.pl + W_LAM * self.lam + 1e-12
        return dict(share_sei=float(self.w @ (self.sei / tot)), share_plating=float(self.w @ (self.pl / tot)),
                    share_lam=float(self.w @ (W_LAM * self.lam / tot)),
                    LLI=float(self.w @ (self.sei + self.pl)), LAM=float(self.w @ self.lam))

    def _paths(self, hs, I=None, T=None):
        recent = np.array(self.hist_IT[-10:]) if self.hist_IT else np.array([[2.0, 24.0]])
        I = recent[:, 0].mean() if I is None else I
        T = recent[:, 1].mean() if T is None else T
        hs = np.asarray(hs, int)
        out = np.empty((len(self.w), len(hs)))
        sei, pl, lam = self.sei.copy(), self.pl.copy(), self.lam.copy()
        h_prev = 0
        for j in np.argsort(hs):
            for kk in range(self.k + h_prev + 1, self.k + hs[j] + 1):
                sei, pl, lam = self._step(kk, I, T, self.a, self.b, self.c, sei, pl, lam)
            h_prev = max(h_prev, hs[j])
            out[:, j] = self._soh(sei, pl, lam)
        return out

    def forecast(self, hs, alpha=0.1, I=None, T=None):
        P = self._paths(hs, I, T)
        lo, med, hi = weighted_quantile(P, self.w, [alpha / 2, 0.5, 1 - alpha / 2])
        return med, lo - 0.004, hi + 0.004   # median: robust to heavy-tailed knee particles

    def forecast_moments(self, hs):
        P = self._paths(hs)
        m = self.w @ P
        return m, np.sqrt(self.w @ (P - m) ** 2)
