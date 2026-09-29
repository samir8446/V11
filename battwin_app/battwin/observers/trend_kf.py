"""Adaptive trend Kalman filter: state [SOH, slope]; process noise scaled by recent NIS."""
from __future__ import annotations
import numpy as np
from .base import Observer, Estimate


class TrendKF(Observer):
    name = "Trend KF"

    def __init__(self, r_sd=0.005, q=2e-6, gate=16.0):
        super().__init__()
        self.R, self.q0, self.gate = r_sd**2, q, gate
        self.x, self.P, self.scale, self.nis_avg, self.rejects = None, None, 1.0, 1.0, 0
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])

    def _Q(self):
        return self.q0 * self.scale * np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]])

    def update(self, obs):
        if self.x is None:
            if not np.isfinite(obs.soh):
                return Estimate(np.nan, np.nan)
            self.x = np.array([obs.soh, -1e-3]); self.P = np.diag([0.01**2, 2e-3**2])
            return Estimate(obs.soh, 0.01)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self._Q()
        nis = np.nan
        if np.isfinite(obs.soh):
            S = self.P[0, 0] + self.R
            nu = obs.soh - self.x[0]
            nis = nu**2 / S
            if nis <= self.gate:
                K = self.P[:, 0] / S
                self.x = self.x + K * nu
                self.P = self.P - np.outer(K, self.P[0, :])
                self.nis_avg = 0.9 * self.nis_avg + 0.1 * nis
                self.scale = float(np.clip(self.nis_avg, 0.3, 20))
            else:
                self.rejects += 1
        return Estimate(float(self.x[0]), float(np.sqrt(self.P[0, 0])), dict(nis=nis, slope=self.x[1]))

    def forecast_moments(self, hs):
        if self.x is None:
            return np.full(len(hs), np.nan), np.full(len(hs), np.nan)
        m = self.x[0] + self.x[1] * hs
        q = self.q0 * self.scale
        var = (self.P[0, 0] + 2 * hs * self.P[0, 1] + hs**2 * self.P[1, 1]
               + q * (hs**3 / 3 + hs**2 / 2 + hs / 6))
        return m, np.sqrt(var)
