"""ECM twin with a dual-time-scale EKF.

Fast scale (samples within a discharge): scalar EKF on SOC with V = V_ref(SOC) - dR-drop.
Slow scale (cycles): Kalman filter on [Q, dQ/dk, R0]. Each cycle's voltage curve is condensed
into a least-squares pseudo-measurement of (Q, R0) whose covariance is inflated for correlated
voltage errors (the curve counts as N_EFF independent samples). A normalised-innovation-squared
(NIS) guard rejects inconsistent updates; three rejections in a row force a re-sync with inflated
covariance. A periodic capacity check (every CHECK_EVERY cycles) uses the measured capacity.
V_ref is the cell's own first valid discharge curve, so no OCV table is assumed. Invalid for
pulsed (square-wave) loads: the constant-current voltage model does not hold there.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2
from .base import Observer, Estimate

N_EFF = 3.0
CHECK_EVERY = 10


class ECMTwin(Observer):
    name = "ECM twin"
    uses_voltage = True

    def __init__(self, pulsed=False, check_every=CHECK_EVERY, n_eff=N_EFF):
        super().__init__()
        if pulsed:
            self.valid, self.reason = False, "constant-current voltage model invalid for pulsed load"
        self.check_every, self.n_eff = check_every, n_eff
        self.ref = None
        self.x = self.P = None
        self.bol = None
        self.k = 0
        self.rejects = self.consec = 0
        self.q_proc = np.diag([1e-7, 1e-9, 1e-7])
        self.guard = chi2.ppf(0.999, 2)
        self.last_nis = np.nan
        self.fast_nis = np.nan

    # --- reference curve -------------------------------------------------------------
    def _set_ref(self, obs):
        c = obs.curve
        q, v, i = c["q"], c["v"], c["i"]
        Q0 = q[-1]
        soc = 1 - q / Q0
        o = np.argsort(soc)
        self.ref = dict(soc=soc[o], v=v[o], I=float(np.median(i)), R0=0.08 if not np.isfinite(obs.R0) else obs.R0)
        self.x = np.array([Q0, -1e-3, self.ref["R0"]])
        self.P = np.diag([0.02**2, 1e-3**2, 0.02**2])
        self.bol = obs.bol_ah if np.isfinite(obs.bol_ah) else Q0

    def _vref(self, soc):
        return np.interp(soc, self.ref["soc"], self.ref["v"])

    def _model(self, th, q, i):
        Q, R0 = th
        return self._vref(1 - q / Q) - (i * R0 - self.ref["I"] * self.ref["R0"])

    # --- fast scale ---------------------------------------------------------------------
    def _fast_ekf(self, c, Q, R0):
        soc, p, r = 1.0, 1e-4, 0.01**2
        nis = []
        for j in range(1, len(c["t"])):
            dt = c["t"][j] - c["t"][j - 1]
            soc -= c["i"][j] * dt / 3600 / Q
            p += 1e-6
            h = (self._vref(soc + 1e-3) - self._vref(soc - 1e-3)) / 2e-3
            vp = self._vref(soc) - (c["i"][j] * R0 - self.ref["I"] * self.ref["R0"])
            s = h * p * h + r
            nu = c["v"][j] - vp
            nis.append(nu**2 / s)
            kgain = p * h / s
            soc += kgain * nu
            p *= (1 - kgain * h)
        return float(np.mean(nis)) if nis else np.nan

    # --- slow scale ---------------------------------------------------------------------
    def update(self, obs):
        if self.ref is None:
            if obs.curve is None or not self.valid:
                if np.isfinite(obs.cap_norm) and not obs.outlier and self.x is None:
                    self.x = np.array([obs.cap_norm, -1e-3, 0.08]); self.P = np.diag([0.02**2, 1e-3**2, 0.02**2])
                    self.bol = obs.bol_ah
                if self.x is None:
                    return Estimate(np.nan, np.nan)
            else:
                self._set_ref(obs)
                self.k = obs.k
                return self._estimate()
        steps = max(obs.k - self.k, 1)
        F = np.array([[1, steps, 0], [0, 1, 0], [0, 0, 1.0]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.q_proc * steps
        self.k = obs.k
        if self.valid and obs.curve is not None and self.ref is not None:
            c = obs.curve
            try:
                r = least_squares(lambda th: self._model(th, c["q"], c["i"]) - c["v"],
                                  [self.x[0], self.x[2]], bounds=([0.3, 0.0], [3.0, 1.0]))
                N = len(c["v"])
                s2 = max(r.fun @ r.fun / max(N - 2, 1), 1e-6)
                cov = np.linalg.pinv(r.jac.T @ r.jac) * s2 * N / self.n_eff
                cov += np.diag([0.004**2, 0.002**2])
                H = np.array([[1.0, 0, 0], [0, 0, 1.0]])
                nu = r.x - H @ self.x
                S = H @ self.P @ H.T + cov
                nis = float(nu @ np.linalg.solve(S, nu))
                self.last_nis = nis
                if nis > self.guard and self.consec < 3:
                    self.rejects += 1; self.consec += 1
                else:
                    if nis > self.guard:           # re-sync after repeated rejection
                        self.P = self.P * 10
                        S = H @ self.P @ H.T + cov
                    self.consec = 0
                    K = self.P @ H.T @ np.linalg.inv(S)
                    self.x = self.x + K @ nu
                    self.P = (np.eye(3) - K @ H) @ self.P
                self.fast_nis = self._fast_ekf(c, self.x[0], self.x[2])
            except Exception:
                self.rejects += 1
        do_check = (not self.valid) or self.ref is None or (obs.k % self.check_every == 0)
        if do_check and np.isfinite(obs.cap_norm) and not obs.outlier and np.isfinite(obs.soh):
            H = np.array([1.0, 0, 0])
            S = H @ self.P @ H + 0.01**2
            K = self.P @ H / S
            self.x = self.x + K * (obs.cap_norm - self.x[0])
            self.P = self.P - np.outer(K, H @ self.P)
        return self._estimate()

    def _estimate(self):
        return Estimate(float(self.x[0] / self.bol), float(np.sqrt(self.P[0, 0]) / self.bol),
                        dict(R0=float(self.x[2]), nis=self.last_nis, fast_nis=self.fast_nis,
                             rejects=self.rejects))

    def forecast_moments(self, hs):
        if self.x is None:
            return np.full(len(hs), np.nan), np.full(len(hs), np.nan)
        hs = np.asarray(hs, float)
        m = (self.x[0] + self.x[1] * hs) / self.bol
        var = self.P[0, 0] + 2 * hs * self.P[0, 1] + hs**2 * self.P[1, 1] + self.q_proc[0, 0] * hs
        return m, np.sqrt(var) / self.bol
