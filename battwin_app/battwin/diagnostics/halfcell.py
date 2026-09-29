"""Half-cell OCV model and fitting for quantitative LLI / LAM_PE / LAM_NE.

Electrode OCVs are published parametric fits (literature references, not measured on the
NASA cells): graphite (Doyle/Fuller-type MCMB fit) and LiCoO2 (Ramadass et al. rational fit).
At 1C the fit is only weakly identifiable; always read the reported residual and parameter std.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq, least_squares

V_MAX = 4.2
Y_MIN = 0.45   # LiCoO2 fit is only valid above ~0.45
KAPPA = 0.25   # concentration overpotential coefficient (ohm), fixed


def u_neg(x):
    x = np.clip(np.asarray(x, float), 1e-3, 1.0)
    return (0.7222 + 0.1387 * x + 0.029 * np.sqrt(x) - 0.0172 / x + 0.0019 / x ** 1.5
            + 0.2808 * np.exp(0.9 - 15 * x) - 0.7984 * np.exp(0.4465 * x - 0.4108))


def u_pos(y):
    y = np.clip(np.asarray(y, float), Y_MIN, 0.999)
    n = -4.656 + 88.669 * y**2 - 401.119 * y**4 + 342.909 * y**6 - 462.471 * y**8 + 433.434 * y**10
    d = -1 + 18.933 * y**2 - 79.532 * y**4 + 37.311 * y**6 - 73.083 * y**8 + 95.96 * y**10
    return n / d


@dataclass
class Electrodes:
    Qp: float = 3.8      # positive electrode capacity (Ah)
    Qn: float = 2.8      # negative electrode capacity (Ah)
    Li: float = 3.9      # cyclable lithium (Ah-equivalent, y*Qp + x*Qn)
    R: float = 0.08      # lumped ohmic + polarisation resistance (ohm)


BOL = Electrodes()


def top_of_charge(e: Electrodes) -> tuple[float, float]:
    """Stoichiometries (x0, y0) at the end of CC-CV charge (OCV = V_MAX)."""
    def f(x0):
        return u_pos((e.Li - x0 * e.Qn) / e.Qp) - u_neg(x0) - V_MAX
    lo, hi = max(1e-3, (e.Li - 0.999 * e.Qp) / e.Qn), min(1.0, (e.Li - Y_MIN * e.Qp) / e.Qn)
    if lo >= hi:
        return hi, (e.Li - hi * e.Qn) / e.Qp
    try:
        x0 = brentq(f, lo + 1e-6, hi - 1e-6)
    except ValueError:
        x0 = hi if f(hi) < 0 else lo
    return x0, (e.Li - x0 * e.Qn) / e.Qp


def ocv_of_q(e: Electrodes, q):
    x0, y0 = top_of_charge(e)
    q = np.asarray(q, float)
    return u_pos(y0 + q / e.Qp) - u_neg(x0 - q / e.Qn)


def discharge(e: Electrodes, current: float, cutoff: float, n: int = 200):
    """Simulated constant-current discharge. Returns (q [Ah], V [V], capacity [Ah])."""
    x0, y0 = top_of_charge(e)
    qlim = max(min(x0 * e.Qn, (0.999 - y0) * e.Qp), 1e-3)
    q = np.linspace(0, qlim * 0.999, 700)
    v = (u_pos(y0 + q / e.Qp) - u_neg(x0 - q / e.Qn)
         - current * (e.R + KAPPA * (q / qlim) ** 6))
    below = np.nonzero(v < cutoff)[0]
    cap = q[below[0]] if below.size and below[0] > 0 else q[-1]
    qq = np.linspace(0, cap, n)
    return qq, np.interp(qq, q, v), float(cap)


def profile(e: Electrodes, n: int = 400):
    """Dense (q, OCV, concentration shape) table for fast repeated evaluation."""
    x0, y0 = top_of_charge(e)
    qlim = max(min(x0 * e.Qn, (0.999 - y0) * e.Qp), 1e-3)
    q = np.linspace(0, qlim * 0.999, n)
    return q, u_pos(y0 + q / e.Qp) - u_neg(x0 - q / e.Qn), KAPPA * (q / qlim) ** 6


def overpotential(e: Electrodes, q, current):
    """Ohmic drop plus a concentration term rising towards the end of discharge
    (makes capacity rate-dependent, as observed on the NASA cells)."""
    x0, y0 = top_of_charge(e)
    qlim = max(min(x0 * e.Qn, (0.999 - y0) * e.Qp), 1e-3)
    return current * (e.R + KAPPA * (np.asarray(q, float) / qlim) ** 6)


def mechanisms(e: Electrodes, ref: Electrodes = BOL) -> dict:
    return {"LLI": 1 - e.Li / ref.Li, "LAM_PE": 1 - e.Qp / ref.Qp, "LAM_NE": 1 - e.Qn / ref.Qn}


@dataclass
class HalfCellFit:
    electrodes: Electrodes
    rmse_mV: float
    std: dict
    success: bool


def fit_curve(q, v, current: float, init: Electrodes | None = None) -> HalfCellFit:
    """Fit (Qp, Qn, Li, R) to one discharge curve V(q)."""
    q, v = np.asarray(q, float), np.asarray(v, float)
    m = np.isfinite(q) & np.isfinite(v)
    q, v = q[m], v[m]
    init = init or BOL

    def resid(p):
        e = Electrodes(*p)
        x0, y0 = top_of_charge(e)
        qlim = max(min(x0 * e.Qn, (0.999 - y0) * e.Qp), 1e-3)
        model = (u_pos(y0 + q / e.Qp) - u_neg(x0 - q / e.Qn)
                 - current * (e.R + KAPPA * (q / qlim) ** 6))
        return (model - v) * 1000

    lb, ub = [2.0, 1.5, 2.0, 0.0], [6.0, 5.0, 6.0, 0.6]
    starts = [init] + ([BOL] if init is not BOL else [])
    try:
        fits = [least_squares(resid, np.clip([s.Qp, s.Qn, s.Li, s.R], lb, ub), bounds=(lb, ub),
                              x_scale=[0.5, 0.5, 0.5, 0.05]) for s in starts]
        r = min(fits, key=lambda f: f.cost)          # multi-start guards against local minima
        dof = max(len(q) - 4, 1)
        s2 = (r.fun @ r.fun) / dof
        try:
            cov = np.linalg.pinv(r.jac.T @ r.jac) * s2
            sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            sd = np.full(4, np.nan)
        return HalfCellFit(Electrodes(*r.x), float(np.sqrt(np.mean(r.fun**2))),
                           dict(zip(["Qp", "Qn", "Li", "R"], sd)), bool(r.success))
    except Exception:  # honest failure, never silent
        return HalfCellFit(init, float("nan"), {}, False)


def fit_series(curves) -> list[dict]:
    """Fit a sequence of (k, q, v, I); mechanisms relative to the first successful fit (cell's own BOL)."""
    out, prev, ref = [], None, None
    for k, q, v, cur in curves:
        f = fit_curve(q, v, cur, prev)
        if ref is None and f.success:
            ref = f.electrodes
        if f.success:
            prev = f.electrodes
        row = {"k": k, "rmse_mV": f.rmse_mV, "success": f.success, **vars(f.electrodes)}
        row.update(mechanisms(f.electrodes, ref or f.electrodes))
        out.append(row)
    return out
