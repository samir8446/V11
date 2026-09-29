"""Dynamic programming for joint operation (discharge current) and replacement.

Stage = one period of `period_days`; state = SOH on a grid; action = (current, replace?).
Replacement happens at the start of a period (cost + downtime), then the new cell operates.
Below the hard floor s_floor the cell must be replaced. Backward induction with linear
interpolation; terminal value = salvage of the remaining useful SOH.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .economics import Economics, daily_flows

CURRENTS = (1.0, 2.0, 3.0, 4.0)


@dataclass
class DPResult:
    grid: np.ndarray
    currents: tuple
    policy_I: np.ndarray        # (T, S) chosen current
    policy_rep: np.ndarray      # (T, S) bool replace
    value: np.ndarray           # (T+1, S)
    period_days: int


def _period(s, I, plant, eco, days):
    """Simulate `days` at constant I from each s (vectorised). Returns end soh and cash totals."""
    s = np.array(s, float)
    tot = dict(revenue=0.0, energy_cost=0.0, perf_loss=0.0, energy=0.0)
    tot = {k: np.zeros_like(s) for k in tot}
    for _ in range(days):
        f = daily_flows(s, I, plant, eco)
        for k in tot:
            tot[k] += f[k]
        s = s - f["dsoh"]
    return np.clip(s, 0.3, 1.0), tot


def solve(plant, eco: Economics, years: float = 1.0, period_days: int = 1, currents=CURRENTS,
          n_grid: int = 121) -> DPResult:
    T = int(np.ceil(years * 365 / period_days))
    grid = np.linspace(plant.s_floor - 0.05, 1.0, n_grid)
    gamma = (1 + eco.discount_year) ** (-period_days / 365)
    V = np.zeros((T + 1, n_grid))
    V[T] = eco.salvage_frac * eco.battery_cost * np.clip((grid - plant.s_floor) / (1 - plant.s_floor), 0, 1)
    pol_I = np.zeros((T, n_grid)); pol_rep = np.zeros((T, n_grid), bool)
    # precompute transitions (stationary)
    trans = {}
    down = eco.downtime_days / period_days
    for I in currents:
        s_end, c = _period(grid, I, plant, eco, period_days)
        r_keep = c["revenue"] - c["energy_cost"] - c["perf_loss"]
        s_new, cn = _period(np.array([1.0]), I, plant, eco, period_days)
        r_new = (1 - down) * (cn["revenue"] - cn["energy_cost"])[0] - cn["perf_loss"][0] \
            - down * eco.penalty * eco.demand_kwh_day * period_days - eco.battery_cost
        trans[I] = (s_end, r_keep, float(s_new[0]), float(r_new))
    for t in range(T - 1, -1, -1):
        best = np.full(n_grid, -np.inf)
        for I, (s_end, r_keep, s_new, r_new) in trans.items():
            v_keep = r_keep + gamma * np.interp(s_end, grid, V[t + 1])
            v_keep = np.where(grid < plant.s_floor, -np.inf, v_keep)
            v_rep = r_new + gamma * np.interp(s_new, grid, V[t + 1])
            for rep, v in ((False, v_keep), (True, np.full(n_grid, v_rep))):
                better = v > best + 1e-12
                best = np.where(better, v, best)
                pol_I[t] = np.where(better, I, pol_I[t])
                pol_rep[t] = np.where(better, rep, pol_rep[t])
        V[t] = best
    return DPResult(grid, tuple(currents), pol_I, pol_rep, V, period_days)


def simulate(plant, eco: Economics, s0: float, years: float, period_days: int, decide) -> pd.DataFrame:
    """Forward simulation with decide(t, s) -> (I, replace). Returns a per-period ledger."""
    T = int(np.ceil(years * 365 / period_days))
    gamma = (1 + eco.discount_year) ** (-period_days / 365)
    s, rows = float(s0), []
    down = eco.downtime_days / period_days
    for t in range(T):
        I, rep = decide(t, s)
        if s < plant.s_floor:
            rep = True
        maint = eco.battery_cost if rep else 0.0
        if rep:
            s = 1.0
        s_end, c = _period(np.array([s]), I, plant, eco, period_days)
        f = 1 - down if rep else 1.0
        rev, ec = f * c["revenue"][0], f * c["energy_cost"][0]
        perf = c["perf_loss"][0] + (down * eco.penalty * eco.demand_kwh_day * period_days if rep else 0.0)
        rows.append(dict(period=t, day=t * period_days, soh_start=s, current=I, replace=rep,
                         revenue=rev, energy_cost=ec, maintenance=maint, perf_loss=perf,
                         J_op=rev - ec, J_maint=maint + perf, profit=rev - ec - maint - perf,
                         discount=gamma**t, energy_kwh=c["energy"][0]))
        s = float(s_end[0])
    df = pd.DataFrame(rows)
    df["profit_pv"] = df.profit * df.discount
    return df


def dp_policy(res: DPResult):
    def decide(t, s):
        t = min(t, res.policy_I.shape[0] - 1)
        j = int(np.clip(np.searchsorted(res.grid, s), 0, len(res.grid) - 1))
        return float(res.policy_I[t, j]), bool(res.policy_rep[t, j])
    return decide


def fixed_policy(current: float, replace_below: float):
    return lambda t, s: (current, s < replace_below)
