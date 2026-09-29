"""Scenario planner and grid study: fixed (current, replacement-SOH) policies vs the DP optimum."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .dp import simulate, fixed_policy, dp_policy, solve, CURRENTS


def summarise(ledger: pd.DataFrame) -> dict:
    return dict(profit_pv=float(ledger.profit_pv.sum()), profit=float(ledger.profit.sum()),
                revenue=float(ledger.revenue.sum()), energy_cost=float(ledger.energy_cost.sum()),
                maintenance=float(ledger.maintenance.sum()), perf_loss=float(ledger.perf_loss.sum()),
                J_op=float(ledger.J_op.sum()), J_maint=float(ledger.J_maint.sum()),
                replacements=int(ledger["replace"].sum()), energy_kwh=float(ledger.energy_kwh.sum()))


def grid_study(plant, eco, s0, years=1.0, period_days=1, currents=CURRENTS,
               thresholds=(0.65, 0.7, 0.75, 0.8, 0.85, 0.9)) -> pd.DataFrame:
    rows = []
    for I in currents:
        for th in thresholds:
            if th < plant.s_floor:
                continue
            led = simulate(plant, eco, s0, years, period_days, fixed_policy(I, th))
            rows.append(dict(current=I, replace_soh=th, **summarise(led)))
    return pd.DataFrame(rows)


def compare(plant, eco, s0, years=1.0, period_days=1, baseline=(2.0, None)):
    res = solve(plant, eco, years, period_days)
    led_dp = simulate(plant, eco, s0, years, period_days, dp_policy(res))
    base_th = baseline[1] if baseline[1] is not None else plant.s_floor + 1e-6
    led_base = simulate(plant, eco, s0, years, period_days, fixed_policy(baseline[0], base_th))
    grid = grid_study(plant, eco, s0, years, period_days)
    best = grid.loc[grid.profit_pv.idxmax()] if len(grid) else None
    return dict(dp=res, ledger_dp=led_dp, ledger_base=led_base, grid=grid, best_fixed=best,
                summary=pd.DataFrame([dict(policy="DP optimum", **summarise(led_dp)),
                                      dict(policy=f"Baseline {baseline[0]:g} A, run to floor", **summarise(led_base)),
                                      *([dict(policy=f"Best fixed {best.current:g} A, replace < {best.replace_soh:.2f}",
                                              **{k: best[k] for k in summarise(led_dp)})] if best is not None else [])]))
