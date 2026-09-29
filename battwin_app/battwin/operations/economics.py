"""Economic parameters (generic currency units, CU) and per-day cash flows.

Profit = Revenue - EnergyCost - MaintenanceCost - PerformanceLoss
       = J_op - J_maint,   J_op = Revenue - EnergyCost,   J_maint = MaintenanceCost + PerformanceLoss
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np

V_NOM = 3.6


@dataclass
class Economics:
    # Defaults are scaled to the accelerated laboratory ageing of the NASA cells (life of order
    # 100-200 cycles), so that operation is profitable and the trade-offs are visible. Set your own.
    price_sell: float = 1.50        # CU/kWh delivered (value of the service)
    price_buy: float = 0.30         # CU/kWh charged
    battery_cost: float = 0.40      # CU per replacement (cell + labour)
    downtime_days: float = 1.0      # days lost per replacement
    demand_kwh_day: float = 0.012   # contracted daily throughput
    penalty: float = 3.0            # CU per kWh of unmet demand
    hours_per_day: float = 6.0      # available cycling hours per day
    charge_current: float = 1.5
    eta_charge: float = 0.95
    discount_year: float = 0.05
    salvage_frac: float = 0.2       # residual value of a used cell at horizon end (fraction of cost)

    def dict(self):
        return asdict(self)


def daily_flows(soh, current, plant, eco: Economics):
    """Vectorised over soh. Returns dict of per-day quantities for constant `current`."""
    soh = np.asarray(soh, float)
    cap = soh * plant.bol_ah
    R = plant.resistance(soh)
    t_cycle = cap / current + cap / eco.charge_current + 0.5
    n = eco.hours_per_day / t_cycle
    e_out = cap * np.clip(V_NOM - current * R, 0.5, None) / 1000        # kWh per cycle
    e_in = cap * (V_NOM + eco.charge_current * R) / 1000 / eco.eta_charge
    rev = n * e_out * eco.price_sell
    ecost = n * e_in * eco.price_buy
    perf = eco.penalty * np.maximum(eco.demand_kwh_day - n * e_out, 0.0)
    dsoh = n * plant.fade_per_cycle(soh, current)
    return dict(cycles=n, revenue=rev, energy_cost=ecost, perf_loss=perf, dsoh=dsoh, energy=n * e_out)
