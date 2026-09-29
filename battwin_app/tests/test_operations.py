import numpy as np
from battwin.operations.plant import calibrate, Plant
from battwin.operations.economics import Economics, daily_flows
from battwin.operations.dp import solve, simulate, dp_policy, fixed_policy
from battwin.operations.scenarios import compare, grid_study


def test_calibration_has_sources(synth, cells):
    cd, _ = synth
    p = calibrate(cd, cells["reference"])
    assert p.r_ref > 0 and {"r_ref", "alpha", "resistance", "bol_ah"} <= set(p.sources)


def test_aggressive_current_degrades_faster():
    p = Plant(alpha=1.2)
    f1 = daily_flows(0.9, 1.0, p, Economics()); f4 = daily_flows(0.9, 4.0, p, Economics())
    assert f4["dsoh"] > f1["dsoh"] and f4["revenue"] > f1["revenue"]


def test_dp_beats_every_fixed_policy():
    p, eco = Plant(), Economics()
    res = solve(p, eco, years=0.5)
    dp = simulate(p, eco, 1.0, 0.5, 1, dp_policy(res)).profit_pv.sum()
    g = grid_study(p, eco, 1.0, years=0.5)
    assert dp >= g.profit_pv.max() - 0.02 * abs(g.profit_pv.max())


def test_floor_forces_replacement():
    p, eco = Plant(), Economics()
    led = simulate(p, eco, p.s_floor - 0.01, 0.1, 1, fixed_policy(2.0, 0.0))
    assert led["replace"].iloc[0]


def test_compare_ledger_consistent(synth, cells):
    cd, _ = synth
    r = compare(calibrate(cd, cells["hot"]), Economics(), 0.95, years=0.3)
    s = r["summary"].iloc[0]
    assert abs(s.profit - (s.revenue - s.energy_cost - s.maintenance - s.perf_loss)) < 1e-9
    assert abs(s.profit - (s.J_op - s.J_maint)) < 1e-9
