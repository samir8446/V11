"""Mission 3 — joint operation and replacement policy (DP) on a calibrated battery; scenario planner."""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from battwin.operations.economics import Economics
from battwin.operations.dp import simulate, fixed_policy
from battwin.operations.scenarios import summarise
from battwin.study import answers
from ui import blocks, charts, cache


def render(ctx):
    cd, key, v, eol = ctx["cd"], ctx["key"], ctx["v"], ctx["eol"]
    st.markdown("## Operations & maintenance")
    st.markdown("<div class='bt-note'>Profit = Revenue − EnergyCost − MaintenanceCost − PerformanceLoss = "
                "J_op − J_maint. Currency units (CU) are generic.</div>", unsafe_allow_html=True)
    cell = st.selectbox("Battery to calibrate the plant on", ctx["cells"], key="op_cell")
    calibrated = st.toggle("Calibrate plant on this battery", value=True, key="op_cal")
    g = cd.cell(cell)
    s_meas = float(g.soh_clean.dropna().iloc[-5:].median()) if g.soh_clean.notna().any() else 1.0
    s0 = st.slider("Starting SOH", 0.6, 1.0, float(np.clip(round(s_meas, 2), 0.6, 1.0)), 0.01, key="op_s0",
                   help="Default: the battery's latest measured SOH.")
    years = st.select_slider("Horizon (years)", [0.25, 0.5, 1.0, 2.0], value=1.0, key="op_years")
    d = Economics()
    with st.expander("Economics", expanded=False):
        eco = {}
        for f, lab, step in [("price_sell", "Sell price (CU/kWh)", 0.1), ("price_buy", "Energy price (CU/kWh)", 0.05),
                             ("battery_cost", "Replacement cost (CU)", 0.05), ("downtime_days", "Downtime (days)", 0.5),
                             ("demand_kwh_day", "Demand (kWh/day)", 0.001), ("penalty", "Penalty (CU/kWh unmet)", 0.5),
                             ("hours_per_day", "Operating hours/day", 1.0), ("discount_year", "Discount rate (/yr)", 0.01),
                             ("salvage_frac", "Salvage (fraction of cost)", 0.05)]:
            eco[f] = st.number_input(lab, value=float(getattr(d, f)), step=step, format="%.3f", key=f"op_{f}")
    base_I = st.select_slider("Baseline current (A)", [1.0, 2.0, 3.0, 4.0], value=2.0, key="op_base")
    plant, economics, r = cache.operations(key, v, eol, cell, calibrated, tuple(sorted(eco.items())) if eco else
                                           tuple(sorted(Economics().dict().items())), years, s0, base_I)
    ptab = pd.DataFrame([dict(parameter=k, value=getattr(plant, k), source=plant.sources.get(
        {"kappa": "kappa", "r_bol": "resistance", "gamma": "resistance"}.get(k, k), "derived"))
        for k in ["bol_ah", "r_ref", "i_ref", "alpha", "kappa", "r_bol", "gamma", "s_floor"]])
    for line in answers.m3(r["summary"], plant):
        st.markdown(line)

    res = r["dp"]
    days = np.arange(res.policy_I.shape[0]) * res.period_days
    z = np.where(res.policy_rep.T, 0, res.policy_I.T)
    text = np.where(res.policy_rep.T, "R", "")
    blocks.chart_block("Optimal policy (DP): current by SOH and day",
                       charts.heatmap(z, days, np.round(res.grid, 3), "Day", "SOH", "Viridis", "A (0 = replace)",
                                      text=text if z.size < 6000 else None), ptab,
                       note="Table: plant parameters and where each comes from (calibrated vs default).", key="op_pol")
    led = {"DP optimum": r["ledger_dp"], f"Baseline {base_I:g} A": r["ledger_base"]}
    blocks.chart_block("Simulated trajectories", charts.ledger_soh(led, plant.s_floor), r["summary"], key="op_traj")
    blocks.chart_block("Profit breakdown", charts.profit_breakdown(r["summary"]), key="op_profit")
    gr = r["grid"]
    if len(gr):
        piv = gr.pivot_table(index="replace_soh", columns="current", values="profit_pv")
        blocks.chart_block("Grid study: fixed current × replacement SOH",
                           charts.heatmap(piv.to_numpy(), [f"{c:g} A" for c in piv.columns], piv.index, "Current",
                                          "Replace below SOH", "Tealgrn", "PV profit (CU)",
                                          text=np.round(piv.to_numpy(), 2)),
                           gr.sort_values("profit_pv", ascending=False).head(10), key="op_grid")
        st.markdown(f"DP optimum PV profit: **{r['summary'].profit_pv.iloc[0]:.3f} CU** "
                    f"(best grid cell {gr.profit_pv.max():.3f} CU).")

    st.markdown("### Scenario planner")
    I = st.select_slider("Current (A)", [1.0, 2.0, 3.0, 4.0], value=3.0, key="op_sc_I")
    th = st.slider("Replace below SOH", float(round(plant.s_floor, 2)), 0.95, 0.8, 0.01, key="op_sc_th")
    sc = simulate(plant, economics, s0, years, 1, fixed_policy(I, th))
    comp = pd.DataFrame([dict(policy="DP optimum", **summarise(r["ledger_dp"])),
                         dict(policy=f"Scenario {I:g} A, replace < {th:.2f}", **summarise(sc))])
    comp["gap_to_DP"] = comp.profit_pv - comp.profit_pv.iloc[0]
    blocks.chart_block("Scenario vs optimum", charts.ledger_soh({"DP optimum": r["ledger_dp"], "Scenario": sc},
                                                                plant.s_floor), comp, key="op_sc")
    st.session_state["ops_result"] = dict(summary=r["summary"], plant=plant, cell=cell)
