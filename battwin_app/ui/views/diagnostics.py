"""Mission 1 — health: capacity/SOH, HI ranking, PCA, stress factors, ICA/DVA, half-cell LLI/LAM."""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from battwin.data import registry
from battwin.diagnostics import icadva
from battwin.diagnostics.stress import FACTORS
from ui import blocks, charts, cache


def render(ctx):
    cd, key, v, eol = ctx["cd"], ctx["key"], ctx["v"], ctx["eol"]
    st.markdown("## Diagnostics")
    scope = blocks.seg("Scope", ["Single cell", "Selected cells", "Fleet"], "Selected cells", key="dg_scope")
    if scope == "Single cell":
        cells = [st.selectbox("Cell", ctx["cells"], key="dg_one")]
    elif scope == "Selected cells":
        cells = st.multiselect("Cells", ctx["cells"], default=ctx["cells"][:6], key="dg_many") or ctx["cells"][:1]
    else:
        cells = ctx["cells"]
    tcells = tuple(cells)
    summ = cd.summary[cd.summary.cell.isin(cells)]

    # capacity
    basis = blocks.seg("Capacity basis", ["Rate-normalised", "Measured", "SOH"], key="dg_basis")
    col = {"Rate-normalised": "cap_norm", "Measured": "capacity", "SOH": "soh"}[basis]
    view = summ[["cell", "group", "bol_ah", "eol_k", "eol_status", "n_cycles", "n_outliers", "crash_flag",
                 "pulsed", "cutoff_V", "I_levels", "normalised"]].rename(columns={"bol_ah": "BOL (Ah)"})
    blocks.chart_block("Capacity fade", charts.capacity(cd.table, cells, eol, col), view,
                       note="× = Hampel outlier / logging artefact (excluded). Pulsed-load cells: capacity from "
                            "integrated current; voltage-curve methods disabled.")

    # HI ranking
    if len(cells) >= 2:
        rank, sev = cache.hi_ranking(key, v, eol, tcells)
        if len(rank):
            blocks.chart_block("Health-indicator ranking", charts.hi_ranking(rank),
                               rank[["indicator", "score", "monotonicity", "trendability", "prognosability",
                                     "loco_rmse", "n_cells"]],
                               note="LOCO = SOH estimated from the indicator by a model trained on the other cells.")
            if len(sev["path"]):
                st.markdown(f"**One or several indicators?** → **{sev['verdict']}** "
                            f"(non-capacity HIs, forward selection under LOCO)")
                blocks.table(sev["path"])
        # PCA
        p = cache.pca(key, v, eol, tcells)
        if p.get("ok"):
            ev = p["explained"]
            blocks.chart_block("PCA of health indicators", charts.pca_scores(p["scores"]),
                               p["loadings"].reset_index().rename(columns={"index": "indicator"}),
                               note=f"Explained variance PC1 {ev[0]:.0%}, PC2 {ev[1]:.0%}; "
                                    f"corr(PC1, SOH) = {p['pc1_soh_corr']:+.2f}.")
        # stress
        st.markdown("### Stress factors")
        f_sel = st.multiselect("Factors", list(FACTORS), default=["ambient", "I_mean", "cutoff_V"],
                               format_func=FACTORS.get, key="dg_factors")
        max_k = st.slider("Fade-rate window (first N cycles)", 30, 300, 120, 10, key="dg_maxk")
        rates, res = cache.stress(key, v, eol, tcells, tuple(f_sel), max_k)
        if len(rates):
            blocks.plot(charts.fade_vs(rates, "ambient", "Ambient temperature (°C)"), key="dg_fade_amb")
            blocks.plot(charts.fade_vs(rates, "I_mean", "Discharge current (A)"), key="dg_fade_I")
            blocks.table(rates.drop(columns=["n"]))
        if res.get("ok"):
            blocks.chart_block("Regression of fade rate on stress factors", charts.stress_coef(res["coef"]),
                               res["coef"], key="dg_coef")
            blocks.badges({"n cells": res["n"], "R²": f"{res['r2']:.2f}", "adj. R²": f"{res['adj_r2']:.2f}",
                           "power": res["power"]})
            if res["confounding"]:
                st.warning("Confounded factors (|r| > 0.7): " + "; ".join(res["confounding"]) +
                           ". Their separate effects are not identifiable from this cohort.")
        else:
            st.info(f"Regression not estimable: {res.get('reason', '')} ({res.get('power', '')}).")
    else:
        st.info("Select at least two cells for ranking, PCA and stress analysis.")

    # ICA / DVA and half-cell for one cell
    cell = cells[0] if len(cells) == 1 else st.selectbox("Cell for voltage-curve analysis", cells, key="dg_curve")
    if bool(cd.summary.set_index("cell").loc[cell, "pulsed"]) or registry.is_pulsed(cell):
        st.info(f"{cell}: pulsed load — ICA/DVA and half-cell fitting assume constant current and are disabled.")
        return
    g = cd.cell(cell)
    g = g[~g.outlier]
    pick = g.iloc[np.unique(np.linspace(0, len(g) - 1, 6).astype(int))]
    cur = cd.curves(cell, pick.Cycle_Index, n=300)
    ic, dv, peaks = {}, {}, []
    for r in pick.itertuples():
        c = cur.get(int(r.Cycle_Index))
        if c is None:
            continue
        ic[f"cycle {r.k}"] = icadva.ica(c)
        dv[f"cycle {r.k}"] = icadva.dva(c)
        pv, ph = icadva.ica_peak(c)
        peaks.append(dict(cycle=r.k, soh=r.soh, peak_V=pv, peak_dQdV=ph))
    blocks.chart_block(f"Incremental capacity — {cell}", charts.curves(ic, "Voltage (V)", "dQ/dV (Ah/V)"),
                       pd.DataFrame(peaks), key="dg_ica")
    blocks.chart_block(f"Differential voltage — {cell}", charts.curves(dv, "Discharged charge (Ah)", "dV/dQ (V/Ah)"),
                       key="dg_dva")
    st.markdown("### Half-cell fitting (LLI / LAM)")
    n_fits = st.slider("Number of cycles fitted", 4, 20, 8, key="dg_nfit")
    if st.toggle("Fit half-cell model", value=False, key="dg_hc"):
        hc = cache.halfcell(key, v, eol, cell, n_fits)
        blocks.chart_block(f"Degradation modes — {cell}", charts.mechanisms(hc),
                           hc[["k", "LLI", "LAM_PE", "LAM_NE", "rmse_mV", "success"]],
                           note="Literature half-cell OCVs (graphite, LiCoO₂); at 1C LAM_NE is weakly identifiable — "
                                "read it with the fit residual.", key="dg_hcfig")
