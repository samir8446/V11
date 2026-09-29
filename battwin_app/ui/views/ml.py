"""ML models (max 4): SOH estimation without capacity inputs; fade-rate forecasting with ΔQ(V)."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from battwin.ml.models import MODELS
from battwin.prognostics import metrics as M
from ui import blocks, charts, cache


def render(ctx):
    cd, key, v, eol = ctx["cd"], ctx["key"], ctx["v"], ctx["eol"]
    st.markdown("## ML models")
    blocks.badges({"validation": "grouped by cell (nested tuning)", "leakage": "no capacity inputs for SOH",
                   "models": "≤ 4"})
    cells = tuple(ctx["cells"])
    if len(cells) < 3:
        st.info("Grouped cross-validation needs at least 3 cells in the selected groups.")
        return
    models = st.multiselect("Models", MODELS, default=["Bayesian Ridge", "Hist. Gradient Boosting"],
                            max_selections=4, key="ml_models")
    tune = st.toggle("Nested hyper-parameter tuning", value=False, key="ml_tune",
                     help="Inner GroupKFold on the training cells only.")
    if not models:
        return

    st.markdown("### SOH estimation from non-capacity measurements")
    if st.toggle("Run SOH estimation", key="ml_soh_run"):
        res = {m: cache.ml_soh(key, v, eol, cells, m, tune) for m in models}
        tab = pd.DataFrame([dict(model=m, rmse=r["rmse"], mae=r["mae"], r2=r["r2"], coverage90=r["coverage90"])
                            for m, r in res.items()]).sort_values("rmse")
        best = tab.model.iloc[0]
        blocks.chart_block(f"Held-out cells — {best}", charts.parity(res[best]["frame"]), tab,
                           note="Each point is predicted by a model that never saw its cell. R² unclipped.", key="ml_par")
        pc = pd.concat([r["per_cell"].assign(model=m) for m, r in res.items()])
        blocks.chart_block("Per-cell error", charts.grouped_bars(pc, "cell", "rmse", "model", None, "RMSE (SOH)"),
                           key="ml_pc")
        imp = res[best]["importance"].reset_index()
        imp.columns = ["feature", "importance"]
        blocks.chart_block(f"Permutation importance on held-out cells — {best}",
                           charts.bars(imp, "feature", "importance", y_title="Δ RMSE when permuted"), imp, key="ml_imp")

    st.markdown("### Fade-rate forecasting (ΔQ(V) + history)")
    horizon = st.select_slider("Horizon (cycles)", [10, 25, 50], value=25, key="ml_h")
    if st.toggle("Run fade-rate models", key="ml_fade_run"):
        res = {m: cache.ml_fade(key, v, eol, cells, horizon, m, tune) for m in models}
        rows = []
        for m, r in res.items():
            f = r["frame"]
            soh_pred = f.soh_now - f.pred * horizon / 100
            rows.append(dict(model=m, rate_rmse=r["rmse"], rate_r2=r["r2"],
                             soh_rmse=M.rmse(f.soh_future, soh_pred), soh_r2=M.r2(f.soh_future, soh_pred),
                             persistence_rmse=M.rmse(f.soh_future, f.soh_now), n=len(f)))
        tab = pd.DataFrame(rows).sort_values("soh_rmse")
        best = tab.model.iloc[0]
        f = res[best]["frame"]
        blocks.chart_block(f"Fade rate, held-out cells — {best}",
                           charts.parity(f.rename(columns={"target": "y"}), "y", "pred"), tab,
                           note="Fade in SOH %/100 cycles. Persistence = 'no further fade' baseline.", key="ml_fp")
        imp = res[best]["importance"].reset_index(); imp.columns = ["feature", "importance"]
        blocks.chart_block("Which features carry the forecast?", charts.bars(imp, "feature", "importance",
                                                                             y_title="Δ RMSE when permuted"), key="ml_fimp")
        cell = st.selectbox("Backtest cell (model trained on all other cells)", list(cells), key="ml_bt_cell")
        bt = cache.ml_backtest(key, v, eol, cells, horizon, cell, tuple(models))
        if len(bt):
            g = cd.cell(cell)
            meas = pd.DataFrame(dict(k=g.k, soh=g.soh_clean, model="Measured"))
            pred = bt.assign(k=bt.k0 + bt.horizon, soh=bt.soh_pred)[["k", "soh", "model"]]
            fig = charts.lines_by(pd.concat([meas, pred]), "k", "soh", "model", "Discharge cycle", "SOH",
                                  hline=cd.eol_ah / float(g.bol_ah.iloc[0]), hline_label="EOL")
            acc = bt.groupby("model").apply(lambda d: pd.Series(dict(rmse=M.rmse(d.soh_future, d.soh_pred),
                                                                     r2=M.r2(d.soh_future, d.soh_pred), n=len(d))),
                                            include_groups=False).reset_index()
            blocks.chart_block(f"Backtest {cell}: SOH at origin + {horizon}", fig, acc, key="ml_bt")
        else:
            st.info("Not enough history for a backtest on this cell.")
