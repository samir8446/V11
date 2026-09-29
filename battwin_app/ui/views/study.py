"""Study results: per-group validation, fair comparison, coverage, answers, export."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from battwin import config as C
from battwin.observers.stream import MODEL_NAMES, ENSEMBLE
from battwin.study.cohort import run_cohort
from battwin.study.summary import study_tables
from battwin.study import answers, export
from ui import blocks, charts, cache


@st.cache_data(show_spinner=False)
def _cohort(key, version, eol, cells: tuple, models: tuple, alpha):
    prog = st.progress(0.0, text="Starting…")
    r = run_cohort(cache._cd(key, eol), list(cells), list(models), alpha=alpha,
                   progress=lambda j, n, c: prog.progress(j / n, text=f"{c} ({j + 1}/{n})"))
    prog.empty()
    return r


def render(ctx):
    cd, key, v, eol = ctx["cd"], ctx["key"], ctx["v"], ctx["eol"]
    st.markdown("## Study results")
    cells = st.multiselect("Cells", ctx["cells"], default=ctx["cells"], key="st_cells")
    models = st.multiselect("Live models", MODEL_NAMES, default=MODEL_NAMES, key="st_models")
    h = st.select_slider("Horizon (cycles)", list(C.HORIZONS), value=25, key="st_h")
    if not st.toggle("Run / show cohort study", key="st_run", help="Cached per data, engine version and settings."):
        st.info("Streams every selected cell through every model (≈2 s per cell).")
        return
    if len(cells) < 2 or not models:
        st.warning("Select at least two cells and one model.")
        return
    res = _cohort(key, v, eol, tuple(cells), tuple(models), C.BAND_ALPHA)
    tabs = study_tables(res, h)
    lb = tabs["leaderboard"]
    blocks.badges({k: res["meta"][k] for k in ("engine", "cells", "eol_ah", "seconds")})

    blocks.chart_block(f"Accuracy by group (median RMSE, h={h}, common cells)",
                       charts.grouped_bars(lb, "group", "rmse_median", "model", None, "Median RMSE (SOH)"),
                       lb[["group", "model", "rmse_median", "r2_median", "coverage", "fade_skill", "n_cells", "not_ok"]],
                       note="Only cells where every compared model produced forecasts; failures counted in 'not_ok'.",
                       key="st_lb")
    groups = ["All"] + sorted(res["metrics"].group.dropna().unique())
    gsel = st.selectbox("Pairwise comparison group", groups, key="st_g")
    w = tabs["wilcoxon"][tabs["wilcoxon"].group == gsel]
    if len(w):
        names = sorted(set(w.model_a) | set(w.model_b))
        mat = pd.DataFrame(1.0, index=names, columns=names)
        for r in w.itertuples():
            mat.loc[r.model_a, r.model_b] = mat.loc[r.model_b, r.model_a] = r.p_holm
        blocks.chart_block(f"Paired Wilcoxon (per cell), Holm-adjusted p — {gsel}",
                           charts.heatmap(mat.to_numpy(), names, names, None, None, "Blues_r", "p (Holm)",
                                          text=mat.round(3).to_numpy()), w.drop(columns="group"), key="st_wil")
    cov = tabs["coverage"]
    cov_h = cov[cov.h == h]
    if len(cov_h):
        blocks.chart_block(f"Band coverage (nominal {1 - C.BAND_ALPHA:.0%}, h={h})",
                           charts.coverage_plot(cov_h, 1 - C.BAND_ALPHA), cov_h, key="st_cov")
    if len(res["rul"]):
        st.markdown("### End-of-life prediction")
        blocks.table(res["rul"])
    if len(res["failures"]):
        st.markdown("### Failures (kept, not dropped)")
        blocks.table(res["failures"])

    rank, sev = cache.hi_ranking(key, v, eol, tuple(cells))
    rates, sres = cache.stress(key, v, eol, tuple(cells), ("ambient", "I_mean", "cutoff_V"), 120)
    ops = st.session_state.get("ops_result")
    ans = {"Mission 1 — Health": answers.m1(rank, sev, sres, res["shares"]),
           "Mission 2 — Self-updating twin": answers.m2(lb[lb.group == "All"], tabs["wilcoxon"][tabs["wilcoxon"].group == "All"],
                                                        h=h),
           "Mission 3 — Operations": answers.m3(ops["summary"], ops["plant"]) if ops else
           ["Open 'Operations & maintenance' to compute the Mission 3 answer."]}
    st.markdown("### Answers")
    for mname, lines in ans.items():
        st.markdown(f"**{mname}**")
        for l in lines:
            st.markdown(f"- {l}")
    tables = dict(cells=cd.summary, metrics=res["metrics"], estimation=res["estimation"], rul=res["rul"],
                  failures=res["failures"], shares=res["shares"], hi_ranking=rank, **tabs)
    c1, c2 = st.columns(2)
    c1.download_button("⬇ CSV bundle (zip)", export.csv_zip(tables), "battwin_study.zip", "application/zip")
    c2.download_button("⬇ HTML report", export.html_report("Battery digital twin — study", ans, tables, meta=res["meta"]),
                       "battwin_study.html", "text/html")
