"""Mission 2 — the self-updating twin, replayed cycle by cycle, plus the M2 studies."""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import chi2
from battwin import config as C
from battwin.observers.stream import (MODEL_NAMES, ENSEMBLE, Stream, make_observations, default_members)
from battwin.prognostics import metrics as M
from ui import blocks, charts, cache
from ui.theme import MECH_COLORS, model_color


def _fan(s: Stream, hmax=60):
    """Ensemble forecast fan from the current state; conformal radius interpolated over horizons."""
    if not s.w_rows or s.i == 0:
        return None
    k_now = s.obs[s.i - 1].k
    last = {r["model"]: r["weight"] for r in s.w_rows[-len(s.members):]}
    hs = np.arange(1, hmax + 1)
    parts = []
    for n, w in last.items():
        if w <= 0:
            continue
        try:
            m, lo, hi = (np.asarray(a, float) for a in s.members[n].forecast(hs, s.alpha))
        except Exception:
            continue
        parts.append((w, m, lo, hi))
    if not parts:
        return None
    tot = sum(p[0] for p in parts)
    mean = sum(p[0] * p[1] for p in parts) / tot
    lo = sum(p[0] * p[2] for p in parts) / tot
    hi = sum(p[0] * p[3] for p in parts) / tot
    rad = [s.conf.radius((ENSEMBLE, int(h))) for h in s.h]
    ok = [(h, r) for h, r in zip(s.h, rad) if np.isfinite(r)]
    if len(ok) >= 2:
        r = np.interp(hs, [o[0] for o in ok], [o[1] for o in ok])
        lo, hi = mean - r, mean + r
    return dict(k=k_now + hs, mean=mean, lo=lo, hi=hi)


def _accuracy(fr, h):
    fc, w = fr["forecasts"], fr["weights"]
    rows = []
    if fc.empty:
        return pd.DataFrame()
    wl = w[w.k == w.k.max()].set_index("model") if len(w) else pd.DataFrame()
    for m, g in fc.groupby("model"):
        g1, gh = g[g.h == 1], g[g.h == h]
        rows.append({"model": m, "1-step RMSE": M.rmse(g1.actual, g1.pred), f"{h}-step RMSE": M.rmse(gh.actual, gh.pred),
                     f"{h}-step R²": M.r2(gh.actual, gh.pred), "coverage": M.coverage(gh.actual, gh.lo, gh.hi),
                     "n resolved": int(gh.actual.notna().sum()),
                     "weight": float(wl.loc[m, "weight"]) if m in wl.index else np.nan,
                     "status": (wl.loc[m, "excluded"] or "active") if m in wl.index else ("—" if m == ENSEMBLE else "")})
    for m, (ok, why) in fr["status"].items():
        if not ok:
            rows = [dict(r, status=f"invalid: {why}") if r["model"] == m else r for r in rows]
    return pd.DataFrame(rows)


def _draw(s: Stream, eol_soh, h, models, true_eol):
    fr = s.frames()
    est = fr["estimates"]
    if est.empty:
        st.info("Press Step or Play to start the replay.")
        return
    k_now = int(est.k.max())
    meas = est[est.model == est.model.iloc[0]][["k", "measured"]].rename(columns={"measured": "soh"})
    blocks.badges({"cycle": k_now, "of": s.obs[-1].k, "measure every": s.m,
                   "EOL SOH": f"{eol_soh:.3f}", "true EOL": true_eol})
    blocks.chart_block("State of health — live", charts.twin_soh(est, _fan(s), eol_soh, k_now,
                                                                 models + [ENSEMBLE], meas),
                       _accuracy(fr, h), note="Band: conformal (online) once ≥8 residuals exist, else the "
                                              "members' own bands.", key="lt_soh")
    w = fr["weights"]
    if len(w) and len(models) > 1:
        wp = w.pivot_table(index="k", columns="model", values="weight").reset_index()
        blocks.chart_block("Ensemble weights", charts.stacked(wp, "k", {m: (m, model_color(m)) for m in models
                                                                        if m in wp}, "Weight"), key="lt_w")
    if "Mechanistic PF" in models:
        mp = est[est.model == "Mechanistic PF"].dropna(subset=["share_sei"])
        if len(mp):
            blocks.chart_block("Mechanism shares (mechanistic PF)", charts.stacked(
                mp, "k", {"share_sei": ("SEI", MECH_COLORS["SEI"]), "share_plating": ("Plating", MECH_COLORS["Plating"]),
                          "share_lam": ("LAM", MECH_COLORS["LAM"])}, "Share of capacity loss"),
                mp[["k", "share_sei", "share_plating", "share_lam", "LLI", "LAM"]].tail(1),
                note="Model-based attribution; only partially identifiable from capacity alone.", key="lt_mech")
    if "ECM twin" in models:
        ecm = est[est.model == "ECM twin"]
        if "nis" in ecm and ecm.nis.notna().any():
            e = ecm.assign(model="NIS (slow)")[["k", "nis", "model"]]
            blocks.chart_block("ECM twin consistency (NIS)", charts.lines_by(
                e, "k", "nis", "model", "Discharge cycle", "NIS", hline=chi2.ppf(0.999, 2),
                hline_label="guard χ²(0.999, 2)", log_y=True, color_fn=lambda _: model_color("ECM twin")),
                ecm[["k", "R0", "nis", "fast_nis", "rejects"]].tail(1), key="lt_nis")
    rul = fr["rul"]
    if len(rul):
        blocks.chart_block("Predicted end of life", charts.lines_by(
            rul.dropna(subset=["eol_pred"]), "k0", "eol_pred", "model", "Forecast origin (cycle)",
            "Predicted EOL cycle", hline=true_eol if isinstance(true_eol, (int, float)) else None,
            hline_label="measured EOL"), key="lt_rul",
            note="Gaps = EOL not predicted within 600 cycles of the origin.")
    if len(fr["failures"]):
        st.error("Model failures (reported, not hidden)")
        blocks.table(fr["failures"])


def render(ctx):
    cd, key, v, eol = ctx["cd"], ctx["key"], ctx["v"], ctx["eol"]
    st.markdown("## Live twin")
    cell = st.selectbox("Cell", ctx["cells"], key="lt_cell")
    models = st.multiselect("Models", MODEL_NAMES, default=MODEL_NAMES, key="lt_models") or ["Trend KF"]
    h = st.select_slider("Scored horizon (cycles)", options=list(C.HORIZONS), value=25, key="lt_h")
    every = st.select_slider("Capacity measured every … cycles", options=[1, 2, 5, 10, 20], value=1, key="lt_every")
    speed = st.select_slider("Cycles per tick", options=[1, 2, 5, 10], value=2, key="lt_speed")
    sig = (key, v, eol, cell, tuple(models), every)
    if st.session_state.get("lt_sig") != sig:
        obs = make_observations(cd, cell, with_curves="ECM twin" in models)
        eol_soh = cd.eol_ah / float(cd.summary.set_index("cell").loc[cell, "bol_ah"])
        st.session_state.lt_stream = Stream(obs, default_members(cd, cell, models), eol_soh, measure_every=every)
        st.session_state.lt_sig, st.session_state.lt_play = sig, False
    s: Stream = st.session_state.lt_stream
    c1, c2, c3, c4, c5 = st.columns(5)
    if c1.button("▶ Play" if not st.session_state.lt_play else "⏸ Pause"):
        st.session_state.lt_play = not st.session_state.lt_play
    if c2.button("Step +1"):
        s.step()
    if c3.button("Step +10"):
        s.run(10)
    if c4.button("Run to end"):
        with st.spinner("Streaming…"):
            s.run()
    if c5.button("Reset"):
        st.session_state.lt_sig = None
        st.rerun()
    summ = cd.summary.set_index("cell").loc[cell]
    true_eol = int(summ.eol_k) if pd.notna(summ.eol_k) else summ.eol_status

    @st.fragment(run_every=0.6 if st.session_state.lt_play else None)
    def live():
        if st.session_state.lt_play and not s.done:
            s.run(speed)
        _draw(s, s.eol_soh, h, models, true_eol)
        if st.session_state.lt_play and s.done:
            st.session_state.lt_play = False
            st.rerun()
    live()

    st.markdown("---")
    st.markdown("## Mission 2 studies")
    st.markdown("### How often should the twin update?")
    uf_cells = st.multiselect("Cells", ctx["cells"], default=ctx["cells"][:4], key="lt_uf_cells")
    uf_models = st.multiselect("Models", ["Trend KF", "Power-law PF", "Hierarchical Bayes", "Mechanistic PF"],
                               default=["Trend KF", "Power-law PF", "Hierarchical Bayes"], key="lt_uf_models")
    if st.toggle("Run update-frequency study", key="lt_uf_run") and uf_cells and uf_models:
        d = cache.update_freq(key, v, eol, tuple(uf_cells), (1, 2, 5, 10, 20), h, tuple(uf_models))
        agg = d.groupby(["model", "interval"]).agg(rmse=("rmse", "median"), coverage=("coverage", "mean")).reset_index()
        blocks.chart_block(f"{h}-cycle forecast error vs measurement interval",
                           charts.lines_by(agg, "interval", "rmse", "model", "Capacity measured every … cycles",
                                           f"Median RMSE at h={h}"), agg, key="lt_uf")
    if "ECM twin" in models and not bool(summ.pulsed):
        if st.toggle("ECM: capacity-check interval study", key="lt_ecm_run"):
            d = cache.ecm_check(key, v, eol, cell, h)
            blocks.chart_block("ECM twin — periodic capacity check", charts.bars(
                d.assign(check=d.check_every.astype(str)), "check", "rmse", horizontal=False,
                x_title="Capacity check every … cycles", y_title=f"RMSE at h={h}"), d, key="lt_ecm")
    st.markdown("### Which measurements are most informative?")
    if st.toggle("Rank measurements (grouped CV, whole cells held out)", key="lt_inf_run"):
        r = cache.informativeness(key, v, eol, tuple(ctx["cells"]))
        blocks.chart_block("SOH estimation error from each measurement alone",
                           charts.bars(r["single"], "measurement", "rmse", x_title=None, y_title="Grouped-CV RMSE (SOH)"),
                           r["single"], note=f"Baseline (predict the mean): {r['baseline_rmse']:.3f}.", key="lt_inf")
        if len(r["path"]):
            st.markdown("**Greedy combination**")
            blocks.table(r["path"])

