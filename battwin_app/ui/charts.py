"""Plotly figure builders. One legend per figure; stable colours/markers from ui.theme."""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from .theme import cell_color, cell_marker, model_color, MECH_COLORS, ACCENT, WARN


def _layout(fig, x=None, y=None, height=420, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="closest",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
                      showlegend=legend, xaxis_title=x, yaxis_title=y, font=dict(family="IBM Plex Sans"))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,.15)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,.15)", zeroline=False)
    return fig


def capacity(table: pd.DataFrame, cells, eol_ah: float, col="cap_norm", show_outliers=True):
    fig = go.Figure()
    for c in cells:
        g = table[table.Cell_ID == c]
        ok = g[~g.outlier]
        fig.add_trace(go.Scatter(x=ok.k, y=ok[col], mode="lines+markers", name=c, legendgroup=c,
                                 line=dict(color=cell_color(c), width=1.6),
                                 marker=dict(symbol=cell_marker(c), size=4, color=cell_color(c))))
        bad = g[g.outlier]
        if show_outliers and len(bad):
            fig.add_trace(go.Scatter(x=bad.k, y=bad[col], mode="markers", name=f"{c} outlier", legendgroup=c,
                                     showlegend=False, marker=dict(symbol="x-thin-open", size=8, color=cell_color(c))))
    if col in ("cap_norm", "capacity"):
        fig.add_hline(y=eol_ah, line=dict(color=WARN, dash="dash"), annotation_text=f"EOL {eol_ah:g} Ah")
    return _layout(fig, "Discharge cycle", "Capacity (Ah)" if col != "soh" else "SOH")


def hi_ranking(r: pd.DataFrame):
    fig = go.Figure()
    for m, c in [("monotonicity", "#0077b6"), ("trendability", "#2a9d8f"), ("prognosability", "#e9c46a")]:
        fig.add_trace(go.Bar(y=r.indicator, x=r[m], name=m.capitalize(), orientation="h", marker_color=c))
    fig.update_layout(barmode="group", yaxis=dict(autorange="reversed"))
    return _layout(fig, "Metric (0-1)", None, height=120 + 38 * len(r))


def pca_scores(scores: pd.DataFrame):
    fig = go.Figure()
    for c, g in scores.groupby("Cell_ID"):
        fig.add_trace(go.Scatter(x=g.PC1, y=g.PC2, mode="markers", name=c,
                                 marker=dict(symbol=cell_marker(c), color=cell_color(c), size=5, opacity=.8)))
    return _layout(fig, "PC1", "PC2")


def stress_coef(coef: pd.DataFrame):
    col = [WARN if c else ACCENT for c in coef.confounded]
    fig = go.Figure(go.Bar(x=coef.std_coef, y=coef.factor, orientation="h", marker_color=col,
                           error_x=dict(type="data", array=1.96 * coef.se), name="Std. coefficient (±95 %)"))
    fig.add_vline(x=0, line=dict(color="gray"))
    return _layout(fig, "Standardised effect on fade rate (SOH %/100 cycles)", None, height=120 + 45 * len(coef))


def fade_vs(rates: pd.DataFrame, x: str, xlabel: str):
    fig = go.Figure()
    for r in rates.itertuples():
        fig.add_trace(go.Scatter(x=[getattr(r, x)], y=[r.fade_per_100], mode="markers", name=r.cell,
                                 marker=dict(symbol=cell_marker(r.cell), color=cell_color(r.cell), size=11)))
    return _layout(fig, xlabel, "Fade (SOH %/100 cycles)")


def curves(xs: dict, x_title, y_title):
    """xs: {label: (x, y)} with a sequential colour scale over the labels (cycles)."""
    import plotly.express as px
    labels = list(xs)
    cols = px.colors.sample_colorscale("Viridis", np.linspace(0, 0.9, max(len(labels), 2)))
    fig = go.Figure()
    for (lab, (x, y)), col in zip(xs.items(), cols):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=lab, line=dict(color=col, width=2)))
    return _layout(fig, x_title, y_title)


def mechanisms(df: pd.DataFrame, cols=("LLI", "LAM_PE", "LAM_NE")):
    fig = go.Figure()
    for c in cols:
        fig.add_trace(go.Scatter(x=df.k, y=df[c] * 100, mode="lines+markers", name=c,
                                 line=dict(color=MECH_COLORS.get(c, "#888"))))
    return _layout(fig, "Discharge cycle", "Loss vs own BOL (%)")


def twin_soh(est: pd.DataFrame, fc_path: dict | None, eol_soh: float, k_now: int, models, measured: pd.DataFrame):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=measured.k, y=measured.soh, mode="markers", name="Measured",
                             marker=dict(color="rgba(128,128,128,.7)", size=5)))
    for m in models:
        e = est[est.model == m]
        if len(e):
            fig.add_trace(go.Scatter(x=e.k, y=e.soh, mode="lines", name=m,
                                     line=dict(color=model_color(m), width=3 if m == "Ensemble" else 1.6)))
    if fc_path:
        k, mu, lo, hi = fc_path["k"], fc_path["mean"], fc_path["lo"], fc_path["hi"]
        fig.add_trace(go.Scatter(x=np.r_[k, k[::-1]], y=np.r_[hi, lo[::-1]], fill="toself",
                                 fillcolor="rgba(0,180,216,.18)", line=dict(width=0), name="Ensemble band",
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=k, y=mu, mode="lines", name="Ensemble forecast",
                                 line=dict(color=model_color("Ensemble"), dash="dot", width=2)))
    fig.add_hline(y=eol_soh, line=dict(color=WARN, dash="dash"), annotation_text="EOL")
    fig.add_vline(x=k_now, line=dict(color="gray", dash="dot"))
    return _layout(fig, "Discharge cycle", "SOH (rate-normalised)", height=460)


def stacked(df: pd.DataFrame, x: str, cols: dict, y_title: str):
    """cols: {column: (label, colour)} -> stacked area."""
    fig = go.Figure()
    for c, (lab, col) in cols.items():
        fig.add_trace(go.Scatter(x=df[x], y=df[c], mode="lines", stackgroup="one", name=lab,
                                 line=dict(width=0.5, color=col)))
    return _layout(fig, "Discharge cycle", y_title, height=320)


def lines_by(df: pd.DataFrame, x: str, y: str, by: str, x_title, y_title, color_fn=model_color,
             hline=None, hline_label=None, log_y=False, markers=True):
    fig = go.Figure()
    for name, g in df.groupby(by, sort=False):
        fig.add_trace(go.Scatter(x=g[x], y=g[y], mode="lines+markers" if markers else "lines", name=str(name),
                                 line=dict(color=color_fn(name))))
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color=WARN, dash="dash"), annotation_text=hline_label)
    if log_y:
        fig.update_yaxes(type="log")
    return _layout(fig, x_title, y_title, height=360)


def bars(df: pd.DataFrame, x: str, y: str, color=ACCENT, horizontal=True, x_title=None, y_title=None):
    if horizontal:
        fig = go.Figure(go.Bar(x=df[y], y=df[x], orientation="h", marker_color=color))
        fig.update_layout(yaxis=dict(autorange="reversed"))
        return _layout(fig, y_title, x_title, height=120 + 36 * len(df), legend=False)
    fig = go.Figure(go.Bar(x=df[x], y=df[y], marker_color=color))
    return _layout(fig, x_title, y_title, legend=False)


def grouped_bars(df: pd.DataFrame, x: str, y: str, by: str, x_title, y_title, error=None):
    fig = go.Figure()
    for name, g in df.groupby(by, sort=False):
        fig.add_trace(go.Bar(x=g[x], y=g[y], name=str(name), marker_color=model_color(name),
                             error_y=dict(type="data", array=g[error]) if error else None))
    fig.update_layout(barmode="group")
    return _layout(fig, x_title, y_title)


def parity(df: pd.DataFrame, y="y", p="pred", by="cell"):
    fig = go.Figure()
    for c, g in df.groupby(by):
        fig.add_trace(go.Scatter(x=g[y], y=g[p], mode="markers", name=c,
                                 marker=dict(color=cell_color(c), symbol=cell_marker(c), size=5, opacity=.8)))
    lo, hi = np.nanmin(df[[y, p]].to_numpy()), np.nanmax(df[[y, p]].to_numpy())
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="ideal", line=dict(color="gray", dash="dash")))
    return _layout(fig, "Measured", "Predicted")


def heatmap(z, x, y, x_title, y_title, colorscale="Viridis", zlabel="", text=None):
    fig = go.Figure(go.Heatmap(z=z, x=x, y=y, colorscale=colorscale, colorbar=dict(title=zlabel),
                               text=text, texttemplate="%{text}" if text is not None else None))
    return _layout(fig, x_title, y_title, legend=False)


def ledger_soh(ledgers: dict, floor: float):
    fig = go.Figure()
    cols = [ACCENT, WARN, "#8d6a9f", "#2a9d8f"]
    for (name, led), col in zip(ledgers.items(), cols):
        fig.add_trace(go.Scatter(x=led.day, y=led.soh_start, mode="lines", name=name, line=dict(color=col)))
        rep = led[led["replace"]]
        fig.add_trace(go.Scatter(x=rep.day, y=np.ones(len(rep)), mode="markers", name=f"{name}: replace",
                                 marker=dict(symbol="triangle-down", size=9, color=col), showlegend=False))
    fig.add_hline(y=floor, line=dict(color="gray", dash="dot"), annotation_text="hard floor")
    return _layout(fig, "Day", "SOH")


def profit_breakdown(summary: pd.DataFrame):
    comp = [("revenue", "Revenue", 1, "#2a9d8f"), ("energy_cost", "Energy cost", -1, "#e9c46a"),
            ("maintenance", "Maintenance", -1, "#e76f51"), ("perf_loss", "Performance loss", -1, "#8d6a9f")]
    fig = go.Figure()
    for c, lab, sgn, col in comp:
        fig.add_trace(go.Bar(x=summary.policy, y=sgn * summary[c], name=lab, marker_color=col))
    fig.add_trace(go.Scatter(x=summary.policy, y=summary.profit, mode="markers", name="Profit",
                             marker=dict(symbol="diamond", size=14, color=ACCENT,
                                         line=dict(color="white", width=1))))
    fig.update_layout(barmode="relative")
    return _layout(fig, None, "CU over horizon (undiscounted)")


def coverage_plot(cov: pd.DataFrame, nominal: float):
    fig = go.Figure()
    for m, g in cov.groupby("model"):
        fig.add_trace(go.Scatter(x=g.group, y=g.achieved, mode="markers", name=m,
                                 marker=dict(color=model_color(m), size=11),
                                 error_y=dict(type="data", symmetric=False, array=g.ci_high - g.achieved,
                                              arrayminus=g.achieved - g.ci_low)))
    fig.add_hline(y=nominal, line=dict(color=WARN, dash="dash"), annotation_text=f"nominal {nominal:.0%}")
    return _layout(fig, None, "Achieved coverage")
