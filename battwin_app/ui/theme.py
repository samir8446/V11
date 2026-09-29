"""Visual identity: stable colour per cell, marker per group, colour per model, page CSS."""
from __future__ import annotations
import hashlib
from battwin.data import registry

# 34 distinguishable colours, assigned once to the 34 NASA cells (sorted), never reshuffled
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173", "#3182bd",
            "#e6550d", "#31a354", "#756bb1", "#636363", "#6baed6", "#fd8d3c", "#74c476", "#9e9ac8",
            "#e7ba52", "#ad494a", "#a55194", "#00a3c4", "#c49c94", "#f7b6d2", "#5254a3", "#8ca252",
            "#bd9e39", "#d6616b"]
CELL_COLORS = {c: _PALETTE[i] for i, c in enumerate(registry.ALL_CELLS)}

GROUP_MARKERS = {"Reference": "circle", "High current": "square", "Hot": "diamond",
                 "Mixed conditions": "triangle-up", "Cold": "star", "Pulsed load": "cross",
                 "Corrupted logging": "x", "Unknown": "hexagon"}

MODEL_COLORS = {"ECM twin": "#0077b6", "Mechanistic PF": "#e76f51", "Power-law PF": "#2a9d8f",
                "Trend KF": "#8d6a9f", "Hierarchical Bayes": "#e9c46a", "Ensemble": "#00b4d8",
                "Gaussian Process": "#0077b6", "Extra Trees": "#2a9d8f",
                "Hist. Gradient Boosting": "#e76f51", "Bayesian Ridge": "#8d6a9f"}
MECH_COLORS = {"SEI": "#2a9d8f", "Plating": "#e76f51", "LAM": "#8d6a9f", "LLI": "#0077b6",
               "LAM_PE": "#e9c46a", "LAM_NE": "#8d6a9f"}
ACCENT = "#00b4d8"
WARN = "#f4a261"


def cell_color(cid: str) -> str:
    if cid in CELL_COLORS:
        return CELL_COLORS[cid]
    return _PALETTE[int(hashlib.md5(cid.encode()).hexdigest(), 16) % len(_PALETTE)]


def cell_marker(cid: str) -> str:
    return GROUP_MARKERS.get(registry.group_of(cid), "circle")


def model_color(name: str) -> str:
    return MODEL_COLORS.get(name, "#888888")


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText { font-family: 'IBM Plex Sans', system-ui, sans-serif; }
code, .stCode, [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', ui-monospace, monospace; }
.block-container { padding-top: 1.2rem; max-width: 1280px; }
h1, h2, h3 { letter-spacing: -0.01em; font-weight: 600; }
h3 { border-left: 3px solid #00b4d8; padding-left: .55rem; margin-top: 1.4rem; }
[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.25); border-radius: 6px; padding: .5rem .8rem;
  background: rgba(0,180,216,.04); }
[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.2); }
.bt-note { font-size: .82rem; opacity: .75; margin: -.2rem 0 .6rem 0; }
.bt-badge { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:.72rem; padding:.05rem .45rem;
  border:1px solid rgba(128,128,128,.4); border-radius:3px; margin-right:.3rem; opacity:.85; }
</style>
"""
