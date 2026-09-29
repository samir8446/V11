"""Reusable layout blocks: hero, full-width chart + table, notes, version-tolerant widgets."""
from __future__ import annotations
import pandas as pd
import streamlit as st

HERO = """
<div class="bt-hero">
<svg viewBox="0 0 900 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="physical battery and its digital twin">
 <defs>
  <linearGradient id="bt-fill" x1="0" x2="1"><stop offset="0" stop-color="#00b4d8"/><stop offset="1" stop-color="#2a9d8f"/></linearGradient>
  <pattern id="bt-grid" width="10" height="10" patternUnits="userSpaceOnUse">
   <path d="M10 0H0V10" fill="none" stroke="#00b4d8" stroke-opacity=".35" stroke-width=".6"/></pattern>
 </defs>
 <!-- physical cell -->
 <g transform="translate(60,30)">
  <rect x="0" y="0" width="190" height="90" rx="10" fill="none" stroke="currentColor" stroke-width="3"/>
  <rect x="190" y="30" width="12" height="30" rx="3" fill="currentColor"/>
  <rect x="8" y="8" height="74" rx="5" fill="url(#bt-fill)"><animate attributeName="width" values="30;174;174;30" dur="6s" repeatCount="indefinite"/></rect>
  <text x="95" y="112" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="12" fill="currentColor" opacity=".8">PHYSICAL CELL · 18650</text>
 </g>
 <!-- data link -->
 <g font-family="IBM Plex Mono,monospace" font-size="11" fill="currentColor" opacity=".85">
  <path id="bt-up" d="M290 60 C 380 20, 520 20, 610 60" fill="none" stroke="currentColor" stroke-opacity=".35" stroke-dasharray="4 5"/>
  <path id="bt-dn" d="M610 90 C 520 130, 380 130, 290 90" fill="none" stroke="currentColor" stroke-opacity=".35" stroke-dasharray="4 5"/>
  <circle r="5" fill="#00b4d8"><animateMotion dur="2.4s" repeatCount="indefinite"><mpath href="#bt-up"/></animateMotion></circle>
  <circle r="5" fill="#00b4d8" opacity=".6"><animateMotion dur="2.4s" begin="1.2s" repeatCount="indefinite"><mpath href="#bt-up"/></animateMotion></circle>
  <circle r="5" fill="#f4a261"><animateMotion dur="3s" repeatCount="indefinite"><mpath href="#bt-dn"/></animateMotion></circle>
  <text x="450" y="22" text-anchor="middle">V · I · T · Q · EIS  →</text>
  <text x="450" y="140" text-anchor="middle">←  SOH · RUL · policy</text>
 </g>
 <!-- digital twin -->
 <g transform="translate(640,30)">
  <rect x="0" y="0" width="190" height="90" rx="10" fill="url(#bt-grid)" stroke="#00b4d8" stroke-width="2" stroke-dasharray="6 4">
   <animate attributeName="stroke-dashoffset" values="0;20" dur="1.5s" repeatCount="indefinite"/></rect>
  <rect x="190" y="30" width="12" height="30" rx="3" fill="none" stroke="#00b4d8" stroke-width="2"/>
  <polyline fill="none" stroke="#00b4d8" stroke-width="2.5" points="12,22 45,26 80,33 115,42 150,56 178,76">
   <animate attributeName="stroke-dasharray" values="0 300;300 0" dur="3s" repeatCount="indefinite"/></polyline>
  <text x="95" y="112" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="12" fill="#00b4d8">DIGITAL TWIN · live</text>
 </g>
</svg>
</div>
<style>.bt-hero{width:100%;max-width:980px;margin:0 auto .4rem auto;color:inherit}
.bt-hero svg{width:100%;height:auto;display:block}
@media (prefers-reduced-motion: reduce){.bt-hero animate,.bt-hero animateMotion{display:none}}</style>
"""


def hero():
    try:
        st.html(HERO)
    except AttributeError:          # very old Streamlit
        st.markdown(HERO, unsafe_allow_html=True)


def plot(fig, key=None):
    try:
        st.plotly_chart(fig, theme="streamlit", width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, theme="streamlit", use_container_width=True, key=key)


def table(df: pd.DataFrame, height=None, fmt: dict | None = None):
    if df is None or len(df) == 0:
        return
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round((fmt or {}).get(c, 4))
    kw = dict(hide_index=True)
    if height:
        kw["height"] = height
    try:
        st.dataframe(d, width="stretch", **kw)
    except TypeError:
        st.dataframe(d, use_container_width=True, **kw)


def chart_block(title: str, fig, df: pd.DataFrame | None = None, note: str | None = None, key=None):
    """Full-width chart, then its table directly below. Never side by side."""
    st.markdown(f"### {title}")
    if note:
        st.markdown(f"<div class='bt-note'>{note}</div>", unsafe_allow_html=True)
    if fig is not None:
        plot(fig, key=key)
    if df is not None:
        table(df)


def badges(items: dict):
    st.markdown(" ".join(f"<span class='bt-badge'>{k}: {v}</span>" for k, v in items.items()),
                unsafe_allow_html=True)


def seg(label, options, default=None, key=None):
    """Segmented control with a radio fallback for older Streamlit versions."""
    default = default if default is not None else options[0]
    if hasattr(st, "segmented_control"):
        v = st.segmented_control(label, options, default=default, key=key)
        return v if v is not None else default
    return st.radio(label, options, index=options.index(default), horizontal=True, key=key)
