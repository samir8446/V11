"""Streamlit front end. The engine (battwin) knows nothing about Streamlit."""
from __future__ import annotations
import os
import streamlit as st
import battwin
from battwin import config as C
from battwin.data import registry

REQUIRED_ENGINE = "1.0"
st.set_page_config(page_title="Battery Digital Twin", page_icon="🔋", layout="wide")

if battwin.ENGINE_MAJOR_MINOR != REQUIRED_ENGINE:
    st.error(f"Engine version mismatch: app needs battwin {REQUIRED_ENGINE}.x, found {battwin.__version__}.")
    st.stop()

from ui import theme, blocks, cache  # noqa: E402
from ui.views import diagnostics, live_twin, ml, operations, study  # noqa: E402

st.markdown(theme.CSS, unsafe_allow_html=True)

DEFAULT_MASTER = os.environ.get("BATTWIN_MASTER", "data/battery_master_data.parquet")
DEFAULT_IMP = os.environ.get("BATTWIN_IMP", next((p for p in ("data/impedance_ground_truth.parquet",
                                                                "data/impedance.parquet") if os.path.exists(p)),
                                                   "data/impedance_ground_truth.parquet"))


def sidebar_source() -> dict:
    st.sidebar.markdown("#### Data")
    has_local = os.path.exists(DEFAULT_MASTER)
    options = ["NASA files (local)", "Upload parquet", "Synthetic demo"]
    kind = st.sidebar.radio("Source", options, index=0 if has_local else 2, label_visibility="collapsed")
    if kind == "Synthetic demo":
        st.sidebar.caption("Synthetic electrode-level ground truth (16 cells). Replace with NASA data for results.")
        return dict(kind="synthetic", key=cache.SYNTH_KEY)
    if kind == "NASA files (local)":
        mp = st.sidebar.text_input("Master parquet", DEFAULT_MASTER)
        ip = st.sidebar.text_input("Impedance parquet", DEFAULT_IMP)
        if not os.path.exists(mp):
            st.sidebar.error("Master file not found — using the synthetic demo.")
            return dict(kind="synthetic", key=cache.SYNTH_KEY)
        s = os.stat(mp)
        return dict(kind="files", master=mp, imp=ip, key=cache.file_key(mp, s.st_mtime, s.st_size))
    fm = st.sidebar.file_uploader("battery_master_data.parquet", type="parquet")
    fi = st.sidebar.file_uploader("impedance_ground_truth.parquet", type="parquet")
    if fm is None:
        st.sidebar.info("Upload the master file — synthetic demo shown meanwhile.")
        return dict(kind="synthetic", key=cache.SYNTH_KEY)
    mb = fm.getvalue(); ib = fi.getvalue() if fi else None
    return dict(kind="upload", master_bytes=mb, imp_bytes=ib,
                key=f"upload-{fm.file_id}-{len(mb)}-{fi.file_id if fi else 'none'}")


def main():
    source = sidebar_source()
    st.session_state["_source"] = source
    st.sidebar.markdown("#### Definitions")
    eol = st.sidebar.number_input("EOL capacity (Ah)", 1.0, 1.9, C.EOL_AH, 0.05,
                                  help=f"One EOL definition everywhere: rate-normalised capacity below this for "
                                       f"{C.EOL_CONSECUTIVE} consecutive cycles.")
    try:
        cd = cache.cycle_data(source, eol)
    except Exception as e:
        st.error(f"Could not build the cycle table: {e}")
        st.stop()
    groups = [g for g in registry.GROUPS if g in set(cd.summary.group)]
    sel_groups = st.sidebar.multiselect("Condition groups", groups, default=groups)
    cells = [c for c in cd.cells if registry.group_of(c) in sel_groups] or cd.cells
    st.sidebar.markdown("---")
    st.sidebar.caption(f"engine {battwin.__version__} · {len(cd.cells)} cells · {len(cd.table)} cycles · "
                       f"data {source['key'][:18]}")
    ctx = dict(source=source, eol=float(eol), cd=cd, cells=cells, key=source["key"], v=cache.V)
    blocks.hero()
    pages = [
        st.Page(lambda: diagnostics.render(ctx), title="Diagnostics", icon="🔬", url_path="diagnostics", default=True),
        st.Page(lambda: live_twin.render(ctx), title="Live twin", icon="🛰️", url_path="live-twin"),
        st.Page(lambda: ml.render(ctx), title="ML models", icon="🧠", url_path="ml"),
        st.Page(lambda: operations.render(ctx), title="Operations & maintenance", icon="⚙️", url_path="operations"),
        st.Page(lambda: study.render(ctx), title="Study results", icon="📊", url_path="study"),
    ]
    st.navigation(pages, position="top" if _supports_top() else "sidebar").run()


def _supports_top():
    try:
        from packaging.version import Version
        return Version(st.__version__) >= Version("1.46")
    except Exception:
        return False


main()
