"""Loading and dtype normalisation of the master and impedance parquet files."""
from __future__ import annotations
import hashlib
import io as _io
import numpy as np
import pandas as pd
from .registry import normalise_id

MASTER_COLS = ["Cell_ID", "Cycle_Index", "Cycle_Type", "Time_s", "Voltage_V", "Current_A",
               "Temp_C", "Capacity_Ah"]


def _read(src) -> pd.DataFrame:
    if isinstance(src, pd.DataFrame):
        return src.copy()
    if isinstance(src, (bytes, bytearray)):
        return pd.read_parquet(_io.BytesIO(src))
    return pd.read_parquet(src)


def normalise_master(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in MASTER_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"master data is missing columns: {missing}")
    df = df.copy()
    df["Cell_ID"] = df["Cell_ID"].map(normalise_id).astype(str)
    df["Cycle_Index"] = pd.to_numeric(df["Cycle_Index"], errors="coerce").round().astype("int64")
    df["Cycle_Type"] = df["Cycle_Type"].astype(str).str.strip().str.lower()
    for c in ["Time_s", "Voltage_V", "Current_A", "Temp_C", "Capacity_Ah"] + (
            ["Ambient_C"] if "Ambient_C" in df else []):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df.sort_values(["Cell_ID", "Cycle_Index", "Time_s"], kind="stable").reset_index(drop=True)


def normalise_impedance(df: pd.DataFrame | None) -> pd.DataFrame:
    empty = pd.DataFrame({"Cell_ID": pd.Series(dtype=str), "Cycle_Index": pd.Series(dtype="int64"),
                          "Re_ohm": pd.Series(dtype=float), "Rct_ohm": pd.Series(dtype=float)})
    if df is None or len(df) == 0 or not {"Cell_ID", "Cycle_Index"} <= set(df.columns):
        return empty        # impedance is optional; without keys it cannot be aligned to cycles
    df = df.copy()
    df["Cell_ID"] = df["Cell_ID"].map(normalise_id).astype(str)
    df["Cycle_Index"] = pd.to_numeric(df["Cycle_Index"], errors="coerce").round().astype("int64")
    for c in ["Re_ohm", "Rct_ohm"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce").astype("float64")
    df = df[["Cell_ID", "Cycle_Index", "Re_ohm", "Rct_ohm"]]
    # physically impossible values (negative / absurd) are logging artefacts
    for c in ["Re_ohm", "Rct_ohm"]:
        df.loc[(df[c] <= 0) | (df[c] > 5), c] = np.nan
    return df.sort_values(["Cell_ID", "Cycle_Index"]).reset_index(drop=True)


def load(master_src, imp_src=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = normalise_master(_read(master_src))
    i = normalise_impedance(_read(imp_src) if imp_src is not None else None)
    return m, i


def data_hash(*objs) -> str:
    h = hashlib.sha1()
    for o in objs:
        if o is None:
            h.update(b"none")
        elif isinstance(o, (bytes, bytearray)):
            h.update(o)
        elif isinstance(o, pd.DataFrame):
            h.update(pd.util.hash_pandas_object(o, index=False).values.tobytes())
        else:
            with open(o, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
    return h.hexdigest()[:16]
