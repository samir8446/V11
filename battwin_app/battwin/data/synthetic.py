"""Synthetic ground truth for tests and the demo mode.

Degradation is simulated at electrode level (half-cell model), so capacity, voltage curves, rate
dependence, knees and cold behaviour all emerge from LLI / LAM_PE / LAM_NE:
  SEI     : LLI growth ~ sqrt(k), Arrhenius in T
  plating : LLI growth, strong at low T and high I, amplified as LAM_NE lowers N/P  -> knees
  LAM     : LAM_NE ~ I^1.5, LAM_PE ~ I * Arrhenius
"""
from __future__ import annotations
import dataclasses
import numpy as np
import pandas as pd
from ..diagnostics.halfcell import BOL, discharge, profile
from . import registry

SCENARIOS = {
    #            T    I       cutoff  sei    plating  lam_ne  lam_pe  n
    "reference": (24, 2.0,    2.7,    1.2e-2, 2e-5,   1.0e-4, 5e-5,   150),
    "knee":      (24, 2.0,    2.5,    6e-3,  4e-4,    7.0e-4, 5e-5,   150),
    "hot":       (43, 4.0,    2.2,    8e-3,  1e-5,    1.5e-4, 1e-4,   120),
    "cold":      (4,  2.0,    2.5,    3e-3,  2.5e-4,  2.0e-4, 5e-5,   120),
    "high":      (24, 4.0,    2.2,    1.1e-2, 4e-5,   2.0e-4, 6e-5,   130),
    "mixed":     (24, "mix",  2.5,    8e-3,  3e-5,    1.5e-4, 6e-5,   150),
    "pulsed":    (24, "pulse", 2.5,   1.0e-2, 3e-5,   1.5e-4, 6e-5,   120),
    "crash":     (4,  2.0,    2.7,    3e-3,  2e-4,    2.0e-4, 5e-5,   110),
}
GROUP = {"reference": "Reference", "knee": "Reference", "hot": "Hot", "cold": "Cold",
         "high": "High current", "mixed": "Mixed conditions", "pulsed": "Pulsed load",
         "crash": "Corrupted logging"}


def _arr(T, Ea=30e3):
    return np.exp(Ea / 8.314 * (1 / 297.15 - 1 / (T + 273.15)))


def _r_temp(T):
    return np.exp(1800 * (1 / (T + 273.15) - 1 / 297.15))


def _conditions(kind, T, I, n):
    if I == "mix":
        cur = np.array([[2.0] * 8 + [4.0] * 8 + [1.0] * 8][0] * (n // 24 + 1))[:n]
        amb = np.where((np.arange(n) // 36) % 2 == 0, 24.0, 44.0)
    else:
        cur = np.full(n, 4.0 if I == "pulse" else float(I))
        amb = np.full(n, float(T))
    return cur, amb


def simulate_cell(cid: str, kind: str, seed: int = 0, n: int | None = None, noise: float = 0.004,
                  outlier_p: float = 0.02):
    T0, I, cutoff, a_sei, b_pl, c_ne, c_pe, n_def = SCENARIOS[kind]
    n = n or n_def
    rng = np.random.default_rng(seed)
    scale = np.exp(rng.normal(0, 0.12, 4))
    a_sei, b_pl, c_ne, c_pe = a_sei * scale[0], b_pl * scale[1], c_ne * scale[2], c_pe * scale[3]
    cur, amb = _conditions(kind, T0, I, n)
    li = sei = pl = lam_ne = lam_pe = 0.0
    rows, samples, imp = [], [], []
    ci = 0
    for k in range(1, n + 1):
        Ik, Tk = cur[k - 1], amb[k - 1]
        i_eff = Ik * (0.5 if kind == "pulsed" else 1.0) / 2.0
        d_sei = a_sei * _arr(Tk) / (2 * np.sqrt(k))
        d_pl = b_pl * i_eff * np.exp((15 - Tk) / 7) * (1 + (max(0.0, lam_ne - 0.05) / 0.015) ** 3)
        sei += d_sei
        pl += d_pl
        lam_ne += c_ne * i_eff ** 1.5
        lam_pe += c_pe * i_eff * _arr(Tk)
        lli = sei + pl
        e = dataclasses.replace(BOL, Li=BOL.Li * (1 - lli), Qn=BOL.Qn * (1 - lam_ne),
                                Qp=BOL.Qp * (1 - lam_pe), R=BOL.R * (1 + 2 * lli + lam_ne))
        e_meas = dataclasses.replace(e, R=e.R * _r_temp(Tk))
        q, v, cap = discharge(e_meas, Ik, cutoff, n=40)
        cap_ref = (discharge(dataclasses.replace(e, R=e.R * _r_temp(24.0)), 2.0, cutoff, n=10)[2]
                   if kind == "mixed" else cap)
        meas = cap + rng.normal(0, noise)
        if rng.random() < outlier_p:
            meas *= 0.85
        if kind == "crash" and k <= 6:
            meas = [2.6, 0.4, 2.45, 0.2, 2.5, 3.1][k - 1]
        # charge cycle
        ci += 1
        cc_t = cap / 1.5 * 3600 * (0.75 - 0.2 * lli)
        tc = np.linspace(0, cc_t * 1.4, 12)
        vc = np.where(tc <= cc_t, 3.9 + 0.3 * tc / cc_t, 4.2)
        ic = np.where(tc <= cc_t, 1.5, 1.5 * np.exp(-(tc - cc_t) / (0.2 * cc_t + 1)))
        for a, b, c in zip(tc, vc, ic):
            samples.append((cid, ci, "charge", a, b, c, Tk + 1, np.nan, Tk))
        # discharge cycle
        ci += 1
        rise = 0.6 * Ik ** 2 * e_meas.R * 10
        if kind == "pulsed":
            pq, pocv, pshape = profile(e_meas)
            m = int(np.ceil(cap / (4.0 * 10 / 3600))) * 4 + 1
            t = np.arange(m) * 5.0
            on = (t % 20) < 10
            ii = np.where(on, -4.0, 0.0)
            ii[0] = 0.0
            qq = np.concatenate([[0.0], np.cumsum(-ii[:-1] * 5 / 3600)])
            keep = qq <= cap
            t, ii, qq = t[keep], ii[keep], qq[keep]
            vv = np.interp(qq, pq, pocv) - np.abs(ii) * (e_meas.R + np.interp(qq, pq, pshape))
        else:
            t = np.concatenate([[0.0], 10 + q / Ik * 3600])
            vv = np.concatenate([[float(profile(e_meas, 3)[1][0])], v])
            ii = np.concatenate([[0.0], np.full(len(q), -Ik)])
            qq = np.concatenate([[0.0], q])
        vv = vv + rng.normal(0, 0.002, len(vv))
        Tt = Tk + rise * qq / max(cap, 1e-3)
        for a, b, c, d in zip(t, vv, ii, Tt):
            samples.append((cid, ci, "discharge", a, b, c, d, meas, Tk))
        if k % 10 == 1:
            ci += 1
            imp.append((cid, ci, 0.045 * (1 + lli) + rng.normal(0, 0.001),
                        0.07 * (1 + 3 * lam_ne + 2 * lli) * _r_temp(Tk) ** 0.5 + rng.normal(0, 0.002)))
            samples.append((cid, ci, "impedance", 0.0, np.nan, np.nan, Tk, np.nan, Tk))
        tot = sei + pl + lam_ne + lam_pe + 1e-12
        rows.append(dict(Cell_ID=cid, k=k, cap_true=cap, cap_ref_true=cap_ref, current=Ik, ambient=Tk,
                         LLI=lli, LAM_NE=lam_ne, LAM_PE=lam_pe, share_sei=sei / tot,
                         share_plating=pl / tot, share_lam=(lam_ne + lam_pe) / tot, R=e.R))
    truth = pd.DataFrame(rows)
    truth["soh_true"] = truth["cap_ref_true"] / truth["cap_ref_true"].iloc[:3].mean()
    return samples, imp, truth


def make_synthetic(kinds=("reference", "knee", "hot", "cold", "high", "mixed", "pulsed", "crash"),
                   seed: int = 0, n: int | None = None, replicas: int = 1):
    """Return (master, impedance, truth). Cell ids S001.. are registered with their group."""
    samples, imps, truths = [], [], []
    j = 0
    for r in range(replicas):
        for kind in kinds:
            j += 1
            cid = f"S{j:03d}"
            T0, I, cutoff = SCENARIOS[kind][:3]
            registry.register(cid, GROUP[kind], ambient=[24, 44] if I == "mix" else T0,
                              load="pulsed" if I == "pulse" else ([1.0, 2.0, 4.0] if I == "mix" else I),
                              cutoff=cutoff)
            s, i, t = simulate_cell(cid, kind, seed=seed * 1000 + j, n=n)
            t["kind"] = kind
            samples += s
            imps += i
            truths.append(t)
    master = pd.DataFrame(samples, columns=["Cell_ID", "Cycle_Index", "Cycle_Type", "Time_s", "Voltage_V",
                                            "Current_A", "Temp_C", "Capacity_Ah", "Ambient_C"])
    master["Cycle_Index"] = master["Cycle_Index"].astype("float64")   # mimic the parquet dtype trap
    imp = pd.DataFrame(imps, columns=["Cell_ID", "Cycle_Index", "Re_ohm", "Rct_ohm"])
    imp["Cycle_Index"] = imp["Cycle_Index"].astype("int32")
    return master, imp, pd.concat(truths, ignore_index=True)
