import numpy as np
import pandas as pd
from battwin.data import io, cycles, registry
from battwin.data.quality import hampel, detect_eol, robust_bol, EOL_NOT_REACHED
from battwin.data.soh import rate_normalise
from battwin.data.synthetic import make_synthetic


def test_dtype_normalisation_and_merge():
    m, i, _ = make_synthetic(kinds=("reference",), n=30)
    assert m.Cycle_Index.dtype == "float64" and i.Cycle_Index.dtype == "int32"
    m2, i2 = io.load(m, i)
    assert m2.Cycle_Index.dtype == "int64" and i2.Cycle_Index.dtype == "int64"
    cd = cycles.build(m2, i2)
    assert cd.table.Re_ohm.notna().all()       # impedance merged on every cycle


def test_id_normalisation():
    assert registry.normalise_id(5) == "B0005"
    assert registry.normalise_id("b0049") == "B0049"
    assert registry.group_of("B0038") == "Mixed conditions"
    assert registry.group_of("B0050") == "Corrupted logging"
    assert registry.info("B0033")["stop_ah"] == 1.6


def test_nasa_registry_complete():
    assert len(registry.ALL_CELLS) == 34
    assert all(registry.group_of(c) != "Unknown" for c in registry.ALL_CELLS)


def test_hampel_flags_dips():
    x = np.linspace(1.9, 1.6, 100)
    x[40] -= 0.25
    out, _ = hampel(x)
    assert out[40] and out.sum() <= 2


def test_eol_requires_consecutive_cycles():
    k = np.arange(1, 11)
    cap = np.array([1.6, 1.55, 1.38, 1.5, 1.45, 1.39, 1.38, 1.37, 1.36, 1.35])
    assert detect_eol(k, cap, 1.4, 3) == (6, "reached")
    assert detect_eol(k, cap + 0.2, 1.4, 3) == (None, EOL_NOT_REACHED)


def test_robust_bol_ignores_crash_values():
    cap = np.r_[[2.6, 0.4, 2.45, 0.2, 2.5], np.linspace(1.88, 1.7, 40)]
    bol, crash = robust_bol(cap, np.zeros(len(cap), bool))
    assert crash and abs(bol - 1.87) < 0.03


def test_rate_normalisation_removes_load_steps():
    rng = np.random.default_rng(0)
    k = np.arange(1, 151)
    I = np.tile([2.0] * 8 + [4.0] * 8 + [1.0] * 8, 7)[:150]
    true = 1.9 - 0.002 * k
    cap = true * np.exp(np.select([I == 4, I == 1], [-0.07, 0.01], 0)) + rng.normal(0, 0.003, 150)
    cn, info = rate_normalise(k, cap, I, np.full(150, 24.0), np.ones(150, bool))
    assert info["normalised"]
    assert np.sqrt(np.mean((cn - true) ** 2)) < 0.006
    assert np.sqrt(np.mean((cap - true) ** 2)) > 0.05


def test_cycle_table_against_truth(synth, cells):
    cd, truth = synth
    x = cd.table.merge(truth, on=["Cell_ID", "k"])
    for kind in ["reference", "knee", "cold", "mixed", "pulsed"]:
        d = x[x.Cell_ID == cells[kind]]
        err = np.sqrt(np.nanmean((d.soh_clean - d.soh_true) ** 2))
        assert err < 0.01, (kind, err)


def test_crash_cell_flagged(synth, cells):
    cd, _ = synth
    s = cd.summary.set_index("cell").loc[cells["crash"]]
    assert s.crash_flag and 1.8 < s.bol_ah < 1.95
    assert cd.cell(cells["crash"]).head(6).outlier.all()


def test_pulsed_detected_and_eol_status(synth, cells):
    cd, _ = synth
    s = cd.summary.set_index("cell")
    assert s.loc[cells["pulsed"], "pulsed"]
    assert s.loc[cells["reference"], "eol_status"] in ("reached", EOL_NOT_REACHED)
    assert s.eol_status.notna().all()                 # never blank
    assert s.loc[cells["cold"], "eol_status"] == "reached"
