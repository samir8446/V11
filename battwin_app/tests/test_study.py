import numpy as np
import pandas as pd
from battwin.study.cohort import run_cohort
from battwin.study.compare import pairwise_wilcoxon, leaderboard, power_label, common_cells
from battwin.study.coverage import coverage_table
from battwin.study.summary import study_tables
from battwin.study import export, answers
from battwin import __version__


def _fake_metrics():
    rng = np.random.default_rng(0)
    rows = []
    for c in range(12):
        for m, base in [("A", 0.010), ("B", 0.020)]:
            rows.append(dict(cell=f"C{c}", group="G1" if c < 8 else "G2", model=m, h=25, n=50,
                             rmse=base + rng.normal(0, 0.002), r2=0.9, coverage=0.9, coverage_model=0.8,
                             fade_skill=0.5, status="ok"))
    rows.append(dict(cell="C0", group="G1", model="C", h=25, n=0, rmse=np.nan, status="invalid: x"))
    return pd.DataFrame(rows)


def test_wilcoxon_per_cell_pairing():
    m = _fake_metrics()
    w = pairwise_wilcoxon(m, 25, ["A", "B"])
    r = w.iloc[0]
    assert r.n == 12 and r.p < 0.01 and r.significant and r.a_better_frac == 1.0


def test_power_labels():
    assert power_label(5).startswith("insufficient")
    assert power_label(8).startswith("low")
    assert power_label(12).startswith("adequate")


def test_common_cells_excludes_failures():
    m = _fake_metrics()
    assert common_cells(m, ["A", "B", "C"], 25) == []
    lb = leaderboard(m, 25, ["A", "B"])
    assert lb.iloc[0].model == "A" and lb.n_cells.iloc[0] == 12


def test_coverage_table():
    c = coverage_table(_fake_metrics(), 0.1)
    assert {"achieved", "nominal", "ci_low", "ci_high", "note"} <= set(c.columns)


def test_cohort_end_to_end(synth, cells):
    cd, _ = synth
    sel = [cells["reference"], cells["pulsed"], cells["cold"]]
    res = run_cohort(cd, sel, ["Trend KF", "ECM twin", "Hierarchical Bayes"])
    m = res["metrics"]
    assert set(m.cell) == set(sel) and res["meta"]["engine"] == __version__
    assert (m[(m.cell == cells["pulsed"]) & (m.model == "ECM twin")].status.str.startswith("invalid")).all()
    t = study_tables(res, 25)
    assert len(t["leaderboard"]) and "p_holm" in t["wilcoxon"]
    html = export.html_report("t", {"M2": answers.m2(t["leaderboard"], t["wilcoxon"], h=25)}, t)
    assert "<table" in html
    assert len(export.csv_zip(t)) > 100
