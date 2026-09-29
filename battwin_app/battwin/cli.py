"""Offline cohort study.

python -m battwin.cli study --master battery_master_data.parquet --imp impedance_ground_truth.parquet \
       --out results/study [--eol 1.4] [--cells B0005 B0006] [--models "Trend KF" ...] [--synthetic]
Writes metrics CSVs, summary.md and report.html.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
from . import __version__, config as C
from .data import io, cycles
from .data.synthetic import make_synthetic
from .diagnostics import indicators, stress
from .study.cohort import run_cohort
from .study.summary import study_tables
from .study import answers, export


def main(argv=None):
    p = argparse.ArgumentParser(prog="battwin")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("study", help="run the cohort study")
    s.add_argument("--master"); s.add_argument("--imp"); s.add_argument("--out", default="results/study")
    s.add_argument("--eol", type=float, default=C.EOL_AH); s.add_argument("--cells", nargs="*")
    s.add_argument("--models", nargs="*"); s.add_argument("--h", type=int, default=25)
    s.add_argument("--synthetic", action="store_true", help="use synthetic ground truth instead of files")
    a = p.parse_args(argv)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.synthetic:
        m, i, _ = make_synthetic()
        m, i = io.load(m, i)
    else:
        if not a.master:
            p.error("--master is required unless --synthetic")
        m, i = io.load(a.master, a.imp)
    cd = cycles.build(m, i, a.eol)
    print(f"battwin {__version__}: {len(cd.cells)} cells, {len(cd.table)} discharge cycles", flush=True)
    res = run_cohort(cd, a.cells, a.models, progress=lambda j, n, c: print(f"[{j+1}/{n}] {c}", flush=True))
    tabs = study_tables(res, a.h)
    hi = indicators.rank_indicators(cd.table)
    sev = indicators.one_or_several(cd.table)
    st = stress.regress(stress.fade_rates(cd))
    ans = {"Mission 1 — Health": answers.m1(hi, sev, st, res["shares"]),
           "Mission 2 — Self-updating twin": answers.m2(tabs["leaderboard"][tabs["leaderboard"].group == "All"],
                                                        tabs["wilcoxon"][tabs["wilcoxon"].group == "All"], h=a.h)}
    tables = dict(cells=cd.summary, metrics=res["metrics"], estimation=res["estimation"], rul=res["rul"],
                  failures=res["failures"], shares=res["shares"], hi_ranking=hi, **tabs)
    for k, v in tables.items():
        v.to_csv(out / f"{k}.csv", index=False)
    (out / "report.html").write_text(export.html_report("Battery twin — cohort study", ans, tables,
                                                        meta=res["meta"]), encoding="utf-8")
    md = [f"# Cohort study (engine {__version__})", "", "```", json.dumps(res["meta"], indent=1), "```"]
    for k, lines in ans.items():
        md += ["", f"## {k}", *[f"- {l}" for l in lines]]
    md += ["", f"## Leaderboard (h={a.h}, all cells)", tabs["leaderboard"][tabs["leaderboard"].group == "All"]
           .to_markdown(index=False) if _has_tabulate() else tabs["leaderboard"].to_string()]
    md += ["", f"Failures recorded: {len(res['failures'])}"]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out}/summary.md, report.html and CSVs")
    return 0


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    sys.exit(main())
