"""Cohort-wide validation: stream every cell through every live model; collect per-cell metrics.
Failures are recorded as rows with status != 'ok', never dropped."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from .. import config as C, __version__
from ..observers.stream import run_cell, MODEL_NAMES, ENSEMBLE
from ..prognostics.evaluate import forecast_accuracy, estimation_accuracy, rul_accuracy


def run_cohort(cd, cells=None, models=None, horizons=C.HORIZONS, alpha=C.BAND_ALPHA, progress=None) -> dict:
    cells = cells or cd.cells
    models = models or MODEL_NAMES
    summ = cd.summary.set_index("cell")
    rows, est_rows, rul_rows, fails, shares = [], [], [], [], []
    t0 = time.time()
    for j, cell in enumerate(cells):
        if progress:
            progress(j, len(cells), cell)
        group = summ.loc[cell, "group"]
        try:
            r = run_cell(cd, cell, models, horizons, alpha)
        except Exception as e:
            fails.append(dict(cell=cell, model="all", error=repr(e)))
            for m in models + [ENSEMBLE]:
                for h in horizons:
                    rows.append(dict(cell=cell, group=group, model=m, h=h, status=f"failed: {e!r}"))
            continue
        status = r["status"]
        acc = forecast_accuracy(r["forecasts"])
        for a in acc.itertuples(index=False):
            ok = status.get(a.model, (True, ""))
            rows.append(dict(cell=cell, group=group, **a._asdict(),
                             status="ok" if ok[0] else f"invalid: {ok[1]}"))
        for m in models + [ENSEMBLE]:        # models with no forecasts at all
            if m not in set(acc.model) and (m != ENSEMBLE or len(models) > 1):
                rows.append(dict(cell=cell, group=group, model=m, h=np.nan, status="no forecasts"))
        for a in estimation_accuracy(r["estimates"]).itertuples(index=False):
            est_rows.append(dict(cell=cell, group=group, **a._asdict()))
        ru = rul_accuracy(r["rul"], summ.loc[cell, "eol_k"])
        for a in ru.itertuples(index=False):
            rul_rows.append(dict(cell=cell, group=group, **a._asdict()))
        for f in r["failures"].itertuples(index=False):
            fails.append(dict(cell=cell, model=f.model, error=f.error))
        e = r["estimates"]
        mp = e[e.model == "Mechanistic PF"]
        if len(mp) and "share_sei" in mp:
            last = mp.iloc[-1]
            shares.append(dict(cell=cell, group=group, share_sei=last.share_sei,
                               share_plating=last.share_plating, share_lam=last.share_lam))
    return dict(metrics=pd.DataFrame(rows), estimation=pd.DataFrame(est_rows), rul=pd.DataFrame(rul_rows),
                failures=pd.DataFrame(fails, columns=["cell", "model", "error"]), shares=pd.DataFrame(shares),
                meta=dict(engine=__version__, cells=len(cells), models=models, horizons=list(horizons),
                          alpha=alpha, eol_ah=cd.eol_ah, seconds=round(time.time() - t0, 1)))
