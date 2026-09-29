"""M2: how often should the twin update? Forecast error vs measurement interval."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..observers.stream import make_observations, default_members, Stream
from ..observers.ecm_ekf import ECMTwin
from .evaluate import forecast_accuracy

FAST_MODELS = ["Trend KF", "Power-law PF", "Hierarchical Bayes", "Mechanistic PF"]


def update_frequency(cd, cells, intervals=(1, 2, 5, 10, 20), h: int = 25, models=None) -> pd.DataFrame:
    """Models see a capacity measurement only every m cycles; forecasts are still issued every cycle
    and scored against all measured cycles."""
    models = models or FAST_MODELS
    rows = []
    for cell in cells:
        obs = make_observations(cd, cell, with_curves=False)
        eol = cd.eol_ah / float(cd.summary.set_index("cell").loc[cell, "bol_ah"])
        for m in intervals:
            s = Stream(obs, default_members(cd, cell, models), eol, horizons=(h,), measure_every=m,
                       rul_every=0).run()
            acc = forecast_accuracy(s.frames()["forecasts"])
            for r in acc[acc.h == h].itertuples():
                rows.append(dict(cell=cell, interval=m, model=r.model, rmse=r.rmse, coverage=r.coverage,
                                 measurements=int(np.ceil(len(obs) / m))))
    return pd.DataFrame(rows)


def ecm_check_interval(cd, cell, intervals=(1, 5, 10, 20, 10_000), h: int = 25) -> pd.DataFrame:
    """ECM twin with the periodic capacity check every n cycles (10 000 = voltage only)."""
    obs = make_observations(cd, cell, with_curves=True)
    eol = cd.eol_ah / float(cd.summary.set_index("cell").loc[cell, "bol_ah"])
    rows = []
    for n in intervals:
        s = Stream(obs, {"ECM twin": ECMTwin(check_every=n)}, eol, horizons=(h,), rul_every=0,
                   ensemble=False).run()
        acc = forecast_accuracy(s.frames()["forecasts"])
        a = acc[acc.h == h].iloc[0] if len(acc) else None
        rows.append(dict(cell=cell, check_every="never" if n >= 10_000 else n,
                         rmse=a.rmse if a is not None else np.nan, coverage=a.coverage if a is not None else np.nan))
    return pd.DataFrame(rows)
