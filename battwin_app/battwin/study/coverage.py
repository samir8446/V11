"""Achieved vs nominal coverage of the forecast bands, pooled per cell first (one value per cell)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..prognostics.metrics import wilson


def coverage_table(metrics: pd.DataFrame, alpha: float) -> pd.DataFrame:
    d = metrics[(metrics.status == "ok") & metrics.coverage.notna()]
    rows = []
    for (g, m, h), x in d.groupby(["group", "model", "h"]):
        n = len(x)
        k = (x.coverage * x.n).sum(); N = x.n.sum()
        lo, hi = wilson(k, N)
        rows.append(dict(group=g, model=m, h=h, n_cells=n, nominal=1 - alpha,
                         achieved=float(x.coverage.mean()), ci_low=lo, ci_high=hi,
                         model_band=float(x.coverage_model.mean()),
                         note="calibrated" if lo <= 1 - alpha <= hi else ("under-covers" if hi < 1 - alpha else "conservative")))
    return pd.DataFrame(rows)
