"""Fair model comparison: common cells only, per-cell paired Wilcoxon tests (one value per cell,
no pseudo-replication across cycles), Holm correction, power labels for small groups."""
from __future__ import annotations
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def power_label(n: int) -> str:
    # two-sided exact Wilcoxon: smallest attainable p is 2 / 2^n
    if n < 6:
        return f"insufficient (n={n}: p<0.05 unattainable)"
    if n < 10:
        return f"low power (n={n})"
    return f"adequate (n={n})"


def common_cells(metrics: pd.DataFrame, models, h) -> list[str]:
    m = metrics[(metrics.h == h) & (metrics.status == "ok") & metrics.rmse.notna()]
    sets = [set(m[m.model == x].cell) for x in models]
    return sorted(set.intersection(*sets)) if sets else []


def leaderboard(metrics: pd.DataFrame, h: int, models=None, group=None) -> pd.DataFrame:
    m = metrics if group is None else metrics[metrics.group == group]
    models = models or sorted(m.model.dropna().unique())
    cells = common_cells(m, models, h)
    d = m[(m.h == h) & m.cell.isin(cells) & m.model.isin(models)]
    lb = d.groupby("model").agg(rmse_median=("rmse", "median"), rmse_mean=("rmse", "mean"),
                                r2_median=("r2", "median"), coverage=("coverage", "mean"),
                                fade_skill=("fade_skill", "median")).reset_index()
    lb["n_cells"] = len(cells)
    lb["rank"] = lb.rmse_median.rank()
    fails = metrics[(metrics.h == h) | metrics.h.isna()]
    fails = fails[fails.status != "ok"].groupby("model").size()
    lb["not_ok"] = lb.model.map(fails).fillna(0).astype(int)
    return lb.sort_values("rank").reset_index(drop=True)


def pairwise_wilcoxon(metrics: pd.DataFrame, h: int, models=None, group=None) -> pd.DataFrame:
    m = metrics if group is None else metrics[metrics.group == group]
    models = models or sorted(m.model.dropna().unique())
    cells = common_cells(m, models, h)
    d = m[(m.h == h) & m.cell.isin(cells)].pivot_table(index="cell", columns="model", values="rmse")
    rows = []
    for a, b in combinations(models, 2):
        if a not in d or b not in d:
            continue
        x, y = d[a].to_numpy(), d[b].to_numpy()
        n = len(x)
        diff = x - y
        if n < 2 or np.allclose(diff, 0):
            p = np.nan
        else:
            try:
                p = float(wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue)
            except ValueError:
                p = np.nan
        rows.append(dict(model_a=a, model_b=b, n=n, median_diff=float(np.median(diff)) if n else np.nan,
                         a_better_frac=float(np.mean(diff < 0)) if n else np.nan, p=p, power=power_label(n)))
    out = pd.DataFrame(rows)
    if len(out):
        order = out.p.fillna(1).argsort().to_numpy()
        k = len(out)
        adj = np.empty(k)
        running = 0.0
        for rank, j in enumerate(order):
            running = max(running, min(1.0, (k - rank) * (out.p.iloc[j] if np.isfinite(out.p.iloc[j]) else 1.0)))
            adj[j] = running
        out["p_holm"] = adj
        out["significant"] = out.p_holm < 0.05
    return out
