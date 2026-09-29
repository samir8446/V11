"""PCA of standardised health indicators (cycle-level rows)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .indicators import HI_COLUMNS


def hi_pca(table: pd.DataFrame, cells=None, n_components: int = 3) -> dict:
    t = table[~table["outlier"]]
    if cells is not None:
        t = t[t["Cell_ID"].isin(cells)]
    cols = [c for c in HI_COLUMNS.values() if c in t and t[c].notna().mean() > 0.6 and t[c].std() > 0]
    d = t[["Cell_ID", "k", "soh"] + cols].dropna()
    if len(d) < 5 or len(cols) < 2:
        return dict(ok=False, reason="not enough complete HI rows")
    X = d[cols].to_numpy(float)
    Z = (X - X.mean(0)) / X.std(0)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    ev = S**2 / (S**2).sum()
    n = min(n_components, len(cols))
    scores = pd.DataFrame(U[:, :n] * S[:n], columns=[f"PC{i+1}" for i in range(n)])
    scores[["Cell_ID", "k", "soh"]] = d[["Cell_ID", "k", "soh"]].to_numpy()
    names = {v: k for k, v in HI_COLUMNS.items()}
    load = pd.DataFrame(Vt[:n].T, index=[names[c] for c in cols], columns=scores.columns[:n])
    corr_pc1_soh = float(np.corrcoef(scores["PC1"].astype(float), scores["soh"].astype(float))[0, 1])
    return dict(ok=True, explained=ev, scores=scores, loadings=load, pc1_soh_corr=corr_pc1_soh)
