"""Outlier flags (Hampel), robust beginning-of-life capacity, crash detection, EOL detection."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .. import config as C

EOL_NOT_REACHED = "EOL not reachable in data"


def hampel(x, window: int = C.HAMPEL_WINDOW, k: float = C.HAMPEL_K, floor: float = 0.004):
    """Return (outlier mask, rolling median). MAD floor avoids flagging noise on flat segments."""
    s = pd.Series(np.asarray(x, float))
    med = s.rolling(2 * window + 1, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(2 * window + 1, center=True, min_periods=1).median() * 1.4826
    scale = np.maximum(mad.to_numpy(), floor * np.nanmedian(np.abs(s)) if s.notna().any() else floor)
    out = (np.abs(s - med).to_numpy() > k * scale) & s.notna().to_numpy()
    return out, med.to_numpy()


def robust_bol(cap, outlier) -> tuple[float, bool]:
    """Robust beginning-of-life capacity and a crash/bogus-early-logging flag."""
    cap = np.asarray(cap, float)
    lo, hi = C.PLAUSIBLE_AH
    early = cap[:20]
    implaus = ~((early >= lo) & (early <= hi))
    ok = (~implaus) & (~np.asarray(outlier)[:20]) & np.isfinite(early)
    cand = early[ok]
    if cand.size == 0:
        cand = cap[np.isfinite(cap)]
        return (float(np.median(cand[: C.BOL_CYCLES])) if cand.size else float("nan")), True
    first = float(np.median(cand[: C.BOL_CYCLES]))
    later = cand[C.BOL_CYCLES:]
    mismatch = later.size >= 5 and abs(first / np.median(later[: 2 * C.BOL_CYCLES]) - 1) > 0.08
    crash = bool(implaus.mean() > 0.2 or mismatch)
    if mismatch:     # plausible-looking but bogus early level: trust the later, stable level
        return float(np.median(later[: 3 * C.BOL_CYCLES])), True
    return first, crash


def detect_eol(k, cap, threshold: float, consecutive: int = C.EOL_CONSECUTIVE, outlier=None):
    """First cycle of the first run of ``consecutive`` non-outlier values below threshold."""
    k, cap = np.asarray(k), np.asarray(cap, float)
    m = np.isfinite(cap) if outlier is None else np.isfinite(cap) & ~np.asarray(outlier)
    k, cap = k[m], cap[m]
    if cap.size < consecutive:
        return None, "insufficient data"
    below = cap < threshold
    run = 0
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= consecutive:
            return int(k[i - consecutive + 1]), "reached"
    return None, EOL_NOT_REACHED
