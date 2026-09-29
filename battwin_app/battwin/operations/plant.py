"""Plant model for Mission 3, calibrated on a real battery.

fade per cycle r(s, I) = r_ref (I / I_ref)^alpha (1 + kappa (1 - s))
  r_ref, kappa : fitted on the selected cell's own SOH history (local slopes vs 1 - SOH)
  alpha        : current exponent from the fleet (per-cell fade rate vs current, cells at 24 °C);
                 literature-style default 1.0 when the fleet cannot identify it
R(s) = R_bol + gamma (1 - s) from the cell's R0 history (default slope if unavailable)
Every parameter carries its source so the app can show what was calibrated vs assumed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy import stats


@dataclass
class Plant:
    bol_ah: float = 1.9
    r_ref: float = 1.5e-3
    i_ref: float = 2.0
    alpha: float = 1.0
    kappa: float = 2.0
    r_bol: float = 0.08
    gamma: float = 0.15
    s_floor: float = 0.6
    sources: dict = field(default_factory=dict)

    def fade_per_cycle(self, soh, current):
        return self.r_ref * (current / self.i_ref) ** self.alpha * (1 + self.kappa * (1 - np.asarray(soh)))

    def resistance(self, soh):
        return self.r_bol + self.gamma * (1 - np.asarray(soh))


def _alpha_from_fleet(cd):
    rows = []
    for c in cd.cells:
        g = cd.cell(c)
        g = g[~g.outlier & (g.ambient.between(20, 28)) & (g.k <= 80)]
        s = cd.summary.set_index("cell").loc[c]
        if len(g) < 20 or s.I_levels > 1 or s.pulsed:
            continue
        rate = -stats.theilslopes(g.soh, g.k).slope
        if rate > 0:
            rows.append((np.log(g.I_mean.median()), np.log(rate)))
    if len(rows) < 3 or np.ptp([r[0] for r in rows]) < 0.3:
        return 1.0, "default (fleet cannot identify current dependence)"
    x, y = np.array(rows).T
    a = float(np.clip(np.polyfit(x, y, 1)[0], 0.0, 3.0))
    return a, f"fleet regression on {len(rows)} cells at 24 °C"


def calibrate(cd, cell: str, soh_floor: float | None = None) -> Plant:
    g = cd.cell(cell)
    g = g[~g.outlier & g.soh.notna()]
    p = Plant(bol_ah=float(g.bol_ah.iloc[0]), i_ref=float(round(g.I_mean.median() * 2) / 2 or 2.0))
    p.sources["bol_ah"] = f"{cell} robust beginning-of-life capacity"
    if len(g) >= 20:
        w = 15
        ks, rs, ss = [], [], []
        for j in range(0, len(g) - w, 5):
            seg = g.iloc[j:j + w]
            ks.append(seg.k.mean()); ss.append(seg.soh.mean())
            rs.append(-stats.theilslopes(seg.soh, seg.k).slope)
        x, y = 1 - np.array(ss), np.array(rs)
        b1, b0 = np.polyfit(x, y, 1)
        if b0 <= 0:
            b0 = max(np.median(y), 1e-5); b1 = 0.0
        p.r_ref, p.kappa = float(b0), float(np.clip(b1 / b0, 0, 20))
        p.sources["r_ref"] = p.sources["kappa"] = f"fitted on {cell} ({len(g)} cycles)"
    else:
        p.sources["r_ref"] = p.sources["kappa"] = "default (too few cycles)"
    p.alpha, p.sources["alpha"] = _alpha_from_fleet(cd)
    r0 = g[["soh", "R0"]].dropna()
    if len(r0) > 10 and np.ptp(r0.soh) > 0.03:
        gamma, rb = np.polyfit(1 - r0.soh, r0.R0, 1)
        p.r_bol, p.gamma = float(max(rb, 0.01)), float(max(gamma, 0.0))
        p.sources["resistance"] = f"R0 history of {cell}"
    else:
        p.sources["resistance"] = "default"
    p.s_floor = soh_floor if soh_floor is not None else float(min(cd.eol_ah / p.bol_ah - 0.05, 0.8))
    return p
