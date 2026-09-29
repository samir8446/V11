"""Replay a cell one discharge cycle at a time through all live models.

At every cycle: score the previous one-step forecasts, update ensemble weights, calibrate
conformal bands with residuals whose target has arrived, update each member, then issue
forecasts at the study horizons (and EOL predictions every few cycles).
"""
from __future__ import annotations
import dataclasses
import numpy as np
import pandas as pd
from .. import config as C
from ..data import registry
from .base import CycleObs
from .conformal import Conformal
from .ensemble import EnsembleWeights
from .trend_kf import TrendKF
from .powerlaw_pf import PowerLawPF, fleet_prior
from .hier_bayes import HierBayes, population_prior
from .mech_pf import MechanisticPF
from .ecm_ekf import ECMTwin
from ..prognostics.rul import RUL_GRID, crossing

ENSEMBLE = "Ensemble"
MODEL_NAMES = ["ECM twin", "Mechanistic PF", "Power-law PF", "Trend KF", "Hierarchical Bayes"]


def make_observations(cd, cell: str, with_curves: bool = True) -> list[CycleObs]:
    g = cd.cell(cell)
    curves = cd.curves(cell) if with_curves and cd.master is not None else {}
    obs = []
    for r in g.itertuples(index=False):
        T = r.T_mean if np.isfinite(r.T_mean) else r.ambient
        obs.append(CycleObs(cell=cell, k=int(r.k), soh=float(r.soh_clean), cap_norm=float(r.cap_norm),
                            current=float(r.I_mean), temperature=float(T), ambient=float(r.ambient),
                            bol_ah=float(r.bol_ah), R0=float(r.R0), Re=float(r.Re_ohm), Rct=float(r.Rct_ohm),
                            curve=curves.get(int(r.Cycle_Index)), outlier=bool(r.outlier)))
    return obs


def default_members(cd, cell: str, which=None, seed: int = C.SEED) -> dict:
    which = which or MODEL_NAMES
    fleet = cd.table[cd.table.Cell_ID != cell]
    pulsed = bool(cd.summary.set_index("cell").loc[cell, "pulsed"]) or registry.is_pulsed(cell)
    make = {
        "ECM twin": lambda: ECMTwin(pulsed=pulsed),
        "Mechanistic PF": lambda: MechanisticPF(seed=seed),
        "Power-law PF": lambda: PowerLawPF(prior=fleet_prior(fleet), seed=seed),
        "Trend KF": lambda: TrendKF(),
        "Hierarchical Bayes": lambda: HierBayes(prior=population_prior(fleet)),
    }
    return {n: make[n]() for n in which if n in make}


class Stream:
    def __init__(self, obs, members: dict, eol_soh: float, horizons=C.HORIZONS, alpha=C.BAND_ALPHA,
                 measure_every: int = 1, rul_every: int = 5, ensemble: bool = True):
        self.obs, self.members = obs, members
        self.h = np.array(sorted(set([1, *horizons])))
        self.alpha, self.eol_soh, self.m, self.rul_every = alpha, eol_soh, measure_every, rul_every
        self.ens = ensemble and len(members) > 1
        names = list(members) + ([ENSEMBLE] if self.ens else [])
        self.weights = EnsembleWeights(list(members))
        self.conf = Conformal(alpha)
        self.i = 0
        self.est, self.fc, self.w_rows, self.rul, self.failures = [], [], [], [], []
        self.pending: dict[int, list[int]] = {}
        self.one_step: dict[str, float] = {}
        self.names = names

    @property
    def done(self):
        return self.i >= len(self.obs)

    def _masked(self, o):
        if self.m > 1 and (self.i % self.m) != 0:
            return dataclasses.replace(o, soh=np.nan, cap_norm=np.nan, curve=None)
        return o

    def step(self):
        if self.done:
            return False
        o_true = self.obs[self.i]
        o = self._masked(o_true)
        measured = np.isfinite(o.soh)
        # 1) score one-step forecasts; resolve pending forecasts at this cycle
        if measured:
            for n, p in self.one_step.items():
                if n in self.members:
                    self.weights.observe(n, abs(o.soh - p))
        for j in self.pending.pop(o.k, []):
            rec = self.fc[j]
            rec["actual"] = o_true.soh
            self.conf.add((rec["model"], rec["h"]), o_true.soh - rec["pred"])
        # 2) update members
        ests = {}
        for n, mdl in self.members.items():
            try:
                ests[n] = mdl.update(o)
            except Exception as e:  # reported, never hidden
                self.failures.append(dict(k=o.k, model=n, error=repr(e)))
                mdl.valid, mdl.reason = False, f"failed: {e!r}"
                ests[n] = None
        valid = {n: m.valid for n, m in self.members.items()}
        finite = {n: ests[n] is not None and np.isfinite(ests[n].soh) for n in self.members}
        w = self.weights.weights(valid, finite) if self.ens else {}
        for n, e in ests.items():
            self.est.append(dict(k=o.k, model=n, soh=e.soh if e else np.nan, std=e.std if e else np.nan,
                                 measured=o_true.soh, valid=valid[n], **(e.extras if e else {})))
        if self.ens and w:
            ms = np.array([ests[n].soh for n in w]); ss = np.array([ests[n].std for n in w])
            wv = np.array(list(w.values()))
            mu = float(wv @ ms)
            self.est.append(dict(k=o.k, model=ENSEMBLE, soh=mu, std=float(np.sqrt(wv @ (ss**2 + (ms - mu) ** 2))),
                                 measured=o_true.soh, valid=True))
        for n in self.members:
            self.w_rows.append(dict(k=o.k, model=n, weight=w.get(n, 0.0),
                                    excluded=self.weights.excluded.get(n, "")))
        # 3) forecasts
        do_rul = self.rul_every and (self.i % self.rul_every == 0)
        hs = np.concatenate([self.h, RUL_GRID]) if do_rul else self.h
        paths = {}
        for n, mdl in self.members.items():
            if not finite[n]:
                continue
            try:
                paths[n] = [np.asarray(a, float) for a in mdl.forecast(hs, self.alpha)]
            except Exception as e:
                self.failures.append(dict(k=o.k, model=n, error=f"forecast: {e!r}"))
        if self.ens and w:
            ok = [n for n in w if n in paths]
            if ok:
                wv = np.array([w[n] for n in ok]); wv /= wv.sum()
                paths[ENSEMBLE] = [sum(wi * paths[n][j] for wi, n in zip(wv, ok)) for j in range(3)]
        self.one_step = {n: p[0][0] for n, p in paths.items()}
        nh = len(self.h)
        for n, (m, lo, hi) in paths.items():
            for j, h in enumerate(self.h):
                clo, chi, cal = self.conf.band((n, int(h)), m[j], lo[j], hi[j])
                self.fc.append(dict(model=n, k0=o.k, h=int(h), k_target=o.k + int(h), pred=m[j],
                                    lo_model=lo[j], hi_model=hi[j], lo=clo, hi=chi, calibrated=cal,
                                    soh_now=o_true.soh, actual=np.nan))
                self.pending.setdefault(o.k + int(h), []).append(len(self.fc) - 1)
            if do_rul:
                self.rul.append(dict(model=n, k0=o.k,
                                     eol_pred=o.k + crossing(RUL_GRID, m[nh:], self.eol_soh),
                                     eol_early=o.k + crossing(RUL_GRID, lo[nh:], self.eol_soh),
                                     eol_late=o.k + crossing(RUL_GRID, hi[nh:], self.eol_soh)))
        self.i += 1
        return True

    def run(self, n: int | None = None):
        steps = 0
        while not self.done and (n is None or steps < n):
            self.step(); steps += 1
        return self

    def frames(self) -> dict:
        return dict(estimates=pd.DataFrame(self.est), forecasts=pd.DataFrame(self.fc),
                    weights=pd.DataFrame(self.w_rows), rul=pd.DataFrame(self.rul),
                    failures=pd.DataFrame(self.failures, columns=["k", "model", "error"]),
                    status={n: (m.valid, m.reason) for n, m in self.members.items()})


def run_cell(cd, cell, which=None, horizons=C.HORIZONS, alpha=C.BAND_ALPHA, measure_every=1,
             with_curves=True, rul_every=5, seed=C.SEED) -> dict:
    obs = make_observations(cd, cell, with_curves=with_curves and (which is None or "ECM twin" in which))
    members = default_members(cd, cell, which, seed)
    eol_soh = cd.eol_ah / float(cd.summary.set_index("cell").loc[cell, "bol_ah"])
    s = Stream(obs, members, eol_soh, horizons, alpha, measure_every, rul_every).run()
    out = s.frames()
    out["eol_soh"] = eol_soh
    return out
