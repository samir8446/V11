"""Plain-language answers per Mission, generated from results with explicit evidence strength."""
from __future__ import annotations
import numpy as np


def m1(hi_rank, one_several, stress_res, shares=None) -> list[str]:
    out = []
    if hi_rank is not None and len(hi_rank):
        top = hi_rank.iloc[0]
        nc = hi_rank[~hi_rank.indicator.str.contains("Capacity|energy", case=False)]
        out.append(f"Best overall health indicator: **{top.indicator}** (score {top.score:.2f}, "
                   f"LOCO RMSE {top.loco_rmse:.3f} SOH).")
        if len(nc):
            b = nc.iloc[0]
            out.append(f"Best non-capacity indicator: **{b.indicator}** (score {b.score:.2f}).")
    if one_several and one_several.get("verdict") != "undetermined":
        sel = ", ".join(one_several["selected"])
        out.append(f"One or several? **{one_several['verdict'].capitalize()}** — forward selection keeps "
                   f"[{sel}] (LOCO RMSE {one_several['single_rmse']:.3f} → {one_several['set_rmse']:.3f}).")
    if stress_res and stress_res.get("ok"):
        c = stress_res["coef"]
        sig = c[c.p_value < 0.05]
        conf = "; confounding: " + ", ".join(stress_res["confounding"]) if stress_res["confounding"] else ""
        if len(sig):
            out.append("Stress factors with p<0.05: " + ", ".join(f"{r.factor} ({r.std_coef:+.2f})" for r in sig.itertuples())
                       + f" — {stress_res['power']}, R²={stress_res['r2']:.2f}{conf}.")
        else:
            out.append(f"No stress factor reaches p<0.05 ({stress_res['power']}, n={stress_res['n']}){conf}.")
    if shares is not None and len(shares):
        g = shares.groupby("group")[["share_sei", "share_plating", "share_lam"]].mean()
        dom = g.idxmax(axis=1)
        out.append("Dominant mechanism (mechanistic PF, model-based): " +
                   ", ".join(f"{k}: {v.replace('share_', '').upper()}" for k, v in dom.items()) + ".")
    return out or ["Not enough data for Mission 1 answers."]


def m2(leader, wilcox, update_df=None, info=None, h=None) -> list[str]:
    out = []
    if leader is not None and len(leader):
        b = leader.iloc[0]
        out.append(f"Most accurate at h={h}: **{b.model}** (median RMSE {b.rmse_median:.3f} SOH, "
                   f"median R² {b.r2_median:.2f}, coverage {b.coverage:.0%}, {int(b.n_cells)} common cells).")
        if wilcox is not None and len(wilcox):
            w = wilcox[(wilcox.model_a == b.model) | (wilcox.model_b == b.model)]
            sig = w[w.significant]
            out.append(f"Significantly different from {len(sig)} of {len(w)} other models (Holm-corrected Wilcoxon; "
                       f"{w.power.iloc[0] if len(w) else ''}).")
    if update_df is not None and len(update_df):
        u = update_df.groupby("interval").rmse.median()
        base = u.iloc[0]
        ok = u[u <= base * 1.10]
        out.append(f"Update interval: measuring every **{int(ok.index.max())}** cycles keeps the error within 10 % of "
                   f"every-cycle updating (RMSE {base:.3f} → {ok.iloc[-1]:.3f}).")
    if info is not None and len(info.get("single", [])):
        s = info["single"]
        out.append(f"Most informative measurement besides capacity: **{s.iloc[0].measurement}** "
                   f"(grouped-CV RMSE {s.iloc[0].rmse:.3f}).")
    return out or ["Run the cohort study for Mission 2 answers."]


def m3(summary, plant) -> list[str]:
    if summary is None or not len(summary):
        return ["Run the operations optimisation for Mission 3 answers."]
    dp = summary.iloc[0]
    others = summary.iloc[1:]
    gain = [f"{(dp.profit_pv - r.profit_pv):+.2f} CU vs {r.policy}" for r in others.itertuples()]
    calib = ", ".join(f"{k}: {v}" for k, v in plant.sources.items()) if plant else ""
    return [f"Joint optimum (DP): discounted profit **{dp.profit_pv:.2f} CU**, {int(dp.replacements)} replacements; "
            + "; ".join(gain) + ".", f"Plant calibration — {calib}."]
