"""One call that turns a cohort run into the study tables and answers (used by app and CLI)."""
from __future__ import annotations
import pandas as pd
from .compare import leaderboard, pairwise_wilcoxon
from .coverage import coverage_table


def study_tables(cohort: dict, h: int) -> dict:
    m = cohort["metrics"]
    groups = sorted(m.group.dropna().unique())
    lbs = [leaderboard(m, h).assign(group="All")] + [leaderboard(m, h, group=g).assign(group=g) for g in groups]
    wil = [pairwise_wilcoxon(m, h).assign(group="All")] + [pairwise_wilcoxon(m, h, group=g).assign(group=g)
                                                          for g in groups]
    return dict(leaderboard=pd.concat(lbs, ignore_index=True), wilcoxon=pd.concat(wil, ignore_index=True),
                coverage=coverage_table(m, cohort["meta"]["alpha"]))
