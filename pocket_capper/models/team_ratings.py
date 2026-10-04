"""Blind, market-free team power ratings.

One ridge regression over every team-game in the window:

    points_scored(team vs opp) = mu + OFF[team] + DEF[opp] + HFA * is_home

Each team-game is weighted by recency (exponential half-life in weeks), with an
extra discount on last season. The target can be a blend of actual points and an
efficiency-implied points number (EPA for NFL, PPA for CFB), which is far less
noisy than the scoreboard. Ridge shrinkage regresses every team to league average,
which is what you want early in a season.

Nothing in here ever sees a betting line.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge


@dataclass
class Ratings:
    mu: float
    hfa: float
    off: dict[str, float]
    deff: dict[str, float]
    games_played: dict[str, float] = field(default_factory=dict)

    def project(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        h = 0.0 if neutral else self.hfa
        hp = self.mu + self.off.get(home, 0.0) + self.deff.get(away, 0.0) + h / 2
        ap = self.mu + self.off.get(away, 0.0) + self.deff.get(home, 0.0) - h / 2
        return hp, ap

    def table(self) -> pd.DataFrame:
        teams = sorted(set(self.off) | set(self.deff))
        df = pd.DataFrame(
            {
                "team": teams,
                "off": [self.off.get(t, 0.0) for t in teams],
                "def": [-self.deff.get(t, 0.0) for t in teams],  # positive = good defense
            }
        )
        df["net"] = df["off"] + df["def"]
        return df.sort_values("net", ascending=False).reset_index(drop=True)


def long_format(games: pd.DataFrame) -> pd.DataFrame:
    """games: season, week, home, away, home_pts, away_pts, neutral, [home_eff_pts, away_eff_pts]."""
    rows = []
    for side, opp, pts, eff, home_flag in (
        ("home", "away", "home_pts", "home_eff_pts", 1),
        ("away", "home", "away_pts", "away_eff_pts", -1),
    ):
        d = pd.DataFrame(
            {
                "season": games["season"].values,
                "week": games["week"].values,
                "team": games[side].values,
                "opp": games[opp].values,
                "pts": games[pts].values,
                "eff_pts": games[eff].values if eff in games else np.nan,
                "home": np.where(games["neutral"].fillna(False).astype(bool), 0, home_flag),
            }
        )
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def fit_ratings(
    games: pd.DataFrame,
    as_of_season: int,
    as_of_week: int,
    half_life_weeks: float,
    prior_season_weight: float,
    alpha: float,
    eff_blend: float = 0.0,
    weeks_per_season: int = 18,
) -> Ratings:
    """Fit ratings using only games strictly before (as_of_season, as_of_week)."""
    g = games[
        (games["season"] < as_of_season)
        | ((games["season"] == as_of_season) & (games["week"] < as_of_week))
    ]
    g = g[g["season"] >= as_of_season - 1].dropna(subset=["home_pts", "away_pts"])
    lf = long_format(g)
    if lf.empty:
        return Ratings(mu=22.0, hfa=1.5, off={}, deff={})

    # Recency weights: weeks ago on a continuous timeline
    t_now = as_of_season * (weeks_per_season + 4) + as_of_week
    t = lf["season"] * (weeks_per_season + 4) + lf["week"]
    weeks_ago = (t_now - t).clip(lower=0)
    w = 0.5 ** (weeks_ago / half_life_weeks)
    w = np.where(lf["season"] < as_of_season, w * prior_season_weight, w)

    y = lf["pts"].astype(float).values
    if eff_blend > 0 and lf["eff_pts"].notna().any():
        eff = lf["eff_pts"].astype(float).values
        y = np.where(np.isnan(eff), y, (1 - eff_blend) * y + eff_blend * eff)

    teams = sorted(set(lf["team"]) | set(lf["opp"]))
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(lf), len(teams)
    rows = np.arange(n)
    off_cols = lf["team"].map(idx).values
    def_cols = lf["opp"].map(idx).values + k
    X = sparse.csr_matrix(
        (
            np.concatenate([np.ones(n), np.ones(n), lf["home"].values.astype(float) / 2]),
            (np.concatenate([rows, rows, rows]), np.concatenate([off_cols, def_cols, np.full(n, 2 * k)])),
        ),
        shape=(n, 2 * k + 1),
    )
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y, sample_weight=w)
    coef = model.coef_
    gp = lf.groupby("team").size().to_dict()
    return Ratings(
        mu=float(model.intercept_),
        hfa=float(coef[2 * k]),
        off={t: float(coef[idx[t]]) for t in teams},
        deff={t: float(coef[idx[t] + k]) for t in teams},
        games_played={t: float(v) for t, v in gp.items()},
    )
