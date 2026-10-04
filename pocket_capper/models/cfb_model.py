"""CFB blind handicap. Same ridge engine as the NFL, fed by CollegeFootballData.

Non-FBS opponents are pooled into a single 'FCS' team so their scores still
inform FBS ratings without needing 100+ extra thinly-sampled teams.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pocket_capper.data import cfbd
from pocket_capper.models import distributions as dist
from pocket_capper.models.team_ratings import Ratings, fit_ratings


def build_games(season: int) -> pd.DataFrame:
    frames = []
    for yr in (season - 1, season):
        for st in ("regular", "postseason"):
            try:
                frames.append(games_with_ppa(yr, st))
            except Exception:
                continue
    g = pd.concat(frames, ignore_index=True)
    fbs = set(cfbd.fbs_teams(season)) | set(cfbd.fbs_teams(season - 1))
    if fbs:
        for col in ("home", "away"):
            g[col + "_raw"] = g[col]
            g[col] = np.where(g[col].isin(fbs), g[col], "FCS")
    return g


def games_with_ppa(year: int, season_type: str) -> pd.DataFrame:
    g = cfbd.games(year, season_type)
    if g.empty:
        return g
    ppa = cfbd.ppa_games(year) if season_type == "regular" else pd.DataFrame()
    if not ppa.empty:
        done = g.dropna(subset=["home_pts"])
        pts = pd.concat(
            [
                done[["game_id", "home", "home_pts"]].rename(columns={"home": "team", "home_pts": "pts"}),
                done[["game_id", "away", "away_pts"]].rename(columns={"away": "team", "away_pts": "pts"}),
            ]
        )
        m = ppa.merge(pts, on=["game_id", "team"]).dropna()
        if len(m) > 100:
            b, a = np.polyfit(m["off_ppa"].astype(float), m["pts"].astype(float), 1)
            eff = ppa.set_index(["game_id", "team"])["off_ppa"].astype(float) * b + a
            g["home_eff_pts"] = [eff.get((gid, t), np.nan) for gid, t in zip(g["game_id"], g["home"])]
            g["away_eff_pts"] = [eff.get((gid, t), np.nan) for gid, t in zip(g["game_id"], g["away"])]
    return g


def fit(games: pd.DataFrame, season: int, week: int, cfg: dict) -> Ratings:
    return fit_ratings(
        games,
        as_of_season=season,
        as_of_week=week,
        half_life_weeks=cfg["decay_half_life_weeks"],
        prior_season_weight=cfg["prior_season_weight"],
        alpha=cfg["ridge_alpha"],
        eff_blend=cfg["ppa_blend"],
        weeks_per_season=17,
    )


def project_games(upcoming: pd.DataFrame, ratings: Ratings, cfg: dict, weather: dict | None = None) -> pd.DataFrame:
    rows = []
    for _, g in upcoming.iterrows():
        hp, ap = ratings.project(g["home"], g["away"], bool(g.get("neutral", False)))
        hp += cfg.get("total_adjust", 0.0) / 2
        ap += cfg.get("total_adjust", 0.0) / 2
        notes = []
        wx = (weather or {}).get(g["game_id"])
        if wx and not wx.get("dome"):
            adj = 0.0
            if wx.get("wind_mph", 0) > 12:
                adj += -0.4 * (wx["wind_mph"] - 12)
            if wx.get("precip_prob", 0) >= 60:
                adj += -1.5
            if adj:
                hp += adj / 2
                ap += adj / 2
                notes.append(f"Weather: {wx.get('wind_mph', 0):.0f} mph wind, {wx.get('precip_prob', 0):.0f}% precip ({adj:+.1f} total)")
        thin = [t for t in (g["home"], g["away"]) if ratings.games_played.get(t, 0) < 4]
        if thin:
            notes.append(f"Thin sample: {', '.join(thin)}")
        margin = hp - ap
        rows.append(
            {
                "game_id": g["game_id"],
                "home": g.get("home_raw", g["home"]),
                "away": g.get("away_raw", g["away"]),
                "home_pts": round(hp, 1),
                "away_pts": round(ap, 1),
                "fair_margin": margin,
                "fair_spread_home": dist.margin_to_spread(margin),
                "fair_total": round((hp + ap) * 2) / 2,
                "home_win_prob": dist.win_prob(margin, cfg["margin_sigma"], "cfb"),
                "adjust_notes": notes,
            }
        )
    return pd.DataFrame(rows)
