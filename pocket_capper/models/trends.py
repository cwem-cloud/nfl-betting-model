"""Situational trends from nflverse history (context only - never fed into the number).

Small-sample trends are mostly noise, so each one is shown with its sample size and
is not used to move the projection or the units.
"""
from __future__ import annotations

import pandas as pd


def _ats(df: pd.DataFrame, side: str) -> tuple[int, int]:
    margin = df["result"] - df["spread_line"]  # >0 home covered
    m = margin if side == "home" else -margin
    return int((m > 0).sum()), int((m < 0).sum())


def situations(sched: pd.DataFrame, game: pd.Series, since: int = 2012) -> list[str]:
    h = sched[(sched["season"] >= since) & sched["result"].notna() & sched["spread_line"].notna()]
    out = []
    line = game.get("spread_line_dk")  # home expected margin convention when available
    checks = []
    if pd.notna(game.get("away_rest")) and game["away_rest"] >= 13:
        checks.append(("Road team off bye", h[h["away_rest"] >= 13], "away"))
    if pd.notna(game.get("home_rest")) and game["home_rest"] >= 13:
        checks.append(("Home team off bye", h[h["home_rest"] >= 13], "home"))
    if pd.notna(game.get("home_rest")) and game["home_rest"] <= 4:
        checks.append(("Home team on short week", h[h["home_rest"] <= 4], "home"))
    if game.get("div_game") == 1 and line is not None:
        dog = "home" if line < 0 else "away"
        sel = h[(h["div_game"] == 1) & ((h["spread_line"] < 0) if dog == "home" else (h["spread_line"] > 0))]
        checks.append(("Divisional underdog", sel, dog))
    if line is not None and line < 0:
        checks.append(("Home underdog", h[h["spread_line"] < 0], "home"))
    for name, df, side in checks:
        w, l = _ats(df, side)
        if w + l >= 50:
            out.append(f"{name}: {w}-{l} ATS since {since} ({w / (w + l):.1%}) - context only")
    return out


def team_ats(sched: pd.DataFrame, team: str, season: int) -> str:
    s = sched[(sched["season"] == season) & sched["result"].notna() & sched["spread_line"].notna()]
    hw, hl = _ats(s[s["home_team"] == team], "home")
    aw, al = _ats(s[s["away_team"] == team], "away")
    return f"{team} {hw + aw}-{hl + al} ATS this season"
