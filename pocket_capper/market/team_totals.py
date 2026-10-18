"""Vegas implied team totals from consensus spread + total.

team total = total / 2 - team spread / 2, using the median line across books. A low
opponent implied total is the standard fantasy D/ST streaming signal.
"""
from __future__ import annotations

import pandas as pd

from pocket_capper.data.teams import nfl_abbr

COLUMNS = ["team", "opp", "venue", "kickoff", "spread", "total", "team_total", "opp_total", "books"]


def week_window(now: pd.Timestamp | None = None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Now through the coming Tuesday 12:00 UTC (after Monday night), i.e. the rest of this NFL week."""
    now = now or pd.Timestamp.now(tz="UTC")
    days = (1 - now.dayofweek) % 7 or 7  # Tuesday = 1
    return now, (now + pd.Timedelta(days=days)).normalize() + pd.Timedelta(hours=12)


def implied_team_totals(odds: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game (one per team) from parse_game_odds output, lowest opponent total first."""
    if odds.empty:
        return pd.DataFrame(columns=COLUMNS)
    sp = odds[(odds["market"] == "spread") & (odds["side"] == "home")]
    tot = odds[(odds["market"] == "total") & (odds["side"] == "over")]
    games = (
        odds.groupby("event_id")[["commence_time", "home_name", "away_name"]].first()
        .join(sp.groupby("event_id")["point"].median().rename("home_spread"))
        .join(tot.groupby("event_id")["point"].median().rename("total"))
        .join(pd.concat([sp, tot]).groupby("event_id")["book"].nunique().rename("books"))
        .dropna(subset=["home_spread", "total"])
    )
    rows = []
    for g in games.itertuples():
        home_tt = g.total / 2 - g.home_spread / 2
        away_tt = g.total - home_tt
        kickoff = pd.Timestamp(g.commence_time)
        for team, opp, venue, spread, tt, ott in (
            (g.home_name, g.away_name, "home", g.home_spread, home_tt, away_tt),
            (g.away_name, g.home_name, "away", -g.home_spread, away_tt, home_tt),
        ):
            rows.append({
                "team": nfl_abbr(team) or team, "opp": nfl_abbr(opp) or opp, "venue": venue, "kickoff": kickoff,
                "spread": spread, "total": g.total, "team_total": round(tt, 1), "opp_total": round(ott, 1),
                "books": g.books,
            })
    return pd.DataFrame(rows, columns=COLUMNS).sort_values(["opp_total", "spread"]).reset_index(drop=True)


def this_week(tt: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    start, end = week_window(now)
    k = pd.to_datetime(tt["kickoff"], utc=True)
    return tt[(k > start) & (k <= end)].reset_index(drop=True)
