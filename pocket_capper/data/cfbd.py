"""CollegeFootballData.com API client (free key: collegefootballdata.com/key).

Handles both v2 camelCase and legacy snake_case field names.
"""
from __future__ import annotations

import pandas as pd

from pocket_capper.data.http import cached_json
from pocket_capper.settings import secret

BASE = "https://api.collegefootballdata.com"


def _g(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _get(path: str, params: dict, max_age_hours: float = 3) -> list:
    key = secret("CFBD_API_KEY")
    if not key:
        raise RuntimeError("CFBD_API_KEY not set (free at collegefootballdata.com/key)")
    return cached_json(f"{BASE}{path}", params, {"Authorization": f"Bearer {key}"}, max_age_hours)


def games(year: int, season_type: str = "regular", max_age_hours: float = 3) -> pd.DataFrame:
    rows = []
    for d in _get("/games", {"year": year, "seasonType": season_type}, max_age_hours):
        rows.append(
            {
                "game_id": str(_g(d, "id")),
                "season": int(_g(d, "season", default=year)),
                "week": int(_g(d, "week", default=0)),
                "start": _g(d, "startDate", "start_date"),
                "home": _g(d, "homeTeam", "home_team"),
                "away": _g(d, "awayTeam", "away_team"),
                "home_pts": _g(d, "homePoints", "home_points"),
                "away_pts": _g(d, "awayPoints", "away_points"),
                "neutral": bool(_g(d, "neutralSite", "neutral_site", default=False)),
                "home_div": _g(d, "homeClassification", "home_division"),
                "away_div": _g(d, "awayClassification", "away_division"),
                "home_conf": _g(d, "homeConference", "home_conference"),
                "away_conf": _g(d, "awayConference", "away_conference"),
                "venue_id": _g(d, "venueId", "venue_id"),
                "completed": bool(_g(d, "completed", default=False)),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["start"] = pd.to_datetime(df["start"], utc=True, errors="coerce")
        if season_type == "postseason":
            df["week"] = df["week"] + 16
    return df


def ppa_games(year: int) -> pd.DataFrame:
    """Per-team-game offensive PPA (EPA-style) per play, garbage time excluded."""
    rows = []
    try:
        data = _get("/ppa/games", {"year": year, "excludeGarbageTime": "true"}, 6)
    except Exception:
        return pd.DataFrame()
    for d in data:
        off = _g(d, "offense", default={}) or {}
        rows.append(
            {
                "game_id": str(_g(d, "gameId", "game_id")),
                "team": _g(d, "team"),
                "off_ppa": _g(off, "overall"),
            }
        )
    return pd.DataFrame(rows)


def lines(year: int, week: int | None = None, season_type: str = "regular") -> pd.DataFrame:
    """Consensus/opening lines by provider (used for line history + CFB opens)."""
    params = {"year": year, "seasonType": season_type}
    if week:
        params["week"] = week
    rows = []
    for d in _get("/lines", params, 1):
        for ln in _g(d, "lines", default=[]) or []:
            rows.append(
                {
                    "game_id": str(_g(d, "id")),
                    "home": _g(d, "homeTeam", "home_team"),
                    "away": _g(d, "awayTeam", "away_team"),
                    "provider": _g(ln, "provider"),
                    "spread": _g(ln, "spread"),
                    "spread_open": _g(ln, "spreadOpen", "spread_open"),
                    "total": _g(ln, "overUnder", "over_under"),
                    "total_open": _g(ln, "overUnderOpen", "over_under_open"),
                }
            )
    return pd.DataFrame(rows)


def venues() -> pd.DataFrame:
    rows = []
    for d in _get("/venues", {}, 24 * 30):
        loc = _g(d, "location", default={}) or {}
        rows.append(
            {
                "venue_id": _g(d, "id"),
                "name": _g(d, "name"),
                "lat": _g(loc, "x", "latitude"),
                "lon": _g(loc, "y", "longitude"),
                "dome": bool(_g(d, "dome", default=False)),
            }
        )
    return pd.DataFrame(rows)


def fbs_teams(year: int) -> list[str]:
    try:
        return [_g(d, "school") for d in _get("/teams/fbs", {"year": year}, 24 * 7)]
    except Exception:
        return []
