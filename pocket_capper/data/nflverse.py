"""NFL historical + current-season data from nflverse (free, updated nightly in season)."""
from __future__ import annotations

import time

import pandas as pd

from pocket_capper.data.http import cached_download

BASE = "https://github.com/nflverse/nflverse-data/releases/download"

PBP_COLS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team",
    "play_type", "epa", "success", "yardline_100", "pass_attempt", "rush_attempt", "qb_dropback",
    "rusher_player_id", "receiver_player_id", "passer_player_id", "touchdown", "td_player_id",
    "wp", "qtr", "half_seconds_remaining", "score_differential", "two_point_attempt",
]


PS_COLS = [
    "player_id", "player_display_name", "position", "season", "week", "season_type", "game_id", "team",
    "opponent_team", "completions", "attempts", "passing_yards", "passing_tds", "passing_epa", "carries",
    "rushing_yards", "rushing_tds", "rushing_epa", "receptions", "targets", "receiving_yards", "receiving_tds",
    "target_share",
]


def schedules() -> pd.DataFrame:
    """Every game since 1999 with scores, closing lines, rest, roof, QBs, refs, coaches."""
    path = cached_download(f"{BASE}/schedules/games.csv", "nfl_games.csv", max_age_hours=3)
    g = pd.read_csv(path)
    g["gameday"] = pd.to_datetime(g["gameday"])
    return g


def pbp(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        age = 6 if s >= max(seasons) else 24 * 30
        try:
            path = cached_download(f"{BASE}/pbp/play_by_play_{s}.parquet", f"nfl_pbp_{s}.parquet", age)
        except Exception:
            continue
        import pyarrow.parquet as pq

        cols = [c for c in PBP_COLS if c in pq.read_schema(path).names]
        frames.append(pd.read_parquet(path, columns=cols))  # ~25 of 370 columns: keeps memory small
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PBP_COLS)


def player_stats(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        age = 6 if s >= max(seasons) else 24 * 30
        try:
            path = cached_download(
                f"{BASE}/stats_player/stats_player_week_{s}.parquet", f"nfl_ps_{s}.parquet", age
            )
        except Exception:
            continue
        frames.append(pd.read_parquet(path, columns=PS_COLS))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def injuries(season: int) -> pd.DataFrame:
    try:
        path = cached_download(f"{BASE}/injuries/injuries_{season}.parquet", f"nfl_inj_{season}.parquet", 3)
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def team_game_epa(seasons: list[int]) -> pd.DataFrame:
    """Per team-game offensive EPA totals/plays (non-garbage time). Cached per season as a tiny parquet,
    so only the current season's play-by-play is ever loaded into memory."""
    from pocket_capper.settings import CACHE_DIR

    frames = []
    for s in seasons:
        path = CACHE_DIR / f"nfl_tge_{s}.parquet"
        fresh_hours = 6 if s >= max(seasons) else 24 * 30
        if path.exists() and (time.time() - path.stat().st_mtime) < fresh_hours * 3600:
            frames.append(pd.read_parquet(path))
            continue
        plays = pbp([s])
        if plays.empty:
            continue
        agg = _aggregate_epa(plays)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        agg.to_parquet(path)
        frames.append(agg)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _aggregate_epa(plays: pd.DataFrame) -> pd.DataFrame:
    p = plays[plays["play_type"].isin(["pass", "run"]) & plays["epa"].notna()]
    p = p[p.get("two_point_attempt", 0).fillna(0) == 0]
    # drop garbage time: win prob outside 5-95% in the 4th quarter
    garbage = (p["qtr"] >= 4) & ((p["wp"] < 0.05) | (p["wp"] > 0.95))
    p = p[~garbage]
    return (
        p.groupby(["game_id", "season", "week", "posteam", "defteam"])
        .agg(off_epa=("epa", "sum"), plays=("epa", "size"), success=("success", "mean"))
        .reset_index()
    )
