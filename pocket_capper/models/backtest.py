"""Walk-forward backtest of the blind NFL model against real closing lines.

For every week, ratings are fit only on games before that week, then compared to
nflverse's closing spread/total. Reports accuracy and ATS/OU hit rates at
different disagreement thresholds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pocket_capper.models import nfl_model


def run(seasons: list[int], cfg: dict, games: pd.DataFrame | None = None, min_week: int = 2) -> pd.DataFrame:
    if games is None:
        games = nfl_model.build_games(sorted(set(seasons) | {min(seasons) - 1}))
    out = []
    for s in seasons:
        sg = games[(games["season"] == s) & games["home_pts"].notna() & (games["game_type"] == "REG")]
        for wk in sorted(sg["week"].unique()):
            if wk < min_week:
                continue
            r = nfl_model.fit(games, s, wk, cfg)
            wg = sg[sg["week"] == wk]
            proj = nfl_model.project_games(wg, r, cfg)
            m = wg[["game_id", "season", "week", "home", "away", "home_pts", "away_pts", "spread_line", "total_line"]].merge(
                proj[["game_id", "fair_margin", "home_pts", "away_pts"]].rename(
                    columns={"home_pts": "p_home", "away_pts": "p_away"}
                ),
                on="game_id",
            )
            out.append(m)
    bt = pd.concat(out, ignore_index=True)
    bt["result"] = bt["home_pts"] - bt["away_pts"]
    bt["total"] = bt["home_pts"] + bt["away_pts"]
    bt["fair_total"] = bt["p_home"] + bt["p_away"]
    return bt


def summarize(bt: pd.DataFrame) -> dict:
    res = {
        "games": len(bt),
        "mae_margin_model": float((bt["fair_margin"] - bt["result"]).abs().mean()),
        "mae_margin_close": float((bt["spread_line"] - bt["result"]).abs().mean()),
        "mae_total_model": float((bt["fair_total"] - bt["total"]).abs().mean()),
        "mae_total_close": float((bt["total_line"] - bt["total"]).abs().mean()),
    }
    # spread_line in nflverse: positive = home favored (expected home margin)
    diff = bt["fair_margin"] - bt["spread_line"]
    cover = np.sign(bt["result"] - bt["spread_line"])
    for th in (1, 2, 3):
        sel = diff.abs() >= th
        hit = (np.sign(diff[sel]) == cover[sel])[cover[sel] != 0]
        res[f"ats_{th}"] = f"{hit.sum()}-{(~hit).sum()} ({hit.mean():.3f})" if len(hit) else "n/a"
    tdiff = bt["fair_total"] - bt["total_line"]
    ou = np.sign(bt["total"] - bt["total_line"])
    for th in (1.5, 3, 4.5):
        sel = tdiff.abs() >= th
        hit = (np.sign(tdiff[sel]) == ou[sel])[ou[sel] != 0]
        res[f"ou_{th}"] = f"{hit.sum()}-{(~hit).sum()} ({hit.mean():.3f})" if len(hit) else "n/a"
    return res


def run_props(season: int, weeks: list[int], cfg: dict, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Walk-forward player projections vs actual box scores (for calibration)."""
    from pocket_capper.data import nflverse
    from pocket_capper.models import nfl_props

    if games is None:
        games = nfl_model.build_games([season - 1, season])
    ps = nflverse.player_stats([season])
    plays = nflverse.pbp([season])
    out = []
    for wk in weeks:
        r = nfl_model.fit(games, season, wk, cfg)
        wg = games[(games["season"] == season) & (games["week"] == wk)]
        proj = nfl_model.project_games(wg, r, cfg)
        tp = pd.concat([
            pd.DataFrame({"team": proj["home"], "opp": proj["away"], "pts": proj["home_pts"], "margin": proj["fair_margin"]}),
            pd.DataFrame({"team": proj["away"], "opp": proj["home"], "pts": proj["away_pts"], "margin": -proj["fair_margin"]}),
        ])
        actual = ps[(ps["season"] == season) & (ps["week"] == wk)]
        pp = nfl_props.project_players(ps, plays[plays["week"] < wk], season, wk, tp, out_players=set())
        if pp.empty:
            continue
        m = pp.merge(
            actual[["player_id", "receptions", "receiving_yards", "rushing_yards", "carries", "passing_yards",
                    "attempts", "passing_tds", "rushing_tds", "receiving_tds"]],
            on="player_id", how="inner", suffixes=("", "_act"),
        )
        m["week"] = wk
        out.append(m)
    return pd.concat(out, ignore_index=True)


def run_cfb(seasons: list[int], cfg: dict, min_week: int = 3) -> pd.DataFrame:
    """Walk-forward CFB blind model vs CFBD closing lines (FBS vs FBS only). Needs CFBD_API_KEY."""
    from pocket_capper.data import cfbd
    from pocket_capper.models import cfb_model

    out = []
    for s in seasons:
        games = cfb_model.build_games(s)
        ln = cfbd.lines(s)
        if ln.empty:
            continue
        close = ln.groupby("game_id").agg(spread=("spread", "median"), total_line=("total", "median"))
        sg = games[(games["season"] == s) & games["home_pts"].notna() & (games["home"] != "FCS") & (games["away"] != "FCS")]
        for wk in sorted(sg["week"].unique()):
            if wk < min_week:
                continue
            r = cfb_model.fit(games, s, wk, cfg)
            wg = sg[sg["week"] == wk]
            proj = cfb_model.project_games(wg, r, cfg)
            m = wg[["game_id", "season", "week", "home_pts", "away_pts"]].merge(
                proj[["game_id", "fair_margin", "home_pts", "away_pts"]].rename(columns={"home_pts": "p_home", "away_pts": "p_away"}),
                on="game_id",
            ).merge(close, left_on="game_id", right_index=True)
            out.append(m)
    bt = pd.concat(out, ignore_index=True)
    bt["result"] = bt["home_pts"] - bt["away_pts"]
    bt["total"] = bt["home_pts"] + bt["away_pts"]
    bt["fair_total"] = bt["p_home"] + bt["p_away"]
    bt["spread_line"] = -bt["spread"]  # CFBD spread is the home line; convert to expected home margin
    return bt.dropna(subset=["spread_line", "total_line"])


def tune(sport: str, seasons: list[int], base_cfg: dict, grid: dict) -> pd.DataFrame:
    """Grid-search the blind model by out-of-sample MAE (margin + total) vs results."""
    import itertools

    keys = list(grid)
    rows = []
    nfl_games = nfl_model.build_games(sorted(set(seasons) | {min(seasons) - 1})) if sport == "nfl" else None
    for combo in itertools.product(*grid.values()):
        cfg = dict(base_cfg, **dict(zip(keys, combo)))
        bt = run(seasons, cfg, nfl_games) if sport == "nfl" else run_cfb(seasons, cfg)
        s = summarize(bt)
        rows.append({**dict(zip(keys, combo)), "mae_margin": s["mae_margin_model"], "mae_total": s["mae_total_model"],
                     "close_margin": s["mae_margin_close"], "close_total": s["mae_total_close"],
                     "ats_2": s["ats_2"], "ou_3": s["ou_3"]})
    out = pd.DataFrame(rows)
    out["score"] = out["mae_margin"] + out["mae_total"]
    return out.sort_values("score")
