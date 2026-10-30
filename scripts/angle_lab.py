"""Angle lab, the research half: re-test every game angle and every matchup factor on history.

The live half (shadow picks, the scorecard) runs inside every slate and report. Run this weekly, or
whenever a new idea is added to models/angles.py or models/matchups.py:

    python scripts/angle_lab.py               # game angles 2010-25 + matchup factors for the last 2 seasons
    python scripts/angle_lab.py --seasons 2023 2024 2025

Matchup factors are measured as each starter's game vs his own season average to that point, with
defenses measured on prior weeks only (no look-ahead apart from who actually started).
"""
import argparse

import _path  # noqa: F401
import numpy as np
import pandas as pd

from pocket_capper.data import nflverse
from pocket_capper.models import angles, matchups

K = matchups.SHRINK


def game_angles() -> None:
    bt = angles.backtest(nflverse.schedules())
    print("===== GAME ANGLES (ATS / totals vs the close; 52.4% breaks even) =====")
    for r in bt.itertuples():
        o, n = r.old, r.new
        print(f"{r.label:44} 2010-19 {o['pct']:.1%} n={o['w'] + o['l']:4} z={o['z']:+.1f} | "
              f"2020+ {n['pct']:.1%} n={n['w'] + n['l']:4} z={n['z']:+.1f} | {'HOLDS' if r.holds else '-'}")


def _rows(season: int) -> pd.DataFrame:
    d = nflverse.def_coverage(season)
    d = d[d["game_type"] == "REG"].copy()
    sn = nflverse.snap_counts(season).copy()
    if d.empty or sn.empty:
        return pd.DataFrame()
    sn["key"], sn["grp"] = sn["team"] + "|" + sn["player"], sn["position"].map(matchups.GROUP)
    d["key"] = d["team"] + "|" + d["pfr_player_name"]
    d["grp"] = d["key"].map(sn.groupby("key")["grp"].first())
    ps = nflverse.player_stats([season])
    ps = ps[(ps["season"] == season) & (ps["season_type"] == "REG")]
    lg = d["def_yards_allowed"].sum() / d["def_targets"].sum()
    out = []
    for wk in range(matchups.MIN_WEEK, 19):
        pr, snp, now = d[d["week"] < wk], sn[sn["week"] < wk], sn[sn["week"] == wk]
        if now.empty:
            continue
        cov = pr.groupby(["team", "grp"])[["def_targets", "def_yards_allowed"]].sum()
        cov = (cov["def_yards_allowed"] + K * lg) / (cov["def_targets"] + K)
        per = pr.groupby("key")[["def_targets", "def_yards_allowed"]].sum()
        per["ypt"] = (per["def_yards_allowed"] + K * lg) / (per["def_targets"] + K)
        cb_now = now[(now["grp"] == "CB") & (now["defense_pct"] >= 0.5)]

        def missing(col):
            reg = snp[snp[col] >= 0.6].groupby(["team", "grp", "key"]).size()
            reg = reg[reg >= 3].reset_index()
            reg["out"] = ~reg["key"].isin(set(now[now[col] >= 0.2]["key"]))
            return reg.groupby(["team", "grp"])["out"].sum()
        mdef, moff = missing("defense_pct"), missing("offense_pct")
        cb = {}
        for team, g in cb_now.groupby("team"):
            ypt = g["key"].map(per["ypt"]).fillna(lg)
            cb[team] = (ypt.max(), int((g["key"].map(per["def_targets"]).fillna(0) < matchups.NEW_CB_TARGETS).any()))
        hist = ps[ps["week"] < wk]
        for r in ps[ps["week"] == wk].itertuples():
            if r.position not in ("WR", "TE", "RB", "QB"):
                continue
            h = hist[hist["player_id"] == r.player_id]
            if len(h) < 3:
                continue
            by = {"QB": "attempts", "RB": "carries"}.get(r.position, "targets")
            rank = hist[(hist["team"] == r.team) & (hist["position"] == r.position)].groupby("player_id")[by].sum() \
                .rank(ascending=False).get(r.player_id, 9)
            if rank > (3 if r.position == "WR" else 1):
                continue
            o = r.opponent_team
            worst, new = cb.get(o, (np.nan, 0))
            out.append(dict(season=season, pos=r.position, rank=rank,
                            res_rec=r.receptions - h["receptions"].mean(), res_recyds=r.receiving_yards - h["receiving_yards"].mean(),
                            res_rush=r.rushing_yards - h["rushing_yards"].mean(), res_pass=r.passing_yards - h["passing_yards"].mean(),
                            new_cb=new, weak_cb=int(worst >= matchups.WEAK_CB), shutdown_cb=int(worst <= matchups.SHUTDOWN_CB),
                            lb=cov.get((o, "LB"), np.nan), dl_out=float(mdef.get((o, "DL"), 0)),
                            ol_out=float(moff.get((r.team, "OL"), 0))))
    return pd.DataFrame(out)


def _ols(q: pd.DataFrame, dv: str, xs: list[str]) -> str:
    q = q.dropna(subset=[dv] + xs)
    if len(q) < 30:
        return "n too small"
    X = np.column_stack([np.ones(len(q))] + [q[x] for x in xs])
    y = q[dv].values
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    se = np.sqrt(np.diag(e @ e / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)))
    return "  ".join(f"{n}={bb:+.2f} (t {bb / ss:+.1f})" for n, bb, ss in list(zip(["c"] + xs, b, se))[1:])


def matchup_factors(seasons: list[int]) -> None:
    o = pd.concat([_rows(s) for s in seasons], ignore_index=True)
    if o.empty:
        print("no matchup data")
        return
    print(f"\n===== MATCHUP FACTORS ({', '.join(map(str, seasons))}; effect on the player vs his own average) =====")
    tests = [("WR", "res_rec", ["new_cb", "weak_cb", "shutdown_cb"]), ("WR", "res_recyds", ["new_cb", "weak_cb", "shutdown_cb"]),
             ("TE", "res_rec", ["lb"]), ("TE", "res_recyds", ["lb"]), ("RB", "res_rec", ["lb"]),
             ("RB", "res_rush", ["dl_out", "ol_out"]), ("QB", "res_pass", ["ol_out", "dl_out"])]
    for pos, dv, xs in tests:
        for s in seasons + [None]:
            q = o[(o["pos"] == pos) & ((o["season"] == s) if s else True)]
            print(f"{pos} {dv:10} {str(s or 'all'):4} n={len(q):5} | {_ols(q, dv, xs)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*")
    a = ap.parse_args()
    game_angles()
    sched = nflverse.schedules()
    done = sorted(sched.loc[sched["result"].notna() & (sched["game_type"] == "REG"), "season"].unique())
    matchup_factors(a.seasons or [int(x) for x in done[-2:]])
