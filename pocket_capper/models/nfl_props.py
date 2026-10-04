"""NFL player projections (blind): volume x share x efficiency x opponent, then a distribution.

Team volume and game script come from the blind team projection, never from a line.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

# league-ish priors used for shrinkage
PRIOR_YPT = {"WR": 8.3, "TE": 7.2, "RB": 5.9}
PRIOR_CATCH = {"WR": 0.64, "TE": 0.70, "RB": 0.78}
PRIOR_YPC = {"RB": 4.3, "QB": 5.0, "WR": 6.0, "TE": 4.0}
PRIOR_YPA, PRIOR_PASS_TD = 6.6, 0.042
# Calibrated walk-forward on 2025 weeks 4-17 (scripts/backtest.py --props):
# CV = sd(actual - proj) / proj, which folds projection error into the spread of outcomes.
CV = {"rec_yds": 0.85, "rush_yds": 0.75, "pass_yds": 0.34}
# Raw projections ran ~10% light on volume stats for players who suited up.
VOLUME_CAL = {"rec_yds": 1.10, "receptions": 1.10, "rush_yds": 1.10, "rush_att": 1.09, "targets": 1.10}
# Anytime-TD logistic recalibration: logit(p) -> a + b*logit(p) (raw model was overconfident at the top).
ATTD_CAL = (-0.176, 0.729)
TDS_PER_POINT = 0.108

MARKET_STAT = {
    "player_pass_yds": "pass_yds", "player_pass_tds": "pass_tds", "player_pass_attempts": "pass_att",
    "player_rush_yds": "rush_yds", "player_rush_attempts": "rush_att", "player_receptions": "receptions",
    "player_reception_yds": "rec_yds", "player_anytime_td": "anytime_td",
}
STAT_LABEL = {
    "pass_yds": "Pass Yds", "pass_tds": "Pass TDs", "pass_att": "Pass Att", "rush_yds": "Rush Yds",
    "rush_att": "Rush Att", "receptions": "Receptions", "rec_yds": "Rec Yds", "anytime_td": "Anytime TD",
}


def _decay_weights(weeks_ago: pd.Series, half_life: float = 4.0) -> np.ndarray:
    return 0.5 ** (weeks_ago.values / half_life)


def _wavg(df: pd.DataFrame, col: str, w: np.ndarray) -> float:
    v = df[col].fillna(0).values
    return float((v * w).sum() / max(w.sum(), 1e-9))


def red_zone_shares(pbp: pd.DataFrame, season: int) -> pd.DataFrame:
    """Per player share of team inside-10 opportunities (carries + targets) and TDs, this season."""
    p = pbp[(pbp["season"] == season) & (pbp["yardline_100"] <= 10) & pbp["play_type"].isin(["run", "pass"])]
    rush = p[p["rusher_player_id"].notna()].groupby(["posteam", "rusher_player_id"]).size().rename("rz")
    rec = p[p["receiver_player_id"].notna()].groupby(["posteam", "receiver_player_id"]).size().rename("rz")
    rz = pd.concat([rush.reset_index().rename(columns={"rusher_player_id": "player_id"}),
                    rec.reset_index().rename(columns={"receiver_player_id": "player_id"})])
    rz = rz.groupby(["posteam", "player_id"])["rz"].sum().reset_index()
    rz["rz_share"] = rz["rz"] / rz.groupby("posteam")["rz"].transform("sum")
    return rz.rename(columns={"posteam": "team"})


def project_players(
    ps: pd.DataFrame,
    pbp: pd.DataFrame,
    season: int,
    week: int,
    team_proj: pd.DataFrame,
    out_players: set[str],
) -> pd.DataFrame:
    """Projections for every skill player on teams in team_proj (cols team, opp, pts, margin)."""
    cur = ps[(ps["season"] == season) & (ps["week"] < week) & (ps["season_type"] == "REG") & ps["player_id"].notna()].copy()
    if cur.empty:
        return pd.DataFrame()
    cur["weeks_ago"] = week - cur["week"]

    # team volume per game
    team_g = cur.groupby(["team", "week"]).agg(
        pass_att=("attempts", "sum"), carries=("carries", "sum"), targets=("targets", "sum"),
        rush_tds=("rushing_tds", "sum"), rec_tds=("receiving_tds", "sum"),
    ).reset_index()
    team_g["weeks_ago"] = week - team_g["week"]
    lg = team_g[["pass_att", "carries", "targets"]].mean()

    # opponent defense: yards allowed per pass att / per carry, shrunk
    opp = cur.groupby("opponent_team").agg(
        py=("passing_yards", "sum"), pa=("attempts", "sum"), ry=("rushing_yards", "sum"), ra=("carries", "sum")
    )
    lg_ypa = opp["py"].sum() / max(opp["pa"].sum(), 1)
    lg_ypc = opp["ry"].sum() / max(opp["ra"].sum(), 1)
    opp_pass = ((opp["py"] + 150 * lg_ypa) / (opp["pa"] + 150) / lg_ypa).to_dict()
    opp_rush = ((opp["ry"] + 150 * lg_ypc) / (opp["ra"] + 150) / lg_ypc).to_dict()

    rz = red_zone_shares(pbp, season) if not pbp.empty else pd.DataFrame(columns=["team", "player_id", "rz_share"])
    rz_map = {(r.team, r.player_id): r.rz_share for r in rz.itertuples()}

    rows = []
    for tp in team_proj.itertuples():
        tg = team_g[team_g["team"] == tp.team]
        if tg.empty:
            continue
        w = _decay_weights(tg["weeks_ago"])
        n = len(tg)
        base_pass = (_wavg(tg, "pass_att", w) * n + lg["pass_att"] * 3) / (n + 3)
        base_rush = (_wavg(tg, "carries", w) * n + lg["carries"] * 3) / (n + 3)
        pass_att = max(20.0, base_pass - 0.28 * tp.margin)
        carries = max(15.0, base_rush + 0.32 * tp.margin)
        targets = pass_att * 0.93
        team_tds = tp.pts * TDS_PER_POINT

        players = cur[cur["team"] == tp.team]
        # only players who appeared in the last 3 team games
        recent = players[players["weeks_ago"] <= players["weeks_ago"].min() + 3]
        ids = set(recent["player_id"]) - out_players
        # redistribute share of OUT players proportionally
        shares = {}
        for pid, grp in players.groupby("player_id"):
            wts = _decay_weights(grp["weeks_ago"])
            tgw = tg.set_index("week").loc[grp["week"]]
            tshare = (grp["targets"].fillna(0).values * wts).sum() / max((tgw["targets"].values * wts).sum(), 1)
            cshare = (grp["carries"].fillna(0).values * wts).sum() / max((tgw["carries"].values * wts).sum(), 1)
            shares[pid] = (tshare, cshare)
        active_t = sum(v[0] for k, v in shares.items() if k in ids)
        active_c = sum(v[1] for k, v in shares.items() if k in ids)
        scale_t = min(1.25, 1 / active_t) if active_t > 0 else 1
        scale_c = min(1.25, 1 / active_c) if active_c > 0 else 1

        team_td_tot = tg["rush_tds"].sum() + tg["rec_tds"].sum()
        for pid in ids:
            grp = players[players["player_id"] == pid].sort_values("week")
            last = grp.iloc[-1]
            pos = last["position"]
            if pos not in ("QB", "RB", "WR", "TE"):
                continue
            tshare, cshare = shares[pid]
            tshare *= scale_t
            cshare *= scale_c
            tot_tgt, tot_car = grp["targets"].fillna(0).sum(), grp["carries"].fillna(0).sum()
            rec = {
                "player_id": pid, "player": last["player_display_name"], "position": pos, "team": tp.team,
                "opp": tp.opp, "games": len(grp),
            }
            dpass = opp_pass.get(tp.opp, 1.0)
            drush = opp_rush.get(tp.opp, 1.0)
            # receiving
            if pos in PRIOR_YPT:
                ypt = (grp["receiving_yards"].fillna(0).sum() + 30 * PRIOR_YPT[pos]) / (tot_tgt + 30)
                cr = (grp["receptions"].fillna(0).sum() + 30 * PRIOR_CATCH[pos]) / (tot_tgt + 30)
                exp_t = targets * tshare
                rec.update(targets=exp_t, receptions=exp_t * cr, rec_yds=exp_t * ypt * dpass)
            # rushing
            ypc = (grp["rushing_yards"].fillna(0).sum() + 40 * PRIOR_YPC.get(pos, 4.3)) / (tot_car + 40)
            exp_c = carries * cshare
            rec.update(rush_att=exp_c, rush_yds=exp_c * ypc * drush)
            # passing (primary QB only)
            if pos == "QB":
                qb_share = grp["attempts"].fillna(0).sum() / max(tg["pass_att"].sum(), 1)
                if qb_share >= 0.5:
                    att = grp["attempts"].fillna(0).sum()
                    ypa = (grp["passing_yards"].fillna(0).sum() + 100 * PRIOR_YPA) / (att + 100)
                    tdr = (grp["passing_tds"].fillna(0).sum() + 150 * PRIOR_PASS_TD) / (att + 150)
                    exp_a = pass_att * 0.97
                    rec.update(pass_att=exp_a, pass_yds=exp_a * ypa * dpass, pass_tds=exp_a * tdr * math.sqrt(dpass))
            # anytime TD
            p_tds = grp["rushing_tds"].fillna(0).sum() + grp["receiving_tds"].fillna(0).sum()
            td_share_actual = (p_tds + 1 * (tshare * 0.5 + cshare * 0.5)) / (team_td_tot + 1) if team_td_tot else 0
            opp_share = 0.55 * cshare + 0.45 * tshare if pos != "QB" else 0.5 * cshare
            rz_share = rz_map.get((tp.team, pid), opp_share)
            td_share = 0.45 * rz_share + 0.30 * opp_share + 0.25 * td_share_actual
            if pos == "QB":
                td_share *= 0.6  # passing TDs dominate QB TD usage; rushing only counts here
            lam = team_tds * td_share * (1.0 if dpass * drush == 0 else math.sqrt(dpass * drush))
            raw = min(max(1 - math.exp(-lam), 0.003), 0.99)
            z = ATTD_CAL[0] + ATTD_CAL[1] * math.log(raw / (1 - raw))
            rec.update(exp_tds=lam, anytime_td=1 / (1 + math.exp(-z)))
            for k, mult in VOLUME_CAL.items():
                if k in rec:
                    rec[k] *= mult
            rows.append(rec)
    return pd.DataFrame(rows)


def prob_over(stat: str, mean: float, line: float) -> float:
    """P(stat > line) under the stat's distribution (lines are .5 so no push)."""
    if mean is None or not np.isfinite(mean) or mean <= 0:
        return float("nan")
    if stat in CV:
        cv = CV[stat]
        k = 1 / cv**2
        return float(stats.gamma.sf(line, a=k, scale=mean / k))
    if stat in ("receptions", "pass_tds"):
        return float(stats.poisson.sf(math.floor(line), mean))
    if stat == "rush_att":
        var = mean * 1.6
        r = mean**2 / max(var - mean, 1e-6)
        return float(stats.nbinom.sf(math.floor(line), r, r / (r + mean)))
    if stat == "pass_att":
        return float(stats.norm.sf(line, mean, 6.5))
    return float("nan")
