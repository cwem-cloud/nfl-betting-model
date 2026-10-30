"""Matchup factors for player props: who the player actually lines up against this week.

Backtested on 2024-25 (each starter vs his own season average, defenses measured on prior weeks only).
A defense's weakest link mattered; its unit averages, pressure rate and missed tackles did not:
  new_cb       WR top-3 vs a starting CB with < 15 coverage targets (this + last season): +4 yds, +0.4 rec (both years)
  weak_cb      WR top-3 vs a weakest starting CB in the worst quartile (>= 8.3 yds/target): +2 yds
  shutdown_cb  WR1 when every starting CB is in the best quartile (<= 7.4 yds/target): -5 yds
  weak_lb      TE1 / RB1 vs a LB group allowing >= 7.8 yds/target (+1 SD): TE +2.8 yds +0.3 rec, RB +0.2 rec
  tough_lb     TE1 vs a LB group allowing <= 6.4 yds/target: the mirror image
  dl_out       RB1 vs a defense missing a regular DL starter: +3.6 rush yds per starter
  ol_out       QB1 / RB1 missing 2+ of their own regular OL starters: -6 pass yds per starter
These are tilts of a few yards, not edges against the line on their own: every one is shadow-tracked
against DK's line before it is allowed to move a price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pocket_capper.data import nflverse
from pocket_capper.data.teams import norm_player

GROUP = {"CB": "CB", "DB": "S", "S": "S", "FS": "S", "SS": "S", "LB": "LB", "ILB": "LB", "OLB": "LB", "MLB": "LB",
         "DE": "DL", "DT": "DL", "NT": "DL", "DL": "DL", "T": "OL", "G": "OL", "C": "OL", "OT": "OL", "OG": "OL"}
SHRINK = 30  # targets of league-average coverage added to every defender / unit
NEW_CB_TARGETS = 15
WEAK_CB, SHUTDOWN_CB = 8.3, 7.4
WEAK_LB, TOUGH_LB = 7.8, 6.4
MIN_WEEK = 4  # coverage samples are too thin before this
FACTORS = {  # key -> (label, evidence)
    "new_cb": ("WR vs an inexperienced starting CB", "+4 yds / +0.4 rec vs own average, both seasons"),
    "weak_cb": ("WR vs a weak starting CB", "+2 yds vs own average"),
    "shutdown_cb": ("WR1 vs a strong CB room", "-5 yds vs own average"),
    "weak_lb": ("TE/RB vs weak coverage LBs", "TE +2.8 yds / +0.3 rec, RB +0.2 rec"),
    "tough_lb": ("TE vs strong coverage LBs", "mirror of weak_lb"),
    "dl_out": ("RB vs a defense missing DL starters", "+3.6 rush yds per starter out"),
    "ol_out": ("Own OL missing 2+ starters", "-6 pass yds per starter out (QB), RB rush down"),
}


def _regulars(snaps: pd.DataFrame, week: int, col: str, min_pct: float = 0.6) -> pd.DataFrame:
    """Players at >= min_pct of snaps over their team's last 4 games (2+ games): team, player, grp, pct."""
    s = snaps[(snaps["week"] < week) & (snaps[col] > 0)]
    out = []
    for team, grp in s.groupby("team"):
        recent = grp[grp["week"] >= sorted(grp["week"].unique())[-4:][0]]
        per = recent.groupby("player").agg(pct=(col, "mean"), games=("week", "nunique"), pos=("position", "last"))
        per["team"], per["grp"] = team, per["pos"].map(GROUP)
        out.append(per.reset_index())
    if not out:
        return pd.DataFrame(columns=["player", "pct", "games", "pos", "team", "grp", "regular"])
    r = pd.concat(out, ignore_index=True)
    r["regular"] = (r["pct"] >= min_pct) & (r["games"] >= 2)
    return r


def _coverage(cov: pd.DataFrame, snaps: pd.DataFrame, week: int) -> tuple[pd.DataFrame, pd.Series, float]:
    """(per-defender targets and shrunk yds/target, per team-group shrunk yds/target, league yds/target)."""
    c = cov[(cov["week"] < week) & (cov["game_type"] == "REG")].copy()
    if c.empty or c["def_targets"].sum() == 0:
        return pd.DataFrame(), pd.Series(dtype=float), np.nan
    grp = snaps.assign(k=snaps["team"] + "|" + snaps["player"].map(norm_player)).groupby("k")["position"].last().map(GROUP)
    c["k"] = c["team"] + "|" + c["pfr_player_name"].map(norm_player)
    c["grp"] = c["k"].map(grp)
    lg = c["def_yards_allowed"].sum() / c["def_targets"].sum()
    per = c.groupby("k")[["def_targets", "def_yards_allowed"]].sum()
    per["ypt"] = (per["def_yards_allowed"] + SHRINK * lg) / (per["def_targets"] + SHRINK)
    unit = c.groupby(["team", "grp"])[["def_targets", "def_yards_allowed"]].sum()
    unit = (unit["def_yards_allowed"] + SHRINK * lg) / (unit["def_targets"] + SHRINK)
    return per, unit, lg


def _top(ps: pd.DataFrame, season: int, week: int, pos: str, by: str, n: int, out_ids: set) -> pd.DataFrame:
    s = ps[(ps["season"] == season) & (ps["week"] < week) & (ps["position"] == pos)]
    s = s[~s["player_id"].isin(out_ids)]
    last_team = s.sort_values("week").groupby("player_id")["team"].last()
    tot = s.groupby("player_id").agg(player=("player_display_name", "last"), v=(by, "sum")).join(last_team)
    return tot.sort_values("v", ascending=False).groupby("team").head(n).assign(
        rank=lambda d: d.groupby("team")["v"].rank(ascending=False, method="first"))


def build(season: int, week: int, ps: pd.DataFrame, opp_of: dict[str, str], out_ids: set | None = None,
          def_out: dict | None = None) -> pd.DataFrame:
    """One row per (player, factor): player_id, player, team, opp, key, stats, direction (+1 over / -1 under), note.
    opp_of maps each team playing this week to its opponent."""
    cols = ["player_id", "player", "team", "opp", "key", "stats", "direction", "note"]
    if week < MIN_WEEK or not opp_of:
        return pd.DataFrame(columns=cols)
    out_ids = out_ids or set()
    snaps, inj = nflverse.snap_counts(season), nflverse.injuries(season)
    cov = nflverse.def_coverage(season)
    if snaps.empty:
        return pd.DataFrame(columns=cols)
    from pocket_capper.models.nfl_model import OUT_STATUSES

    ruled_out = set()
    if not inj.empty:
        w = inj[(inj["week"] == week) & inj["report_status"].isin(OUT_STATUSES)]
        ruled_out = set(w["team"] + "|" + w["full_name"].map(norm_player))
    reg_d = _regulars(snaps, week, "defense_pct")
    reg_o = _regulars(snaps, week, "offense_pct")
    for r in (reg_d, reg_o):
        r["out"] = (r["team"] + "|" + r["player"].map(norm_player)).isin(ruled_out)
    per, unit, lg = _coverage(cov, snaps, week)
    # experience counts last season too: a starter back from injury isn't a rookie
    prev = nflverse.def_coverage(season - 1)
    exp = per["def_targets"] if not per.empty else pd.Series(dtype=float)
    if not prev.empty:
        by_name = prev.assign(n=prev["pfr_player_name"].map(norm_player)).groupby("n")["def_targets"].sum()
        exp = exp.add(pd.Series(exp.index.str.split("|").str[1], index=exp.index).map(by_name).fillna(0), fill_value=0)

    rows = []

    def add(pl, team, key, stats, direction, note):
        rows.append({"player_id": pl.name, "player": pl["player"], "team": team, "opp": opp_of[team], "key": key,
                     "stats": stats, "direction": direction, "note": note})

    wr = _top(ps, season, week, "WR", "targets", 3, out_ids)
    te = _top(ps, season, week, "TE", "targets", 1, out_ids)
    rb = _top(ps, season, week, "RB", "carries", 1, out_ids)
    qb = _top(ps, season, week, "QB", "attempts", 1, out_ids)
    for team, opp in opp_of.items():
        # --- corners: projected starters = regulars not ruled out, plus the next CB up for each one out
        cbs = reg_d[(reg_d["team"] == opp) & (reg_d["grp"] == "CB")].sort_values("pct", ascending=False)
        starters = cbs[cbs["regular"] & ~cbs["out"]]
        n_out = int((cbs["regular"] & cbs["out"]).sum())
        fill = cbs[~cbs["regular"] & ~cbs["out"]].head(n_out)
        proj = pd.concat([starters.assign(fill=False), fill.assign(fill=True)])
        if not per.empty and not proj.empty:
            k = opp + "|" + proj["player"].map(norm_player)
            proj = proj.assign(tgts=k.map(exp).fillna(0).values,
                               ypt=k.map(per["ypt"]).fillna(lg).values)
            new = proj[proj["tgts"] < NEW_CB_TARGETS]
            worst = proj.sort_values("ypt").iloc[-1]
            for pid, pl in wr[wr["team"] == team].iterrows():
                if not new.empty:
                    names = ", ".join(f"{r.player} ({int(r.tgts)} coverage targets since last season{', fill-in' if r.fill else ''})"
                                      for r in new.itertuples())
                    add(pl, team, "new_cb", ["receptions", "rec_yds"], 1, f"opp starting CB with little coverage history: {names}")
                elif worst["ypt"] >= WEAK_CB:
                    add(pl, team, "weak_cb", ["rec_yds"], 1,
                        f"opp CB {worst['player']} allows {worst['ypt']:.1f} yds/target (league {lg:.1f})")
                elif pl["rank"] == 1 and worst["ypt"] <= SHUTDOWN_CB:
                    add(pl, team, "shutdown_cb", ["rec_yds"], -1,
                        f"opp CB room all strong: weakest starter {worst['player']} {worst['ypt']:.1f} yds/target")
        # --- linebackers in coverage (TE1, RB1 receiving)
        lb = unit.get((opp, "LB"), np.nan) if not unit.empty else np.nan
        if np.isfinite(lb):
            for pid, pl in te[te["team"] == team].iterrows():
                if lb >= WEAK_LB:
                    add(pl, team, "weak_lb", ["receptions", "rec_yds"], 1, f"opp LBs allow {lb:.1f} yds/target (league {lg:.1f})")
                elif lb <= TOUGH_LB:
                    add(pl, team, "tough_lb", ["receptions", "rec_yds"], -1, f"opp LBs allow just {lb:.1f} yds/target (league {lg:.1f})")
            for pid, pl in rb[rb["team"] == team].iterrows():
                if lb >= WEAK_LB:
                    add(pl, team, "weak_lb", ["receptions"], 1, f"opp LBs allow {lb:.1f} yds/target (league {lg:.1f})")
        # --- trenches
        dl = reg_d[(reg_d["team"] == opp) & (reg_d["grp"] == "DL") & reg_d["regular"] & reg_d["out"]]
        if not dl.empty:
            for pid, pl in rb[rb["team"] == team].iterrows():
                add(pl, team, "dl_out", ["rush_yds"], 1, "opp DL starters out: " + ", ".join(dl["player"]))
        ol = reg_o[(reg_o["team"] == team) & (reg_o["grp"] == "OL") & reg_o["regular"] & reg_o["out"]]
        if len(ol) >= 2:
            note = f"{len(ol)} OL starters out: " + ", ".join(f"{r.pos} {r.player}" for r in ol.itertuples())
            for pid, pl in qb[qb["team"] == team].iterrows():
                add(pl, team, "ol_out", ["pass_yds"], -1, note)
            for pid, pl in rb[rb["team"] == team].iterrows():
                add(pl, team, "ol_out", ["rush_yds"], -1, note)
    return pd.DataFrame(rows, columns=cols)


def notes_by_player(m: pd.DataFrame) -> dict[str, str]:
    """player_id -> '; '-joined matchup notes (for the props board, ladders and the capper)."""
    if m.empty:
        return {}
    return m.groupby("player_id")["note"].apply(lambda s: "; ".join(dict.fromkeys(s))).to_dict()


def for_bet(m: pd.DataFrame, player_id: str, stat: str, side: str) -> tuple[list[dict], list[dict]]:
    """(factors behind this prop bet, factors against it)."""
    if m.empty:
        return [], []
    rows = [r for r in m[m["player_id"] == player_id].to_dict("records") if stat in r["stats"]]
    d = 1 if side == "over" else -1
    return [r for r in rows if r["direction"] == d], [r for r in rows if r["direction"] == -d]
