"""Angle lab: situational betting angles, backtested on closing lines and shadow-tracked every week.

Each angle is a rule over a team-game table built the same way for history (nflverse closing lines)
and for this week's games (DK's current numbers, the kickoff wind forecast), so what we backtest is
exactly what fires live. Angles never stake on their own: every hit is logged as a tracking-only pick
(source "angle") and graded like any other, and the weekly scorecard says when one has earned a place
on the card. When two or more angles land on the same bet, that is logged as its own "stack" pick.

Backtest notes are from 2010-19 vs 2020-25 (52.4% breaks even at -110).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

WEST = {"SEA", "SF", "LA", "LAR", "LAC", "OAK", "LV", "ARI", "DEN", "SD"}
EAST = {"NE", "NYJ", "NYG", "BUF", "MIA", "PHI", "WAS", "BAL", "PIT", "CLE", "CIN", "ATL", "CAR", "TB", "JAX",
        "IND", "DET"}
BREAKEVEN = 0.524


@dataclass(frozen=True)
class Angle:
    key: str
    label: str
    market: str  # "spread" (bet the row's team) or "total"
    bet: str  # "team", "under" or "over"
    rule: Callable[[pd.DataFrame], pd.Series]
    why: str


def _sunday(t):
    return t["dow"] == 6


ANGLES = [
    Angle("west_early", "West team at 1pm ET in the East", "spread", "team",
          lambda t: (t["side"] == "away") & t["team"].isin(WEST) & t["opp"].isin(EAST) & (t["hour"] == 13) & _sunday(t),
          "54.6% / 58.7% ATS (dogs 62% since 2020)"),
    Angle("blowout_bounce", "Underdog after a 20+ point loss", "spread", "team",
          lambda t: (t["prev_margin"] <= -20) & (t["fav_by"] < 0),
          "54.3% / 54.9% ATS"),
    Angle("vs_rested_home", "Road team vs a home team off a bye", "spread", "team",
          lambda t: (t["side"] == "away") & (t["opp_rest"] >= 13) & (t["rest"] <= 8),
          "54.4% / 53.5% ATS"),
    Angle("east_late_west", "East team in a late game out West", "spread", "team",
          lambda t: (t["side"] == "away") & t["team"].isin(EAST) & t["opp"].isin(WEST) & (t["hour"] >= 16) & _sunday(t),
          "53.5% / 55.7% ATS"),
    Angle("wind_under", "Wind 15+ mph: under", "total", "under",
          lambda t: (t["wind"] >= 15) & ~t["roof"].isin(["dome", "closed"]),
          "53.2% / 54.9% unders"),
    Angle("prime_under", "Primetime under", "total", "under",
          lambda t: t["prime"], "49.2% / 54.4% unders (recent only)"),
    Angle("low_total_under", "Total 38.5 or lower: under", "total", "under",
          lambda t: t["total_line"] <= 38.5, "45.8% / 60.2% unders (recent only)"),
    Angle("div_dog_short", "Division underdog +3 to +6.5", "spread", "team",
          lambda t: (t["div"] == 1) & (t["fav_by"] <= -3) & (t["fav_by"] >= -6.5),
          "50.6% / 56.9% ATS (recent only)"),
    Angle("big_home_fav", "Home favorite of 10+", "spread", "team",
          lambda t: (t["side"] == "home") & (t["fav_by"] >= 10), "48.2% / 55.2% ATS (recent only)"),
]
BY_KEY = {a.key: a for a in ANGLES}


def team_table(sched: pd.DataFrame, lines: dict | None = None, wind: dict | None = None) -> pd.DataFrame:
    """One row per team-game (regular season). `lines` = {game_id: (home spread_line, total_line)} and
    `wind` = {game_id: mph} override the schedule for games not yet played (spread_line = home margin)."""
    s = sched[sched["game_type"] == "REG"].copy() if "game_type" in sched else sched.copy()
    for col, default in (("div_game", 0), ("wind", np.nan), ("roof", "outdoors"), ("gametime", "13:00"),
                         ("weekday", ""), ("home_rest", 7), ("away_rest", 7), ("total_line", np.nan),
                         ("spread_line", np.nan), ("result", np.nan), ("home_score", np.nan), ("away_score", np.nan)):
        if col not in s:
            s[col] = default
    s["div_game"] = pd.to_numeric(s["div_game"], errors="coerce").fillna(0)
    for gid, (sp, tot) in (lines or {}).items():
        m = s["game_id"] == gid
        if sp is not None and np.isfinite(sp):
            s.loc[m, "spread_line"] = sp
        if tot is not None and np.isfinite(tot):
            s.loc[m, "total_line"] = tot
    for gid, w in (wind or {}).items():
        s.loc[s["game_id"] == gid, "wind"] = w
    gt = s["gametime"].fillna("13:00").astype(str)
    s["hour"] = pd.to_numeric(gt.str[:2], errors="coerce")
    s["dow"] = pd.to_datetime(s["gameday"]).dt.dayofweek
    s["prime"] = s["weekday"].isin(["Monday", "Thursday"]) | ((s["weekday"] == "Sunday") & (gt >= "19:00"))
    rows = []
    for side, team, opp, rest, orest, sign in (("home", "home_team", "away_team", "home_rest", "away_rest", 1),
                                               ("away", "away_team", "home_team", "away_rest", "home_rest", -1)):
        rows.append(pd.DataFrame({
            "season": s["season"], "week": s["week"], "game_id": s["game_id"], "side": side,
            "team": s[team], "opp": s[opp], "rest": s[rest], "opp_rest": s[orest],
            "fav_by": sign * s["spread_line"], "total_line": s["total_line"], "margin": sign * s["result"],
            "total": s["home_score"] + s["away_score"], "div": s["div_game"], "wind": s["wind"],
            "roof": s["roof"].fillna("outdoors"), "hour": s["hour"], "dow": s["dow"], "prime": s["prime"],
        }))
    t = pd.concat(rows, ignore_index=True).sort_values(["team", "season", "week"])
    t["prev_margin"] = t.groupby(["team", "season"])["margin"].shift()
    t["cover"] = np.sign(t["margin"] - t["fav_by"])
    t["over"] = np.sign(t["total"] - t["total_line"])
    return t.reset_index(drop=True)


def _hits(t: pd.DataFrame, a: Angle) -> pd.DataFrame:
    m = a.rule(t).fillna(False).astype(bool)
    if a.market == "total":  # game-level: one row per game
        m &= t["side"] == "home"
    return t[m]


def _value(rows: pd.DataFrame, a: Angle) -> pd.Series:
    return rows["cover"] if a.market == "spread" else rows["over"] * (-1 if a.bet == "under" else 1)


def _rate(v: pd.Series) -> dict:
    w, l = int((v > 0).sum()), int((v < 0).sum())
    n = w + l
    pct = w / n if n else np.nan
    return {"w": w, "l": l, "pct": pct, "z": (pct - 0.5) / np.sqrt(0.25 / n) if n else np.nan}


def backtest(sched: pd.DataFrame, split: int = 2020, since: int = 2010) -> pd.DataFrame:
    """Hit rate per angle (and for 2+ angle stacks) in the older and recent periods."""
    t = team_table(sched)
    t = t[(t["season"] >= since) & t["cover"].notna()]
    out = []
    for a in ANGLES:
        h = _hits(t, a)
        v = _value(h, a)
        old, new = _rate(v[h["season"] < split]), _rate(v[h["season"] >= split])
        out.append({"key": a.key, "label": a.label, "old": old, "new": new,
                    "holds": bool(old["pct"] > BREAKEVEN and new["pct"] > BREAKEVEN)})
    for k, (old, new) in _stack_rates(t, split).items():
        out.append({"key": k, "label": f"{k[-1]}+ angles on the same side", "old": old, "new": new,
                    "holds": bool(old["pct"] > BREAKEVEN and new["pct"] > BREAKEVEN)})
    return pd.DataFrame(out)


def _stack_rates(t: pd.DataFrame, split: int) -> dict:
    """Spread bets backed by 2+ (and 3+) angles at once."""
    n = pd.Series(0, index=t.index)
    for a in ANGLES:
        if a.market == "spread":
            n += a.rule(t).fillna(False).astype(int)
    out = {}
    for k in (2, 3):
        v = t.loc[n >= k, "cover"]
        s = t.loc[n >= k, "season"]
        out[f"stack{k}"] = (_rate(v[s < split]), _rate(v[s >= split]))
    return out


def fire(t: pd.DataFrame, game_ids) -> dict[str, list[dict]]:
    """Angles that apply to each game: {game_id: [{key, label, market, side, team, why}]}, where side is
    the bet ("home"/"away" for spreads, "under"/"over" for totals)."""
    t = t[t["game_id"].isin(set(game_ids))]
    out: dict[str, list[dict]] = {}
    for a in ANGLES:
        for r in _hits(t, a).itertuples():
            side = r.side if a.market == "spread" else a.bet
            out.setdefault(r.game_id, []).append({"key": a.key, "label": a.label, "market": a.market, "side": side,
                                                  "team": r.team if a.market == "spread" else None, "why": a.why})
    return out


def stacks(fired: list[dict]) -> list[dict]:
    """Bets two or more angles agree on: [{market, side, team, keys, labels}]."""
    by = {}
    for f in fired:
        by.setdefault((f["market"], f["side"]), []).append(f)
    return [{"market": m, "side": s, "team": fs[0]["team"], "keys": [f["key"] for f in fs],
             "labels": [f["label"] for f in fs]} for (m, s), fs in by.items() if len(fs) >= 2]


def agreeing(fired: list[dict], market: str, side: str) -> tuple[list[dict], list[dict]]:
    """(angles behind this bet, angles against it). ML counts as the spread side."""
    m = "spread" if market == "ml" else market
    opp = {"home": "away", "away": "home", "over": "under", "under": "over"}.get(side)
    return ([f for f in fired if f["market"] == m and f["side"] == side],
            [f for f in fired if f["market"] == m and f["side"] == opp])


def fire_game(t: pd.DataFrame, game_id: str, home_spread_line=None, total_line=None) -> list[dict]:
    """Angles for one game at today's numbers (home_spread_line = home's expected margin, i.e. -DK spread)."""
    g = t[t["game_id"] == game_id].copy()
    if g.empty:
        return []
    if home_spread_line is not None and np.isfinite(home_spread_line):
        g["fav_by"] = np.where(g["side"] == "home", home_spread_line, -home_spread_line)
    if total_line is not None and np.isfinite(total_line):
        g["total_line"] = total_line
    return fire(g, [game_id]).get(game_id, [])


# ---------------------------------------------------------------- weekly scorecard

def angle_key(pick_id: str) -> str | None:
    tail = str(pick_id).rsplit("|", 1)[-1]
    return tail[len("angle-"):] if tail.startswith("angle-") else None


def scorecard(picks: pd.DataFrame, cfg: dict, sched: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every angle / matchup factor / stack: backtest (game angles), live shadow record, and a verdict.
    ready  = a candidate whose live record has earned a look for the card (enough picks, winning, not losing CLV)
    fading = a proven angle losing live over a real sample (watch it, retire it if it keeps going)"""
    from pocket_capper.models import matchups

    acfg = cfg.get("angles", {})
    proven, min_n = set(acfg.get("proven", [])), int(acfg.get("promote_min_n", 30))
    labels = {a.key: (a.label, "game") for a in ANGLES}
    labels |= {k: (v[0], "matchup") for k, v in matchups.FACTORS.items()}
    labels |= {"stack": ("2+ angles on the same bet", "stack"), "stack_proven": ("2+ proven angles on the same bet", "stack"),
               "mstack": ("2+ matchup factors on the same prop", "stack")}
    bt = backtest(sched).set_index("key") if sched is not None else pd.DataFrame()
    a = picks[picks["source"].fillna("") == "angle"].copy() if not picks.empty and "source" in picks else pd.DataFrame()
    if not a.empty:
        a["key"] = a["pick_id"].map(angle_key)
    rows = []
    for key, (label, kind) in labels.items():
        g = a[a["key"] == key] if not a.empty else a
        done = g[g["status"].isin(["win", "loss", "push"])] if not g.empty else g
        w, l = (int((done["status"] == s).sum()) if not done.empty else 0 for s in ("win", "loss"))
        risk = float(done["units"].sum()) if not done.empty else 0.0
        pl = float(done["profit_units"].sum()) if not done.empty else 0.0
        clv = float(done["clv_prob"].mean()) if not done.empty and done["clv_prob"].notna().any() else np.nan
        pct = w / (w + l) if w + l else np.nan
        old = bt.loc[key, "old"] if key in bt.index else None
        new = bt.loc[key, "new"] if key in bt.index else None
        if key == "stack_proven" and "stack2" in bt.index:
            old, new = bt.loc["stack2", "old"], bt.loc["stack2", "new"]
        status = "proven" if key in proven else "candidate"
        if w + l >= min_n:
            if status == "candidate" and pct > 0.54 and pl > 0 and not (clv < 0):
                status = "READY: promote?"
            elif status == "proven" and pct < 0.5:
                status = "fading: watch"
        rows.append({"key": key, "angle": label, "kind": kind, "status": status,
                     "backtest": "" if old is None else f"{old['pct']:.1%} / {new['pct']:.1%} (n {old['w'] + old['l']}/{new['w'] + new['l']})",
                     "live": f"{w}-{l}" if w + l else "-", "live_pct": pct, "units": round(pl, 2),
                     "roi": pl / risk if risk else np.nan, "clv_prob": clv,
                     "open": int((g["status"] == "pending").sum()) if not g.empty else 0})
    return pd.DataFrame(rows)


def support(picks: pd.DataFrame) -> pd.Series:
    """For each non-angle pick: how many open angle / matchup shadow picks back the same bet (stacks excluded)."""
    if picks.empty or "source" not in picks:
        return pd.Series(0, index=picks.index)
    a = picks[(picks["source"].fillna("") == "angle") & (picks["status"] == "pending")]
    keys = a["pick_id"].map(angle_key)
    a = a[~keys.isin(["stack", "stack_proven", "mstack"])]
    base = a["pick_id"].str.rsplit("|", n=1).str[0].value_counts()

    def bet_of(pid: str) -> str:
        parts = pid.split("|")
        if len(parts) >= 5 and parts[2] == "ml":
            parts[2] = "spread"
        if parts[2] in ("spread", "total"):
            return "|".join(parts[:4])
        return "|".join(parts[:5])
    return picks["pick_id"].map(lambda pid: int(base.get(bet_of(pid), 0)))
