"""Grade finished picks (win/loss/push), record closing line value (CLV), grade manual sharp plays."""
from __future__ import annotations

import pandas as pd

from pocket_capper import db
from pocket_capper.data import nflverse
from pocket_capper.market import odds_math as om
from pocket_capper.settings import config, secret

PROP_STAT_COL = {
    "player_pass_yds": "passing_yards", "player_pass_tds": "passing_tds", "player_pass_attempts": "attempts",
    "player_rush_yds": "rushing_yards", "player_rush_attempts": "carries", "player_receptions": "receptions",
    "player_reception_yds": "receiving_yards",
}


def _results_nfl() -> pd.DataFrame:
    s = nflverse.schedules()
    s = s[s["result"].notna()]
    return s.rename(columns={"home_score": "home_pts", "away_score": "away_pts"})[["game_id", "home_pts", "away_pts"]]


def _results_cfb(seasons: set[int]) -> pd.DataFrame:
    from pocket_capper.data import cfbd

    frames = []
    for yr in seasons:
        for st in ("regular", "postseason"):
            try:
                frames.append(cfbd.games(yr, st, max_age_hours=1))
            except Exception:
                pass
    if not frames:
        return pd.DataFrame(columns=["game_id", "home_pts", "away_pts"])
    g = pd.concat(frames)
    return g[g["home_pts"].notna()][["game_id", "home_pts", "away_pts"]]


def grade_side(market: str, side: str, line: float | None, hp: float, ap: float) -> tuple[str, float]:
    margin = hp - ap
    if market == "spread":
        v = (margin if side == "home" else -margin) + line
    elif market == "total":
        v = (hp + ap - line) * (1 if side == "over" else -1)
    else:
        v = margin if side == "home" else -margin
    return ("win" if v > 0 else "loss" if v < 0 else "push"), v


def closing_price(sport: str, event_id: str, market: str, side: str, kickoff: str, player: str | None = None):
    """Last primary-book snapshot at or before kickoff."""
    if not event_id:
        return None
    where = "WHERE sport = :s AND event_id = :e AND market = :m AND side = :side AND book = :b"
    params = {"s": sport, "e": event_id, "m": market, "side": side, "b": config()["books"]["primary"]}
    if player:
        where += " AND player = :p"
        params["p"] = player
    snap = db.read(db.odds_snapshots, where, params)
    if snap.empty:
        return None
    snap = snap[pd.to_datetime(snap["fetched_at"], utc=True) <= pd.to_datetime(kickoff, utc=True)]
    if snap.empty:
        return None
    return snap.sort_values("fetched_at").iloc[-1]


def grade_all(log=print) -> dict:
    picks = db.read(db.picks, "WHERE status = 'pending'")
    stats = {"graded": 0, "clv_recorded": 0}
    if picks.empty:
        return stats
    now = pd.Timestamp.now(tz="UTC")
    res_nfl = _results_nfl() if (picks["sport"] == "nfl").any() else pd.DataFrame()
    cfb_seasons = {pd.Timestamp(k).year if pd.Timestamp(k).month >= 7 else pd.Timestamp(k).year - 1
                   for k in picks.loc[picks["sport"] == "cfb", "kickoff"].dropna()}
    res_cfb = _results_cfb(cfb_seasons) if cfb_seasons and secret("CFBD_API_KEY") else pd.DataFrame()
    ps = pd.DataFrame()
    for p in picks.itertuples():
        kickoff = pd.to_datetime(p.kickoff, utc=True, errors="coerce")
        side = p.pick_id.split("|")[3]
        # CLV once the game has kicked off
        if pd.notna(kickoff) and kickoff <= now and pd.isna(p.close_price):
            c = closing_price(p.sport, p.event_id, p.market if p.category not in ("prop", "attd") else p.market,
                              side, p.kickoff, p.player if p.category in ("prop", "attd") else None)
            if c is not None:
                clv_pts = None
                if pd.notna(p.line) and pd.notna(c["point"]):
                    # positive = you beat the close
                    clv_pts = (c["point"] - p.line) if side == "over" else (p.line - c["point"])
                db.update_pick(p.pick_id, close_line=c["point"], close_price=c["price"], clv_points=clv_pts,
                               clv_prob=om.american_to_prob(c["price"]) - om.american_to_prob(p.price))
                stats["clv_recorded"] += 1
        # results
        if p.category in ("side", "total", "ml"):
            res = res_nfl if p.sport == "nfl" else res_cfb
            if res.empty:
                continue
            r = res[res["game_id"].astype(str) == str(p.game_id)]
            if r.empty:
                continue
            status, v = grade_side(p.market, side, p.line, float(r["home_pts"].iloc[0]), float(r["away_pts"].iloc[0]))
        else:
            if ps.empty:
                season = int(str(p.game_id)[:4]) if p.game_id else now.year
                ps = nflverse.player_stats([season])
            gid = str(p.game_id)
            row = ps[(ps["player_id"] == p.player_id) & (ps["game_id"] == gid)] if "game_id" in ps else pd.DataFrame()
            if row.empty:
                # player did not appear: void once the game result is in
                if not res_nfl.empty and gid in set(res_nfl["game_id"]) and kickoff < now - pd.Timedelta(days=2):
                    db.update_pick(p.pick_id, status="void", profit_units=0.0)
                continue
            row = row.iloc[0]
            if p.category == "attd":
                v = float((row.get("rushing_tds") or 0) + (row.get("receiving_tds") or 0))
                status = "win" if v >= 1 else "loss"
            else:
                v = float(row[PROP_STAT_COL[p.market]] or 0)
                status = "win" if ((v > p.line) if side == "over" else (v < p.line)) else "loss"
        db.update_pick(p.pick_id, status=status, result_value=v,
                       profit_units=om.profit_units(p.units, p.price, status))
        stats["graded"] += 1
    grade_sharp_plays(res_nfl, res_cfb)
    log(f"Graded {stats['graded']} picks, CLV on {stats['clv_recorded']}")
    return stats


def grade_sharp_plays(res_nfl: pd.DataFrame, res_cfb: pd.DataFrame) -> None:
    sp = db.read(db.sharp_plays, "WHERE status IS NULL OR status = 'pending'")
    for m in sp.itertuples():
        res = res_nfl if m.sport == "nfl" else res_cfb
        if res.empty:
            continue
        gid = m.game_id
        r = res[res["game_id"].astype(str) == str(gid)] if gid else pd.DataFrame()
        if r.empty:
            continue
        status, _ = grade_side(m.market, m.selection, m.line, float(r["home_pts"].iloc[0]), float(r["away_pts"].iloc[0]))
        price = m.price if pd.notna(m.price) else -110
        db.update_sharp(m.id, status=status, profit_units=om.profit_units(m.units or 1.0, price, status))
