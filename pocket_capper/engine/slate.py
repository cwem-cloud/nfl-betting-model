"""The Run button. Orchestrates: blind handicap -> market -> signals -> plays -> history.

Order matters: blind projections are computed and stored BEFORE any odds are fetched,
so the handicap can never be anchored to the number.
"""
from __future__ import annotations

import json
import traceback
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd

from pocket_capper import db
from pocket_capper.data import nflverse, odds_api, splits as splits_src, teams, weather
from pocket_capper.engine import rationale
from pocket_capper.market import evaluate
from pocket_capper.market.odds_math import american_to_prob, fmt_american
from pocket_capper.models import nfl_model, nfl_props, trends
from pocket_capper.settings import config, secret


class RunLog:
    def __init__(self, cb=None):
        self.lines, self.cb = [], cb

    def __call__(self, msg: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.lines.append(f"[{stamp}] {msg}")
        if self.cb:
            self.cb(msg)


def run(sports=("nfl", "cfb"), include_props: bool = True, progress=None, now: datetime | None = None) -> dict:
    cfg = config()
    log = RunLog(progress)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    now = now or datetime.now(timezone.utc)
    db.insert_df(db.runs, pd.DataFrame([{"run_id": run_id, "started_at": db.now(), "sports": ",".join(sports), "status": "running"}]))
    summary = {"run_id": run_id, "picks": 0, "errors": []}
    for sport in sports:
        try:
            n = (run_nfl if sport == "nfl" else run_cfb)(run_id, cfg, log, now, include_props)
            summary["picks"] += n
        except Exception as e:  # one sport failing should not kill the other
            log(f"{sport.upper()} failed: {e}")
            summary["errors"].append(f"{sport}: {e}")
            traceback.print_exc()
    with db.engine().begin() as conn:
        conn.execute(
            db.runs.update().where(db.runs.c.run_id == run_id).values(
                finished_at=db.now(), status="error" if summary["errors"] else "ok", log="\n".join(log.lines)
            )
        )
    summary["log"] = log.lines
    return summary


# ------------------------------------------------------------------ shared helpers

def _fetch_odds(sport: str, cfg: dict, log, now: datetime, store: bool = True) -> pd.DataFrame:
    if not secret("ODDS_API_KEY"):
        log("No ODDS_API_KEY - blind board only (no market comparison).")
        return pd.DataFrame()
    df = odds_api.game_odds(cfg["sports"][sport]["odds_key"], cfg["books"]["compare"])
    if not df.empty:
        # never store or price in-play odds: they'd pollute closing lines and CLV
        df = df[pd.to_datetime(df["commence_time"], utc=True) > pd.Timestamp(now)]
    df["sport"] = sport
    if store:
        db.insert_snapshots(df)
    log(f"Odds: {df['event_id'].nunique() if not df.empty else 0} events from {df['book'].nunique() if not df.empty else 0} books "
        f"(API requests remaining: {odds_api.LAST_USAGE.get('remaining')})")
    return df


def _history(sport: str, event_ids: list[str]) -> pd.DataFrame:
    if not event_ids:
        return pd.DataFrame()
    ids = ",".join(f"'{e}'" for e in event_ids)
    return db.read(db.odds_snapshots, f"WHERE sport = :s AND player IS NULL AND event_id IN ({ids})", {"s": sport})


def _splits(sport: str, log) -> pd.DataFrame:
    df = splits_src.fetch_dk_splits(sport)
    if not df.empty:
        db.insert_df(db.splits, df)
    log(f"Splits: {len(df)} DK Network rows")
    hist = db.read(db.splits, "WHERE sport = :s", {"s": sport})
    if hist.empty:
        return hist
    return hist.sort_values("fetched_at").groupby(["away", "home", "market", "side"], as_index=False).last()


def _manual_for(sport: str) -> pd.DataFrame:
    return db.read(db.sharp_plays, "WHERE sport = :s AND (status IS NULL OR status = 'pending')", {"s": sport})


def _existing_pick_ids(sport: str) -> set[str]:
    return set(db.read(db.picks, "WHERE sport = :s AND status = 'pending'", {"s": sport})["pick_id"])


_OPPOSITE = {"home": "away", "away": "home", "over": "under", "under": "over"}


def _opposite_ids(pick_id: str) -> set[str]:
    """Ids of bets that would oppose this one (spread and ML on the other team both count)."""
    sport, gid, market, side, player = pick_id.split("|")
    opp = _OPPOSITE.get(side, side)
    markets = ("spread", "ml") if market in ("spread", "ml") else (market,)
    return {f"{sport}|{gid}|{m}|{opp}|{player}" for m in markets}


def _pick_record(sport, run_id, game, cand, proj, extra) -> dict:
    side = cand["side"]
    team = game["home"] if side == "home" else game["away"] if side == "away" else side.title()
    if cand["market"] == "spread":
        sel = f"{team} {cand['line']:+g}"
    elif cand["market"] == "total":
        sel = f"{side.title()} {cand['line']:g}"
    else:
        sel = f"{team} ML"
    return {
        "pick_id": f"{sport}|{game['game_id']}|{cand['market']}|{side}|",
        "run_id": run_id, "sport": sport, "category": cand["category"], "event_id": game.get("event_id"),
        "game_id": str(game["game_id"]), "kickoff": str(game.get("kickoff")), "matchup": f"{game['away']} @ {game['home']}",
        "market": cand["market"], "selection": sel, "player": None, "player_id": None,
        "line": cand["line"], "price": cand["price"], "book": config()["books"]["primary"], "units": cand["units"],
        "model_prob": cand["model_prob"], "market_prob": cand["market_prob"], "final_prob": cand["final_prob"],
        "ev": cand["ev"], "fair_line": cand["fair_line"], "signals": cand["signals"],
        "push_prob": cand["push_prob"], **extra,
    }


def _finalize_pick(p: dict, blind_summary: str, dk_summary: str, sharp_summary: str | None, context: list[str]) -> dict:
    p = dict(p)
    p.update(blind_summary=blind_summary, dk_summary=dk_summary, sharp_summary=sharp_summary,
             context=context, breakeven=american_to_prob(p["price"]))
    bullets = rationale.build(p)
    prose = rationale.polish(p["selection"], bullets)
    p["rationale"] = (prose + "\n\n" if prose else "") + bullets
    return p


def _game_rows(sport, run_id, games, projs, odds, cfg, log, match_event, name_of, trend_fn=None):
    """Shared game evaluation loop. Returns number of picks written."""
    board = pd.DataFrame()
    snap = pd.DataFrame()
    if not odds.empty:
        from pocket_capper.market.market import current_board

        board = current_board(odds)
        snap = _history(sport, list(odds["event_id"].unique()))
    split_df = _splits(sport, log) if not odds.empty else pd.DataFrame()
    if not split_df.empty:  # resolve split team names once, not per game
        split_df = split_df.assign(home_key=split_df["home"].map(name_of), away_key=split_df["away"].map(name_of))
    manual = _manual_for(sport)
    existing = _existing_pick_ids(sport)
    rows, n = [], 0
    for _, g in games.iterrows():
        pr = projs[projs["game_id"] == g["game_id"]]
        if pr.empty:
            continue
        pr = pr.iloc[0].to_dict()
        ev_id = match_event(g, odds) if not odds.empty else None
        g = g.copy()
        g["event_id"] = ev_id
        sp_map = {}
        if not split_df.empty:
            for s in split_df[(split_df["home_key"] == g["home"]) & (split_df["away_key"] == g["away"])].itertuples():
                sp_map[(s.market, s.side)] = {"bets_pct": s.bets_pct, "money_pct": s.money_pct}
        man = []
        if ev_id and not manual.empty:
            for m in manual[manual["event_id"] == ev_id].itertuples():
                man.append({"source": m.source, "market": m.market, "side": m.selection, "direction": 1,
                            "detail": f"{m.selection} {'' if pd.isna(m.line) else f'{m.line:+g}'}"
                                      + ("" if pd.isna(m.units) else f" ({m.units:g}U)")})
        base = {
            "game_id": g["game_id"], "kickoff": str(g.get("kickoff")), "away": g["away"], "home": g["home"],
            "proj_away_pts": pr["away_pts"], "proj_home_pts": pr["home_pts"], "notes": " | ".join(pr["adjust_notes"]),
            "event_id": ev_id,
        }
        if not ev_id:
            rows.append({**base, "fair_spread_home": pr["fair_spread_home"], "fair_total": pr["fair_total"],
                         "home_win_prob": pr["home_win_prob"], "best_play": "No market yet"})
            continue
        row, cands = evaluate.evaluate_game(sport=sport, proj=pr, board=board, snap=snap, event_id=ev_id,
                                            splits=sp_map, manual=man, cfg=cfg)
        plays = [c for c in cands if c["units"] > 0]
        context = list(pr["adjust_notes"])
        if trend_fn:
            context += trend_fn(g, row)
        blind = (f"{g['away']} {pr['away_pts']:.1f} - {g['home']} {pr['home_pts']:.1f} "
                 f"(fair {g['home']} {pr['fair_spread_home']:+g}, total {pr['fair_total']:g}, "
                 f"{g['home']} win {pr['home_win_prob']:.0%})")
        best = []
        for c in sorted(plays, key=lambda c: -c["ev"]):
            p = _pick_record(sport, run_id, g, c, pr, {})
            if p["pick_id"] in existing:  # already issued; refreshed below
                best.append(f"{p['selection']} {fmt_american(c['price'])} (already on card)")
                continue
            if _opposite_ids(p["pick_id"]) & existing:  # never hedge against our own open play
                continue
            if c["market"] == "spread":
                dk_s = f"{p['selection']} {fmt_american(c['price'])}"
                sharp_s = None if row["sharp_spread_home"] is None else f"{g['home']} {row['sharp_spread_home']:+g}"
            elif c["market"] == "total":
                dk_s = f"{p['selection']} {fmt_american(c['price'])}"
                sharp_s = None if row["sharp_total"] is None else f"{row['sharp_total']:g}"
            else:
                dk_s, sharp_s = f"{p['selection']} {fmt_american(c['price'])}", None
            p = _finalize_pick(p, blind, dk_s, sharp_s, context)
            db.upsert_pick(p)
            n += 1
            best.append(f"{p['selection']} {fmt_american(c['price'])} ({c['units']:g}U, {c['ev']:+.1%})")
        # refresh every open pick on this game with the current line and its current EV
        for c in cands:
            pid = _pick_record(sport, run_id, g, c, pr, {})["pick_id"]
            if pid in existing:
                db.refresh_pick(pid, c["line"], c["price"], c["ev"])
        # strongest non-qualifying lean for transparency
        lean = max(cands, key=lambda c: c["ev"]) if cands else None
        rows.append({
            **base, **row, "best_play": " + ".join(best) if best else ("Flagged" if row.get("flag") else "No edge"),
            "lean": None if not lean else f"{lean['market']} {lean['side']} ({lean['ev']:+.1%} EV)",
            "signals": json.dumps([s for c in cands for s in c["signals"] if c.get("units", 0) > 0 or s["direction"] > 0][:8]),
        })
    out = pd.DataFrame(rows)
    db.save_board(run_id, sport, "games", out)
    log(f"{sport.upper()}: {len(out)} games on the board, {n} new plays")
    return n


# ------------------------------------------------------------------ NFL

def run_nfl(run_id, cfg, log, now, include_props) -> int:
    ncfg = cfg["sports"]["nfl"]
    sched = nflverse.schedules()
    horizon = now + timedelta(days=cfg["run"]["lookahead_days"])
    sched["kickoff"] = pd.to_datetime(
        sched["gameday"].dt.strftime("%Y-%m-%d") + " " + sched["gametime"].fillna("13:00"), errors="coerce"
    ).dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    upcoming = sched[(sched["kickoff"] > now) & (sched["kickoff"] <= horizon) & sched["result"].isna()]
    if upcoming.empty:
        log("NFL: no games in the window")
        return 0
    season, week = int(upcoming["season"].iloc[0]), int(upcoming["week"].min())
    log(f"NFL {season} week {week}: {len(upcoming)} games. Building blind ratings...")

    games = nfl_model.build_games([season - 1, season])
    ratings = nfl_model.fit(games, season, week, ncfg)
    ps = nflverse.player_stats([season - 1, season])
    qb_vals = nfl_model.team_qb_value(ps, season)
    wx = {}
    for g in upcoming.itertuples():
        lat, lon, roof = teams.NFL_STADIUMS.get(g.home_team, (None, None, "dome"))
        if lat and roof != "dome":
            w = weather.kickoff_weather(lat, lon, g.kickoff)
            if w:
                wx[g.game_id] = w
    up = upcoming.rename(columns={"home_team": "home", "away_team": "away"}).copy()
    up["neutral"] = up["location"].eq("Neutral")
    # A Monday/Thursday run can span two weeks: injury flags are week-specific.
    projs = pd.concat(
        [nfl_model.project_games(wk_games, ratings, ncfg, nfl_model.qb_status(season, wk), qb_vals, wx)
         for wk, wk_games in up.groupby("week")],
        ignore_index=True,
    )
    projs["kickoff"] = up.set_index("game_id").loc[projs["game_id"], "kickoff"].astype(str).values
    db.insert_df(db.projections, projs.assign(run_id=run_id, sport="nfl", notes=projs["adjust_notes"].map(" | ".join)))
    db.save_board(run_id, "nfl", "ratings", ratings.table())
    log("NFL blind projections stored (before any odds were pulled).")

    odds = _fetch_odds("nfl", cfg, log, now)

    def match_event(g, odds_df):
        m = odds_df[(odds_df["home_name"].map(teams.nfl_abbr) == g["home"]) & (odds_df["away_name"].map(teams.nfl_abbr) == g["away"])]
        return m["event_id"].iloc[0] if not m.empty else None

    nick = {k.split()[-1].lower(): v for k, v in teams.NFL_NAME_TO_ABBR.items()}

    def name_of(s):
        s = str(s)
        return teams.nfl_abbr(s) or nick.get(s.split()[-1].lower()) if s else None

    def trend_fn(g, row):
        out = [trends.team_ats(sched, g["home"], season), trends.team_ats(sched, g["away"], season)]
        gg = g.copy()
        if row.get("dk_spread_home") is not None:
            gg["spread_line_dk"] = -row["dk_spread_home"]
        return out + trends.situations(sched, gg)

    n = _game_rows("nfl", run_id, up, projs, odds, cfg, log, match_event, name_of, trend_fn)

    if include_props:
        n += _run_nfl_props(run_id, cfg, log, season, week, up, projs, odds, ps)
    return n


def _run_nfl_props(run_id, cfg, log, season, week, up, projs, odds, ps) -> int:
    tp = pd.concat([
        pd.DataFrame({"team": projs["home"], "opp": projs["away"], "pts": projs["home_pts"], "margin": projs["fair_margin"]}),
        pd.DataFrame({"team": projs["away"], "opp": projs["home"], "pts": projs["away_pts"], "margin": -projs["fair_margin"]}),
    ])
    inj = nflverse.injuries(season)
    out_ids = set()
    if not inj.empty:
        wk = inj[inj["week"] == week]
        out_ids = set(wk[wk["report_status"].isin(nfl_model.OUT_STATUSES)]["gsis_id"])
    plays = nflverse.pbp([season])
    pproj = nfl_props.project_players(ps, plays, season, week, tp, out_ids)
    log(f"Player projections: {len(pproj)} players ({len(out_ids)} ruled out/doubtful removed)")
    if odds.empty or pproj.empty:
        db.save_board(run_id, "nfl", "projections_players", pproj)
        return 0
    frames = []
    for ev_id in odds["event_id"].unique():
        try:
            df = odds_api.event_props(cfg["sports"]["nfl"]["odds_key"], ev_id, cfg["props"]["nfl_markets"], cfg["books"]["compare"])
            frames.append(df)
        except Exception as e:
            log(f"props fetch failed for {ev_id}: {e}")
    props = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if props.empty:
        log("No props posted yet.")
        return 0
    props["sport"] = "nfl"
    db.insert_snapshots(props)
    log(f"Props: {len(props)} prices across {props['book'].nunique()} books "
        f"(API requests remaining: {odds_api.LAST_USAGE.get('remaining')})")
    ev = evaluate.evaluate_props(props, pproj, cfg)
    if ev.empty:
        return 0
    existing_all = _existing_pick_ids("nfl")
    gid_by_event = {}
    for r in projs.itertuples():
        m = odds[(odds["home_name"].map(teams.nfl_abbr) == r.home) & (odds["away_name"].map(teams.nfl_abbr) == r.away)]
        if not m.empty:
            gid_by_event[m["event_id"].iloc[0]] = r.game_id
    for r in ev.to_dict("records"):  # current EV on open prop picks, qualifying or not
        pid = f"nfl|{gid_by_event.get(r['event_id'])}|{r['market']}|{r['side']}|{r['player_id']}"
        if pid in existing_all:
            db.refresh_pick(pid, r["line"], r["price"], r["ev"])
    ev_games = odds.drop_duplicates("event_id").set_index("event_id")
    ev["matchup"] = ev["event_id"].map(lambda e: f"{teams.nfl_abbr(ev_games.loc[e, 'away_name'])} @ {teams.nfl_abbr(ev_games.loc[e, 'home_name'])}")
    ev["kickoff"] = ev["event_id"].map(lambda e: ev_games.loc[e, "commence_time"])
    n = 0
    gid = {(r.home, r.away): r.game_id for r in projs.itertuples()}
    existing = _existing_pick_ids("nfl")
    for r in ev[ev["qualifies"]].to_dict("records"):
        away, home = r["matchup"].split(" @ ")
        is_td = r["stat"] == "anytime_td"
        label = nfl_props.STAT_LABEL[r["stat"]]
        sel = f"{r['player']} {label}" if is_td else f"{r['player']} {r['side'].title()} {r['line']:g} {label}"
        p = {
            "pick_id": f"nfl|{gid.get((home, away))}|{r['market']}|{r['side']}|{r['player_id']}",
            "run_id": run_id, "sport": "nfl", "category": "attd" if is_td else "prop", "event_id": r["event_id"],
            "game_id": gid.get((home, away)), "kickoff": r["kickoff"], "matchup": r["matchup"], "market": r["market"],
            "selection": sel, "player": r["player"], "player_id": r["player_id"], "line": r["line"], "price": r["price"],
            "book": cfg["books"]["primary"], "units": r["units"], "model_prob": r["model_prob"],
            "market_prob": r["market_prob"], "final_prob": r["final_prob"], "ev": r["ev"],
            "fair_line": r["projection"], "signals": [],
        }
        if p["pick_id"] in existing or _opposite_ids(p["pick_id"]) & existing:
            continue
        blind = (f"{r['player']} ({r['position']}, {r['team']} vs {r['opp']}) projects "
                 + (f"{r['model_prob']:.0%} to score ({r['projection']:.2f} expected TDs)" if is_td
                    else f"{r['projection']:.1f} {label}"))
        p = _finalize_pick(p, blind, f"{sel} {fmt_american(r['price'])}", None,
                           [f"Market consensus from {r['books']} book(s)"])
        db.upsert_pick(p)
        n += 1
    db.save_board(run_id, "nfl", "props", ev[ev["stat"] != "anytime_td"])
    db.save_board(run_id, "nfl", "attd", ev[ev["stat"] == "anytime_td"])
    log(f"Props: {int(ev['qualifies'].sum())} plays ({int((ev['qualifies'] & (ev['stat'] == 'anytime_td')).sum())} TD Zone)")
    return n


# ------------------------------------------------------------------ CFB

def run_cfb(run_id, cfg, log, now, include_props) -> int:
    from pocket_capper.data import cfbd
    from pocket_capper.models import cfb_model

    if not secret("CFBD_API_KEY"):
        log("CFB skipped: set CFBD_API_KEY (free at collegefootballdata.com/key).")
        return 0
    ccfg = cfg["sports"]["cfb"]
    season = now.year if now.month >= 7 else now.year - 1
    games = cfb_model.build_games(season)
    horizon = now + timedelta(days=cfg["run"]["lookahead_days"])
    games["kickoff"] = games["start"]
    up = games[(games["season"] == season) & (games["start"] > now) & (games["start"] <= horizon)
               & games["home_pts"].isna() & ((games["home"] != "FCS") & (games["away"] != "FCS"))]
    if up.empty:
        log("CFB: no FBS games in the window")
        return 0
    week = int(up["week"].min())
    log(f"CFB {season} week {week}: {len(up)} FBS games. Building blind ratings...")
    ratings = cfb_model.fit(games, season, week, ccfg)
    wx = {}
    try:
        ven = cfbd.venues().set_index("venue_id")
        for g in up.itertuples():
            if g.venue_id in ven.index:
                v = ven.loc[g.venue_id]
                if not v["dome"] and pd.notna(v["lat"]):
                    w = weather.kickoff_weather(float(v["lat"]), float(v["lon"]), g.start)
                    if w:
                        wx[g.game_id] = w
    except Exception as e:
        log(f"CFB weather skipped: {e}")
    projs = cfb_model.project_games(up, ratings, ccfg, wx)
    projs["kickoff"] = up.set_index("game_id").loc[projs["game_id"], "kickoff"].astype(str).values
    db.insert_df(db.projections, projs.assign(run_id=run_id, sport="cfb", notes=projs["adjust_notes"].map(" | ".join)))
    db.save_board(run_id, "cfb", "ratings", ratings.table())
    log("CFB blind projections stored (before any odds were pulled).")

    odds = _fetch_odds("cfb", cfg, log, now, store=False)  # stored after opening lines are seeded
    schools = sorted(set(up["home_raw"]) | set(up["away_raw"]))
    up = up.assign(home=up["home_raw"], away=up["away_raw"])

    def match_event(g, odds_df):
        hn = odds_df["home_name"].map(lambda s: teams.match_cfb(s, schools))
        an = odds_df["away_name"].map(lambda s: teams.match_cfb(s, schools))
        # orientation must match: the projection's "home" is the side the odds call home
        m = odds_df[(hn == g["home"]) & (an == g["away"])]
        return m["event_id"].iloc[0] if not m.empty else None

    def name_of(s):
        return teams.match_cfb(str(s), schools)

    if not odds.empty:
        _seed_cfb_opens(season, up, odds, match_event, log)
        db.insert_snapshots(odds)
    return _game_rows("cfb", run_id, up, projs, odds, cfg, log, match_event, name_of)


def _seed_cfb_opens(season, up, odds, match_event, log) -> None:
    """The first time we see a CFB game, back-fill DK's opening number from CFBD so movement starts at the open."""
    from pocket_capper.data import cfbd

    try:
        lines = cfbd.lines(season)
    except Exception as e:
        log(f"CFB opening lines skipped: {e}")
        return
    if lines.empty:
        return
    dk = lines[lines["provider"].fillna("").str.lower().str.contains("draftkings")]
    seeded = 0
    for _, g in up.iterrows():
        ev_id = match_event(g, odds)
        ln = dk[dk["game_id"] == str(g["game_id"])]
        if not ev_id or ln.empty:
            continue
        hist = db.read(db.odds_snapshots, "WHERE event_id = :e AND player IS NULL AND book = 'draftkings'", {"e": ev_id})
        if not hist.empty:
            continue  # seen before: history already starts earlier
        ln = ln.iloc[0]
        at = (pd.to_datetime(odds["fetched_at"], utc=True).min() - pd.Timedelta(hours=1)).isoformat()
        home_name, away_name, commence = odds.loc[odds["event_id"] == ev_id, ["home_name", "away_name", "commence_time"]].iloc[0]
        rows = []
        if pd.notna(ln["spread_open"]):
            rows += [("spread", "home", float(ln["spread_open"])), ("spread", "away", -float(ln["spread_open"]))]
        if pd.notna(ln["total_open"]):
            rows += [("total", "over", float(ln["total_open"])), ("total", "under", float(ln["total_open"]))]
        if rows:
            db.insert_df(db.odds_snapshots, pd.DataFrame([{
                "sport": "cfb", "event_id": ev_id, "commence_time": commence, "home_name": home_name,
                "away_name": away_name, "book": "draftkings", "market": m, "side": sd, "point": pt,
                "price": -110.0, "fetched_at": at} for m, sd, pt in rows]))
            seeded += 1
    if seeded:
        log(f"CFB: seeded DK opening lines for {seeded} games from CFBD")
