"""The reasoning capper: a second handicapper that forms its own opinions from ball knowledge.

The EV model prices numbers. The capper reads the whole situation for one game at a time - the
model's projections and edges, line movement and sharp signals, the injury report with snap shares
on both sides, who inherits work from absent teammates, recent usage, the props and alt-line ladders
DK is offering - plus the latest news it looks up itself, and decides what it actually likes and why.
It can agree with the model, disagree with it, or pass. Every pick it makes has to be one of the DK
offers listed in its dossier, so it is graded and CLV-tracked exactly like the model's picks, in its
own record (picks.source = "capper").

It keeps a notebook: after enough of its picks are graded it reviews its theses against the results
and rewrites a short list of lessons, which it reads before every game.
"""
from __future__ import annotations

import json
from datetime import timedelta

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from pocket_capper import db
from pocket_capper.data import nflverse
from pocket_capper.market.odds_math import fmt_american
from pocket_capper.models import nfl_model
from pocket_capper.models.nfl_props import MARKET_STAT, STAT_LABEL
from pocket_capper.settings import config, secret

ALT_TO_MAIN = {
    "player_reception_yds_alternate": "player_reception_yds",
    "player_receptions_alternate": "player_receptions",
    "player_rush_yds_alternate": "player_rush_yds",
    "player_pass_yds_alternate": "player_pass_yds",
}

SYSTEM = """You are the reasoning capper for Pocket Capper, an NFL betting model. You work next to a \
quantitative EV model. The model is good at prices; your job is ball knowledge: reading situations \
the box scores and the market have not caught up to.

Things that create real edges (look for them, and say which one a pick rests on):
- Role changes: who inherits targets/carries when a teammate is out, a player returning from a missed \
game whose recent numbers are depressed, a backup QB changing target distribution, a new play-caller.
- Matchups: a backup or third-string corner forced into the lineup, a missing run-stopper or pass \
rusher, a slot receiver against a weak nickel, a defense that funnels production to RBs or TEs.
- Game script: a big favourite leaning on the run late, a trailing team throwing, pace, weather \
(wind hurts deep passing and kicking more than rushing).
- Market psychology: overs and favourites carry public money, recency bias after a big or quiet game, \
lines that have not moved despite news, stale numbers at DK versus the sharp books.
- Spots: short weeks, travel, lookahead/letdown, division familiarity.

Discipline:
- Use web search to check the latest news before you commit: injury designations and inactives, \
practice reports, beat-reporter notes on roles and snap plans, weather. Prefer recent sources.
- Only pick from the offers listed in the dossier, by offer id. The price you get is the DK price shown.
- The model's numbers are an input, not an order. Agreeing with it is fine; disagreeing is fine when \
you can say what it is missing. Never pick something just because the EV column is positive.
- Passing is a good outcome. Most games deserve zero or one pick; never more than three.
- Size in units from 0.25 to 1.0: 0.25 for a lean or a ladder long shot, 0.5 standard, 0.75-1.0 only \
when several independent reasons line up and the price is right. Ladder rungs above the first should \
be small; rungs on the same player win and lose together.
- Write theses a sharp bettor would respect: specific, checkable, no hype, no guarantees.
- Read your notebook. It holds lessons from your own graded picks; apply them or say why not.

Finish by calling submit_card exactly once."""

SUBMIT_TOOL = {
    "name": "submit_card",
    "description": "Submit your read on this game and your picks (zero to three). Call exactly once, at the end.",
    "strict": True,
    "eager_input_streaming": True,
    "input_schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["game_read", "picks", "passes"],
        "properties": {
            "game_read": {"type": "string", "description": "Your overall read on the game, 2-4 sentences."},
            "passes": {"type": "string", "description": "What you looked at and passed on, and why (short)."},
            "picks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["offer_id", "units", "confidence", "thesis", "factors", "risk", "vs_model", "edge_type"],
                    "properties": {
                        "offer_id": {"type": "string", "description": "An offer id from the dossier, e.g. o17"},
                        "units": {"type": "number", "description": "0.25 to 1.0"},
                        "confidence": {"type": "integer", "description": "1 (lean) to 5 (strongest)"},
                        "thesis": {"type": "string", "description": "Why, in 2-4 specific sentences."},
                        "factors": {"type": "array", "items": {"type": "string"}, "description": "Key facts it rests on."},
                        "risk": {"type": "string", "description": "What would make this wrong."},
                        "vs_model": {"type": "string", "enum": ["agrees", "disagrees", "independent"]},
                        "edge_type": {"type": "string", "enum": ["role_change", "matchup", "game_script", "weather",
                                                                 "market", "spot", "price", "other"]},
                    },
                },
            },
        },
    },
}


class CapperPick(BaseModel):
    offer_id: str
    units: float
    confidence: int = Field(ge=1, le=5)
    thesis: str
    factors: list[str]
    risk: str
    vs_model: str
    edge_type: str


class CapperCard(BaseModel):
    game_read: str
    passes: str
    picks: list[CapperPick]


# ---------------------------------------------------------------- dossier

def _dk_latest(event_id: str, markets_like: str) -> pd.DataFrame:
    s = db.read(db.odds_snapshots, "WHERE event_id = :e AND book = 'draftkings' AND market LIKE :m",
                {"e": event_id, "m": markets_like})
    if s.empty:
        return s
    return s.sort_values("fetched_at").groupby(["market", "side", "player"], dropna=False, as_index=False).last()


def _line_history(event_id: str) -> list[str]:
    s = db.read(db.odds_snapshots, "WHERE event_id = :e AND book = 'draftkings' AND market IN ('spread', 'total')",
                {"e": event_id})
    out = []
    for (market, side), g in s.groupby(["market", "side"]):
        if side not in ("home", "over"):
            continue
        g = g.sort_values("fetched_at")
        o, n = g.iloc[0], g.iloc[-1]
        out.append(f"DK {market} ({side}): opened {o['point']:+g} {fmt_american(o['price'])} -> now {n['point']:+g} "
                   f"{fmt_american(n['price'])} ({len(g)} snapshots)")
    return out


def _usage_lines(ps: pd.DataFrame, snaps: pd.DataFrame, team: str, season: int, week: int, top: int = 9) -> list[str]:
    cur = ps[(ps["season"] == season) & (ps["week"] < week) & (ps["team"] == team) & ps["position"].isin(["QB", "RB", "WR", "TE"])]
    if cur.empty:
        return []
    recent_weeks = sorted(cur["week"].unique())[-3:]
    r = cur[cur["week"].isin(recent_weeks)]
    vol = r.groupby("player_display_name")[["targets", "carries", "attempts"]].sum().fillna(0)
    names = (vol["targets"] + vol["carries"] + vol["attempts"] * 0.3).sort_values(ascending=False).head(top).index
    sn = snaps[(snaps["team"] == team) & snaps["week"].isin(recent_weeks)] if not snaps.empty else pd.DataFrame()
    lines = []
    for name in names:
        g = r[r["player_display_name"] == name].sort_values("week")
        pos = g["position"].iloc[-1]
        games = []
        for x in g.itertuples():
            if pos == "QB":
                games.append(f"wk{x.week}: {x.completions:.0f}/{x.attempts:.0f} {x.passing_yards:.0f}y, {x.carries:.0f}-{x.rushing_yards:.0f} rush")
            else:
                games.append(f"wk{x.week}: {x.targets:.0f} tgt {x.receptions:.0f}-{x.receiving_yards:.0f}, {x.carries:.0f}-{x.rushing_yards:.0f} rush")
        snap = ""
        if not sn.empty:
            m = sn[sn["player"] == name]
            if not m.empty:
                snap = f" | snaps {', '.join(f'{v:.0%}' for v in m.sort_values('week')['offense_pct'])}"
        missed = [w for w in recent_weeks if w not in set(g["week"])]
        lines.append(f"{name} ({pos}): " + "; ".join(games) + snap + (f" | DID NOT PLAY wk{','.join(map(str, missed))}" if missed else ""))
    return lines


def _injury_lines(inj: pd.DataFrame, snaps: pd.DataFrame, team: str, week: int) -> list[str]:
    if inj.empty:
        return []
    w = inj[(inj["week"] == week) & (inj["team"] == team) & inj["report_status"].notna()]
    sn = snaps[(snaps["team"] == team) & (snaps["week"] < week)] if not snaps.empty else pd.DataFrame()
    out = []
    for r in w.itertuples():
        share = ""
        if not sn.empty:
            m = sn[sn["player"] == r.full_name].sort_values("week").tail(4)
            if not m.empty:
                pct = m["defense_pct"].mean() if m["defense_pct"].mean() > m["offense_pct"].mean() else m["offense_pct"].mean()
                side = "def" if m["defense_pct"].mean() > m["offense_pct"].mean() else "off"
                share = f", {pct:.0%} {side} snaps recently"
        out.append(f"{r.position} {r.full_name}: {r.report_status} ({r.practice_status or 'no practice note'}){share}")
    return out


def build_offers(game: dict, props: pd.DataFrame, ladder: pd.DataFrame, cfg: dict) -> list[dict]:
    """Every DK price the capper may pick from, each with an id."""
    offers = []

    def add(**o):
        o["id"] = f"o{len(offers) + 1}"
        offers.append(o)

    g = game
    if pd.notna(g.get("dk_spread_home")):
        add(kind="game", market="spread", side="home", line=float(g["dk_spread_home"]), price=g.get("dk_spread_home_price"),
            label=f"{g['home']} {g['dk_spread_home']:+g}")
        add(kind="game", market="spread", side="away", line=-float(g["dk_spread_home"]), price=g.get("dk_spread_away_price"),
            label=f"{g['away']} {-g['dk_spread_home']:+g}")
    if pd.notna(g.get("dk_total")):
        add(kind="game", market="total", side="over", line=float(g["dk_total"]), price=g.get("dk_over_price"), label=f"Over {g['dk_total']:g}")
        add(kind="game", market="total", side="under", line=float(g["dk_total"]), price=g.get("dk_under_price"), label=f"Under {g['dk_total']:g}")
    if pd.notna(g.get("dk_ml_home")):
        add(kind="game", market="ml", side="home", line=None, price=g.get("dk_ml_home"), label=f"{g['home']} ML")
        add(kind="game", market="ml", side="away", line=None, price=g.get("dk_ml_away"), label=f"{g['away']} ML")
    # main props, both sides, at DK's current prices
    if g.get("event_id") and not props.empty:
        dk = _dk_latest(g["event_id"], "player_%")
        dk = dk[dk["market"].isin(MARKET_STAT)] if not dk.empty else dk
        ids = props.drop_duplicates("player")[["player", "player_id", "team", "position"]].set_index("player")
        for r in dk.itertuples():
            if r.player not in ids.index or pd.isna(r.price):
                continue
            pi = ids.loc[r.player]
            stat = MARKET_STAT[r.market]
            if stat == "anytime_td":
                if r.side != "yes":
                    continue
                label = f"{r.player} anytime TD"
            else:
                label = f"{r.player} {r.side.title()} {r.point:g} {STAT_LABEL[stat]}"
            add(kind="prop", market=r.market, side="over" if r.side == "yes" else r.side, line=None if stat == "anytime_td" else float(r.point),
                price=float(r.price), player=r.player, player_id=pi["player_id"], team=pi["team"], position=pi["position"], label=label)
    # ladder rungs with a DK alt price
    if not ladder.empty:
        lad = ladder[(ladder["event_id"] == g.get("event_id")) & ladder["dk_price"].notna()]
        for r in lad.itertuples():
            main = {v: k for k, v in MARKET_STAT.items()}[r.stat]
            add(kind="ladder", market=main, side="over", line=float(r.point), price=float(r.dk_price), player=r.player,
                player_id=r.player_id, team=r.team, position=r.position,
                label=f"{r.player} {r.rung}+ {STAT_LABEL[r.stat]} (alt)", fair=r.fair_odds)
    return offers


def build_dossier(game: dict, offers: list[dict], props: pd.DataFrame, ladder: pd.DataFrame, pproj: pd.DataFrame,
                  ps: pd.DataFrame, inj: pd.DataFrame, snaps: pd.DataFrame, def_out: dict, model_picks: pd.DataFrame,
                  season: int, week: int) -> str:
    g = game
    home, away = g["home"], g["away"]
    L = [f"# {away} @ {home} - kickoff {g.get('kickoff')} (NFL {season} week {week})", ""]
    L += ["## EV model (blind numbers, formed before seeing any line)",
          f"Projection: {away} {g.get('proj_away_pts', float('nan')):.1f} - {home} {g.get('proj_home_pts', float('nan')):.1f} | "
          f"fair {home} {g.get('fair_spread_home', float('nan')):+g}, total {g.get('fair_total', float('nan')):g}, "
          f"{home} win {g.get('home_win_prob', float('nan')):.0%}",
          f"Adjustments: {g.get('notes') or 'none'}",
          f"Sharp-book consensus: {home} {g.get('sharp_spread_home')}, total {g.get('sharp_total')}",
          f"Model's lean: {g.get('lean')} | best play: {g.get('best_play')} | flags: {g.get('flag') or 'none'}"]
    if g.get("signals"):
        try:
            sig = json.loads(g["signals"]) if isinstance(g["signals"], str) else g["signals"]
            if sig:
                L.append("Market signals: " + "; ".join(f"{s['name']}: {s.get('detail', '')}" for s in sig))
        except (ValueError, TypeError, KeyError):
            pass
    if not model_picks.empty:
        L.append("Model's open picks on this game: " + "; ".join(f"{r.selection} {fmt_american(r.price)} {r.units:g}U"
                                                                for r in model_picks.itertuples()))
    if g.get("event_id"):
        L += _line_history(g["event_id"])
    for team, opp in ((away, home), (home, away)):
        L += ["", f"## {team}"]
        il = _injury_lines(inj, snaps, team, week)
        L += ["Injury report:"] + [f"- {x}" for x in il] if il else ["Injury report: nothing listed"]
        if def_out.get(team):
            L.append("Defensive starters out: " + ", ".join(f"{a['pos']} {a['name']} ({a['snap_pct']:.0%} snaps)" for a in def_out[team]))
        ul = _usage_lines(ps, snaps, team, season, week)
        if ul:
            L += ["Recent usage (last 3 games):"] + [f"- {x}" for x in ul]
        if not pproj.empty and "situation" in pproj:
            sit = pproj[(pproj["team"] == team) & (pproj["situation"].fillna("") != "")]
            if not sit.empty:
                L += ["Situations the model flagged:"] + [f"- {r.player}: {r.situation}" for r in sit.itertuples()]
        if not props.empty:
            pt = props[props["team"] == team].sort_values("ev", ascending=False)
            if not pt.empty:
                L.append("Model prop view (projection vs DK main line):")
                for r in pt.head(18).itertuples():
                    L.append(f"- {r.player} {STAT_LABEL.get(r.stat, r.stat)}: projects {r.projection:.1f} vs line {r.line:g}; "
                             f"model leans {r.side} ({r.ev:+.1%} EV)")
    L += ["", "## Offers you may pick from (DK prices; fair = the model's fair price for ladder rungs)"]
    for o in offers:
        fair = f" | model fair {fmt_american(o['fair'])}" if o.get("fair") is not None else ""
        L.append(f"{o['id']}: {o['label']} @ {fmt_american(o['price']) if o.get('price') is not None else 'n/a'}{fair}")
    return "\n".join(L)


# ---------------------------------------------------------------- Claude

def _client():
    import anthropic

    return anthropic.Anthropic(api_key=secret("ANTHROPIC_API_KEY"))


def notebook() -> str:
    n = db.read(db.capper_notes)
    if n.empty:
        return "(empty - no graded picks yet)"
    return n.sort_values("id").iloc[-1]["notes"]


def ask_capper(client, dossier: str, cfg: dict) -> tuple[CapperCard | None, str]:
    """One game -> a validated card. Returns (card, model_that_served) or (None, reason)."""
    import anthropic

    ccfg = cfg.get("capper", {})
    system = [
        {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "## Your notebook (lessons from your own graded picks)\n" + notebook()},
    ]
    tools = [
        {"type": "web_search_20260209", "name": "web_search", "max_uses": int(ccfg.get("web_searches", 5))},
        SUBMIT_TOOL,
    ]
    messages = [{"role": "user", "content": dossier}]
    for _ in range(int(ccfg.get("max_continuations", 4)) + 1):
        try:
            with client.beta.messages.stream(
                model=cfg["run"]["claude_model"], max_tokens=32000, system=system, tools=tools, messages=messages,
                output_config={"effort": ccfg.get("effort", "high")},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            ) as stream:
                msg = stream.get_final_message()
        except ValueError as e:  # tool input JSON the SDK could not parse at all
            return None, f"unparseable tool input: {e}"
        except anthropic.APIError as e:
            return None, f"API error: {e}"
        if msg.stop_reason == "refusal":
            return None, "refused"
        if msg.stop_reason == "max_tokens":
            return None, "hit max_tokens"
        if msg.stop_reason == "pause_turn":  # server-side search loop paused: resend, it resumes
            messages = [messages[0], {"role": "assistant", "content": msg.content}]
            continue
        call = next((b for b in msg.content if b.type == "tool_use" and b.name == "submit_card"), None)
        if call is None:
            messages = [messages[0], {"role": "assistant", "content": msg.content},
                        {"role": "user", "content": "Submit your card now with submit_card (picks may be empty)."}]
            continue
        try:
            return CapperCard.model_validate(call.input), msg.model
        except ValidationError as e:
            return None, f"invalid card: {e}"
    return None, "no card after continuations"


NOTEBOOK_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["notes"],
    "properties": {"notes": {"type": "string", "description": "The rewritten notebook: at most 15 short lessons."}},
}


def update_notebook(client, cfg: dict, log=print) -> bool:
    """Rewrite the notebook once enough new capper picks have been graded since the last version."""
    p = db.read(db.picks, "WHERE source = 'capper' AND status IN ('win', 'loss', 'push')")
    n_prev = db.read(db.capper_notes)
    last = 0 if n_prev.empty else int(n_prev.sort_values("id").iloc[-1]["graded_count"] or 0)
    if len(p) - last < int(cfg.get("capper", {}).get("notebook_every", 6)):
        return False
    p = p.sort_values("kickoff")
    lines = []
    for r in p.itertuples():
        clv = "" if pd.isna(r.clv_points) else f", CLV {r.clv_points:+g} pts"
        lines.append(f"- {r.kickoff[:10]} {r.matchup}: {r.selection} {fmt_american(r.price)} {r.units:g}U -> {r.status} "
                     f"({r.profit_units:+.2f}U, result {r.result_value}{clv})\n  thesis: {(r.rationale or '')[:600]}")
    w = (p["status"] == "win").sum()
    prompt = (f"Your graded picks so far ({w}-{(p['status'] == 'loss').sum()}, {p['profit_units'].sum():+.2f}U):\n"
              + "\n".join(lines) + "\n\nYour current notebook:\n" + notebook() +
              "\n\nRewrite the notebook. Keep lessons that the results support, drop ones they contradict, add new ones. "
              "Each lesson: one or two sentences, specific (situation -> what happened -> what to do), with the evidence "
              "count, e.g. '(3-1)'. Separate real signal from small-sample noise and from bad luck on good reads "
              "(beat the close but lost). At most 15 lessons.")
    try:
        resp = client.beta.messages.create(
            model=cfg["run"]["claude_model"], max_tokens=16000,
            system="You are the reasoning capper reviewing your own track record honestly.",
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": "high", "format": {"type": "json_schema", "schema": NOTEBOOK_SCHEMA}},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        if resp.stop_reason == "refusal":
            return False
        notes = json.loads(next(b.text for b in resp.content if b.type == "text"))["notes"]
    except Exception as e:
        log(f"Capper notebook update failed: {e}")
        return False
    db.insert_df(db.capper_notes, pd.DataFrame([{"created_at": db.now(), "graded_count": len(p), "notes": notes}]))
    log(f"Capper notebook rewritten from {len(p)} graded picks")
    return True


# ---------------------------------------------------------------- run

def _pick_record(card_pick: CapperPick, offer: dict, game: dict, run_id: str, served_by: str, read: str) -> dict:
    gid = str(game["game_id"])
    alt = "-alt" if offer["kind"] == "ladder" else ""
    line_tag = "" if offer["line"] is None else f"{offer['line']:g}"
    pid = f"nfl|{gid}|{offer['market']}|{offer['side']}|{offer.get('player_id') or ''}|capper{alt}@{line_tag}"
    if offer["kind"] == "game":
        category = {"spread": "side", "total": "total", "ml": "ml"}[offer["market"]]
    else:
        category = "attd" if MARKET_STAT.get(offer["market"]) == "anytime_td" else "prop"
    rationale = (f"**Capper ({card_pick.edge_type.replace('_', ' ')}, confidence {card_pick.confidence}/5, "
                 f"{card_pick.vs_model} with the model):** {card_pick.thesis}\n\n"
                 + "".join(f"- {f}\n" for f in card_pick.factors)
                 + f"- **Risk:** {card_pick.risk}\n\n_Game read: {read}_")
    return {
        "pick_id": pid, "run_id": run_id, "sport": "nfl", "category": category, "event_id": game.get("event_id"),
        "game_id": gid, "kickoff": str(game.get("kickoff")), "matchup": f"{game['away']} @ {game['home']}",
        "market": offer["market"], "selection": offer["label"], "player": offer.get("player"),
        "player_id": offer.get("player_id"), "line": offer["line"], "price": float(offer["price"]),
        "book": "draftkings", "units": card_pick.units, "signals": [], "rationale": rationale, "source": "capper",
    }


def run(run_id: str, cfg: dict, log=print, now=None) -> int:
    """Analyse each NFL game kicking off within the window that the capper hasn't read recently."""
    ccfg = cfg.get("capper", {})
    if not ccfg.get("enabled", False):
        return 0
    if not secret("ANTHROPIC_API_KEY"):
        log("Capper skipped: ANTHROPIC_API_KEY not set")
        return 0
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC"))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    client = _client()
    update_notebook(client, cfg, log)
    games, _ = db.latest_board("nfl", "games")
    if games.empty:
        return 0
    games["kick"] = pd.to_datetime(games["kickoff"], utc=True, errors="coerce", format="ISO8601")
    window = games[(games["kick"] > now) & (games["kick"] <= now + timedelta(hours=float(ccfg.get("window_hours", 36))))]
    if window.empty:
        return 0
    reviews = db.read(db.capper_reviews)
    recent = set()
    if not reviews.empty:
        made = pd.to_datetime(reviews["created_at"], utc=True, format="ISO8601")
        recent = set(reviews.loc[made > now - timedelta(hours=float(ccfg.get("reread_hours", 20))), "game_id"].astype(str))
    props, _ = db.latest_board("nfl", "props")
    ladder, _ = db.latest_board("nfl", "ladder")
    pproj, _ = db.latest_board("nfl", "projections_players")
    gid0 = str(window["game_id"].iloc[0])
    season, week = int(gid0[:4]), int(gid0[5:7])
    ps = nflverse.player_stats([season])
    inj, snaps = nflverse.injuries(season), nflverse.snap_counts(season)
    try:
        def_out = nfl_model.defensive_absences(season, week)
    except Exception:
        def_out = {}
    open_picks = db.read(db.picks, "WHERE sport = 'nfl' AND status = 'pending'")
    n = 0
    day_used: dict[str, float] = {}
    if not open_picks.empty:
        cap_open = open_picks[db.is_capper(open_picks)]
        for r in cap_open.itertuples():
            d = pd.to_datetime(r.kickoff, utc=True, format="ISO8601").tz_convert("America/New_York").strftime("%Y-%m-%d")
            day_used[d] = day_used.get(d, 0.0) + float(r.units)
    for g in window.sort_values("kick").to_dict("records"):
        if str(g["game_id"]) in recent or not g.get("event_id"):
            continue
        offers = build_offers(g, props, ladder, cfg)
        if not offers:
            continue
        model_picks = open_picks[(open_picks["game_id"].astype(str) == str(g["game_id"])) & ~db.off_book(open_picks)] \
            if not open_picks.empty else pd.DataFrame()
        dossier = build_dossier(g, offers, props[props["team"].isin([g["home"], g["away"]])] if not props.empty else props,
                                ladder, pproj, ps, inj, snaps, def_out, model_picks, season, week)
        card, served = ask_capper(client, dossier, cfg)
        if card is None:
            log(f"Capper {g['away']} @ {g['home']}: no card ({served})")
            continue
        by_id = {o["id"]: o for o in offers}
        kept = []
        day = g["kick"].tz_convert("America/New_York").strftime("%Y-%m-%d")
        for cp in card.picks[: int(ccfg.get("max_picks_per_game", 3))]:
            o = by_id.get(cp.offer_id)
            if o is None or o.get("price") is None or pd.isna(o["price"]):
                log(f"Capper: ignored unknown offer {cp.offer_id}")
                continue
            cp.units = round(min(max(cp.units, 0.25), float(ccfg.get("max_units", 1.0))) * 4) / 4
            if day_used.get(day, 0.0) + cp.units > float(ccfg.get("units_per_day", 4.0)) + 1e-9:
                log(f"Capper: daily cap reached, skipped {o['label']}")
                continue
            rec = _pick_record(cp, o, g, run_id, served, card.game_read)
            if db.upsert_pick(rec) == "inserted":
                day_used[day] = day_used.get(day, 0.0) + cp.units
                kept.append(f"{o['label']} {fmt_american(o['price'])} {cp.units:g}U")
                n += 1
        db.insert_df(db.capper_reviews, pd.DataFrame([{
            "created_at": now.isoformat(), "sport": "nfl", "game_id": str(g["game_id"]), "matchup": f"{g['away']} @ {g['home']}",
            "kickoff": str(g.get("kickoff")), "model": served, "read": card.game_read, "passes": card.passes,
            "payload": card.model_dump_json(),
        }]))
        log(f"Capper {g['away']} @ {g['home']}: " + ("; ".join(kept) if kept else "pass"))
    return n
