"""Live integration probe: hits every external source and prints what came back.

Run from the Actions tab (task: diagnose). Uses a throwaway SQLite DB; never touches the real history.
"""
import json
import re
import traceback

import _path  # noqa: F401
import pandas as pd

from pocket_capper.data import cfbd, nflverse, odds_api, splits, teams, weather
from pocket_capper.settings import config

cfg = config()
pd.set_option("display.width", 200, "display.max_columns", 30)


def section(name):
    print(f"\n{'=' * 20} {name} {'=' * 20}", flush=True)


def probe(fn):
    try:
        fn()
    except Exception:
        traceback.print_exc()


def odds():
    section("ODDS API: sports")
    sports = odds_api._get("/sports", {})
    for s in sports:
        if s["key"] in ("americanfootball_nfl", "americanfootball_ncaaf"):
            print(s)
    print("usage:", odds_api.LAST_USAGE)
    for sport in ("nfl", "cfb"):
        section(f"ODDS API: {sport} game odds")
        df = odds_api.game_odds(cfg["sports"][sport]["odds_key"], cfg["books"]["compare"])
        print("rows", len(df), "events", df["event_id"].nunique() if len(df) else 0)
        if len(df):
            print("books:", df.groupby("book")["event_id"].nunique().to_dict())
            print("markets/sides:", df.groupby(["market", "side"]).size().to_dict())
            print(df.head(8).to_string())
            names = sorted(set(df["home_name"]) | set(df["away_name"]))
            if sport == "nfl":
                print("unmapped NFL names:", [n for n in names if not teams.nfl_abbr(n)])
            else:
                globals()["CFB_ODDS_NAMES"] = names
        print("usage:", odds_api.LAST_USAGE)
        if sport == "nfl" and len(df):
            probe(lambda: nfl_props_probe(df))


def nfl_props_probe(df):
    if True:
        if True:
            section("ODDS API: NFL props (first event)")
            ev = df["event_id"].iloc[0]
            p = odds_api.event_props(cfg["sports"]["nfl"]["odds_key"], ev, cfg["props"]["nfl_markets"], cfg["books"]["compare"])
            print("rows", len(p))
            if len(p):
                print("books:", p.groupby("book").size().to_dict())
                print("markets x sides:", p.groupby(["market", "side"]).size().to_dict())
                print(p.head(12).to_string())
                ps = nflverse.player_stats([2026])
                known = set(ps["player_display_name"].map(teams.norm_player))
                players = sorted(set(p["player"]))
                miss = [x for x in players if teams.norm_player(x) not in known]
                print(f"players {len(players)}, unmatched to nflverse: {len(miss)} -> {miss[:25]}")
            print("usage:", odds_api.LAST_USAGE)


def cfb():
    section("CFBD")
    g = cfbd.games(2026)
    print("games", len(g), "completed", int(g["completed"].sum()) if len(g) else 0)
    print(g.head(3).to_string())
    raw = cfbd._get("/games", {"year": 2026, "seasonType": "regular", "week": 6})
    print("raw game keys:", sorted(raw[0].keys()) if raw else None)
    ppa = cfbd.ppa_games(2026)
    print("ppa rows", len(ppa), ppa.head(3).to_dict("records"))
    rawp = cfbd._get("/ppa/games", {"year": 2026, "excludeGarbageTime": "true"})
    print("raw ppa sample:", json.dumps(rawp[0])[:600] if rawp else None)
    ln = cfbd.lines(2026)
    print("lines rows", len(ln), "providers", ln["provider"].value_counts().to_dict() if len(ln) else {})
    print(ln.head(5).to_string())
    fbs = cfbd.fbs_teams(2026)
    print("fbs teams", len(fbs))
    v = cfbd.venues()
    print("venues", len(v), v.head(3).to_dict("records"))
    names = globals().get("CFB_ODDS_NAMES", [])
    if names:
        un = [n for n in names if not teams.match_cfb(n, fbs)]
        print(f"CFB odds names {len(names)}, unmatched to CFBD FBS schools: {un}")


def wx():
    section("OPEN-METEO")
    lat, lon, _ = teams.NFL_STADIUMS["GB"]
    print(weather.kickoff_weather(lat, lon, pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2)))


def dk_splits():
    section("DK NETWORK SPLITS")
    r = splits.SESSION.get(splits.DK_SPLITS_URL, params={"tb_eg": splits.DK_EVENT_GROUPS["nfl"], "tb_edate": "n7days", "tb_emt": "0"}, timeout=30)
    print("status", r.status_code, "bytes", len(r.text))
    html = r.text
    classes = re.findall(r'class="([^"]*tb[^"]*)"', html)
    print("tb* classes:", pd.Series(classes).value_counts().head(30).to_dict())
    i = html.find("%")
    j = html.find("tb-sodd")
    k = html.rfind("<div", 0, max(0, j - 3000))
    print("HTML around first tb-sodd:\n", html[k: j + 6000])
    print("parsed rows:", len(splits.parse_dk_splits_html(html, "nfl")))


def cfb_backtest():
    section("CFB BACKTEST 2025 (bias check)")
    from pocket_capper.models import backtest

    bt = backtest.run_cfb([2025], cfg["sports"]["cfb"])
    print(backtest.summarize(bt))
    print("margin bias (actual - model):", round((bt["result"] - bt["fair_margin"]).mean(), 2),
          "total bias (actual - model):", round((bt["total"] - bt["fair_total"]).mean(), 2),
          "close total bias:", round((bt["total"] - bt["total_line"]).mean(), 2))
    print("model total - close total:", round((bt["fair_total"] - bt["total_line"]).mean(), 2),
          "| sd margin resid", round((bt["result"] - bt["fair_margin"]).std(), 2),
          "| sd total resid", round((bt["total"] - bt["fair_total"]).std(), 2))
    for w in (0.0, 0.1, 0.2, 0.3, 0.5):
        m = w * bt["fair_margin"] + (1 - w) * bt["spread_line"]
        t = w * bt["fair_total"] + (1 - w) * bt["total_line"]
        print(f"blend w={w}: margin MAE {(m - bt['result']).abs().mean():.3f}  total MAE {(t - bt['total']).abs().mean():.3f}")


for f in (odds, cfb, wx, dk_splits, cfb_backtest):
    probe(f)
