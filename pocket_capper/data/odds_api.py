"""The Odds API v4 client (the-odds-api.com). Multi-book game lines + player props.

Usage cost: game odds = markets x regions per call; event props = markets x regions
per event. Passing <=10 `bookmakers` counts as one region.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from pocket_capper.data.http import SESSION
from pocket_capper.settings import secret

BASE = "https://api.the-odds-api.com/v4"
LAST_USAGE: dict = {}


def _get(path: str, params: dict) -> object:
    key = secret("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY not set (the-odds-api.com)")
    r = SESSION.get(f"{BASE}{path}", params={"apiKey": key, **params}, timeout=30)
    LAST_USAGE.update(
        remaining=r.headers.get("x-requests-remaining"), used=r.headers.get("x-requests-used")
    )
    r.raise_for_status()
    return r.json()


def parse_game_odds(data: list, fetched_at: str | None = None) -> pd.DataFrame:
    """Flatten /odds payload into one row per (event, book, market, side)."""
    fetched_at = fetched_at or datetime.now(timezone.utc).isoformat()
    rows = []
    for ev in data:
        home, away = ev["home_team"], ev["away_team"]
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                market = {"h2h": "ml", "spreads": "spread", "totals": "total"}.get(mk["key"], mk["key"])
                for o in mk.get("outcomes", []):
                    name = o.get("name")
                    side = (
                        "home" if name == home else "away" if name == away else str(name).lower()
                    )
                    rows.append(
                        {
                            "event_id": ev["id"],
                            "commence_time": ev["commence_time"],
                            "home_name": home,
                            "away_name": away,
                            "book": bk["key"],
                            "market": market,
                            "side": side,
                            "point": o.get("point"),
                            "price": o.get("price"),
                            "book_update": mk.get("last_update") or bk.get("last_update"),
                            "fetched_at": fetched_at,
                        }
                    )
    return pd.DataFrame(rows)


def parse_props(ev: dict, fetched_at: str | None = None) -> pd.DataFrame:
    fetched_at = fetched_at or datetime.now(timezone.utc).isoformat()
    rows = []
    for bk in ev.get("bookmakers", []):
        for mk in bk.get("markets", []):
            for o in mk.get("outcomes", []):
                rows.append(
                    {
                        "event_id": ev["id"],
                        "commence_time": ev.get("commence_time"),
                        "home_name": ev.get("home_team"),
                        "away_name": ev.get("away_team"),
                        "book": bk["key"],
                        "market": mk["key"],
                        "player": o.get("description") or o.get("name"),
                        "side": str(o.get("name", "")).lower(),  # over / under / yes / no
                        "point": o.get("point"),
                        "price": o.get("price"),
                        "fetched_at": fetched_at,
                    }
                )
    return pd.DataFrame(rows)


def game_odds(sport_key: str, books: list[str]) -> pd.DataFrame:
    data = _get(
        f"/sports/{sport_key}/odds",
        {"bookmakers": ",".join(books[:10]), "markets": "h2h,spreads,totals", "oddsFormat": "american"},
    )
    return parse_game_odds(data)


def event_props(sport_key: str, event_id: str, markets: list[str], books: list[str]) -> pd.DataFrame:
    data = _get(
        f"/sports/{sport_key}/events/{event_id}/odds",
        {"bookmakers": ",".join(books[:10]), "markets": ",".join(markets), "oddsFormat": "american"},
    )
    return parse_props(data)
