import os, requests, pandas as pd
from typing import Dict, Any
import numpy as np

def _get(path: str, params: Dict[str, Any]) -> Any:
    base = os.environ.get("ODDS_API_BASE", "https://api.the-odds-api.com/v4")
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError("Missing ODDS_API_KEY in Streamlit secrets.")
    url = f"{base.rstrip('/')}/{path.lstrip('/')}"
    params = {"apiKey": key, **params}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def fetch_upcoming_odds() -> pd.DataFrame:
    sport = os.environ.get("ODDS_SPORT", "americanfootball_nfl")
    markets = os.environ.get("ODDS_MARKETS", "spreads,totals,h2h")
    region = os.environ.get("ODDS_REGION", "us")
    odds_format = os.environ.get("ODDS_ODDS_FORMAT", "american")

    data = _get(f"sports/{sport}/odds", {
        "regions": region,
        "markets": markets,
        "oddsFormat": odds_format,
    })

    rows = []
    for game in data:
        gid = game.get("id")
        home = game.get("home_team")
        away = game.get("away_team")
        commence = game.get("commence_time")

        acc = {"spreads": [], "totals": [], "h2h_home": [], "h2h_away": []}
        for bk in game.get("bookmakers", []):
            for mk in bk.get("markets", []):
                key = mk.get("key")
                for out in mk.get("outcomes", []):
                    if key == "spreads" and out.get("name") == home:
                        acc["spreads"].append(out.get("point"))
                    elif key == "totals":
                        pt = out.get("point")
                        if pt is not None:
                            acc["totals"].append(pt)
                    elif key == "h2h":
                        if out.get("name") == home:
                            acc["h2h_home"].append(out.get("price"))
                        elif out.get("name") == away:
                            acc["h2h_away"].append(out.get("price"))

        def med(x):
            x = [v for v in x if v is not None]
            return float(np.median(x)) if x else None

        rows.append({
            "game_id": gid,
            "commence_time": commence,
            "home_team": home,
            "away_team": away,
            "spread_close_home_consensus": med(acc["spreads"]),
            "total_close_consensus": med(acc["totals"]),
            "ml_home_consensus": med(acc["h2h_home"]),
            "ml_away_consensus": med(acc["h2h_away"]),
        })
    return pd.DataFrame(rows)
