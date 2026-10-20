"""NBA data: team names, ESPN scoreboard / box scores / injuries (public site API), Odds API keys."""
from __future__ import annotations

import pandas as pd

from pocket_capper.data.http import SESSION
from pocket_capper.data.teams import norm_player

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ODDS_KEY = "basketball_nba"
# Odds API team name -> ESPN abbreviation
TEAMS = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GS", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NO",
    "New York Knicks": "NY", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SA",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTAH", "Washington Wizards": "WSH",
}
PROP_MARKETS = ["player_points", "player_rebounds", "player_assists", "player_threes", "player_points_rebounds_assists"]
ALT_MARKETS = ["player_points_alternate", "player_rebounds_alternate", "player_assists_alternate",
               "player_threes_alternate"]
STAT_LABEL = {"player_points": "Points", "player_rebounds": "Rebounds", "player_assists": "Assists",
              "player_threes": "Threes", "player_points_rebounds_assists": "Pts+Reb+Ast"}


def abbr(name: str | None) -> str | None:
    return TEAMS.get(name or "")


def player_id(name: str) -> str:
    return "nba:" + norm_player(name).replace(" ", "-")


def main_market(market: str) -> str:
    return market.replace("_alternate", "")


def game_id(commence_iso: str, away: str, home: str) -> str:
    """ET calendar date + teams, e.g. 20261020_BOS_DET (how ESPN's scoreboard is keyed)."""
    d = pd.Timestamp(commence_iso).tz_convert("America/New_York").strftime("%Y%m%d")
    return f"{d}_{away}_{home}"


def _get(path: str, params: dict | None = None) -> dict:
    r = SESSION.get(f"{ESPN}/{path}", params=params or {}, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    return r.json()


def scoreboard(date_yyyymmdd: str) -> list[dict]:
    """[{espn_id, away, home, away_pts, home_pts, completed}] for an ET date."""
    out = []
    for e in _get("scoreboard", {"dates": date_yyyymmdd}).get("events", []):
        comp = e["competitions"][0]
        teams = {c["homeAway"]: c for c in comp["competitors"]}
        out.append({
            "espn_id": e["id"], "away": teams["away"]["team"]["abbreviation"], "home": teams["home"]["team"]["abbreviation"],
            "away_pts": float(teams["away"].get("score") or 0), "home_pts": float(teams["home"].get("score") or 0),
            "completed": bool(e["status"]["type"].get("completed")),
        })
    return out


def parse_box(summary: dict) -> pd.DataFrame:
    """One row per player who played: points, rebounds, assists, threes made."""
    rows = []
    for team in summary.get("boxscore", {}).get("players", []):
        tab = team.get("team", {}).get("abbreviation")
        for st in team.get("statistics", []):
            labels = st.get("labels") or st.get("names") or []
            idx = {lab: i for i, lab in enumerate(labels)}
            for a in st.get("athletes", []):
                stats = a.get("stats") or []
                if not stats or a.get("didNotPlay"):
                    continue

                def num(lab, s=stats):
                    if lab not in idx or idx[lab] >= len(s):
                        return 0.0
                    v = str(s[idx[lab]]).split("-")[0]
                    try:
                        return float(v)
                    except ValueError:
                        return 0.0
                rows.append({"team": tab, "player": a["athlete"]["displayName"],
                             "key": norm_player(a["athlete"]["displayName"]),
                             "points": num("PTS"), "rebounds": num("REB"), "assists": num("AST"), "threes": num("3PT")})
    return pd.DataFrame(rows)


def box_score(espn_id: str) -> pd.DataFrame:
    return parse_box(_get("summary", {"event": espn_id}))


def stat_value(market: str, row: pd.Series) -> float:
    m = main_market(market)
    if m == "player_points_rebounds_assists":
        return float(row["points"] + row["rebounds"] + row["assists"])
    return float(row[{"player_points": "points", "player_rebounds": "rebounds", "player_assists": "assists",
                      "player_threes": "threes"}[m]])


def injuries() -> dict[str, list[str]]:
    """ESPN abbreviation -> ['Name (POS): Status - note', ...]."""
    by_name = {v: k for k, v in TEAMS.items()}
    out: dict[str, list[str]] = {}
    for t in _get("injuries").get("injuries", []):
        ab = abbr(t.get("displayName")) or by_name.get(t.get("displayName"))
        for i in t.get("injuries", []):
            ath = i.get("athlete") or {}
            name = ath.get("displayName") or "?"
            pos = (ath.get("position") or {}).get("abbreviation", "")
            note = (i.get("shortComment") or "")[:220]
            out.setdefault(ab or t.get("displayName"), []).append(f"{name} ({pos}): {i.get('status', '?')} - {note}")
    return out
