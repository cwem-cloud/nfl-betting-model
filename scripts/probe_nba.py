"""Throwaway probe: can Actions reach the NBA data we need?"""
import json
import os

import requests

S = requests.Session()
S.headers["User-Agent"] = "Mozilla/5.0"
E = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
for d in ("20261020", "20261021", "20261022"):
    r = S.get(f"{E}/scoreboard", params={"dates": d}, timeout=20)
    print("scoreboard", d, r.status_code, len(r.text))
    if r.ok:
        ev = r.json().get("events", [])
        print("  events", len(ev), [(e["id"], e["shortName"], e["status"]["type"]["name"]) for e in ev[:6]])
        done = [e for e in ev if e["status"]["type"]["completed"]]
        if done:
            sm = S.get(f"{E}/summary", params={"event": done[0]["id"]}, timeout=20)
            print("  summary", sm.status_code)
            if sm.ok:
                bs = sm.json().get("boxscore", {}).get("players", [])
                if bs:
                    st = bs[0]["statistics"][0]
                    print("  box labels", st.get("labels") or st.get("keys"))
                    a = st["athletes"][0]
                    print("  athlete", a["athlete"]["displayName"], a["stats"])
r = S.get(f"{E}/injuries", timeout=20)
print("injuries", r.status_code)
if r.ok:
    j = r.json()
    teams = j.get("injuries", [])
    print("  teams", len(teams))
    if teams:
        print("  sample", json.dumps(teams[0])[:800])
k = os.environ.get("ODDS_API_KEY")
if k:
    r = S.get("https://api.the-odds-api.com/v4/sports/basketball_nba/events", params={"apiKey": k}, timeout=20)
    print("odds events", r.status_code, len(r.json()) if r.ok else r.text[:200])
    if r.ok and r.json():
        ev = r.json()[0]
        rr = S.get(f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{ev['id']}/odds",
                   params={"apiKey": k, "bookmakers": "draftkings", "oddsFormat": "american",
                           "markets": "player_points,player_rebounds,player_assists,player_threes,player_points_rebounds_assists,player_points_alternate"},
                   timeout=20)
        print("props", rr.status_code, ev["home_team"], ev["away_team"], ev["commence_time"])
        if rr.ok:
            for bk in rr.json().get("bookmakers", []):
                for m in bk["markets"]:
                    print("  ", m["key"], len(m["outcomes"]), m["outcomes"][:2])
