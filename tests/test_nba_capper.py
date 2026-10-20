import pandas as pd

from pocket_capper.data import nba
from pocket_capper.engine import capper, grading
from pocket_capper.settings import config
from tests.test_capper import FakeClient

KICK = "2030-01-06T00:00:00Z"  # 7pm ET on Jan 5
NOW = pd.Timestamp("2030-01-05T20:00:00Z")
GID = nba.game_id(KICK, "BOS", "DET")


def _summary():
    labels = ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO", "STL", "BLK", "OREB", "DREB", "PF", "+/-"]
    return {"boxscore": {"players": [{"team": {"abbreviation": "BOS"}, "statistics": [{"labels": labels, "athletes": [
        {"athlete": {"displayName": "Jayson Tatum"}, "stats": ["36", "31", "11-22", "4-9", "5-6", "9", "6", "3", "1", "0", "1", "8", "2", "+7"]},
        {"athlete": {"displayName": "Bench Guy"}, "stats": [], "didNotPlay": True},
    ]}]}]}}


def test_box_score_parsing_and_stats():
    box = nba.parse_box(_summary())
    assert list(box["player"]) == ["Jayson Tatum"]  # DNP dropped
    t = box.iloc[0]
    assert (t["points"], t["rebounds"], t["assists"], t["threes"]) == (31, 9, 6, 4)
    assert nba.stat_value("player_points_rebounds_assists", t) == 46
    assert nba.stat_value("player_points_alternate", t) == 31
    assert GID == "20300105_BOS_DET"


def _seed(db):
    db.save_board("r0", "nba", "games", pd.DataFrame([{
        "game_id": GID, "event_id": "nev1", "kickoff": KICK, "away": "BOS", "home": "DET",
        "away_name": "Boston Celtics", "home_name": "Detroit Pistons", "dk_spread_home": 2.5,
        "dk_spread_home_price": -110, "dk_spread_away_price": -110, "dk_total": 221.5, "dk_over_price": -110,
        "dk_under_price": -110, "dk_ml_home": 120, "dk_ml_away": -140, "consensus_spread_home": 2.5,
        "consensus_total": 222.0, "books": 9}]))
    rows = [("player_points", "over", 26.5, -113), ("player_points", "under", 26.5, -113),
            ("player_points_alternate", "over", 29.5, 165), ("player_points_alternate", "under", 29.5, -300)]
    db.save_board("r0", "nba", "props", pd.DataFrame([{
        "event_id": "nev1", "book": "draftkings", "market": m, "player": "Jayson Tatum", "side": s, "point": pt, "price": pr}
        for m, s, pt, pr in rows]))


def test_nba_capper_flow(tmp_db, monkeypatch):
    _seed(tmp_db)
    games, _ = tmp_db.latest_board("nba", "games")
    props, _ = tmp_db.latest_board("nba", "props")
    offers = capper.nba_offers(games.iloc[0].to_dict(), props)
    labels = {o["id"]: o["label"] for o in offers}
    assert "Jayson Tatum 30+ Points (alt)" in labels.values()
    assert not any("29.5" in v and "Under" in v for v in labels.values() if "alt" in v)  # alt unders skipped
    main = next(k for k, v in labels.items() if v == "Jayson Tatum Over 26.5 Points")
    alt = next(k for k, v in labels.items() if v.endswith("30+ Points (alt)"))
    card = {"game_read": "DET without its starting C.", "passes": "total", "picks": [
        {"offer_id": main, "units": 0.5, "confidence": 3, "thesis": "Usage up.", "factors": [], "risk": "Blowout.",
         "vs_model": "independent", "edge_type": "role_change"},
        {"offer_id": alt, "units": 0.25, "confidence": 2, "thesis": "Ladder.", "factors": [], "risk": "x",
         "vs_model": "independent", "edge_type": "role_change"}]}
    client = FakeClient(card)
    monkeypatch.setattr(capper, "secret", lambda n, d=None: "k" if n == "ANTHROPIC_API_KEY" else None)
    monkeypatch.setattr(capper, "_client", lambda: client)
    monkeypatch.setattr(nba, "injuries", lambda: {"DET": ["Jalen Duren (C): Out - ankle"]})
    assert capper.run("r1", config(), print, NOW, sport="nba") == 2
    sent = client.kwargs[0]
    assert "Jalen Duren (C): Out" in sent["messages"][0]["content"]
    assert "NBA betting model" in sent["system"][0]["text"] and "back-to-back" in sent["system"][0]["text"]
    p = tmp_db.read(tmp_db.picks)
    assert set(p["sport"]) == {"nba"} and set(p["market"]) == {"player_points"}
    assert p["pick_id"].str.startswith(f"nba|{GID}|player_points|over|nba:jaysontatum|").all()
    rv = tmp_db.read(tmp_db.capper_reviews)
    assert list(rv["sport"]) == ["nba"] and "via api" in rv["model"].iloc[0]


def test_nba_grading(tmp_db, monkeypatch):
    _seed(tmp_db)
    base = dict(sport="nba", event_id="nev1", game_id=GID, matchup="BOS @ DET", kickoff=KICK, units=0.5,
                signals=[], source="capper", book="draftkings")
    tmp_db.upsert_pick(dict(base, pick_id=f"nba|{GID}|player_points|over|nba:jayson-tatum|capper@26.5",
                            category="prop", market="player_points", player="Jayson Tatum", line=26.5, price=-113,
                            selection="Tatum Over 26.5"))
    tmp_db.upsert_pick(dict(base, pick_id=f"nba|{GID}|player_points|over|nba:bench-guy|capper@8.5",
                            category="prop", market="player_points", player="Bench Guy", line=8.5, price=-110,
                            selection="Bench Over 8.5"))
    tmp_db.upsert_pick(dict(base, pick_id=f"nba|{GID}|spread|home||capper@2.5", category="side", market="spread",
                            line=2.5, price=-110, selection="DET +2.5"))
    monkeypatch.setattr(nba, "scoreboard", lambda d: [{"espn_id": "1", "away": "BOS", "home": "DET",
                                                       "away_pts": 110.0, "home_pts": 104.0, "completed": True}])
    monkeypatch.setattr(nba, "box_score", lambda e: nba.parse_box(_summary()))
    monkeypatch.setattr(pd.Timestamp, "now", classmethod(lambda cls, tz=None: pd.Timestamp("2030-01-07T12:00:00Z")))
    grading.grade_all()
    p = tmp_db.read(tmp_db.picks).set_index("selection")
    assert p.loc["Tatum Over 26.5", "status"] == "win" and p.loc["Tatum Over 26.5", "result_value"] == 31
    assert p.loc["Bench Over 8.5", "status"] == "void"
    assert p.loc["DET +2.5", "status"] == "loss"  # lost by 6


def test_backend_selection(monkeypatch):
    cfg = config()
    env = {}
    monkeypatch.setattr(capper, "secret", lambda n, d=None: env.get(n))
    assert capper.backend(cfg) is None
    env["CLAUDE_CODE_OAUTH_TOKEN"] = "t"
    assert capper.backend(cfg) == "agent_sdk"
    env["ANTHROPIC_API_KEY"] = "k"
    assert capper.backend(cfg) == "api"  # a key wins, like Claude Code itself


def test_sdk_brain_card(monkeypatch):
    out = {"game_read": "r", "passes": "p", "picks": [{"offer_id": "o1", "units": 0.5, "confidence": 3, "thesis": "t",
           "factors": [], "risk": "r", "vs_model": "agrees", "edge_type": "matchup"}]}
    seen = {}

    def fake_run(self, prompt, system, schema, cfg, tools, max_turns):
        seen.update(system=system, tools=tools, schema=schema)
        return out, "claude-opus-5-5"
    monkeypatch.setattr(capper.SdkBrain, "_run", fake_run)
    card, served = capper.SdkBrain().card(capper.system_prompt("nba"), "dossier", config())
    assert card.picks[0].edge_type == "matchup" and served == "claude-opus-5-5"
    assert seen["tools"] == ["WebSearch", "WebFetch"] and "submit_card" not in seen["system"]
    assert seen["schema"]["required"] == ["game_read", "picks", "passes"]
