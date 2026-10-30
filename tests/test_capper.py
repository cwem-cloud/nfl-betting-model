import json
from types import SimpleNamespace

import pandas as pd

from pocket_capper.engine import capper, slate
from pocket_capper.settings import config

KICK = "2030-01-06T18:00:00Z"
NOW = pd.Timestamp("2030-01-06T08:00:00Z")


class FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class FakeClient:
    """Stands in for anthropic.Anthropic: returns a scripted submit_card call."""

    def __init__(self, card, notes="1. Test lesson (1-0)"):
        self.card, self.notes, self.calls, self.kwargs = card, notes, 0, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream, create=self._create))

    def _stream(self, **kw):
        self.calls += 1
        self.kwargs.append(kw)
        block = SimpleNamespace(type="tool_use", name="submit_card", input=self.card, id="toolu_1")
        return FakeStream(SimpleNamespace(stop_reason="tool_use", content=[block], model="claude-opus-5-5"))

    def _create(self, **kw):
        return SimpleNamespace(stop_reason="end_turn",
                               content=[SimpleNamespace(type="text", text=json.dumps({"notes": self.notes}))])


def _seed(db):
    games = pd.DataFrame([{
        "game_id": "2030_18_WAS_SF", "kickoff": KICK, "away": "WAS", "home": "SF", "event_id": "ev1",
        "proj_away_pts": 20.1, "proj_home_pts": 25.3, "fair_spread_home": -5.0, "fair_total": 45.5,
        "home_win_prob": 0.66, "notes": "Rest edge home", "sharp_spread_home": -4.5, "sharp_total": 46.0,
        "dk_spread_home": -4.5, "dk_spread_home_price": -110, "dk_spread_away_price": -110,
        "dk_total": 46.5, "dk_over_price": -108, "dk_under_price": -112, "dk_ml_home": -210, "dk_ml_away": 175,
        "lean": "spread home (+1.2% EV)", "flag": None, "best_play": "No edge", "signals": "[]",
    }])
    db.save_board("r0", "nfl", "games", games)
    props = pd.DataFrame([{"event_id": "ev1", "player": "Rachaad White", "player_id": "00-1", "team": "WAS", "opp": "SF",
                           "position": "RB", "stat": "receptions", "side": "over", "line": 2.5, "price": 130,
                           "projection": 2.9, "model_prob": 0.55, "market_prob": 0.42, "final_prob": 0.45, "ev": 0.03}])
    db.save_board("r0", "nfl", "props", props)
    lad = pd.DataFrame([{"event_id": "ev1", "player": "Rachaad White", "player_id": "00-1", "team": "WAS", "opp": "SF",
                         "position": "RB", "stat": "receptions", "rung": 4, "point": 3.5, "fair_odds": 255,
                         "dk_price": 320.0, "ev": 0.18, "value": True, "context": ""}])
    db.save_board("r0", "nfl", "ladder", lad)
    db.save_board("r0", "nfl", "projections_players", pd.DataFrame([{
        "player_id": "00-1", "player": "Rachaad White", "team": "WAS", "situation": "+20% carries with RB X out"}]))
    snaps = []
    for side, pt, pr in (("over", 2.5, 130), ("under", 2.5, -160)):
        snaps.append({"sport": "nfl", "event_id": "ev1", "commence_time": KICK, "home_name": "San Francisco 49ers",
                      "away_name": "Washington Commanders", "book": "draftkings", "market": "player_receptions",
                      "side": side, "player": "Rachaad White", "point": pt, "price": pr, "fetched_at": "2030-01-06T07:00:00+00:00"})
    db.insert_snapshots(pd.DataFrame(snaps))


def _patch(monkeypatch, client):
    monkeypatch.setattr(capper, "secret", lambda n, d=None: "test-key" if n == "ANTHROPIC_API_KEY" else None)
    monkeypatch.setattr(capper, "_client", lambda: client)
    monkeypatch.setattr(capper.nflverse, "player_stats", lambda seasons: pd.DataFrame(columns=[
        "season", "week", "team", "position", "player_display_name", "targets", "carries", "attempts"]))
    monkeypatch.setattr(capper.nflverse, "injuries", lambda season: pd.DataFrame([{
        "week": 18, "team": "WAS", "full_name": "Bill Croskey-Merritt", "position": "RB", "report_status": "Out",
        "practice_status": "Did Not Participate In Practice"}]))
    monkeypatch.setattr(capper.nflverse, "snap_counts", lambda season: pd.DataFrame())
    monkeypatch.setattr(capper.nfl_model, "defensive_absences", lambda s, w: {})


def _offer_id(label_part):
    games, _ = capper.db.latest_board("nfl", "games")
    props, _ = capper.db.latest_board("nfl", "props")
    ladder, _ = capper.db.latest_board("nfl", "ladder")
    offers = capper.build_offers(games.iloc[0].to_dict(), props, ladder, config())
    return next(o["id"] for o in offers if label_part in o["label"]), offers


def test_capper_picks_are_stored_validated_and_capped(tmp_db, monkeypatch):
    _seed(tmp_db)
    alt_id, offers = _offer_id("4+ Receptions (alt)")
    main_id = next(o["id"] for o in offers if o["label"].startswith("Rachaad White Over 2.5"))
    card = {"game_read": "WAS leans on White with Croskey-Merritt out.", "passes": "Side: no read.",
            "picks": [
                {"offer_id": main_id, "units": 3.0, "confidence": 4, "thesis": "Role grows.", "factors": ["BCM out"],
                 "risk": "Ekeler vultures passing downs.", "vs_model": "agrees", "edge_type": "role_change"},
                {"offer_id": alt_id, "units": 0.25, "confidence": 2, "thesis": "Ladder.", "factors": [],
                 "risk": "Game script.", "vs_model": "agrees", "edge_type": "role_change"},
                {"offer_id": "o999", "units": 0.5, "confidence": 3, "thesis": "x", "factors": [], "risk": "x",
                 "vs_model": "independent", "edge_type": "other"},
            ]}
    client = FakeClient(card)
    _patch(monkeypatch, client)
    n = capper.run("r1", config(), print, NOW)
    assert n == 2 and client.calls == 1
    dossier = client.kwargs[0]["messages"][0]["content"]
    assert "Bill Croskey-Merritt: Out" in dossier and "+20% carries" in dossier and f"{alt_id}: Rachaad White 4+" in dossier
    p = tmp_db.read(tmp_db.picks)
    assert set(p["source"]) == {"capper"}
    main = p[p["line"] == 2.5].iloc[0]
    assert main["units"] == 1.0  # clipped to capper max_units
    assert main["category"] == "prop" and main["market"] == "player_receptions" and main["price"] == 130
    alt = p[p["line"] == 3.5].iloc[0]
    assert "|capper-alt@" in alt["pick_id"] and alt["price"] == 320 and alt["pick_id"].split("|")[3] == "over"
    assert "Role grows." in main["rationale"] and "Ekeler" in main["rationale"]
    # one review row, and the same game isn't re-read within reread_hours
    assert len(tmp_db.read(tmp_db.capper_reviews)) == 1
    capper.run("r2", config(), print, NOW + pd.Timedelta(hours=2))
    assert client.calls == 1


def test_capper_picks_stay_out_of_model_card_and_caps(tmp_db, monkeypatch):
    _seed(tmp_db)
    main_id, _ = _offer_id("Rachaad White Over 2.5")
    card = {"game_read": "r", "passes": "p", "picks": [
        {"offer_id": main_id, "units": 1.0, "confidence": 3, "thesis": "t", "factors": [], "risk": "r",
         "vs_model": "independent", "edge_type": "role_change"}]}
    _patch(monkeypatch, FakeClient(card))
    capper.run("r1", config(), print, NOW)
    b = slate._DayBudget("nfl", "props", config())
    assert sum(b.used.values()) == 0  # capper units don't use the model's daily props budget
    assert slate._existing_pick_ids("nfl", real_only=True) == set()


def test_capper_skips_without_api_key(tmp_db, monkeypatch):
    _seed(tmp_db)
    monkeypatch.setattr(capper, "secret", lambda n, d=None: None)
    assert capper.run("r1", config(), print, NOW) == 0


def test_notebook_rewrites_after_enough_graded_picks(tmp_db, monkeypatch):
    for i in range(6):
        tmp_db.upsert_pick({"pick_id": f"nfl|g{i}|player_receptions|under|p|capper@3.5", "sport": "nfl", "category": "prop",
                            "market": "player_receptions", "selection": f"P{i} Under 3.5", "matchup": "A @ B",
                            "kickoff": "2030-01-01T18:00:00Z", "units": 0.5, "price": -110, "line": 3.5,
                            "signals": [], "source": "capper", "rationale": "thesis"})
        tmp_db.update_pick(f"nfl|g{i}|player_receptions|under|p|capper@3.5", status="win" if i % 2 else "loss",
                           profit_units=0.45 if i % 2 else -0.5, result_value=2.0)
    brain = capper.ApiBrain(FakeClient({}, notes="1. Unders on capped-role WRs held up (3-3)."))
    assert capper.update_notebook(brain, config(), print, "nfl")
    assert "capped-role" in capper.notebook("nfl")
    assert not capper.update_notebook(brain, config(), print, "nfl")  # nothing new graded since
