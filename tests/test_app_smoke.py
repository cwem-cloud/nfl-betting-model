"""Render the whole dashboard (every tab) against a seeded DB and fail on any exception."""

import pandas as pd
from streamlit.testing.v1 import AppTest


def test_dashboard_renders(tmp_db):
    tmp_db.save_board("r1", "nfl", "games", pd.DataFrame([{
        "game_id": "g1", "kickoff": "2030-01-01T18:00:00Z", "away": "BUF", "home": "KC", "proj_away_pts": 23.1,
        "proj_home_pts": 25.0, "notes": "", "event_id": "e1", "fair_spread_home": -2.0, "fair_total": 48.0,
        "home_win_prob": 0.56, "dk_spread_home": -2.5, "sharp_spread_home": -2.0, "dk_total": 47.5,
        "best_play": "BUF +2.5 -110 (1U, +3.0%)", "flag": None,
    }]))
    tmp_db.insert_df(tmp_db.odds_snapshots, pd.DataFrame([
        {"sport": "nfl", "event_id": "e1", "book": b, "market": m, "side": sd, "point": pt, "price": -110,
         "fetched_at": f"2029-12-3{d}T12:00:00+00:00", "home_name": "Kansas City Chiefs", "away_name": "Buffalo Bills"}
        for d in (0, 1) for b in ("draftkings", "pinnacle")
        for m, sd, pt in (("spread", "home", -2.5 - d * 0.5), ("spread", "away", 2.5 + d * 0.5), ("total", "over", 47.5))]))
    sig = [{"name": "DK off-market", "detail": "x", "direction": 1}]
    base = dict(sport="nfl", category="side", event_id="e1", game_id="g1", matchup="BUF @ KC", market="spread",
                price=-110.0, units=1.0, model_prob=0.55, market_prob=0.5, final_prob=0.52, ev=0.03, signals=sig,
                rationale="- test", line=2.5)
    # game picks store "YYYY-MM-DD HH:MM:SS+00:00", prop picks the Odds API's "...T...Z": both must parse
    tmp_db.upsert_pick(dict(base, pick_id="nfl|g1|spread|away|", selection="BUF +2.5", kickoff="2030-01-01 18:00:00+00:00"))
    tmp_db.upsert_pick(dict(base, pick_id="nfl|g1|player_receptions|over|p1", category="prop", market="player_receptions",
                            selection="Josh Allen Over 0.5 Receptions", kickoff="2030-01-01T18:00:00Z"))
    tmp_db.upsert_pick(dict(base, pick_id="nfl|g0|spread|away|", selection="BUF +3", kickoff="2025-01-01T18:00:00Z"))
    tmp_db.update_pick("nfl|g0|spread|away|", status="win", profit_units=0.91, clv_prob=0.02)

    # angle lab: two angles behind the BUF +2.5 pick (one graded elsewhere), shown as a chip and on History
    for k in ("west_early", "stack"):
        tmp_db.upsert_pick(dict(base, pick_id=f"nfl|g1|spread|away|angle-{k}", selection="BUF +2.5", source="angle",
                                tracking=1, kickoff="2030-01-01 18:00:00+00:00", ev=None, rationale="**ANGLE LAB - x** y"))
    tmp_db.upsert_pick(dict(base, pick_id="nfl|g1|spread|away|angle-blowout_bounce", selection="BUF +2.5", source="angle",
                            tracking=1, kickoff="2030-01-01 18:00:00+00:00", ev=None, rationale="**ANGLE LAB - x** y"))
    tmp_db.upsert_pick(dict(base, pick_id="nfl|g0|spread|away|angle-west_early", selection="BUF +3", source="angle",
                            tracking=1, kickoff="2025-01-01T18:00:00Z", ev=None, rationale="**ANGLE LAB - x** y"))
    tmp_db.update_pick("nfl|g0|spread|away|angle-west_early", status="win", profit_units=0.91)

    # the capper's own picks, a game read and its notebook (Capper tab)
    tmp_db.upsert_pick(dict(base, pick_id="nfl|g1|player_receptions|over|p1|capper@2.5", category="prop",
                            market="player_receptions", selection="Josh Allen Over 2.5 Receptions", source="capper",
                            kickoff="2030-01-01T18:00:00Z", model_prob=None, final_prob=None, ev=None,
                            rationale="**Capper:** thesis"))
    tmp_db.upsert_pick(dict(base, pick_id="nfl|g0|spread|home|capper@-3", selection="KC -3", source="capper",
                            kickoff="2025-01-01T18:00:00Z", ev=None))
    tmp_db.update_pick("nfl|g0|spread|home|capper@-3", status="loss", profit_units=-1.0)
    tmp_db.insert_df(tmp_db.capper_reviews, pd.DataFrame([{"created_at": "2029-12-31T12:00:00+00:00", "sport": "nfl",
        "game_id": "g1", "matchup": "BUF @ KC", "kickoff": "2030-01-01T18:00:00Z", "model": "claude-opus-5-5",
        "read": "Read.", "passes": "Side.", "payload": "{}"}]))
    tmp_db.insert_df(tmp_db.capper_notes, pd.DataFrame([{"created_at": "2029-12-31", "graded_count": 6, "notes": "1. Lesson"}]))

    # ladder board with an anchor (builder) and a graded paper ladder
    from tests.test_ladder import _lemon_rows
    from pocket_capper.market import ladder as ladmod
    from pocket_capper.settings import config as _cfg
    tmp_db.save_board("r1", "nfl", "ladder", _lemon_rows())
    pl = ladmod.plan(_lemon_rows(), _cfg())
    tmp_db.insert_df(tmp_db.ladder_plans, pd.DataFrame([{
        "created_at": "2026-10-26", "source": "paper", "sport": "nfl", "event_id": "e1", "game_id": "2026_07_DAL_PHI",
        "matchup": "DAL @ PHI", "kickoff": "2026-10-27T00:15:00Z", "player": "Makai Lemon", "player_id": "00-L",
        "stat": "receptions", "market": "player_receptions", "scale": pl["scale"], "risk": pl["risk"], "ev": pl["ev"],
        "legs": __import__("json").dumps(pl["legs"]), "context": "", "status": "win", "result_value": 7.0,
        "profit_units": ladmod.settle(pl["legs"], 7)}]))

    at = AppTest.from_file("../app/app.py", default_timeout=60).run()
    assert not at.exception, at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "BUF +2.5" in text
    assert "2 angles behind it" in text  # stacks don't double count
    assert "Angle lab" in text
    code = " ".join(c.value for c in at.code)
    assert "Josh Allen Over 0.5 Receptions" in code  # prop shows on the copyable card
