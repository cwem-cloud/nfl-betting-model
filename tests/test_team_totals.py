import pandas as pd

from pocket_capper.data import odds_api
from pocket_capper.market.team_totals import implied_team_totals, this_week, week_window
from tests.test_parsers import GAME_PAYLOAD


def test_implied_team_totals():
    tt = implied_team_totals(odds_api.parse_game_odds(GAME_PAYLOAD))
    assert len(tt) == 2
    kc = tt[tt.team == "KC"].iloc[0]
    buf = tt[tt.team == "BUF"].iloc[0]
    # 47.5 total, KC -2.5: KC 25.0, BUF 22.5
    assert kc.team_total == 25.0 and buf.team_total == 22.5
    assert kc.opp_total == 22.5 and kc.spread == -2.5 and buf.spread == 2.5
    assert tt.iloc[0].team == "KC"  # lowest opponent total first


def test_consensus_is_median_across_books():
    payload = [dict(GAME_PAYLOAD[0])]
    dk = payload[0]["bookmakers"][0]
    fd = {**dk, "key": "fanduel", "markets": [
        {"key": "spreads", "outcomes": [{"name": "Buffalo Bills", "price": -110, "point": 3.5},
                                        {"name": "Kansas City Chiefs", "price": -110, "point": -3.5}]},
        {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 48.5},
                                       {"name": "Under", "price": -110, "point": 48.5}]},
    ]}
    payload[0]["bookmakers"] = [dk, fd]
    tt = implied_team_totals(odds_api.parse_game_odds(payload))
    kc = tt[tt.team == "KC"].iloc[0]
    assert kc.total == 48.0 and kc.spread == -3.0 and kc.team_total == 25.5 and kc.books == 2


def test_week_window_runs_through_monday_night():
    sun = pd.Timestamp("2026-10-18T15:00Z")
    _, end = week_window(sun)
    assert end == pd.Timestamp("2026-10-20T12:00Z")
    _, end = week_window(pd.Timestamp("2026-10-20T15:00Z"))  # Tuesday -> next Tuesday
    assert end == pd.Timestamp("2026-10-27T12:00Z")
    tt = implied_team_totals(odds_api.parse_game_odds(GAME_PAYLOAD))  # kicks off 2026-10-11
    assert this_week(tt, pd.Timestamp("2026-10-08T12:00Z")).shape[0] == 2
    assert this_week(tt, sun).empty
