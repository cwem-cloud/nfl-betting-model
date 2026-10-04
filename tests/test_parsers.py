import pandas as pd

from pocket_capper.data import odds_api, splits, teams

GAME_PAYLOAD = [{
    "id": "ev1", "sport_key": "americanfootball_nfl", "commence_time": "2026-10-11T17:00:00Z",
    "home_team": "Kansas City Chiefs", "away_team": "Buffalo Bills",
    "bookmakers": [{
        "key": "draftkings", "title": "DraftKings", "last_update": "2026-10-08T12:00:00Z",
        "markets": [
            {"key": "h2h", "outcomes": [{"name": "Buffalo Bills", "price": 120}, {"name": "Kansas City Chiefs", "price": -140}]},
            {"key": "spreads", "outcomes": [{"name": "Buffalo Bills", "price": -110, "point": 2.5},
                                            {"name": "Kansas City Chiefs", "price": -110, "point": -2.5}]},
            {"key": "totals", "outcomes": [{"name": "Over", "price": -108, "point": 47.5},
                                           {"name": "Under", "price": -112, "point": 47.5}]},
        ],
    }],
}]


def test_parse_game_odds():
    df = odds_api.parse_game_odds(GAME_PAYLOAD)
    assert len(df) == 6
    home_sp = df[(df.market == "spread") & (df.side == "home")].iloc[0]
    assert home_sp.point == -2.5 and home_sp.price == -110
    assert set(df[df.market == "total"].side) == {"over", "under"}
    assert set(df[df.market == "ml"].side) == {"home", "away"}


def test_parse_props():
    ev = {"id": "ev1", "home_team": "Kansas City Chiefs", "away_team": "Buffalo Bills", "bookmakers": [{
        "key": "draftkings", "markets": [
            {"key": "player_reception_yds", "outcomes": [
                {"name": "Over", "description": "Travis Kelce", "price": -115, "point": 52.5},
                {"name": "Under", "description": "Travis Kelce", "price": -105, "point": 52.5}]},
            {"key": "player_anytime_td", "outcomes": [{"name": "Yes", "description": "James Cook", "price": 110}]},
        ]}]}
    df = odds_api.parse_props(ev)
    assert set(df.side) == {"over", "under", "yes"}
    assert df[df.side == "yes"].player.iloc[0] == "James Cook"


def test_team_names():
    assert teams.nfl_abbr("Los Angeles Rams") == "LA"
    assert teams.nfl_abbr("KC") == "KC"
    assert teams.match_cfb("Texas A&M Aggies", ["Texas", "Texas A&M"]) == "Texas A&M"
    assert teams.match_cfb("Louisiana Monroe Warhawks", ["Louisiana", "UL Monroe"]) == "UL Monroe"
    assert teams.norm_player("Marvin Harrison Jr.") == teams.norm_player("marvin harrison")


def test_manual_splits_csv():
    text = "sport,away,home,market,side,bets_pct,money_pct\nnfl,BUF,KC,spread,away,68%,41%\n"
    df = splits.parse_manual_csv(text, source="Action")
    assert df.iloc[0].bets_pct == 68 and df.iloc[0].money_pct == 41 and df.iloc[0].source == "Action"
    assert isinstance(df, pd.DataFrame)
