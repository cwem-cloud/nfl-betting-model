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


def test_snapshots_store_only_changes(tmp_db):
    from tests_helpers import snap_rows

    first = snap_rows("2026-10-01T12:00:00+00:00", -3.0, -110)
    assert tmp_db.insert_snapshots(first) == 2
    # identical prices a few hours later: nothing stored
    assert tmp_db.insert_snapshots(snap_rows("2026-10-01T15:00:00+00:00", -3.0, -110)) == 0
    # line moves: both sides stored
    assert tmp_db.insert_snapshots(snap_rows("2026-10-01T18:00:00+00:00", -3.5, -110)) == 2
    assert len(tmp_db.read(tmp_db.odds_snapshots)) == 4


def test_postgres_url_forms():
    from sqlalchemy import create_engine

    from pocket_capper.db import normalize_pg_url

    want = "postgresql+psycopg://u:p@ep-x.neon.tech/neondb?sslmode=require&channel_binding=require"
    for given in (
        "postgresql://u:p@ep-x.neon.tech/neondb?sslmode=require&channel_binding=require",
        "postgres://u:p@ep-x.neon.tech/neondb?sslmode=require&channel_binding=require",
        "postgresql+psycopg2://u:p@ep-x.neon.tech/neondb?sslmode=require&channel_binding=require",
        "psql 'postgresql://u:p@ep-x.neon.tech/neondb?sslmode=require&channel_binding=require'",
        '  "postgresql://u:p@ep-x.neon.tech/neondb?sslmode=require&channel_binding=require" ',
    ):
        assert normalize_pg_url(given) == want
    # the driver is installed and SQLAlchemy can build an engine for it (no connection is made)
    assert create_engine(want).dialect.driver == "psycopg"
    assert normalize_pg_url("sqlite:///x.db") == "sqlite:///x.db"


def test_dk_splits_live_layout():
    from pathlib import Path

    html = (Path(__file__).parent / "fixtures" / "dk_splits_sample.html").read_text()
    df = splits.parse_dk_splits_html(html, "nfl")
    assert len(df) == 6
    r = df.set_index(["market", "side"])
    assert (r.loc[("ml", "home"), "money_pct"], r.loc[("ml", "home"), "bets_pct"]) == (61, 80)
    assert (r.loc[("spread", "away"), "money_pct"], r.loc[("spread", "away"), "bets_pct"]) == (20, 29)
    assert r.loc[("total", "under"), "money_pct"] == 65
    assert set(df["away"]) == {"NE Patriots"} and set(df["home"]) == {"BUF Bills"}
