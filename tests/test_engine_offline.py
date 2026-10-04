"""Hermetic end-to-end run: synthetic league + synthetic odds, no network."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import fake_market
from pocket_capper.data import nflverse, odds_api, splits, weather
from pocket_capper.engine import grading, slate

TEAMS = ["KC", "BUF", "BAL", "CIN", "DAL", "PHI", "SF", "SEA"]
NOW = datetime(2026, 10, 4, 12, tzinfo=timezone.utc)


def synthetic_schedule():
    rng = np.random.default_rng(7)
    strength = dict(zip(TEAMS, np.linspace(6, -6, len(TEAMS))))
    rows = []
    start = pd.Timestamp("2025-09-07")
    for season in (2025, 2026):
        base = start if season == 2025 else pd.Timestamp("2026-09-13")
        weeks = 17 if season == 2025 else 4
        for wk in range(1, weeks + 1):
            order = rng.permutation(TEAMS)
            for i in range(0, len(TEAMS), 2):
                h, a = order[i], order[i + 1]
                day = base + pd.Timedelta(days=7 * (wk - 1))
                done = season == 2025 or wk < 4
                hs = max(0, round(22 + (strength[h] - strength[a]) / 2 + 1 + rng.normal(0, 9))) if done else np.nan
                as_ = max(0, round(22 - (strength[h] - strength[a]) / 2 + rng.normal(0, 9))) if done else np.nan
                rows.append({
                    "game_id": f"{season}_{wk:02d}_{a}_{h}", "season": season, "game_type": "REG", "week": wk,
                    "gameday": day, "gametime": "13:00", "home_team": h, "away_team": a,
                    "home_score": hs, "away_score": as_, "result": hs - as_ if done else np.nan,
                    "total": hs + as_ if done else np.nan, "location": "Home", "home_rest": 7, "away_rest": 7,
                    "spread_line": 0.0, "total_line": 44.0, "div_game": 0, "roof": "outdoors",
                })
    # put week 4 later today
    df = pd.DataFrame(rows)
    df.loc[(df.season == 2026) & (df.week == 4), "gameday"] = pd.Timestamp("2026-10-04")
    df.loc[(df.season == 2026) & (df.week == 4), "gametime"] = "16:25"
    return df


def test_full_run_and_grade(tmp_db, monkeypatch):
    sched = synthetic_schedule()
    monkeypatch.setattr(nflverse, "schedules", lambda: sched.copy())
    monkeypatch.setattr(nflverse, "team_game_epa", lambda seasons: pd.DataFrame())
    monkeypatch.setattr(nflverse, "player_stats", lambda seasons: pd.DataFrame(columns=["season", "position"]))
    monkeypatch.setattr(nflverse, "injuries", lambda season: pd.DataFrame())
    monkeypatch.setattr(weather, "kickoff_weather", lambda *a, **k: None)
    monkeypatch.setattr(splits, "fetch_dk_splits", lambda sport: pd.DataFrame(columns=splits.SPLIT_COLS))
    monkeypatch.setattr(slate, "secret", lambda n, d=None: "test" if n == "ODDS_API_KEY" else None)

    up = sched[(sched.season == 2026) & (sched.week == 4)]
    # market disagrees with nobody in particular: random lines around zero
    rng = np.random.default_rng(3)
    games = [(f"ev{i}", r.home_team, r.away_team, rng.normal(0, 3), 44 + rng.normal(0, 2), "2026-10-04T20:25:00Z")
             for i, r in enumerate(up.itertuples())]
    payload = fake_market.game_payload(games)
    monkeypatch.setattr(odds_api, "_get", lambda path, params: payload)

    res = slate.run(("nfl",), include_props=False, now=NOW)
    assert res["errors"] == []
    board, _ = tmp_db.latest_board("nfl", "games")
    assert len(board) == 4 and board["event_id"].notna().all()
    # projections were written before odds were fetched, and exist for every game
    proj = tmp_db.read(tmp_db.projections)
    assert len(proj) == 4

    picks = tmp_db.read(tmp_db.picks)
    assert len(picks) > 0, "synthetic market should yield at least one edge"
    for p in picks.itertuples():
        assert 0.5 <= p.units <= 3.0
        assert p.ev >= 0.025
        assert "Blind handicap" in p.rationale

    # re-running does not duplicate locked picks
    slate.run(("nfl",), include_props=False, now=NOW)
    assert len(tmp_db.read(tmp_db.picks)) == len(picks)

    # market flips hard against every open pick: no opposing plays get added, and the
    # open picks record the (now negative) current EV
    flipped = fake_market.game_payload([(e, h, a, -m * 3 + (6 if m < 0 else -6), t, c) for e, h, a, m, t, c in games])
    monkeypatch.setattr(odds_api, "_get", lambda path, params: flipped)
    slate.run(("nfl",), include_props=False, now=NOW)
    after = tmp_db.read(tmp_db.picks)
    for p in after.itertuples():
        assert not (slate._opposite_ids(p.pick_id) & set(after["pick_id"]))
    assert after["latest_ev"].notna().all()

    # final scores arrive -> grading settles every pick
    done = sched.copy()
    m = (done.season == 2026) & (done.week == 4)
    done.loc[m, "home_score"], done.loc[m, "away_score"] = 24, 17
    done.loc[m, "result"] = 7
    monkeypatch.setattr(nflverse, "schedules", lambda: done.copy())
    grading.grade_all(log=lambda *a: None)
    graded = tmp_db.read(tmp_db.picks)
    assert set(graded["status"]) <= {"win", "loss", "push"}


def test_grade_side():
    assert grading.grade_side("spread", "home", -3.0, 24, 21)[0] == "push"
    assert grading.grade_side("spread", "away", 3.5, 24, 21)[0] == "win"
    assert grading.grade_side("total", "over", 44.5, 24, 21)[0] == "win"
    assert grading.grade_side("ml", "away", None, 24, 21)[0] == "loss"


def test_cfb_run(tmp_db, monkeypatch):
    from pocket_capper.data import cfbd

    schools = ["Alabama", "Georgia", "Texas", "Ohio State", "Oregon", "Michigan"]
    rng = np.random.default_rng(11)

    cache = {}

    def games(year, season_type="regular", max_age_hours=3):
        if season_type != "regular":
            return pd.DataFrame()
        if year in cache:
            return cache[year].copy()
        rows = []
        for wk in range(1, 13 if year == 2025 else 7):
            order = rng.permutation(schools)
            for i in range(0, 6, 2):
                upcoming = year == 2026 and wk == 6
                start = pd.Timestamp("2026-10-04 19:30", tz="UTC") if upcoming else pd.Timestamp(f"{year}-09-01", tz="UTC") + pd.Timedelta(days=7 * wk)
                rows.append({"game_id": f"{year}{wk}{i}", "season": year, "week": wk, "start": start,
                             "home": order[i], "away": order[i + 1],
                             "home_pts": np.nan if upcoming else float(rng.integers(10, 45)),
                             "away_pts": np.nan if upcoming else float(rng.integers(10, 45)),
                             "neutral": False, "venue_id": 1, "completed": not upcoming})
        rows.append({"game_id": f"{year}fcs", "season": year, "week": 1, "start": pd.Timestamp(f"{year}-08-30", tz="UTC"),
                     "home": "Alabama", "away": "Mercer", "home_pts": 52.0, "away_pts": 7.0, "neutral": False,
                     "venue_id": 1, "completed": True})
        cache[year] = pd.DataFrame(rows)
        return cache[year].copy()

    monkeypatch.setattr(cfbd, "games", games)
    monkeypatch.setattr(cfbd, "ppa_games", lambda year: pd.DataFrame())

    def lines(year, week=None, season_type="regular"):
        g = games(year).query("week == 6")
        return pd.DataFrame({"game_id": g["game_id"], "home": g["home"], "away": g["away"], "provider": "DraftKings",
                             "spread": -3.5, "spread_open": -1.5, "total": 50.0, "total_open": 52.5})
    monkeypatch.setattr(cfbd, "lines", lines)
    monkeypatch.setattr(cfbd, "fbs_teams", lambda year: schools)
    monkeypatch.setattr(cfbd, "venues", lambda: pd.DataFrame([{"venue_id": 1, "name": "X", "lat": 33.2, "lon": -87.5, "dome": False}]))
    monkeypatch.setattr(weather, "kickoff_weather", lambda *a, **k: {"temp_f": 70, "wind_mph": 20, "gust_mph": 30, "precip_prob": 10})
    monkeypatch.setattr(splits, "fetch_dk_splits", lambda sport: pd.DataFrame(columns=splits.SPLIT_COLS))
    monkeypatch.setattr(slate, "secret", lambda n, d=None: "test" if n in ("ODDS_API_KEY", "CFBD_API_KEY") else None)
    full = {"Alabama": "Alabama Crimson Tide", "Georgia": "Georgia Bulldogs", "Texas": "Texas Longhorns",
            "Ohio State": "Ohio State Buckeyes", "Oregon": "Oregon Ducks", "Michigan": "Michigan Wolverines"}
    up = games(2026).query("week == 6")
    payload = []
    for r in up.itertuples():
        ev = fake_market.game_payload([("e" + r.game_id, "KC", "BUF", 0.0, 50.0, "2026-10-04T19:30:00Z")])[0]
        ev["home_team"], ev["away_team"] = full[r.home], full[r.away]
        for bk in ev["bookmakers"]:
            for mkt in bk["markets"]:
                for o in mkt["outcomes"]:
                    o["name"] = {"Kansas City Chiefs": full[r.home], "Buffalo Bills": full[r.away]}.get(o["name"], o["name"])
        payload.append(ev)
    monkeypatch.setattr(odds_api, "_get", lambda path, params: payload)

    res = slate.run(("cfb",), include_props=False, now=NOW)
    assert res["errors"] == []
    board, _ = tmp_db.latest_board("cfb", "games")
    assert len(board) == 3 and board["event_id"].notna().all()
    assert board["notes"].str.contains("Weather").all()
    # CFBD opening numbers were seeded so movement starts at the open
    assert set(board["dk_open_spread_home"]) == {-1.5}
    assert set(board["dk_open_total"]) == {52.5}
