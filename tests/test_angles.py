import pandas as pd

from pocket_capper.data import nflverse
from pocket_capper.engine import slate
from pocket_capper.models import angles, matchups
from pocket_capper.settings import config


def _sched():
    rows = []
    # week 1: LV loses by 24 at home; week 2: LV plays at NYJ at 1pm ET Sunday as a 1.5-point dog
    rows.append(dict(season=2030, week=1, game_type="REG", game_id="2030_01_DEN_LV", gameday="2030-09-08", gametime="16:25",
                     weekday="Sunday", away_team="DEN", home_team="LV", result=-24, spread_line=-2.0, total_line=44.5,
                     home_score=10, away_score=34, away_rest=7, home_rest=7, div_game=1, roof="dome", wind=None))
    rows.append(dict(season=2030, week=2, game_type="REG", game_id="2030_02_LV_NYJ", gameday="2030-09-15", gametime="13:00",
                     weekday="Sunday", away_team="LV", home_team="NYJ", result=None, spread_line=1.5, total_line=45.5,
                     home_score=None, away_score=None, away_rest=7, home_rest=7, div_game=0, roof="outdoors", wind=None))
    return pd.DataFrame(rows).assign(gameday=lambda d: pd.to_datetime(d["gameday"]))


def test_angles_fire_and_stack_at_todays_number():
    t = angles.team_table(_sched(), wind={"2030_02_LV_NYJ": 18.0})
    fired = angles.fire_game(t, "2030_02_LV_NYJ", home_spread_line=1.5, total_line=45.5)
    keys = {f["key"]: f["side"] for f in fired}
    assert keys == {"west_early": "away", "blowout_bounce": "away", "wind_under": "under"}
    st = angles.stacks(fired)
    assert len(st) == 1 and st[0]["side"] == "away" and set(st[0]["keys"]) == {"west_early", "blowout_bounce"}
    # line flips: LV now favoured, so the underdog bounce no longer applies
    assert "blowout_bounce" not in {f["key"] for f in angles.fire_game(t, "2030_02_LV_NYJ", home_spread_line=-2.5)}
    pro, con = angles.agreeing(fired, "ml", "home")
    assert not pro and {f["key"] for f in con} == {"west_early", "blowout_bounce"}


def test_angle_picks_are_shadow_tracked(tmp_db):
    t = angles.team_table(_sched())
    fired = angles.fire_game(t, "2030_02_LV_NYJ", 1.5, 45.5)
    row = {"dk_spread_home": -1.5, "dk_spread_home_price": -110, "dk_spread_away_price": -110, "dk_total": 45.5,
           "dk_over_price": -110, "dk_under_price": -110}
    g = pd.Series({"game_id": "2030_02_LV_NYJ", "home": "NYJ", "away": "LV", "kickoff": "2030-09-15T17:00:00Z", "event_id": "e"})
    picks = slate._angle_picks("nfl", "r", g, row, fired, set(), config())
    by = {p["pick_id"].rsplit("|", 1)[-1]: p for p in picks}
    assert {"angle-west_early", "angle-blowout_bounce", "angle-stack", "angle-stack_proven"} <= set(by)
    p = by["angle-west_early"]
    assert p["selection"] == "LV +1.5" and p["tracking"] == 1 and p["source"] == "angle" and p["units"] == 1.0
    for p in picks:
        tmp_db.upsert_pick(p)
    model = {"pick_id": "nfl|2030_02_LV_NYJ|spread|away|", "sport": "nfl", "category": "side", "market": "spread",
             "selection": "LV +1.5", "kickoff": "2030-09-15T17:00:00Z", "units": 1.0, "price": -110, "line": 1.5}
    tmp_db.upsert_pick(model)
    allp = tmp_db.read(tmp_db.picks)
    sup = angles.support(allp)
    assert int(sup[allp["pick_id"] == model["pick_id"]].iloc[0]) == 2  # two angles, stacks not double counted
    assert tmp_db.off_book(allp[allp["source"] == "angle"]).all()
    sc = angles.scorecard(allp, config()).set_index("key")
    assert sc.loc["west_early", "open"] == 1 and sc.loc["west_early", "status"] == "proven"


def test_scorecard_flags_ready_candidates():
    rows = [{"pick_id": f"nfl|g{i}|spread|home|angle-prime_under", "source": "angle", "status": "win" if i < 20 else "loss",
             "units": 1.0, "profit_units": 0.91 if i < 20 else -1.0, "clv_prob": 0.01} for i in range(32)]
    sc = angles.scorecard(pd.DataFrame(rows), config()).set_index("key")
    assert sc.loc["prime_under", "live"] == "20-12" and sc.loc["prime_under", "status"].startswith("READY")


def _snaps(week):
    rows = []
    for w in range(1, week):
        rows += [dict(season=2030, week=w, player="Ace Corner", position="CB", team="NYJ", offense_pct=0, defense_pct=1.0),
                 dict(season=2030, week=w, player="Two Corner", position="CB", team="NYJ", offense_pct=0, defense_pct=0.9),
                 dict(season=2030, week=w, player="Rookie Backup", position="CB", team="NYJ", offense_pct=0, defense_pct=0.1)]
        rows += [dict(season=2030, week=w, player=f"Lineman {i}", position="T" if i < 2 else "G", team="LV",
                      offense_pct=1.0, defense_pct=0) for i in range(4)]
    return pd.DataFrame(rows)


def test_matchup_factors_find_fill_in_corner_and_line_injuries(monkeypatch):
    monkeypatch.setattr(nflverse, "snap_counts", lambda s: _snaps(6) if s == 2030 else pd.DataFrame())
    monkeypatch.setattr(nflverse, "injuries", lambda s: pd.DataFrame([
        {"week": 6, "team": "NYJ", "full_name": "Ace Corner", "report_status": "Out", "gsis_id": "x"},
        {"week": 6, "team": "LV", "full_name": "Lineman 0", "report_status": "Out", "gsis_id": "y"},
        {"week": 6, "team": "LV", "full_name": "Lineman 2", "report_status": "Doubtful", "gsis_id": "z"}]))
    cov = pd.DataFrame([dict(season=2030, week=w, game_type="REG", team="NYJ", opponent="X", pfr_player_name=n,
                             def_targets=t, def_completions_allowed=t // 2, def_yards_allowed=t * 7, def_receiving_td_allowed=0)
                        for w in range(1, 6) for n, t in (("Ace Corner", 8), ("Two Corner", 6), ("Rookie Backup", 1))])
    monkeypatch.setattr(nflverse, "def_coverage", lambda s: cov if s == 2030 else pd.DataFrame())
    ps = pd.DataFrame([dict(player_id=pid, player_display_name=n, position=pos, season=2030, week=w, team="LV",
                            targets=tg, carries=0, attempts=at) for w in range(1, 6)
                       for pid, n, pos, tg, at in (("w1", "Top Wideout", "WR", 9, 0), ("q1", "Signal Caller", "QB", 0, 35))])
    m = matchups.build(2030, 6, ps, {"LV": "NYJ", "NYJ": "LV"})
    got = {(r.player, r.key): r for r in m.itertuples()}
    wr = got[("Top Wideout", "new_cb")]
    assert wr.direction == 1 and "Rookie Backup" in wr.note and "fill-in" in wr.note
    qb = got[("Signal Caller", "ol_out")]
    assert qb.direction == -1 and qb.stats == ["pass_yds"] and "Lineman 0" in qb.note
    pro, con = matchups.for_bet(m, "w1", "receptions", "under")
    assert not pro and con  # a receptions under runs into the fill-in corner
    assert matchups.build(2030, 3, ps, {"LV": "NYJ"}).empty  # too early in the season


def test_matchup_shadow_picks_use_dk_main_line():
    mfac = pd.DataFrame([{"player_id": "w1", "player": "Top Wideout", "team": "LV", "opp": "NYJ", "key": "new_cb",
                          "stats": ["receptions"], "direction": 1, "note": "opp CB fill-in"},
                         {"player_id": "w1", "player": "Top Wideout", "team": "LV", "opp": "NYJ", "key": "weak_cb",
                          "stats": ["receptions"], "direction": 1, "note": "weak CB"}])
    ev = pd.DataFrame([{"event_id": "e1", "player": "Top Wideout", "player_id": "w1", "stat": "receptions", "line": 5.5,
                        "side": "under", "game_id": "2030_06_LV_NYJ", "kickoff": "2030-10-12T17:00:00Z", "matchup": "LV @ NYJ"}])
    props = pd.DataFrame([{"event_id": "e1", "book": "draftkings", "market": "player_receptions", "player": "Top Wideout",
                           "side": sd, "point": 5.5, "price": pr, "fetched_at": "2030-10-12T10:00:00Z"}
                          for sd, pr in (("over", 120), ("under", -150))])
    picks = slate._matchup_picks("r", mfac, props, ev, config(), set())
    ids = {p["pick_id"] for p in picks}
    assert ids == {f"nfl|2030_06_LV_NYJ|player_receptions|over|w1|angle-{k}" for k in ("new_cb", "weak_cb", "mstack")}
    assert all(p["price"] == 120 and p["line"] == 5.5 and p["tracking"] == 1 for p in picks)


def test_totals_lab_angles_and_inverse():
    rows = []
    for wk in range(1, 5):  # both teams' first three games fly 10 over their totals; week 4 they meet
        for a_, h in (("LV", "DEN"), ("KC", "LAC")) if wk < 4 else (("LV", "KC"),):
            rows.append(dict(season=2030, week=wk, game_type="REG", game_id=f"2030_0{wk}_{a_}_{h}",
                             gameday=f"2030-09-{wk + 6:02d}", gametime="16:25", weekday="Sunday", away_team=a_,
                             home_team=h, result=0 if wk < 4 else None, spread_line=0.0, total_line=44.0,
                             home_score=27 if wk < 4 else None, away_score=27 if wk < 4 else None,
                             away_rest=7, home_rest=7, div_game=0, roof="dome", wind=None))
    t = angles.team_table(pd.DataFrame(rows).assign(gameday=lambda d: pd.to_datetime(d["gameday"])))
    fired = {f["key"]: f["side"] for f in angles.fire_game(t, "2030_04_LV_KC", 0.0, 47.5, open_total=45.0)}
    assert fired["hot_fade_under"] == "under" and fired["total_steam_over"] == "over"
    assert "total_steam_under" not in fired
    picks = pd.DataFrame([{"pick_id": f"nfl|g{i}|total|under|angle-hot_fade_under", "source": "angle",
                           "status": "loss", "units": 1.0, "profit_units": -1.0, "clv_prob": None} for i in range(3)])
    sc = angles.scorecard(picks, config()).set_index("key")
    assert sc.loc["hot_fade_under", "live"] == "0-3" and sc.loc["hot_fade_under", "inverse"] == "3-0"
    assert abs(sc.loc["hot_fade_under", "inv_units"] - 2.73) < 0.01
