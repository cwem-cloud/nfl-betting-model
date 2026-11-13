import pandas as pd

from pocket_capper.models import nfl_model
from pocket_capper.models.team_ratings import Ratings
from pocket_capper.settings import config

CFG = config()["sports"]["nfl"]
R = Ratings(mu=22.0, hfa=1.5, off={"KC": 2, "BUF": 1}, deff={"KC": -1, "BUF": 0})
GAME = pd.DataFrame([{"game_id": "g", "home": "KC", "away": "BUF", "neutral": False, "home_rest": 7,
                      "away_rest": 7, "roof": "outdoors"}])


def proj(**kw):
    return nfl_model.project_games(GAME, R, CFG, **kw).iloc[0]


def test_qb_out_scales_with_starter_quality():
    base = proj()
    flags = {"KC": [{"id": "qb1", "status": "Out", "name": "QB"}]}
    good = proj(qb_flags=flags, qb_values={"KC": {"id": "qb1", "name": "QB", "epa_pp": 0.2}})
    bad = proj(qb_flags=flags, qb_values={"KC": {"id": "qb1", "name": "QB", "epa_pp": -0.25}})
    assert base.fair_margin - good.fair_margin > 3.5
    assert base.fair_margin - bad.fair_margin < 0.5


def test_skill_out_and_weather():
    base = proj()
    kp = {"BUF": [{"id": "wr1", "name": "WR", "role": "WR1"}, {"id": "rb1", "name": "RB", "role": "lead RB"}]}
    hurt = proj(key_players=kp, out_ids={"wr1", "rb1"})
    assert round(hurt.fair_margin - base.fair_margin, 2) == 2.5
    windy = proj(weather={"g": {"wind_mph": 20, "temp_f": 50, "precip_prob": 0}})
    assert (windy.home_pts + windy.away_pts) < (base.home_pts + base.away_pts) - 3.5


def test_ir_players_count_as_out(monkeypatch):
    import pandas as pd

    from pocket_capper.data import nflverse
    from pocket_capper.models import nfl_model

    monkeypatch.setattr(nflverse, "injuries", lambda s: pd.DataFrame([
        {"week": 9, "gsis_id": "q", "report_status": "Out"}, {"week": 9, "gsis_id": "d", "report_status": "Questionable"}]))
    monkeypatch.setattr(nflverse, "weekly_rosters", lambda s: pd.DataFrame([
        {"week": 8, "gsis_id": "ir", "status": "ACT"}, {"week": 9, "gsis_id": "ir", "status": "RES"},
        {"week": 9, "gsis_id": "ps", "status": "DEV"}, {"week": 9, "gsis_id": "a", "status": "ACT"},
        {"week": 10, "gsis_id": "a", "status": "RES"}]))  # a future week never leaks back
    assert nfl_model.out_player_ids(2030, 9) == {"q", "ir"}


def _rb_week(week, pid, name, carries, team="DEN"):
    return {"player_id": pid, "player_display_name": name, "position": "RB", "season": 2030, "week": week,
            "season_type": "REG", "team": team, "opponent_team": "KC", "completions": 0, "attempts": 0,
            "passing_yards": 0, "passing_tds": 0, "carries": carries, "rushing_yards": carries * 4, "rushing_tds": 0,
            "receptions": 1, "targets": 1, "receiving_yards": 5, "receiving_tds": 0, "target_share": 0.05}


def test_rb_role_shift_is_flagged_unstable():
    import pandas as pd

    from pocket_capper.models import nfl_props

    rows = []
    for wk in range(1, 9):  # Dobbins-like lead back loses the job to a rookie over the last two games
        lead, rook = (18, 4) if wk <= 6 else (6, 12)
        rows += [_rb_week(wk, "lead", "Old Lead", lead), _rb_week(wk, "rook", "New Lead", rook),
                 _rb_week(wk, "qb", "Quarterback", 4) | {"position": "QB", "attempts": 30}]
    ps = pd.DataFrame(rows)
    tp = pd.DataFrame([{"team": "DEN", "opp": "KC", "pts": 22.0, "margin": 0.0}])
    pp = nfl_props.project_players(ps, pd.DataFrame(), 2030, 9, tp, set()).set_index("player")
    assert "role shrinking" in pp.loc["Old Lead", "situation"] and nfl_props.UNSTABLE_ROLE in pp.loc["Old Lead", "situation"]
    assert "role growing" in pp.loc["New Lead", "situation"]
    assert "role" not in (pp.loc["Quarterback", "situation"] or "")


def test_returning_lead_back_is_flagged_himself():
    import pandas as pd

    from pocket_capper.models import nfl_props

    rows = []
    for wk in range(1, 9):  # lead back misses week 8; the backup starts in his place
        if wk < 8:
            rows.append(_rb_week(wk, "lead", "Lead Back", 18))
        rows.append(_rb_week(wk, "bk", "Backup", 4 if wk < 8 else 20))
        rows.append(_rb_week(wk, "qb", "Quarterback", 3) | {"position": "QB", "attempts": 30})
    tp = pd.DataFrame([{"team": "DEN", "opp": "KC", "pts": 22.0, "margin": 0.0}])
    pp = nfl_props.project_players(pd.DataFrame(rows), pd.DataFrame(), 2030, 9, tp, set()).set_index("player")
    assert "returning lead back" in pp.loc["Lead Back", "situation"]
    assert "Lead Back back at RB" in pp.loc["Backup", "situation"]
