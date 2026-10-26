import numpy as np
import pandas as pd

from pocket_capper.market import ladder
from pocket_capper.models import nfl_props
from pocket_capper.settings import config

PS_COLS = dict(attempts=0, completions=0, passing_yards=0, passing_tds=0, carries=0, rushing_yards=0, rushing_tds=0,
               targets=0, receptions=0, receiving_yards=0, receiving_tds=0)


def _week(week, rows):
    out = []
    for pid, name, pos, kw in rows:
        out.append({"season": 2026, "week": week, "season_type": "REG", "player_id": pid, "player_display_name": name,
                    "position": pos, "team": "TB", "opponent_team": "NO", **{**PS_COLS, **kw}})
    return out


def _stats():
    rows = []
    for wk in (1, 2, 3):
        rows += _week(wk, [
            ("rb1", "Lead Back", "RB", dict(carries=18, rushing_yards=80, targets=2, receptions=2, receiving_yards=10)),
            ("rb2", "Backup Back", "RB", dict(carries=4, rushing_yards=18, targets=1, receptions=1, receiving_yards=5)),
            ("wr1", "Wide One", "WR", dict(carries=1, rushing_yards=5, targets=10, receptions=6, receiving_yards=80)),
            ("wr2", "Wide Two", "WR", dict(targets=6, receptions=4, receiving_yards=45)),
            ("qb", "Quarter Back", "QB", dict(attempts=33, completions=21, passing_yards=230, carries=3, rushing_yards=12)),
        ])
    return pd.DataFrame(rows)


def test_out_rb_carries_go_mostly_to_the_other_back():
    ps = _stats()
    tp = pd.DataFrame({"team": ["TB"], "opp": ["NO"], "pts": [23.0], "margin": [0.0]})
    base = nfl_props.project_players(ps, pd.DataFrame(), 2026, 4, tp, out_players=set()).set_index("player_id")
    out = nfl_props.project_players(ps, pd.DataFrame(), 2026, 4, tp, out_players={"rb1"}).set_index("player_id")
    assert "rb1" not in out.index
    rb2_gain = out.loc["rb2", "rush_att"] / base.loc["rb2", "rush_att"]
    wr1_gain = out.loc["wr1", "rush_att"] / base.loc["wr1", "rush_att"]
    assert rb2_gain > 2 and rb2_gain > 2 * wr1_gain  # the backup back inherits most of the work
    assert "carries with RB Lead Back out" in out.loc["rb2", "situation"]


def test_returning_player_is_flagged():
    ps = _stats()
    ps = ps[~((ps["player_id"] == "wr2") & (ps["week"] == 3))]  # missed the last game
    tp = pd.DataFrame({"team": ["TB"], "opp": ["NO"], "pts": [23.0], "margin": [0.0]})
    out = nfl_props.project_players(ps, pd.DataFrame(), 2026, 4, tp, out_players=set()).set_index("player_id")
    assert "returning" in out.loc["wr2", "situation"]
    assert "returning" not in out.loc["wr1", "situation"]


def _eval_row(**kw):
    r = dict(event_id="e1", matchup="NO @ TB", kickoff="2030-01-01T18:00:00Z", player="Wide One", player_id="wr1",
             team="TB", opp="NO", position="WR", stat="receptions", side="over", line=4.5, market_prob=0.5,
             projection=6.0)
    r.update(kw)
    return r


def _alt(points_prices, player="Wide One", market="player_receptions_alternate"):
    return pd.DataFrame([{"event_id": "e1", "book": "draftkings", "market": market, "player": player,
                          "side": "over", "point": pt, "price": pr} for pt, pr in points_prices])


def test_no_model_opinion_means_no_ladder_value():
    """Projection equal to the market's number: the distribution's shape must not create 'value'."""
    from pocket_capper.market import evaluate

    cfg = config()
    m_mkt = evaluate._implied_mean("receptions", 4.5, 0.5)
    ev = pd.DataFrame([_eval_row(projection=m_mkt)])
    lad = ladder.build(ev, _alt([(5.5, 250), (6.5, 500), (7.5, 900)]), pd.DataFrame(), {}, cfg)
    priced = lad[lad["dk_price"].notna()]
    assert len(priced) == 3 and not priced["value"].any()
    # with no opinion, fair = DK's de-vigged price, so EV is about minus the stripped margin
    assert (priced["ev"].abs() < 0.08).all() and (priced["ev"] < 0).all()


def test_ladder_prices_rungs_and_flags_value():
    cfg = config()
    ev = pd.DataFrame([_eval_row(projection=6.5)])  # we're well above the market on this player
    lad = ladder.build(ev, _alt([(5.5, 250), (6.5, 150)]), pd.DataFrame(),
                       {"NO": [{"name": "Corner Guy", "pos": "CB", "snap_pct": 0.9, "group": "pass",
                                "already_missed": False}]}, cfg)
    assert set(lad["rung"]) >= {6, 7, 8}
    six, seven = lad[lad["rung"] == 6].iloc[0], lad[lad["rung"] == 7].iloc[0]
    assert six["ev"] > 0 and seven["ev"] > 0  # higher projection lifts every rung above DK's own price
    assert abs(six["fair_odds"] - (100 / six["model_prob"] - 100)) < 1
    assert "CB Corner Guy" in six["context"]
    no_price = lad[lad["dk_price"].isna()]
    assert len(no_price) and not no_price["value"].any()  # unpriced rungs: fair only, never flagged


def test_lower_projection_kills_overs_value():
    cfg = config()
    ev = pd.DataFrame([_eval_row(projection=3.0, side="under", market_prob=0.5)])
    lad = ladder.build(ev, _alt([(5.5, 250), (6.5, 500)]), pd.DataFrame(), {}, cfg)
    assert not lad["value"].any() and (lad[lad["dk_price"].notna()]["ev"] < -0.05).all()
