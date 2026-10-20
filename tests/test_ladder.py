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


def test_ladder_prices_rungs_and_flags_value():
    cfg = config()
    ev = pd.DataFrame([_eval_row()])
    alt = pd.DataFrame([
        {"event_id": "e1", "book": "draftkings", "market": "player_receptions_alternate", "player": "Wide One",
         "side": "over", "point": 5.5, "price": 600},   # way above fair: value
        {"event_id": "e1", "book": "draftkings", "market": "player_receptions_alternate", "player": "Wide One",
         "side": "over", "point": 6.5, "price": 150},   # below fair: not value
    ])
    lad = ladder.build(ev, alt, pd.DataFrame(), {"NO": [{"name": "Corner Guy", "pos": "CB", "snap_pct": 0.9,
                                                          "group": "pass", "already_missed": False}]}, cfg)
    assert set(lad["rung"]) >= {6, 7, 8}
    six, seven = lad[lad["rung"] == 6].iloc[0], lad[lad["rung"] == 7].iloc[0]
    # rungs get less likely as they climb, and fair odds match the probability
    assert six["model_prob"] > seven["model_prob"]
    assert six["model_prob"] < 0.5 and abs(six["fair_odds"] - (100 / six["model_prob"] - 100)) < 1
    assert six["ev"] > 0.35 and not six["value"]  # +600 is too good to be true: over the sanity cap, not flagged
    assert seven["ev"] < 0 and not seven["value"]
    assert "CB Corner Guy" in six["context"]
    # rungs without a DK alt price are still priced (fair only), never flagged
    no_price = lad[lad["dk_price"].isna()]
    assert len(no_price) and not no_price["value"].any()


def test_ladder_value_flag():
    cfg = config()
    ev = pd.DataFrame([_eval_row(projection=4.6)])
    lad0 = ladder.build(ev, pd.DataFrame(), pd.DataFrame(), {}, cfg)
    p = float(lad0[lad0["rung"] == 6]["model_prob"].iloc[0])
    fair_dec = 1 / p
    good = round((fair_dec * 1.10 - 1) * 100)  # 10% over fair
    alt = pd.DataFrame([{"event_id": "e1", "book": "draftkings", "market": "player_receptions_alternate",
                         "player": "Wide One", "side": "over", "point": 5.5, "price": good}])
    lad = ladder.build(ev, alt, pd.DataFrame(), {}, cfg)
    r = lad[lad["rung"] == 6].iloc[0]
    assert r["value"] and 0.08 < r["ev"] < 0.12
