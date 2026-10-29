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
    assert "missed the last game" in out.loc["wr2", "situation"]
    assert "unstable role" not in out.loc["wr1", "situation"]  # WR rotations aren't flagged


def test_backup_of_returning_lead_back_is_flagged_not_bet():
    """Lloyd/Jacobs: the lead back missed weeks 1-2 and returned in week 3. The backup's role is unstable:
    flagged, and none of his props/TDs/ladders qualify."""
    from pocket_capper.market import evaluate

    rows = []
    for wk in (1, 2, 3):
        rb2 = dict(carries=18, rushing_yards=80, targets=3, receptions=2, receiving_yards=15) if wk < 3 else \
            dict(carries=5, rushing_yards=20, targets=1, receptions=1, receiving_yards=6)
        rows += _week(wk, [("rb2", "Backup Back", "RB", rb2),
                           ("wr1", "Wide One", "WR", dict(targets=10, receptions=6, receiving_yards=80)),
                           ("qb", "Quarter Back", "QB", dict(attempts=33, completions=21, passing_yards=230))])
        if wk == 3:
            rows += _week(3, [("rb1", "Lead Back", "RB", dict(carries=17, rushing_yards=75, targets=2, receptions=2,
                                                              receiving_yards=12))])
    ps = pd.DataFrame(rows)
    tp = pd.DataFrame({"team": ["TB"], "opp": ["NO"], "pts": [23.0], "margin": [0.0]})
    out = nfl_props.project_players(ps, pd.DataFrame(), 2026, 4, tp, out_players=set()).set_index("player_id")
    assert "unstable role: Lead Back back at RB" in out.loc["rb2", "situation"]
    assert "unstable role" not in out.loc["wr1", "situation"]
    assert evaluate._unstable(out.loc["rb2"].to_dict()) and not evaluate._unstable(out.loc["wr1"].to_dict())
    # and its ladder is never flagged or picked as a paper ladder
    ev = pd.DataFrame([_eval_row(player="Backup Back", player_id="rb2", stat="rush_yds", line=40.5,
                                 projection=45.0, position="RB")])
    alt = _alt([(49.5, 200), (59.5, 330)], player="Backup Back", market="player_rush_yds_alternate")
    lad = ladder.build(ev, alt, out.reset_index()[["player_id", "situation"]], {}, config())
    assert not lad["value"].any() and lad["context"].str.contains("unstable role").all()



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


def test_far_off_projection_is_never_flagged():
    """Pierce-style case: the model projects a fill-in's role for a backup when the starter returns."""
    cfg = config()
    ev = pd.DataFrame([_eval_row(stat="rush_yds", line=15.5, projection=39.6, position="RB")])
    lad = ladder.build(ev, _alt([(19.5, 133), (24.5, 205)], market="player_rush_yds_alternate"), pd.DataFrame(), {}, cfg)
    assert (lad[lad["dk_price"].notna()]["ev"] > 0.1).all()  # looks like a huge edge...
    assert not lad["value"].any() and lad["context"].str.contains("missing news").all()  # ...but is not flagged


def _lemon_rows():
    """Last MNF's Makai Lemon receptions ladder as the board stored it (model probs, DK prices)."""
    base = dict(event_id="e1", game_id="2026_07_DAL_PHI", matchup="DAL @ PHI", kickoff="2026-10-27T00:15:00Z",
                player="Makai Lemon", player_id="00-L", team="PHI", opp="DAL", position="WR", stat="receptions",
                market="player_receptions", main_line=3.5, context="+12% targets with WR DeVonta Smith out", value=False,
                market_mean=4.50, projection=5.39, blended_mean=4.68)
    rungs = [("anchor", 3.5, -156, 0.617), ("rung", 4.5, 131, 0.4276), ("rung", 5.5, 251, 0.2867),
             ("rung", 6.5, 467, 0.181), ("rung", 7.5, 820, 0.1138), ("rung", 8.5, 1440, 0.0694)]
    rows = []
    for kind, pt, price, p in rungs:
        rows.append(dict(base, kind=kind, point=pt, rung=int(pt + 0.5), dk_price=float(price), model_prob=p,
                         fair_odds=(100 / p - 100) if p < 0.5 else -100 * p / (1 - p),
                         ev=p * (1 + (price / 100 if price > 0 else 100 / -price)) - 1))
    return pd.DataFrame(rows)


def test_ladder_plan_matches_template_and_payouts():
    cfg = config()
    p = ladder.plan(_lemon_rows(), cfg, scale=1.0)
    assert [l["units"] for l in p["legs"]] == [2.0, 1.5, 1.0, 0.5, 0.25]
    assert [l["rung"] for l in p["legs"]] == [4, 5, 6, 7, 8]
    assert p["risk"] == 5.25
    by = {o["result"]: o for o in p["outcomes"]}
    assert abs(by["under 4"]["profit"] + 5.25) < 1e-9 and abs(by["under 4"]["prob"] - 0.383) < 1e-3
    assert abs(by["4"]["profit"] + 1.968) < 0.01
    assert abs(by["7"]["profit"] - 7.842) < 0.01  # Lemon caught 7
    assert abs(by["8+"]["profit"] - 10.142) < 0.01
    assert abs(sum(o["prob"] for o in p["outcomes"]) - 1) < 1e-9
    assert abs(p["ev"] - 0.04) < 0.02
    assert abs(ladder.settle(p["legs"], 7) - 7.842) < 0.01 and ladder.settle(p["legs"], 3) == -5.25


def test_ladder_auto_scale():
    cfg = config()
    rows = _lemon_rows()
    assert ladder.plan(rows, cfg)["scale"] == 0.5  # anchor isn't model-grade, ladder slightly +EV -> half size
    strong = rows.copy()
    strong.loc[strong["kind"] == "anchor", "ev"] = 0.08
    assert ladder.plan(strong, cfg)["scale"] == 1.0
    weak = rows.copy()
    weak["model_prob"] = weak["model_prob"] * 0.8
    assert ladder.plan(weak, cfg)["scale"] == 0.25


def test_paper_ladder_selection_and_grading(tmp_db, monkeypatch):
    import json

    from pocket_capper.engine import grading, slate

    cfg = config()
    rows = _lemon_rows()
    picked = ladder.pick_paper_ladders(rows, cfg)
    assert len(picked) == 1 and picked[0]["player"] == "Makai Lemon"
    far = rows.assign(context="model far off the market - likely missing news, not flagged")
    assert ladder.pick_paper_ladders(far, cfg) == []
    slate._save_paper_ladders(picked, print)
    slate._save_paper_ladders(picked, print)  # same game again: still one paper ladder
    lp = tmp_db.read(tmp_db.ladder_plans)
    assert len(lp) == 1 and lp["status"].iloc[0] == "pending"
    ps = pd.DataFrame([{"player_id": "00-L", "game_id": "2026_07_DAL_PHI", "receptions": 7, "season": 2026}])
    monkeypatch.setattr(grading.nflverse, "player_stats", lambda seasons: ps)
    grading.grade_ladders(pd.DataFrame(), pd.Timestamp("2026-10-28T12:00:00Z"))
    lp = tmp_db.read(tmp_db.ladder_plans).iloc[0]
    legs = json.loads(lp["legs"])
    assert lp["status"] == "win" and lp["result_value"] == 7
    assert abs(lp["profit_units"] - ladder.settle(legs, 7)) < 1e-9 and lp["profit_units"] > 0


def test_ladder_rungs_are_a_real_step_apart():
    rows = _lemon_rows().assign(stat="rec_yds", market="player_reception_yds")
    pts = [(13.5, -116, 0.55), (14.5, -106, 0.52), (19.5, 143, 0.40), (24.5, 204, 0.30), (29.5, 287, 0.22), (34.5, 400, 0.16)]
    rows = rows.iloc[:len(pts)].copy()
    rows["kind"] = ["anchor"] + ["rung"] * (len(pts) - 1)
    rows["point"] = [p for p, _, _ in pts]
    rows["rung"] = [int(p + 0.5) for p, _, _ in pts]
    rows["dk_price"] = [float(pr) for _, pr, _ in pts]
    rows["model_prob"] = [q for _, _, q in pts]
    p = ladder.plan(rows, config(), scale=1.0)
    assert [l["rung"] for l in p["legs"]] == [14, 20, 25, 30, 35]  # 15+ skipped: one yard above the anchor
