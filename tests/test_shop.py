import pandas as pd

from pocket_capper.market import arbs, evaluate
from pocket_capper.market import market as mk
from pocket_capper.settings import config

MINE = ["draftkings", "fanduel", "betmgm", "fanatics"]


def _rows(*rows):
    return pd.DataFrame(rows, columns=["event_id", "book", "market", "side", "point", "price"])


def test_shop_takes_best_price_at_the_same_number_and_skips_outliers():
    r = _rows(("e", "draftkings", "spread", "home", -3.0, -110), ("e", "fanduel", "spread", "home", -3.0, -102),
              ("e", "betmgm", "spread", "home", -2.5, -105), ("e", "fanatics", "spread", "home", -3.0, 160),
              ("e", "pinnacle", "spread", "home", -3.0, 100))
    assert mk.shop(r, MINE, -3.0, -110) == ("fanduel", -102.0)  # +160 is stale/bad data; pinnacle isn't ours
    assert mk.shop(r, MINE, -2.5, -110) == ("betmgm", -105.0)


def _game_board():
    rows = []
    for bk, hp, ap in (("draftkings", -110, -110), ("fanduel", -104, -116), ("pinnacle", -108, -102)):
        rows += [("e1", bk, "spread", "home", -3.0, hp), ("e1", bk, "spread", "away", 3.0, ap)]
    for bk in ("draftkings", "fanduel", "pinnacle"):
        rows += [("e1", bk, "total", "over", 44.5, -110), ("e1", bk, "total", "under", 44.5, -110),
                 ("e1", bk, "ml", "home", None, -150), ("e1", bk, "ml", "away", None, 130)]
    b = _rows(*rows)
    b["fetched_at"] = "2030-01-01T12:00:00+00:00"
    return b


def test_game_candidates_are_priced_at_the_best_book():
    b = _game_board()
    proj = {"fair_margin": 5.0, "home_pts": 25.0, "away_pts": 20.0, "fair_spread_home": -5.0, "fair_total": 45.0,
            "home_win_prob": 0.65}
    row, cands = evaluate.evaluate_game(sport="nfl", proj=proj, board=b, snap=b, event_id="e1", splits={}, manual=[],
                                        cfg=config())
    home = next(c for c in cands if c["market"] == "spread" and c["side"] == "home")
    away = next(c for c in cands if c["market"] == "spread" and c["side"] == "away")
    assert (home["book"], home["price"]) == ("fanduel", -104.0)
    assert (away["book"], away["price"]) == ("draftkings", -110.0)  # DK already best
    assert row["dk_spread_home_price"] == -110  # the board still shows DK's own number


def test_props_are_shopped():
    snap = pd.DataFrame([{"event_id": "e1", "book": bk, "market": "player_receptions", "player": "A Receiver",
                          "side": sd, "point": 4.5, "price": pr, "fetched_at": "2030-01-01T12:00:00+00:00",
                          "home_name": "Kansas City Chiefs", "away_name": "Buffalo Bills"}
                         for bk, o, u in (("draftkings", -110, -120), ("fanatics", -115, 102), ("pinnacle", -112, -108))
                         for sd, pr in (("over", o), ("under", u))])
    proj = pd.DataFrame([{"player": "A Receiver", "player_id": "p1", "team": "BUF", "opp": "KC", "position": "WR",
                          "receptions": 3.2}])
    ev = evaluate.evaluate_props(snap, proj, config())
    r = ev.iloc[0]
    assert r["side"] == "under" and r["book"] == "fanatics" and r["price"] == 102


def test_arbs_and_middles():
    b = _rows(("e", "draftkings", "spread", "home", -2.5, -110), ("e", "fanduel", "spread", "away", 3.5, -110),
              ("e", "betmgm", "total", "over", 44.5, 105), ("e", "fanatics", "total", "under", 44.5, 105),
              ("e", "draftkings", "ml", "home", None, -150), ("e", "fanduel", "ml", "away", None, 140))
    out = arbs.scan(b, MINE, {"e": "BUF @ KC"})
    arb = out[out["kind"] == "arb"].iloc[0]
    assert arb["market"] == "total" and abs(arb["edge"] - 0.025) < 1e-9 and arb["stake_a"] == 50.0
    mid = out[out["kind"] == "middle"].iloc[0]
    assert mid["market"] == "spread" and mid["window"] == "home margin 3" and mid["matchup"] == "BUF @ KC"
    assert len(out) == 2  # -150 / +140 is no arb
    # whole numbers either side of a gap of 1 only push, never middle
    flat = _rows(("e", "draftkings", "total", "over", 45.0, -110), ("e", "fanduel", "total", "under", 46.0, -110))
    assert arbs.scan(flat, MINE).empty
    props = pd.DataFrame([{"event_id": "e", "book": "draftkings", "market": "player_receptions", "player": "X",
                           "side": "over", "point": 4.5, "price": -110},
                          {"event_id": "e", "book": "fanduel", "market": "player_receptions", "player": "X",
                           "side": "under", "point": 5.5, "price": -110}])
    pm = arbs.scan(props, MINE)
    assert pm.iloc[0]["kind"] == "middle" and pm.iloc[0]["window"] == "result 5" and pm.iloc[0]["player"] == "X"
