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
