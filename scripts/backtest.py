"""Walk-forward backtests of the blind models.

    python scripts/backtest.py                   # NFL sides/totals vs closing lines, 2023-2025
    python scripts/backtest.py --props 2025      # player projection calibration
    python scripts/backtest.py --cfb --seasons 2023,2024,2025
    python scripts/backtest.py --cfb --tune      # grid-search CFB params
"""
import argparse

import _path  # noqa: F401
import numpy as np

from pocket_capper.models import backtest
from pocket_capper.settings import config

ap = argparse.ArgumentParser()
ap.add_argument("--seasons", default="2023,2024,2025")
ap.add_argument("--props", type=int, default=None)
ap.add_argument("--cfb", action="store_true", help="CFB walk-forward vs CFBD closing lines (needs CFBD_API_KEY)")
ap.add_argument("--tune", action="store_true", help="grid-search ratings params; paste the winner into config.yaml")
args = ap.parse_args()
cfg = config()["sports"]["nfl"]

if args.tune:
    sport = "cfb" if args.cfb else "nfl"
    blend = "ppa_blend" if args.cfb else "epa_blend"
    grid = {"decay_half_life_weeks": [4, 6, 8, 10], "prior_season_weight": [0.2, 0.4, 0.7],
            "ridge_alpha": [1.0, 3.0, 8.0], blend: [0.0, 0.3, 0.5]}
    res = backtest.tune(sport, [int(s) for s in args.seasons.split(",")], config()["sports"][sport], grid)
    print(res.head(10).to_string(index=False))
elif args.cfb:
    seasons = [int(s) for s in args.seasons.split(",")]
    bt = backtest.run_cfb(seasons, config()["sports"]["cfb"])
    for k, v in backtest.summarize(bt).items():
        print(f"{k:20s} {v}")
elif args.props:
    bt = backtest.run_props(args.props, list(range(4, 18)), cfg)
    for p, a in [("rec_yds", "receiving_yards"), ("receptions", "receptions_act"), ("rush_yds", "rushing_yards"),
                 ("pass_yds", "passing_yards")]:
        d = bt[bt[p].notna() & (bt[p] > 0)]
        print(f"{p:11s} n={len(d):5d} proj={d[p].mean():6.1f} actual={d[a].mean():6.1f} "
              f"corr={np.corrcoef(d[p], d[a])[0, 1]:.3f} cv={((d[a] - d[p]) / d[p]).std():.3f}")
else:
    bt = backtest.run([int(s) for s in args.seasons.split(",")], cfg)
    for k, v in backtest.summarize(bt).items():
        print(f"{k:20s} {v}")
