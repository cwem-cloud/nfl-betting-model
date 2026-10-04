"""Walk-forward backtests of the blind models.

    python scripts/backtest.py                   # NFL sides/totals vs closing lines, 2023-2025
    python scripts/backtest.py --props 2025      # player projection calibration
"""
import argparse

import _path  # noqa: F401
import numpy as np

from pocket_capper.models import backtest
from pocket_capper.settings import config

ap = argparse.ArgumentParser()
ap.add_argument("--seasons", default="2023,2024,2025")
ap.add_argument("--props", type=int, default=None)
args = ap.parse_args()
cfg = config()["sports"]["nfl"]

if args.props:
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
