"""Fair odds for alt-line ladders, from the latest props board (market-anchored projection).

For each player stat in the chosen games: back out the market-implied mean from the main line,
move it toward our projection by the stat's model weight (same as the main prop pricing), then
price higher rungs off that distribution. Bet a rung only if the book pays more than fair.

Usage: python scripts/ladder.py WAS SF
"""
import sys

import _path  # noqa: F401
import numpy as np

from pocket_capper import db
from pocket_capper.market import evaluate
from pocket_capper.market.odds_math import fmt_american, prob_to_american
from pocket_capper.models import nfl_props
from pocket_capper.settings import config

RUNGS = {
    "rec_yds": [40, 50, 60, 70, 80, 90, 100, 125],
    "rush_yds": [40, 50, 60, 70, 80, 90, 100, 125],
    "pass_yds": [200, 225, 250, 275, 300, 325, 350],
    "receptions": [3, 4, 5, 6, 7, 8, 9],
}
teams = set(a.upper() for a in sys.argv[1:])
board, ts = db.latest_board("nfl", "props")
print(f"props board from {ts}")
b = board[board["stat"].isin(RUNGS) & (board["team"].isin(teams) | board["opp"].isin(teams) if teams else True)]
k_by_stat = config()["props"]["model_weight"]
rows = []
for r in b.itertuples():
    p_over_mkt = r.market_prob if r.side == "over" else 1 - r.market_prob
    m_mkt = evaluate._implied_mean(r.stat, r.line, p_over_mkt)
    m = m_mkt + k_by_stat.get(r.stat, 0.15) * (r.projection - m_mkt)
    rows.append((r, m_mkt, m))
# overs-first: players where our number sits above the market are the ladder candidates
rows.sort(key=lambda x: -(x[2] - x[1]) / max(x[1], 1))
for r, m_mkt, m in rows:
    lean = "OVER lean" if m > m_mkt else "under lean"
    print(f"\n{r.player} ({r.team} vs {r.opp}) {r.stat}: main line {r.line:g}, market mean {m_mkt:.1f}, "
          f"our blind {r.projection:.1f}, blended {m:.1f}  [{lean}]")
    out = []
    for x in RUNGS[r.stat]:
        line = x - 0.5  # "X+" == over X-0.5
        if line <= r.line:
            continue
        p = nfl_props.prob_over(r.stat, m, line)
        if np.isfinite(p) and p > 0.03:
            out.append(f"{x}+ fair {fmt_american(prob_to_american(p))} ({p:.0%})")
    print("   " + " | ".join(out) if out else "   (no rungs above the main line)")
