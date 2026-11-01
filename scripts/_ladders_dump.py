import _path  # noqa: F401
import pandas as pd
from pocket_capper import db
from pocket_capper.market import ladder
from pocket_capper.market.odds_math import fmt_american
from pocket_capper.settings import config
cfg = config()
lad, at = db.latest_board("nfl", "ladder")
print(f"LADDER BOARD as of {at}: {len(lad)} rows")
lad = lad[pd.to_datetime(lad["kickoff"], utc=True, errors="coerce") > pd.Timestamp.now(tz="UTC")]
plans = []
for (pl, stat), rows in lad.groupby(["player", "stat"]):
    ctx = rows["context"].fillna("").iloc[0]
    if "missing news" in ctx:
        continue
    p = ladder.plan(rows, cfg)
    if p:
        p["n_value"] = int(rows["value"].sum())
        plans.append(p)
plans.sort(key=lambda p: -p["ev"])
for p in plans[:15]:
    legs = " / ".join(f"{l['rung']}+ {fmt_american(l['price'])} {l['units']:g}U (p {l['prob']:.0%})" for l in p["legs"])
    print(f"\n{p['matchup']} {p['kickoff']} | {p['player']} {p['stat']} | scale {p['scale']} risk {p['risk']:.2f}U "
          f"EV {p['ev']:+.3f}U P(profit) {p['p_profit']:.0%} value rungs {p['n_value']}\n  {legs}\n  context: {p['context']}")
v = lad[lad["value"] == True].sort_values("ev", ascending=False).head(20)
print("\nTOP VALUE RUNGS")
print(v[["matchup", "player", "stat", "main_line", "projection", "market_mean", "rung", "dk_price", "fair_odds", "ev", "context"]].to_string(index=False))
