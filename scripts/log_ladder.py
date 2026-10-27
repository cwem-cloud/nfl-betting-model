"""Log a ladder by hand (one you bet, or a retro paper entry), optionally already settled.

  python scripts/log_ladder.py --source user --player "Makai Lemon" --stat receptions \
      --matchup "DAL @ PHI" --game-id 2026_07_DAL_PHI --kickoff 2026-10-27T00:15:00Z \
      --legs "4:-156:2,5:131:1.5,6:251:1,7:467:0.5,8:820:0.25" [--player-id 00-...] [--result 7]

--legs is rung:american_price:units for each tier ("4" = 4+, i.e. over 3.5). Without --result it stays
pending and is graded from the box score like the paper ladders (needs --player-id).
"""
import argparse
import json

import _path  # noqa: F401
import pandas as pd

from pocket_capper import db
from pocket_capper.market import odds_math as om
from pocket_capper.market.ladder import settle

ap = argparse.ArgumentParser()
for a in ("--player", "--stat", "--matchup", "--game-id", "--kickoff", "--legs"):
    ap.add_argument(a, required=True)
ap.add_argument("--source", default="user", choices=["user", "paper", "retro"])
ap.add_argument("--player-id", default=None)
ap.add_argument("--market", default=None)
ap.add_argument("--result", type=float, default=None)
ap.add_argument("--note", default="")
a = ap.parse_args()

legs = []
for i, part in enumerate(a.legs.split(",")):
    rung, price, units = part.split(":")
    rung, price, units = int(rung), float(price), float(units)
    legs.append({"tier": i + 1, "rung": rung, "point": rung - 0.5, "price": price, "units": units,
                 "win": units * (om.american_to_decimal(price) - 1), "prob": None, "ev": None})
rec = {"created_at": db.now(), "source": a.source, "sport": "nfl", "event_id": None, "game_id": a.game_id,
       "matchup": a.matchup, "kickoff": a.kickoff, "player": a.player, "player_id": a.player_id, "stat": a.stat,
       "market": a.market, "scale": None, "risk": sum(l["units"] for l in legs), "ev": None,
       "legs": json.dumps(legs), "context": a.note, "status": "pending"}
if a.result is not None:
    p = settle(legs, a.result)
    rec.update(status="win" if p > 0 else "loss", result_value=a.result, profit_units=p)
db.insert_df(db.ladder_plans, pd.DataFrame([rec]))
print(f"logged {a.source} ladder: {a.player} {a.stat}, risk {rec['risk']:g}U"
      + (f", result {a.result:g} -> {rec['profit_units']:+.2f}U" if a.result is not None else ""))
