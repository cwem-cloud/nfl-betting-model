"""One-off cleanup: void pending prop/TD picks whose projection came from the wrong game.

Before the fix in evaluate_props, a team playing twice inside the 7-day window (e.g. Sunday and the
following Thursday) could get the other game's projection. The rationale records the opponent used
("(QB, DEN vs SEA)"); if that opponent isn't in the pick's matchup, the pick is voided (0 units).
"""
import re

import _path  # noqa: F401

from pocket_capper import db

p = db.read(db.picks, "WHERE status = 'pending' AND category IN ('prop', 'attd')")
n = 0
for r in p.itertuples():
    m = re.search(r"\(\w+, (\w+) vs (\w+)\)", r.rationale or "")
    if not m:
        continue
    team, opp = m.groups()
    teams = set(r.matchup.split(" @ "))
    if team not in teams or opp not in teams:
        db.update_pick(r.pick_id, status="void", profit_units=0.0)
        n += 1
        print(f"voided: {r.selection} [{r.matchup}] projected vs {opp}")
print(f"voided {n} mismatched prop picks")
