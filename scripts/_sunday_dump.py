import _path  # noqa: F401
import pandas as pd
from pocket_capper import db
from pocket_capper.market import ladder
from pocket_capper.market.odds_math import fmt_american
from pocket_capper.settings import config
pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 300, "display.max_colwidth", 160)
cfg = config()
now = pd.Timestamp.now(tz="UTC")

print("\n######## RACHAAD WHITE ########")
pj, _ = db.latest_board("nfl", "projections_players")
print(pj[pj["player"].str.contains("Rachaad White", na=False)].T.to_string())
pr, _ = db.latest_board("nfl", "props")
print(pr[pr["player"].str.contains("Rachaad White", na=False)].T.to_string())
s = db.read(db.odds_snapshots, "WHERE player = 'Rachaad White' AND market IN ('player_rush_yds','player_rush_attempts','player_rush_yds_alternate','player_receptions','player_anytime_td')")
if not s.empty:
    s = s[pd.to_datetime(s["fetched_at"], utc=True, format="ISO8601") > now - pd.Timedelta(hours=6)]
    s = s.sort_values("fetched_at").groupby(["book", "market", "side", "point"], as_index=False).last()
    print(s[["market", "book", "side", "point", "price", "fetched_at"]].sort_values(["market", "point", "side", "book"]).to_string(index=False))
lad, at = db.latest_board("nfl", "ladder")
print(lad[lad["player"].str.contains("Rachaad White", na=False)].to_string(index=False))
# same-team backfield
if not pj.empty:
    t = pj.loc[pj["player"].str.contains("Rachaad White", na=False), "team"]
    if len(t):
        print(pj[(pj["team"] == t.iloc[0]) & pj["position"].isin(["RB", "QB"])][["player", "position", "rush_yds", "rush_att", "receptions", "situation", "matchup_notes"]].to_string(index=False))

print(f"\n######## LADDERS (board {at}) ########")
lad = lad[pd.to_datetime(lad["kickoff"], utc=True, errors="coerce") > now]
MEGA = [2.0, 1.5, 1.0, 0.75, 0.5, 0.35, 0.25, 0.15]
for name, tmpl in (("STANDARD", None), ("MEGA", MEGA)):
    plans = []
    for (pl, stat), rows in lad.groupby(["player", "stat"]):
        ctx = rows["context"].fillna("").iloc[0]
        if "missing news" in ctx:
            continue
        p = ladder.plan(rows, cfg, template=tmpl)
        if p:
            p["n_value"] = int(rows["value"].sum())
            plans.append(p)
    plans.sort(key=lambda p: -p["ev"])
    print(f"\n==== {name} ====")
    for p in plans[:12]:
        legs = " / ".join(f"{l['rung']}+ {fmt_american(l['price'])} {l['units']:g}U (p {l['prob']:.0%})" for l in p["legs"])
        print(f"\n{p['matchup']} {p['kickoff']} | {p['player']} {p['stat']} | scale {p['scale']} risk {p['risk']:.2f}U "
              f"EV {p['ev']:+.3f}U P(profit) {p['p_profit']:.0%} value rungs {p['n_value']}\n  {legs}\n  context: {p['context']}")
        print("  outcomes: " + "; ".join(f"{o['result']} {o['prob']:.0%} {o['profit']:+.2f}U" for o in p["outcomes"]))
