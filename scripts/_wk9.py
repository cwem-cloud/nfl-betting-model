import _path  # noqa: F401
import pandas as pd
from pocket_capper import db
from pocket_capper.data import nflverse
pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_colwidth", 60)
ps = nflverse.player_stats([2026]); w = ps[ps["week"] == 9]
print(w[w.player_display_name.isin(["DeMario Douglas", "Jonathan Taylor", "Rachaad White", "J.K. Dobbins", "Jonah Coleman"])]
      [["player_display_name", "carries", "rushing_yards", "rushing_tds", "receptions", "receiving_yards", "receiving_tds"]].to_string(index=False))
p = db.read(db.picks)
p["kick"] = pd.to_datetime(p["kickoff"], utc=True, errors="coerce", format="ISO8601")
wk = p[(p["kick"] >= "2026-11-06") & (p["kick"] < "2026-11-11")]
for name, d in (("MODEL", wk[~db.off_book(wk)]), ("TRACKING (switched-off mkts)", wk[db.is_tracking(wk) & (wk["source"].fillna("") != "angle")])):
    g = d[d.status.isin(["win", "loss", "push"])]
    print(f"\n== {name}: {(g.status=='win').sum()}-{(g.status=='loss').sum()}-{(g.status=='push').sum()}  {g.profit_units.sum():+.2f}U  pending {int((d.status=='pending').sum())}")
    print(d.sort_values("kick")[["kick", "category", "matchup", "selection", "price", "book", "units", "result_value", "close_line", "clv_points", "clv_prob", "status", "profit_units"]].to_string(index=False))
lp = db.read(db.ladder_plans)
lp = lp[lp["kickoff"] >= "2026-11-06"]
print("\n== PAPER LADDERS"); print(lp[["matchup", "player", "stat", "risk", "status", "result_value", "profit_units"]].to_string(index=False))
