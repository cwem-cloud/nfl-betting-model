import _path  # noqa: F401
import pandas as pd
from pocket_capper import db
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 200)
p = db.read(db.picks, "WHERE sport = 'nfl' AND category = 'total'")
print(p[["created_at", "matchup", "selection", "line", "price", "units", "ev", "model_prob", "market_prob", "fair_line",
         "close_line", "clv_points", "result_value", "status", "profit_units", "tracking", "source"]].sort_values("created_at").to_string(index=False))
for r in p.sort_values("created_at").itertuples():
    print("\n###", r.matchup, r.selection, r.status, "\n", (r.rationale or "")[:900])
