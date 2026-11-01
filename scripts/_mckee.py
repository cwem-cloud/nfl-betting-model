import _path  # noqa: F401
import pandas as pd
from pocket_capper import db
pd.set_option("display.width", 250, "display.max_rows", 200)
s = db.read(db.odds_snapshots, "WHERE player LIKE '%McKee%'")
s = s.sort_values("fetched_at").groupby(["book", "market", "side", "point"], as_index=False).last()
print(s[["book", "market", "side", "point", "price", "fetched_at"]].sort_values(["market", "point", "side", "book"]).to_string(index=False))
pr, at = db.latest_board("nfl", "props")
print(at); print(pr[pr["player"].str.contains("McKee", na=False)].to_string(index=False))
