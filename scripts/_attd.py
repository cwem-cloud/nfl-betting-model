import _path  # noqa: F401
import pandas as pd
from pocket_capper import db
pd.set_option("display.width", 250, "display.max_rows", 100)
a, at = db.latest_board("nfl", "attd")
pj, _ = db.latest_board("nfl", "projections_players")
print("board", at)
a = a[pd.to_datetime(a["kickoff"], utc=True, errors="coerce") > pd.Timestamp.now(tz="UTC")]
a = a.merge(pj[["player_id", "situation"]].drop_duplicates("player_id"), on="player_id", how="left")
a["unstable"] = a["situation"].fillna("").str.contains("unstable role")
cols = ["matchup", "kickoff", "player", "position", "price", "book", "model_prob", "market_prob", "final_prob", "ev", "unstable", "situation"]
print(a.sort_values("final_prob", ascending=False)[cols].head(25).to_string(index=False))
