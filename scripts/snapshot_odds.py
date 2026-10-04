"""Cheap odds-only snapshot (game lines, all books) to build line-movement history between runs."""
import _path  # noqa: F401
import pandas as pd

from pocket_capper import db
from pocket_capper.data import odds_api, splits
from pocket_capper.settings import config

cfg = config()
for sport in ("nfl", "cfb"):
    try:
        df = odds_api.game_odds(cfg["sports"][sport]["odds_key"], cfg["books"]["compare"])
        if not df.empty:
            df = df[pd.to_datetime(df["commence_time"], utc=True) > pd.Timestamp.now(tz="UTC")]
        df["sport"] = sport
        db.insert_snapshots(df)
        sp = splits.fetch_dk_splits(sport)
        db.insert_df(db.splits, sp)
        print(f"{sport}: {len(df)} prices, {len(sp)} split rows; API remaining {odds_api.LAST_USAGE.get('remaining')}")
    except Exception as e:
        print(f"{sport}: snapshot failed: {e}")

db.compact()
