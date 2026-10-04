import pandas as pd


def snap_rows(ts, home_point, price):
    return pd.DataFrame([
        {"sport": "nfl", "event_id": "e1", "book": "draftkings", "market": "spread", "side": "home",
         "point": home_point, "price": price, "fetched_at": ts},
        {"sport": "nfl", "event_id": "e1", "book": "draftkings", "market": "spread", "side": "away",
         "point": -home_point, "price": price, "fetched_at": ts},
    ])
