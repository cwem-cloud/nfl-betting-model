"""Slice graded picks by factor to see what's working. Small samples: read as hints, not proof."""
import json

import _path  # noqa: F401
import numpy as np
import pandas as pd

from pocket_capper import db

pd.set_option("display.width", 220, "display.max_columns", 20)
p = db.read(db.picks)
g = p[p["status"].isin(["win", "loss", "push"])].copy()
print(f"graded picks: {len(g)}")
if g.empty:
    raise SystemExit


def tbl(by, d=g):
    def f(x):
        w, l = (x["status"] == "win").sum(), (x["status"] == "loss").sum()
        return pd.Series({"n": len(x), "rec": f"{w}-{l}", "units": round(x["profit_units"].sum(), 2),
                          "roi": f"{x['profit_units'].sum() / x['units'].sum():+.0%}",
                          "avg_ev": f"{x['ev'].mean():+.1%}", "avg_price": int(x["price"].median())})
    return d.groupby(by).apply(f, include_groups=False)


sel = g["selection"].str.lower()
g["direction"] = np.where(sel.str.contains(" under "), "under", np.where(sel.str.contains(" over "), "over",
                 np.where(g["category"] == "attd", "td-yes", np.where(sel.str.startswith("under"), "under",
                 np.where(sel.str.startswith("over"), "over", "side")))))
g["stat"] = g["market"].str.replace("player_", "")
g["price_bucket"] = pd.cut(g["price"], [-1000, -125, -105, 105, 150, 10000],
                           labels=["<= -125", "-125..-105", "pick'em", "+105..+150", "> +150"])
g["ev_bucket"] = pd.cut(g["ev"], [0, 0.04, 0.06, 0.08, 1], labels=["<4%", "4-6%", "6-8%", "8%+"])
sig = g["signals"].map(lambda s: json.loads(s) if isinstance(s, str) and s else [])
g["net_signals"] = sig.map(lambda L: sum(x["direction"] for x in L))
g["signal_view"] = np.where(g["net_signals"] > 0, "sharps agree", np.where(g["net_signals"] < 0, "sharps against", "no signal"))
g["edge_at_close"] = np.where(g["latest_ev"] > 0, "edge held", "edge gone by kickoff")
k = pd.to_datetime(g["kickoff"], utc=True, errors="coerce", format="ISO8601").dt.tz_convert("America/New_York")
g["slot"] = np.where(k.dt.dayofweek == 6, np.where(k.dt.hour < 16, "Sun early", np.where(k.dt.hour < 19, "Sun late", "Sun night")),
                     k.dt.day_name())

for title, by, d in [
    ("BY SPORT/TYPE", ["sport", "category"], g),
    ("PROPS: OVER vs UNDER", ["direction"], g[g["category"] == "prop"]),
    ("PROPS: BY STAT", ["stat"], g[g["category"] == "prop"]),
    ("GAME LINES: DIRECTION", ["sport", "direction"], g[g["category"].isin(["side", "total", "ml"])]),
    ("BY PRICE", ["price_bucket"], g),
    ("BY MODEL EV", ["ev_bucket"], g),
    ("BY SHARP SIGNALS (game lines)", ["signal_view"], g[g["category"].isin(["side", "total", "ml"])]),
    ("EDGE STILL THERE AT LAST RUN?", ["edge_at_close"], g[g["latest_ev"].notna()]),
    ("BY TIME SLOT (NFL)", ["slot"], g[g["sport"] == "nfl"]),
]:
    if len(d):
        print(f"\n== {title} ==")
        print(tbl(by, d).to_string())
