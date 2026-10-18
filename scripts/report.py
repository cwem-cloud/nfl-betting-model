"""Print the betting history (graded results) and the current card from the real database.

    python scripts/report.py            # history + plays for games in the next 36 hours
"""
import _path  # noqa: F401
import pandas as pd

from pocket_capper import db
from pocket_capper.engine.grading import clv_summary
from pocket_capper.market.odds_math import fmt_american
from pocket_capper.settings import config

pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 500, "display.max_colwidth", 60)

p = db.read(db.picks)
print(f"TOTAL PICKS IN DB: {len(p)}")
if p.empty:
    raise SystemExit
trk = p[db.is_tracking(p)].copy()  # switched-off markets, logged but not bet
p = p[~db.is_tracking(p)].copy()
p["kick"] = pd.to_datetime(p["kickoff"], utc=True, errors="coerce", format="ISO8601").dt.tz_convert("America/New_York")
p["price_s"] = p["price"].map(fmt_american)

g = p[p["status"].isin(["win", "loss", "push"])].copy()
print(f"\n===== GRADED RESULTS ({len(g)} picks) =====")
if len(g):
    def summ(d):
        w, l, pu = (d["status"] == "win").sum(), (d["status"] == "loss").sum(), (d["status"] == "push").sum()
        risk = d["units"].sum()
        pl = d["profit_units"].sum()
        c = clv_summary(d)
        return pd.Series({"record": f"{w}-{l}-{pu}", "units_risked": round(risk, 2), "profit_u": round(pl, 2),
                          "roi": f"{pl / risk:+.1%}" if risk else "-",
                          "clv_pts_avg": f"{c['avg_pts']:+.2f}" if c["n"] else "-",
                          "beat_close": f"{c['beat']:.0%}" if c["n"] else "-",
                          "worse_than_close": f"{c['worse']:.0%}" if c["n"] else "-"})
    print("\nOVERALL"); print(summ(g).to_string())
    print("\nBY SPORT/TYPE"); print(g.groupby(["sport", "category"]).apply(summ, include_groups=False).to_string())
    print("\nEVERY GRADED PICK")
    print(g.sort_values("kick")[["kick", "sport", "category", "matchup", "selection", "price_s", "units", "ev",
                                 "close_line", "clv_points", "clv_prob", "result_value", "status", "profit_units"]].to_string(index=False))
pend_past = p[(p["status"] == "pending") & (p["kick"] < pd.Timestamp.now(tz="America/New_York"))]
if len(pend_past):
    print(f"\nKICKED OFF BUT NOT YET GRADED: {len(pend_past)}")
    print(pend_past[["kick", "sport", "category", "matchup", "selection", "price_s", "units"]].to_string(index=False))
v = p[p["status"] == "void"]
if len(v):
    print(f"\nVOID (inactive player or rule change): {len(v)}")

now = pd.Timestamp.now(tz="America/New_York")
allopen = p[(p["status"] == "pending") & (p["kick"] >= now)].sort_values("kick")
print(f"\n===== ALL OPEN PICKS ({len(allopen)}, {allopen['units'].sum():g}U) =====")
if not allopen.empty:
    print(allopen[["kick", "sport", "category", "matchup", "selection", "price_s", "units", "ev"]].to_string(index=False))
# Re-check every open play at the CURRENT DK line/price (latest_* are refreshed on each run, with the
# latest injuries). Only plays that would still qualify today stay on this card.
edges = config()["edges"]
MIN_EV = {"side": edges["min_ev_side"], "total": edges["min_ev_total"], "ml": edges["min_ev_ml"],
          "prop": edges["min_ev_prop"], "attd": edges["min_ev_attd"]}
MAX_EV = {"side": edges["max_ev_game"], "total": edges["max_ev_game"], "ml": edges["max_ev_game"],
          "prop": edges["max_ev_prop"], "attd": edges["max_ev_attd"]}
if not allopen.empty:
    a = allopen.copy()
    a["now_line"] = a["latest_line"].fillna(a["line"])
    a["now_price"] = a["latest_price"].fillna(a["price"])
    a["edge_now"] = a["latest_ev"]
    ok = (a["edge_now"] >= a["category"].map(MIN_EV)) & (a["edge_now"] <= a["category"].map(MAX_EV))
    ok &= ~((a["category"] == "attd") & (a["now_price"] > edges["max_attd_odds"]))
    live, gone = a[ok], a[~ok]
    print(f"\n===== CARD AT CURRENT LINES ({len(live)} still qualify, {live['units'].sum():g}U) =====")
    cols = ["kick", "category", "matchup", "selection", "now_line", "now_price_s", "units", "edge_now_s"]
    for d in (live, gone):
        d["now_price_s"] = d["now_price"].map(fmt_american)
        d["edge_now_s"] = d["edge_now"].map(lambda x: "" if pd.isna(x) else f"{x:+.1%}")
    if len(live):
        print(live[cols].to_string(index=False))
    print(f"\n----- edge gone at current lines, skip if not already bet ({len(gone)}) -----")
    if len(gone):
        print(gone[cols].to_string(index=False))
up = p[(p["status"] == "pending") & (p["kick"] >= now) & (p["kick"] <= now + pd.Timedelta(hours=36))].copy()
print(f"\n===== CARD: PLAYS FOR GAMES IN THE NEXT 36 HOURS ({len(up)}) =====")
if len(up):
    up["edge_now"] = up["latest_ev"].map(lambda x: "" if pd.isna(x) else f"{x:+.1%}")
    for (sport, cat), grp in up.sort_values(["sport", "category", "units", "ev"], ascending=[True, True, False, False]).groupby(["sport", "category"], sort=False):
        print(f"\n--- {sport.upper()} {cat} ---")
        print(grp[["kick", "matchup", "selection", "price_s", "units", "ev", "final_prob", "edge_now"]].to_string(index=False))
    print("\nRATIONALES (top 12 by units)")
    for _, r in up.sort_values(["units", "ev"], ascending=False).head(12).iterrows():
        print(f"\n### {r['selection']} {r['price_s']} ({r['units']:g}U) - {r['matchup']}")
        print(r["rationale"])

if len(trk):
    trk["kick"] = pd.to_datetime(trk["kickoff"], utc=True, errors="coerce", format="ISO8601").dt.tz_convert("America/New_York")
    trk["price_s"] = trk["price"].map(fmt_american)
    tg = trk[trk["status"].isin(["win", "loss", "push"])]
    c = clv_summary(trk)
    print(f"\n===== TRACKING ONLY, NOT BET ({len(trk)} picks, {len(tg)} graded) =====")
    print(f"record {(tg['status'] == 'win').sum()}-{(tg['status'] == 'loss').sum()}-{(tg['status'] == 'push').sum()}"
          f"  units if bet {tg['profit_units'].sum():+.2f}"
          + (f"  CLV pts {c['avg_pts']:+.2f}  beat close {c['beat']:.0%}" if c["n"] else ""))
    print(trk.sort_values("kick")[["kick", "sport", "category", "matchup", "selection", "price_s", "units", "ev",
                                   "close_line", "clv_points", "status", "profit_units"]].to_string(index=False))
