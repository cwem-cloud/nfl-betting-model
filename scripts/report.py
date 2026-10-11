"""Print the betting history (graded results) and the current card from the real database.

    python scripts/report.py            # history + plays for games in the next 36 hours
"""
import _path  # noqa: F401
import pandas as pd

from pocket_capper import db
from pocket_capper.market.odds_math import fmt_american

pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 500, "display.max_colwidth", 60)

p = db.read(db.picks)
print(f"TOTAL PICKS IN DB: {len(p)}")
if p.empty:
    raise SystemExit
p["kick"] = pd.to_datetime(p["kickoff"], utc=True, errors="coerce", format="ISO8601").dt.tz_convert("America/New_York")
p["price_s"] = p["price"].map(fmt_american)

g = p[p["status"].isin(["win", "loss", "push"])].copy()
print(f"\n===== GRADED RESULTS ({len(g)} picks) =====")
if len(g):
    def summ(d):
        w, l, pu = (d["status"] == "win").sum(), (d["status"] == "loss").sum(), (d["status"] == "push").sum()
        risk = d["units"].sum()
        pl = d["profit_units"].sum()
        clv = d["clv_prob"].dropna()
        return pd.Series({"record": f"{w}-{l}-{pu}", "units_risked": round(risk, 2), "profit_u": round(pl, 2),
                          "roi": f"{pl / risk:+.1%}" if risk else "-", "avg_clv": f"{clv.mean():+.1%}" if len(clv) else "-",
                          "beat_close": f"{(clv > 0).mean():.0%}" if len(clv) else "-"})
    print("\nOVERALL"); print(summ(g).to_string())
    print("\nBY SPORT/TYPE"); print(g.groupby(["sport", "category"]).apply(summ, include_groups=False).to_string())
    print("\nEVERY GRADED PICK")
    print(g.sort_values("kick")[["kick", "sport", "category", "matchup", "selection", "price_s", "units", "ev",
                                 "close_line", "clv_prob", "result_value", "status", "profit_units"]].to_string(index=False))
pend_past = p[(p["status"] == "pending") & (p["kick"] < pd.Timestamp.now(tz="America/New_York"))]
if len(pend_past):
    print(f"\nKICKED OFF BUT NOT YET GRADED: {len(pend_past)}")
    print(pend_past[["kick", "sport", "category", "matchup", "selection", "price_s", "units"]].to_string(index=False))
v = p[p["status"] == "void"]
if len(v):
    print(f"\nVOID (player inactive): {len(v)}")

now = pd.Timestamp.now(tz="America/New_York")
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
