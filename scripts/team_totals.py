"""Live Vegas implied team totals for this NFL week, ranked for D/ST streaming.

    python scripts/team_totals.py            # this week, best D/ST matchups first
    python scripts/team_totals.py --all      # every game the books have posted
    python scripts/team_totals.py --csv tt.csv

Costs 2 Odds API credits (spreads + totals, one region).
"""
import argparse

import _path  # noqa: F401

from pocket_capper.data import odds_api
from pocket_capper.market.team_totals import implied_team_totals, this_week
from pocket_capper.settings import config

ap = argparse.ArgumentParser()
ap.add_argument("--all", action="store_true", help="don't limit to this week's games")
ap.add_argument("--csv", help="also write the table to this path")
ap.add_argument("--markdown", action="store_true", help="print a Markdown table (for GitHub job summaries)")
args = ap.parse_args()

cfg = config()
odds = odds_api.game_odds(cfg["sports"]["nfl"]["odds_key"], cfg["books"]["compare"], markets="spreads,totals")
tt = implied_team_totals(odds)
if not args.all:
    tt = this_week(tt)
if tt.empty:
    raise SystemExit("No NFL games with both a spread and a total posted.")

out = tt.assign(kickoff=tt["kickoff"].dt.tz_convert("America/New_York").dt.strftime("%a %-I:%M%p"))
out["matchup"] = out.apply(lambda r: f"vs {r.opp}" if r.venue == "home" else f"@ {r.opp}", axis=1)
out["spread"] = out["spread"].map(lambda v: f"{v:+g}" if v else "PK")
out = out[["team", "matchup", "kickoff", "spread", "total", "team_total", "opp_total", "books"]]
out.columns = ["D/ST", "Matchup", "Kickoff (ET)", "Spread", "Total", "Team TT", "Opp TT", "Books"]
out.index = range(1, len(out) + 1)

if args.csv:
    tt.to_csv(args.csv, index=False)
if args.markdown:
    print("| # | " + " | ".join(out.columns) + " |\n|" + "---|" * (len(out.columns) + 1))
    for i, r in out.iterrows():
        print(f"| {i} | " + " | ".join(str(v) for v in r) + " |")
else:
    print(out.to_string())
print(f"\nConsensus = median line across books. Lowest Opp TT = best D/ST matchup. "
      f"API credits remaining: {odds_api.LAST_USAGE.get('remaining')}")
