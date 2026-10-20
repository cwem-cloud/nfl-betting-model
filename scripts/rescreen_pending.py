"""Re-apply the current betting rules to open picks that haven't kicked off yet.

Picks are locked when issued, so a rule change (a market switched off, the EV cap, no chasing,
no sharp-agreement bonus, CFB half stakes) doesn't touch picks already on the card. This voids
the ones the current rules would not have made and re-sizes the rest. Prices are not re-checked:
a pick that has since lost its edge to line movement is still a pick.

Safe to re-run: it only looks at picks issued before RULES_CHANGED_AT, once each.

Usage: python scripts/rescreen_pending.py [--dry-run]
"""
import json
import sys

import _path  # noqa: F401
import pandas as pd

from pocket_capper import db
from pocket_capper.settings import config

OLD_SIGNAL_BONUS = 0.25  # what agreeing sharp signals used to add per net signal (max 2)
CHASE = {"Line move", "Steam", "RLM"}
GAME = {"side", "total", "ml"}
# Week 6 rules merged (PR #8). Picks issued after this were already made under them; re-screening
# those would apply the half stake / bonus removal twice.
RULES_CHANGED_AT = pd.Timestamp("2026-10-15T22:15:00Z")
MARK = "[Re-sized for week 6 rules]"


def verdict(r, cfg) -> tuple[str | None, float]:
    """(void reason or None, new units)."""
    edges, units = cfg["edges"], cfg["units"]
    sigs = json.loads(r.signals) if isinstance(r.signals, str) and r.signals else []
    if r.category in GAME:
        scfg = cfg["sports"][r.sport]
        if r.market not in set(scfg.get("markets", ["spread", "total", "ml"])):
            return f"{r.sport.upper()} {r.market} is switched off", 0.0
        if r.ev is not None and r.ev > edges.get("max_ev_game", 1.0):
            return f"EV {r.ev:+.1%} over the {edges['max_ev_game']:.0%} cap", 0.0
        if cfg["sharp"].get("skip_chasing") and any(s["direction"] > 0 and s["name"] in CHASE for s in sigs):
            return "line had already moved our way (chasing)", 0.0
        u = float(r.units)
        net = sum(s["direction"] for s in sigs)
        if net > 0:
            u -= (OLD_SIGNAL_BONUS - units.get("sharp_signal_bonus", 0.0)) * min(net, 2)
        u *= float(scfg.get("stake_mult", 1.0))
        u = round(u * 4) / 4
        if u < units["min_units"]:
            return f"re-sized to {u:g}U, under the {units['min_units']:g}U minimum", 0.0
        return None, u
    cap = edges.get("max_ev_attd" if r.category == "attd" else "max_ev_prop", 1.0)
    if r.ev is not None and r.ev > cap:
        return f"EV {r.ev:+.1%} over the {cap:.0%} cap", 0.0
    return None, float(r.units)


def main(dry: bool) -> None:
    cfg = config()
    p = db.read(db.picks, "WHERE status = 'pending'")
    p = p[~db.off_book(p)]
    kick = pd.to_datetime(p["kickoff"], utc=True, errors="coerce", format="ISO8601")
    made = pd.to_datetime(p["created_at"], utc=True, errors="coerce", format="ISO8601")
    p = p[(kick > pd.Timestamp.now(tz="UTC")) & (made < RULES_CHANGED_AT)]
    voided = resized = 0
    for r in p.itertuples():
        why = r.rationale if isinstance(r.rationale, str) else ""
        if why.startswith(MARK):
            continue
        reason, u = verdict(r, cfg)
        if reason:
            voided += 1
            print(f"VOID   {r.sport.upper():3} {r.selection:32} {r.units:g}U  [{r.matchup}] - {reason}")
            if not dry:
                db.update_pick(r.pick_id, status="void", profit_units=0.0,
                               rationale=f"VOIDED by rule change: {reason}. " + why)
        elif abs(u - float(r.units)) > 1e-9:
            resized += 1
            print(f"RESIZE {r.sport.upper():3} {r.selection:32} {r.units:g}U -> {u:g}U  [{r.matchup}]")
            if not dry:
                db.update_pick(r.pick_id, units=u, rationale=f"{MARK} was {r.units:g}U. " + why)
        else:
            print(f"KEEP   {r.sport.upper():3} {r.selection:32} {r.units:g}U  [{r.matchup}]")
    print(f"{len(p)} open picks: {voided} voided, {resized} re-sized{' (dry run)' if dry else ''}")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
