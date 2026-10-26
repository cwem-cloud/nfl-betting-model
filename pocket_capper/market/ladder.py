"""Alt-line ladders: fair odds for every rung, DK's alt prices, and the situation behind each player.

Each player-stat starts from the same market-anchored number as the main prop (market-implied mean
moved toward our projection by the stat's model weight). Where DK posts an alt price, the rung is priced
as DK's own (de-vigged) probability scaled by how much more or less likely our number makes it than the
market's number does, under the same distribution. Our distribution's shape cancels out of that ratio,
so a rung only shows value when our projection genuinely differs from the market - pricing rungs off our
shape alone flagged ~530 "value" rungs a week, nearly all of them shape artefacts on long-shot tails.
A rung is flagged as value only when DK pays more than fair by `min_ev`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pocket_capper.market import evaluate
from pocket_capper.market import odds_math as om
from pocket_capper.models import nfl_props

ALT_MARKET_STAT = {
    "player_reception_yds_alternate": "rec_yds",
    "player_receptions_alternate": "receptions",
    "player_rush_yds_alternate": "rush_yds",
    "player_pass_yds_alternate": "pass_yds",
}
# rungs priced even when DK has no alt line posted ("X+" means over X - 0.5)
DEFAULT_RUNGS = {
    "rec_yds": [40, 50, 60, 70, 80, 90, 100, 125],
    "rush_yds": [40, 50, 60, 70, 80, 90, 100, 125],
    "pass_yds": [200, 225, 250, 275, 300, 325, 350],
    "receptions": [3, 4, 5, 6, 7, 8, 9],
}
RECEIVERS, RUSHERS = {"WR", "TE", "RB"}, {"RB", "QB"}


def defense_note(stat: str, position: str, absences: list[dict]) -> str:
    """Opposing defensive starters out that matter for this stat."""
    want = "pass" if stat in ("rec_yds", "receptions", "pass_yds") else "run"
    hits = [a for a in absences if a["group"] == want]
    if not hits:
        return ""
    names = ", ".join(f"{a['pos']} {a['name']} ({a['snap_pct']:.0%} snaps)" + (" - also missed last game" if a["already_missed"] else "")
                      for a in hits)
    return f"opp defense without {names}"


def build(props_eval: pd.DataFrame, alt: pd.DataFrame, proj: pd.DataFrame, def_out: dict, cfg: dict) -> pd.DataFrame:
    """One row per (player, stat, rung)."""
    from pocket_capper.data.teams import norm_player

    if props_eval.empty:
        return pd.DataFrame()
    lcfg = cfg["props"].get("ladder", {})
    min_ev, max_ev, min_p = lcfg.get("min_ev", 0.05), lcfg.get("max_ev", 0.35), lcfg.get("min_prob", 0.04)
    hold = lcfg.get("alt_hold", 0.07)  # one-sided alt prices: strip roughly this much margin
    primary = cfg["books"]["primary"]
    k_by_stat = cfg["props"]["model_weight"]
    situation = {}
    if not proj.empty and "situation" in proj:
        situation = {r.player_id: r.situation for r in proj.itertuples() if isinstance(r.situation, str) and r.situation}

    dk_alt = {}
    if not alt.empty:
        a = alt[(alt["book"] == primary) & (alt["side"] == "over") & alt["market"].isin(ALT_MARKET_STAT)].copy()
        a["key"] = a["player"].map(norm_player)
        a["stat"] = a["market"].map(ALT_MARKET_STAT)
        for (ev_id, key, stat), g in a.groupby(["event_id", "key", "stat"]):
            dk_alt[(ev_id, key, stat)] = dict(zip(g["point"].astype(float), g["price"].astype(float)))

    rows = []
    b = props_eval[props_eval["stat"].isin(DEFAULT_RUNGS) & props_eval["line"].notna()]
    for r in b.itertuples():
        p_over_mkt = r.market_prob if r.side == "over" else 1 - r.market_prob
        m_mkt = evaluate._implied_mean(r.stat, r.line, p_over_mkt)
        m = m_mkt + k_by_stat.get(r.stat, 0.15) * (r.projection - m_mkt)
        offers = dk_alt.get((r.event_id, norm_player(r.player), r.stat), {})
        points = sorted({float(p) for p in offers if p > r.line} | {x - 0.5 for x in DEFAULT_RUNGS[r.stat] if x - 0.5 > r.line})
        ctx = "; ".join(filter(None, [situation.get(r.player_id, ""),
                                       defense_note(r.stat, r.position, def_out.get(r.opp, []))]))
        for pt in points:
            p_ours = nfl_props.prob_over(r.stat, m, pt)
            if not np.isfinite(p_ours) or p_ours < min_p:
                continue
            price = offers.get(pt)
            p = p_ours
            if price is not None:
                p_shape = nfl_props.prob_over(r.stat, m_mkt, pt)  # same shape, market's number
                if np.isfinite(p_shape) and p_shape > 0:
                    p = float(np.clip(om.american_to_prob(price) / (1 + hold) * p_ours / p_shape, 0.001, 0.99))
            ev = om.expected_value(p, price) if price is not None else np.nan
            rows.append({
                "event_id": r.event_id, "matchup": getattr(r, "matchup", None), "kickoff": getattr(r, "kickoff", None),
                "player": r.player, "player_id": r.player_id, "team": r.team, "opp": r.opp, "position": r.position,
                "stat": r.stat, "main_line": r.line, "market_mean": m_mkt, "projection": r.projection, "blended_mean": m,
                "rung": int(round(pt + 0.5)), "point": pt, "model_prob": p, "fair_odds": om.prob_to_american(p),
                "dk_price": price, "ev": ev,
                "value": bool(price is not None and min_ev <= ev <= max_ev),
                "context": ctx,
            })
    return pd.DataFrame(rows)
