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


def main_over_prices(props_snap: pd.DataFrame, book: str) -> dict:
    """(event_id, player key, stat) -> DK price on the Over of the main line (the ladder's anchor)."""
    from pocket_capper.data.teams import norm_player

    if props_snap.empty:
        return {}
    s = props_snap[(props_snap["book"] == book) & (props_snap["side"] == "over")
                   & props_snap["market"].isin(nfl_props.MARKET_STAT)]
    s = s.sort_values("fetched_at").groupby(["event_id", "market", "player"], as_index=False).last() \
        if "fetched_at" in s else s
    return {(r.event_id, norm_player(r.player), nfl_props.MARKET_STAT[r.market]): float(r.price)
            for r in s.itertuples() if pd.notna(r.price)}


def build(props_eval: pd.DataFrame, alt: pd.DataFrame, proj: pd.DataFrame, def_out: dict, cfg: dict,
          main_over: dict | None = None) -> pd.DataFrame:
    """One row per (player, stat, rung). kind = "anchor" for the main line's Over, "rung" for alt lines above it."""
    from pocket_capper.data.teams import norm_player

    if props_eval.empty:
        return pd.DataFrame()
    lcfg = cfg["props"].get("ladder", {})
    min_ev, max_ev, min_p = lcfg.get("min_ev", 0.05), lcfg.get("max_ev", 0.35), lcfg.get("min_prob", 0.04)
    hold = lcfg.get("alt_hold", 0.07)  # one-sided alt prices: strip roughly this much margin
    max_gap = cfg["edges"].get("max_prop_edge_pct", 0.35)
    primary = cfg["books"]["primary"]
    k_by_stat = cfg["props"]["model_weight"]
    situation = {}
    if not proj.empty and "situation" in proj:
        situation = {r.player_id: r.situation for r in proj.itertuples() if isinstance(r.situation, str) and r.situation}
    if not proj.empty and "matchup_up" in proj:  # matchup factors that favour the over (see models.matchups)
        for r in proj.itertuples():
            if isinstance(r.matchup_up, str) and r.matchup_up:
                situation[r.player_id] = "; ".join(filter(None, [situation.get(r.player_id, ""), r.matchup_up]))

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
        # same guard as the main props: a projection this far from the market usually means missing news
        # (e.g. a starter returning from injury whose fill-in's numbers the model is still projecting)
        far_off = abs(r.projection - m_mkt) / max(m_mkt, 0.5) > max_gap \
            or nfl_props.UNSTABLE_ROLE in situation.get(r.player_id, "")
        ctx = "; ".join(filter(None, [situation.get(r.player_id, ""),
                                       defense_note(r.stat, r.position, def_out.get(r.opp, [])),
                                       "model far off the market - likely missing news, not flagged" if far_off else ""]))
        base = {
            "event_id": r.event_id, "game_id": getattr(r, "game_id", None), "matchup": getattr(r, "matchup", None),
            "kickoff": getattr(r, "kickoff", None), "player": r.player, "player_id": r.player_id, "team": r.team,
            "opp": r.opp, "position": r.position, "stat": r.stat, "market": getattr(r, "market", None),
            "main_line": r.line, "market_mean": m_mkt, "projection": r.projection, "blended_mean": m, "context": ctx,
        }
        anchor_price = (main_over or {}).get((r.event_id, norm_player(r.player), r.stat))
        if anchor_price is not None:
            # the main line: our market-anchored probability is exactly the main prop's (shape-consistent)
            p = float(nfl_props.prob_over(r.stat, m, r.line))
            if np.isfinite(p):
                ev = om.expected_value(p, anchor_price)
                rows.append({**base, "kind": "anchor", "rung": int(round(r.line + 0.5)), "point": float(r.line),
                             "model_prob": p, "fair_odds": om.prob_to_american(p), "dk_price": anchor_price,
                             "ev": ev, "value": False})
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
                **base, "kind": "rung", "rung": int(round(pt + 0.5)), "point": pt, "model_prob": p,
                "fair_odds": om.prob_to_american(p), "dk_price": price, "ev": ev,
                "value": bool(price is not None and not far_off and min_ev <= ev <= max_ev),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ladder builder

DEFAULT_TEMPLATE = [2.0, 1.5, 1.0, 0.5, 0.25]
PAPER_MIN_EV = 0.005  # paper ladders need at least +0.5% of the units risked (not a +0.00U coin flip)
RUNG_STEP = {"receptions": 1, "rec_yds": 5, "rush_yds": 5, "pass_yds": 25}


def plan(rows: pd.DataFrame, cfg: dict, template: list[float] | None = None, scale: float | None = None) -> dict | None:
    """Stake a ladder on one player-stat: the anchor (main-line Over) plus the next DK-priced rungs, sized by the
    template (default 2 / 1.5 / 1 / 0.5 / 0.25U). scale=None picks it from the edge: full size only when the anchor
    itself is a model-grade play, half when the whole ladder is +EV, quarter otherwise (lottery size).
    Returns legs, the payout for every possible finishing rung, and the expected value."""
    lcfg = cfg["props"].get("ladder", {})
    template = template or lcfg.get("template", DEFAULT_TEMPLATE)
    r = rows[rows["dk_price"].notna()].sort_values("point")
    anchor = r[r["kind"] == "anchor"]
    if anchor.empty:
        return None
    step = RUNG_STEP.get(str(anchor["stat"].iloc[0]), 1)
    chosen, last = [anchor.head(1)], float(anchor["point"].iloc[0])
    for _, rr in r[r["kind"] == "rung"].iterrows():  # each tier at least one real step above the last
        if len(chosen) == len(template):
            break
        if rr["point"] >= last + step - 1e-9:
            chosen.append(rr.to_frame().T)
            last = float(rr["point"])
    picked = pd.concat(chosen)
    if len(picked) < 2:
        return None
    picked = picked.astype({"point": float, "dk_price": float, "model_prob": float, "rung": int})
    probs = list(picked["model_prob"].astype(float))
    for i in range(1, len(probs)):  # a higher rung can't be likelier than a lower one
        probs[i] = min(probs[i], probs[i - 1])

    def build_legs(sc):
        legs = []
        for i, (row, p) in enumerate(zip(picked.itertuples(), probs)):
            units = round(template[i] * sc * 100) / 100
            dec = om.american_to_decimal(row.dk_price)
            legs.append({"tier": i + 1, "rung": int(row.rung), "point": float(row.point), "price": float(row.dk_price),
                         "units": units, "win": units * (dec - 1), "prob": p, "ev": units * (p * dec - 1)})
        return legs

    full = build_legs(1.0)
    if scale is None:
        anchor_ev = float(anchor["ev"].iloc[0])
        scale = 1.0 if anchor_ev >= cfg["edges"].get("min_ev_prop", 0.04) else (
            0.5 if sum(l["ev"] for l in full) > 0 else 0.25)
    legs = build_legs(scale)
    risk = sum(l["units"] for l in legs)
    outcomes = [{"result": f"under {legs[0]['rung']}", "prob": 1 - probs[0], "profit": -risk}]
    for k, leg in enumerate(legs):
        p_here = probs[k] - (probs[k + 1] if k + 1 < len(legs) else 0.0)
        profit = sum(l["win"] for l in legs[: k + 1]) - sum(l["units"] for l in legs[k + 1:])
        label = f"{leg['rung']}+" if k + 1 == len(legs) else (
            f"{leg['rung']}" if legs[k + 1]["rung"] - leg["rung"] == 1 else f"{leg['rung']}-{legs[k + 1]['rung'] - 1}")
        outcomes.append({"result": label, "prob": p_here, "profit": profit})
    first = picked.iloc[0]
    return {
        "player": first["player"], "player_id": first["player_id"], "stat": first["stat"], "market": first.get("market"),
        "event_id": first["event_id"], "game_id": first.get("game_id"), "matchup": first.get("matchup"),
        "kickoff": first.get("kickoff"), "context": first.get("context") or "", "scale": scale, "legs": legs,
        "outcomes": outcomes, "risk": risk, "ev": sum(l["ev"] for l in legs),
        "p_profit": sum(o["prob"] for o in outcomes if o["profit"] > 0),
    }


def settle(legs: list[dict], value: float) -> float:
    """Profit in units of a ladder given the player's final stat."""
    return sum(l["win"] if value > l["point"] else -l["units"] for l in legs)



def pick_paper_ladders(lad: pd.DataFrame, cfg: dict) -> list[dict]:
    """The model's suggested ladder for each game: the best +EV plan among players with a situation behind them
    or a value rung (never one the far-off-market guard caught). At most one per game."""
    if lad.empty or "kind" not in lad:
        return []
    out = []
    for ev_id, g in lad.groupby("event_id"):
        best = None
        for _, rows in g.groupby(["player", "stat"]):
            ctx = rows["context"].fillna("").iloc[0]
            if "missing news" in ctx or nfl_props.UNSTABLE_ROLE in ctx or not (ctx or rows["value"].any()):
                continue
            p = plan(rows, cfg)
            if p and p["ev"] >= PAPER_MIN_EV * p["risk"] and (best is None or p["ev"] > best["ev"]):
                best = p
        if best:
            out.append(best)
    return out
