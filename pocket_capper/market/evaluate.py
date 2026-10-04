"""Turn blind projections + market data into graded opportunities (plays or 'no edge')."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pocket_capper.market import market as mk
from pocket_capper.market import odds_math as om
from pocket_capper.models import distributions as dist
from pocket_capper.models import nfl_props


def _units(p_win, price, p_push, cfg, net_sig, cap=None):
    u = om.size_units(p_win, price, p_push, cfg["units"], cap)
    if u <= 0:
        return 0.0
    u += cfg["units"]["sharp_signal_bonus"] * float(np.clip(net_sig, -2, 2))
    cap = cap if cap is not None else cfg["units"]["max_units"]
    u = min(cap, round(u * 4) / 4)
    return u if u >= cfg["units"]["min_units"] else 0.0


def evaluate_game(
    *, sport: str, proj: dict, board: pd.DataFrame, snap: pd.DataFrame, event_id: str,
    splits: dict, manual: list[dict], cfg: dict,
) -> tuple[dict, list[dict]]:
    """Returns (board_row, candidate_picks)."""
    scfg = cfg["sports"][sport]
    primary = cfg["books"]["primary"]
    sharp_books = cfg["books"]["sharp"]
    msig, tsig = scfg["margin_sigma"], scfg["total_sigma"]
    mu, tot = proj["fair_margin"], proj["home_pts"] + proj["away_pts"]

    sharp_mu = mk.sharp_fair_margin(board, event_id, sharp_books, msig, sport)
    sharp_tot = mk.sharp_fair_total(board, event_id, sharp_books, tsig, sport)
    dk_sp = mk.book_line(board, event_id, primary, "spread")
    dk_tot = mk.book_line(board, event_id, primary, "total")
    dk_ml = mk.book_line(board, event_id, primary, "ml")

    row = {
        "fair_spread_home": proj["fair_spread_home"], "fair_total": proj["fair_total"],
        "home_win_prob": proj["home_win_prob"],
        "sharp_spread_home": None if sharp_mu is None else dist.margin_to_spread(sharp_mu),
        "sharp_total": None if sharp_tot is None else round(sharp_tot * 2) / 2,
        "dk_spread_home": dk_sp.get("home", (None, None))[0], "dk_spread_home_price": dk_sp.get("home", (None, None))[1],
        "dk_spread_away_price": dk_sp.get("away", (None, None))[1],
        "dk_total": dk_tot.get("over", (None, None))[0], "dk_over_price": dk_tot.get("over", (None, None))[1],
        "dk_under_price": dk_tot.get("under", (None, None))[1],
        "dk_ml_home": dk_ml.get("home", (None, None))[1], "dk_ml_away": dk_ml.get("away", (None, None))[1],
    }
    # movement + splits for the dashboard
    for side in ("home", "away"):
        mv = mk.movement(snap, event_id, primary, "spread", side)
        if mv:
            row[f"dk_open_spread_{side}"] = mv["open_point"]
        sp = splits.get(("spread", side)) or {}
        row[f"spread_bets_{side}"] = sp.get("bets_pct")
        row[f"spread_money_{side}"] = sp.get("money_pct")
    mv = mk.movement(snap, event_id, primary, "total", "over")
    if mv:
        row["dk_open_total"] = mv["open_point"]
    for side in ("over", "under"):
        sp = splits.get(("total", side)) or {}
        row[f"total_bets_{side}"] = sp.get("bets_pct")
        row[f"total_money_{side}"] = sp.get("money_pct")

    candidates = []
    edges = cfg["edges"]

    # ---- spreads
    if "home" in dk_sp and "away" in dk_sp and dk_sp["home"][0] is not None:
        home_line = float(dk_sp["home"][0])
        pm = dist.spread_probs(mu, msig, home_line, sport)
        if sharp_mu is not None:
            pmk = dist.spread_probs(sharp_mu, msig, home_line, sport)
        else:
            ph, pa = om.devig([dk_sp["home"][1], dk_sp["away"][1]])
            pmk = {"home": ph * (1 - pm["push"]), "away": pa * (1 - pm["push"]), "push": pm["push"]}
        w = scfg["model_weight_side"]
        for side in ("home", "away"):
            price = dk_sp[side][1]
            line = home_line if side == "home" else -home_line
            p_model = pm[side]
            p_mkt = pmk[side]
            p_push = w * pm["push"] + (1 - w) * pmk["push"]
            p_final = w * p_model + (1 - w) * p_mkt
            ev = om.expected_value(p_final, price, p_push)
            fair_line = -mu if side == "home" else mu
            edge_pts = line - fair_line
            sharp_line = None if sharp_mu is None else (-sharp_mu if side == "home" else sharp_mu)
            sig = mk.sharp_signals(
                side=side, market="spread", dk_point=line, sharp_point=sharp_line,
                mv=mk.movement(snap, event_id, primary, "spread", side), split=splits.get(("spread", side)),
                steam=mk.steam_moves(snap, event_id, "spread", side, cfg["sharp"]["steam_points"], cfg["sharp"]["steam_books"]),
                cfg=cfg, manual=[m for m in manual if m["market"] == "spread" and m["side"] == side]
                + [dict(m, direction=-1) for m in manual if m["market"] == "spread" and m["side"] != side],
            )
            candidates.append({
                "category": "side", "market": "spread", "side": side, "line": line, "price": price,
                "fair_line": round(fair_line * 2) / 2, "edge_pts": edge_pts,
                "model_prob": p_model, "market_prob": p_mkt, "final_prob": p_final, "push_prob": p_push,
                "ev": ev, "signals": sig,
                "qualifies": ev >= edges["min_ev_side"] and edges["min_points_side"] <= edge_pts <= edges["max_points_side"],
                "too_big": edge_pts > edges["max_points_side"],
            })
    # ---- totals
    if "over" in dk_tot and "under" in dk_tot and dk_tot["over"][0] is not None:
        tl = float(dk_tot["over"][0])
        pm = dist.total_probs(tot, tsig, tl, sport)
        if sharp_tot is not None:
            pmk = dist.total_probs(sharp_tot, tsig, tl, sport)
        else:
            po, pu = om.devig([dk_tot["over"][1], dk_tot["under"][1]])
            pmk = {"over": po * (1 - pm["push"]), "under": pu * (1 - pm["push"]), "push": pm["push"]}
        w = scfg["model_weight_total"]
        for side in ("over", "under"):
            price = dk_tot[side][1]
            p_model, p_mkt = pm[side], pmk[side]
            p_push = w * pm["push"] + (1 - w) * pmk["push"]
            p_final = w * p_model + (1 - w) * p_mkt
            ev = om.expected_value(p_final, price, p_push)
            edge_pts = (tot - tl) if side == "over" else (tl - tot)
            sig = mk.sharp_signals(
                side=side, market="total", dk_point=tl, sharp_point=sharp_tot,
                mv=mk.movement(snap, event_id, primary, "total", side), split=splits.get(("total", side)),
                steam=mk.steam_moves(snap, event_id, "total", side, cfg["sharp"]["steam_points"], cfg["sharp"]["steam_books"]),
                cfg=cfg, manual=[m for m in manual if m["market"] == "total" and m["side"] == side]
                + [dict(m, direction=-1) for m in manual if m["market"] == "total" and m["side"] != side],
            )
            candidates.append({
                "category": "total", "market": "total", "side": side, "line": tl, "price": price,
                "fair_line": proj["fair_total"], "edge_pts": edge_pts,
                "model_prob": p_model, "market_prob": p_mkt, "final_prob": p_final, "push_prob": p_push,
                "ev": ev, "signals": sig,
                "qualifies": ev >= edges["min_ev_total"] and edges["min_points_total"] <= edge_pts <= edges["max_points_total"],
                "too_big": edge_pts > edges["max_points_total"],
            })
    # ---- moneyline
    if "home" in dk_ml and "away" in dk_ml:
        p_home_model = proj["home_win_prob"]
        sharp_ml = {}
        for bk in sharp_books:
            ln = mk.book_line(board, event_id, bk, "ml")
            if "home" in ln and "away" in ln:
                sharp_ml[bk] = om.devig([ln["home"][1], ln["away"][1]])[0]
        p_home_mkt = float(np.median(list(sharp_ml.values()))) if sharp_ml else om.devig([dk_ml["home"][1], dk_ml["away"][1]])[0]
        w = scfg["model_weight_side"]
        for side in ("home", "away"):
            pm_ = p_home_model if side == "home" else 1 - p_home_model
            pk_ = p_home_mkt if side == "home" else 1 - p_home_mkt
            pf = w * pm_ + (1 - w) * pk_
            price = dk_ml[side][1]
            ev = om.expected_value(pf, price)
            sig = [s for s in mk.sharp_signals(
                side=side, market="ml", dk_point=None, sharp_point=None, mv={}, split=splits.get(("ml", side)),
                steam=[], cfg=cfg, manual=[m for m in manual if m["market"] == "ml" and m["side"] == side]
                + [dict(m, direction=-1) for m in manual if m["market"] == "ml" and m["side"] != side],
            )]
            candidates.append({
                "category": "ml", "market": "ml", "side": side, "line": None, "price": price,
                "fair_line": om.prob_to_american(pm_), "edge_pts": None,
                "model_prob": pm_, "market_prob": pk_, "final_prob": pf, "push_prob": 0.0,
                "ev": ev, "signals": sig,
                "qualifies": (ev >= edges["min_ev_ml"] and edges["ml_price_min"] <= price <= edges["ml_price_max"]
                              and abs(mu - (sharp_mu if sharp_mu is not None else mu)) <= edges["max_points_side"]),
            })

    for c in candidates:
        net = mk.net_signal(c["signals"])
        cap = cfg["units"]["max_units"]
        c["units"] = _units(c["final_prob"], c["price"], c["push_prob"], cfg, net, cap) if c["qualifies"] else 0.0
        c["net_signal"] = net
    # One play per family per game: spread and ML on the same team are the same bet.
    for family in (("spread", "ml"), ("total",)):
        fam = [c for c in candidates if c["market"] in family and c["units"] > 0]
        for c in sorted(fam, key=lambda c: -c["ev"])[1:]:
            c["units"] = 0.0
    row["flag"] = "; ".join(
        f"Model {c['edge_pts']:.1f} pts off market on {c['market']} - likely missing news, no bet"
        for c in candidates if c.get("too_big")
    ) or None
    return row, candidates


# ---------------------------------------------------------------- props

def evaluate_props(props_snap: pd.DataFrame, proj: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per (player, market) DK offer with projection, probabilities and EV."""
    from pocket_capper.data.teams import norm_player

    if props_snap.empty or proj.empty:
        return pd.DataFrame()
    primary = cfg["books"]["primary"]
    edges = cfg["edges"]
    board = props_snap.sort_values("fetched_at").groupby(
        ["event_id", "book", "market", "player", "side"], as_index=False
    ).last()
    board["key"] = board["player"].map(norm_player)
    proj = proj.copy()
    proj["key"] = proj["player"].map(norm_player)
    pidx = proj.set_index("key")
    rows = []
    for (ev_id, market, key), grp in board.groupby(["event_id", "market", "key"]):
        if key not in pidx.index or market not in nfl_props.MARKET_STAT:
            continue
        pj = pidx.loc[key]
        if isinstance(pj, pd.DataFrame):
            pj = pj.iloc[0]
        stat = nfl_props.MARKET_STAT[market]
        dk = grp[grp["book"] == primary]
        if dk.empty:
            continue
        if stat == "anytime_td":
            yes = dk[dk["side"] == "yes"]
            if yes.empty:
                continue
            price = float(yes["price"].iloc[0])
            fair = []
            for bk, b in grp.groupby("book"):
                y, n = b[b["side"] == "yes"], b[b["side"] == "no"]
                if y.empty:
                    continue
                fair.append(om.devig([y["price"].iloc[0], n["price"].iloc[0]])[0] if not n.empty
                            else om.devig_one_sided(y["price"].iloc[0]))
            p_model = float(pj["anytime_td"])
            if not fair or not np.isfinite(p_model):
                continue
            p_mkt = float(np.median(fair))
            p_final = 0.5 * p_model + 0.5 * p_mkt
            ev = om.expected_value(p_final, price)
            qualifies = ev >= edges["min_ev_attd"] and price <= edges["max_attd_odds"]
            rows.append({
                "event_id": ev_id, "player": pj["player"], "player_id": pj["player_id"], "team": pj["team"],
                "opp": pj["opp"], "position": pj["position"], "market": market, "stat": stat, "side": "yes",
                "line": None, "price": price, "projection": float(pj.get("exp_tds", np.nan)),
                "model_prob": p_model, "market_prob": p_mkt, "final_prob": p_final,
                "fair_odds": om.prob_to_american(p_final), "ev": ev, "books": len(fair),
                "units": om.size_units(p_final, price, 0, cfg["units"], cfg["units"]["max_units_attd"]) if qualifies else 0.0,
                "qualifies": qualifies,
            })
            continue
        mean = pj.get(stat)
        if mean is None or not np.isfinite(mean):
            continue
        over, under = dk[dk["side"] == "over"], dk[dk["side"] == "under"]
        if over.empty or under.empty:
            continue
        line = float(over["point"].iloc[0])
        p_over_model = nfl_props.prob_over(stat, float(mean), line)
        if not np.isfinite(p_over_model):
            continue
        fair = []
        for bk, b in grp.groupby("book"):
            o, u = b[(b["side"] == "over") & (b["point"] == line)], b[(b["side"] == "under") & (b["point"] == line)]
            if not o.empty and not u.empty:
                fair.append(om.devig([o["price"].iloc[0], u["price"].iloc[0]])[0])
        p_over_mkt = float(np.median(fair)) if fair else p_over_model
        w = 0.5
        best = None
        for side, pm_, pk_, price in (
            ("over", p_over_model, p_over_mkt, float(over["price"].iloc[0])),
            ("under", 1 - p_over_model, 1 - p_over_mkt, float(under["price"].iloc[0])),
        ):
            pf = w * pm_ + (1 - w) * pk_
            ev = om.expected_value(pf, price)
            cand = (ev, side, pm_, pk_, pf, price)
            best = cand if best is None or ev > best[0] else best
        ev, side, pm_, pk_, pf, price = best
        rel = (mean - line) / max(line, 0.5) if side == "over" else (line - mean) / max(line, 0.5)
        qualifies = ev >= edges["min_ev_prop"] and edges["min_prop_edge_pct"] <= rel <= edges["max_prop_edge_pct"]
        rows.append({
            "event_id": ev_id, "player": pj["player"], "player_id": pj["player_id"], "team": pj["team"],
            "opp": pj["opp"], "position": pj["position"], "market": market, "stat": stat, "side": side,
            "line": line, "price": price, "projection": float(mean), "model_prob": pm_, "market_prob": pk_,
            "final_prob": pf, "fair_odds": om.prob_to_american(pf), "ev": ev, "books": len(fair),
            "units": om.size_units(pf, price, 0, cfg["units"], cfg["units"]["max_units_prop"]) if qualifies else 0.0,
            "qualifies": qualifies,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["qualifies"] = out["qualifies"] & (out["units"] > 0)
    # Portfolio caps: best EV first, limited per category and per game.
    out = out.sort_values("ev", ascending=False).reset_index(drop=True)
    keep, per_game, counts = [], {}, {"attd": 0, "prop": 0}
    for i, r in out.iterrows():
        if not r["qualifies"]:
            keep.append(False)
            continue
        cat = "attd" if r["stat"] == "anytime_td" else "prop"
        limit = edges["max_attd_plays"] if cat == "attd" else edges["max_prop_plays"]
        ok = counts[cat] < limit and per_game.get(r["event_id"], 0) < edges["max_plays_per_game_props"]
        if ok:
            counts[cat] += 1
            per_game[r["event_id"]] = per_game.get(r["event_id"], 0) + 1
        keep.append(ok)
    out["qualifies"] = keep
    out.loc[~out["qualifies"], "units"] = 0.0
    return out
