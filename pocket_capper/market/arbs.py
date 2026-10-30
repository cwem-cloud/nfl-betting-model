"""Cross-book scanner over the books we bet: arbitrage (both sides at a guaranteed profit) and middles
(both sides at numbers far enough apart that one result wins both bets).

Informational, never auto-bet: arbs between our books are usually a stale price that disappears within
minutes, and books limit accounts that arb them. Stakes are for a $100 total outlay.
"""
from __future__ import annotations

import math

import pandas as pd

from pocket_capper.market import odds_math as om
from pocket_capper.market.market import BOOK_NAME

SIDES = {"spread": ("home", "away"), "total": ("over", "under"), "ml": ("home", "away")}
COLS = ["kind", "event_id", "matchup", "market", "player", "a", "b", "edge", "window", "stake_a", "stake_b", "detail"]


def _best(rows: pd.DataFrame) -> pd.Series | None:
    if rows.empty:
        return None
    return rows.loc[rows["price"].map(om.american_to_decimal).idxmax()]


def _leg(r: pd.Series, side: str, point) -> str:
    num = "" if point is None or pd.isna(point) else (f" {point:+g}" if r["market"] in ("spread",) else f" {point:g}")
    return f"{side.title()}{num} {om.fmt_american(r['price'])} @ {BOOK_NAME.get(r['book'], r['book'])}"


def _arb(event_id, matchup, market, player, a, b, side_a, side_b, pa, pb) -> dict | None:
    da, db_ = om.american_to_decimal(a["price"]), om.american_to_decimal(b["price"])
    s = 1 / da + 1 / db_
    if s >= 1:
        return None
    return {"kind": "arb", "event_id": event_id, "matchup": matchup, "market": market, "player": player,
            "a": _leg(a, side_a, pa), "b": _leg(b, side_b, pb), "edge": 1 / s - 1, "window": None,
            "stake_a": round(100 / da / s, 2), "stake_b": round(100 / db_ / s, 2),
            "detail": f"locks {1 / s - 1:+.2%} whatever happens"}


def _middle(event_id, matchup, market, player, a, b, side_a, side_b, pa, pb, window, min_worst) -> dict | None:
    """Equal-risk stakes; worst case = only one side wins."""
    da, db_ = om.american_to_decimal(a["price"]), om.american_to_decimal(b["price"])
    sa = 100 * db_ / (da + db_)
    sb = 100 - sa
    worst = min(sa * da, sb * db_) / 100 - 1
    if worst < min_worst:
        return None
    return {"kind": "middle", "event_id": event_id, "matchup": matchup, "market": market, "player": player,
            "a": _leg(a, side_a, pa), "b": _leg(b, side_b, pb), "edge": worst, "window": window,
            "stake_a": round(sa, 2), "stake_b": round(sb, 2),
            "detail": f"both win on {window}; otherwise {worst:+.1%}"}


def scan(board: pd.DataFrame, books: list[str], matchup_of: dict | None = None, min_worst_middle: float = -0.05,
         min_window: float = 1.0) -> pd.DataFrame:
    """board: latest price per (event_id, book, market, side[, player]) with point and price.
    Handles game markets (spread/total/ml) and over/under player props."""
    if board.empty:
        return pd.DataFrame(columns=COLS)
    b = board[board["book"].isin(books) & board["price"].notna()].copy()
    if "player" not in b:
        b["player"] = None
    b["player"] = b["player"].where(b["player"].notna(), "")
    out = []
    for (ev, market, player), g in b.groupby(["event_id", "market", "player"]):
        mu = (matchup_of or {}).get(ev, ev)
        if market in SIDES:
            sa, sb = SIDES[market]
        elif set(g["side"]) >= {"over", "under"}:
            sa, sb = "over", "under"
        else:
            continue
        A, B = g[g["side"] == sa], g[g["side"] == sb]
        if A.empty or B.empty:
            continue
        if market == "ml":
            r = _arb(ev, mu, market, player, _best(A), _best(B), sa, sb, None, None)
            if r:
                out.append(r)
            continue
        # spreads in the home team's terms: home at h, away at a covers the same game when a == -h
        ka = A["point"]
        kb = -B["point"] if market == "spread" else B["point"]
        for k in sorted(set(ka) & set(kb)):
            a, bb = _best(A[ka == k]), _best(B[kb == k])
            r = _arb(ev, mu, market, player, a, bb, sa, sb, a["point"], bb["point"])
            if r:
                out.append(r)
        # middles: over x / under y with y > x; home h / away a with h + a > 0 (e.g. -2.5 and +3.5 -> 3)
        best_mid = None
        for _, a in A.iterrows():
            for _, bb in B.iterrows():
                gap = (bb["point"] + a["point"]) if market == "spread" else (bb["point"] - a["point"])
                if gap < min_window:
                    continue
                lo, hi = (-a["point"], bb["point"]) if market == "spread" else (a["point"], bb["point"])
                hits = [n for n in range(int(math.floor(lo)) + 1, int(math.ceil(hi))) if lo < n < hi]
                if not hits:  # e.g. over 45 / under 46: each side only pushes
                    continue
                what = "home margin" if market == "spread" else "result"
                win = f"{what} {hits[0]}" if len(hits) == 1 else f"{what} {hits[0]}-{hits[-1]}"
                m = _middle(ev, mu, market, player, a, bb, sa, sb, a["point"], bb["point"], win, min_worst_middle)
                if m and (best_mid is None or (gap, m["edge"]) > best_mid[0]):
                    best_mid = ((gap, m["edge"]), m)
        if best_mid:
            out.append(best_mid[1])
    df = pd.DataFrame(out, columns=COLS)
    return df.sort_values(["kind", "edge"], ascending=[True, False]).reset_index(drop=True) if not df.empty else df
