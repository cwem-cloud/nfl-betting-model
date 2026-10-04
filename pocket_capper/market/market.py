"""Market layer: consensus/sharp fair prices, line movement, splits and sharp signals.

This runs only AFTER the blind projections exist.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pocket_capper.market import odds_math as om
from pocket_capper.models import distributions as dist


def current_board(snap: pd.DataFrame) -> pd.DataFrame:
    """Latest price per (event, book, market, side)."""
    if snap.empty:
        return snap
    s = snap.sort_values("fetched_at")
    return s.groupby(["event_id", "book", "market", "side"], as_index=False).last()


def book_line(board: pd.DataFrame, event_id: str, book: str, market: str) -> dict:
    """{side: (point, price)} for one book/market."""
    b = board[(board["event_id"] == event_id) & (board["book"] == book) & (board["market"] == market)]
    return {r.side: (r.point, r.price) for r in b.itertuples()}


def sharp_fair_margin(board: pd.DataFrame, event_id: str, sharp_books: list[str], sigma: float, sport: str) -> float | None:
    """Market-implied home margin from sharp books (de-vigged spread, price-adjusted).

    Falls back to the all-book median when no sharp book is posted.
    """
    def implied(books):
        mus = []
        for bk in books:
            ln = book_line(board, event_id, bk, "spread")
            if "home" in ln and "away" in ln and ln["home"][0] is not None:
                p_home, _ = om.devig([ln["home"][1], ln["away"][1]])
                mus.append(_solve_margin(ln["home"][0], p_home, sigma, sport))
        return float(np.median(mus)) if mus else None

    mu = implied(sharp_books)
    if mu is None:
        mu = implied(sorted(board.loc[board["event_id"] == event_id, "book"].unique()))
    return mu


def sharp_fair_total(board: pd.DataFrame, event_id: str, sharp_books: list[str], sigma: float, sport: str) -> float | None:
    def implied(books):
        mus = []
        for bk in books:
            ln = book_line(board, event_id, bk, "total")
            if "over" in ln and "under" in ln and ln["over"][0] is not None:
                p_over, _ = om.devig([ln["over"][1], ln["under"][1]])
                mus.append(_solve_total(ln["over"][0], p_over, sigma, sport))
        return float(np.median(mus)) if mus else None

    mu = implied(sharp_books)
    if mu is None:
        mu = implied(sorted(board.loc[board["event_id"] == event_id, "book"].unique()))
    return mu


def _solve_margin(home_line: float, p_home_cover: float, sigma: float, sport: str) -> float:
    """Home margin mean that makes P(home covers home_line), push-excluded, equal p_home_cover."""
    lo, hi = -60.0, 60.0
    for _ in range(40):
        mid = (lo + hi) / 2
        pr = dist.spread_probs(mid, sigma, home_line, sport)
        p = pr["home"] / max(1e-9, pr["home"] + pr["away"])
        lo, hi = (mid, hi) if p < p_home_cover else (lo, mid)
    return (lo + hi) / 2


def _solve_total(line: float, p_over: float, sigma: float, sport: str) -> float:
    lo, hi = 10.0, 110.0
    for _ in range(40):
        mid = (lo + hi) / 2
        pr = dist.total_probs(mid, sigma, line, sport)
        p = pr["over"] / max(1e-9, pr["over"] + pr["under"])
        lo, hi = (mid, hi) if p < p_over else (lo, mid)
    return (lo + hi) / 2


# ---------------------------------------------------------------- movement

def movement(snap: pd.DataFrame, event_id: str, book: str, market: str, side: str) -> dict:
    s = snap[(snap["event_id"] == event_id) & (snap["book"] == book) & (snap["market"] == market) & (snap["side"] == side)]
    if s.empty:
        return {}
    s = s.sort_values("fetched_at")
    o, c = s.iloc[0], s.iloc[-1]
    return {
        "open_point": o["point"], "open_price": o["price"], "open_at": o["fetched_at"],
        "cur_point": c["point"], "cur_price": c["price"], "n": len(s),
    }


def steam_moves(snap: pd.DataFrame, event_id: str, market: str, side: str, pts: float, min_books: int) -> list[dict]:
    """Snapshot intervals where >= min_books moved the same direction by >= pts."""
    s = snap[(snap["event_id"] == event_id) & (snap["market"] == market) & (snap["side"] == side)]
    if s.empty or s["fetched_at"].nunique() < 2:
        return []
    # only changes are stored, so carry each book's last number forward before differencing
    piv = s.pivot_table(index="fetched_at", columns="book", values="point", aggfunc="last").sort_index().ffill()
    d = piv.diff()
    out = []
    for ts, row in d.iloc[1:].iterrows():
        down = int((row <= -pts).sum())
        up = int((row >= pts).sum())
        if max(down, up) >= min_books:
            out.append({"at": ts, "direction": "down" if down > up else "up", "books": max(down, up)})
    return out


# ---------------------------------------------------------------- signals

def sharp_signals(
    *,
    side: str,
    market: str,
    dk_point: float | None,
    sharp_point: float | None,
    mv: dict,
    split: dict | None,
    steam: list[dict],
    cfg: dict,
    manual: list[dict],
) -> list[dict]:
    """Return signals as dicts {name, detail, direction (+1 agrees with `side`, -1 against)}.

    Conventions: spreads are stored from the side's own perspective (side -3 means
    that side lays 3). A side 'gets better' when its number goes up.
    """
    sig = []
    sc = cfg["sharp"]
    # 1) DK stale vs sharp books
    if dk_point is not None and sharp_point is not None and market in ("spread", "total"):
        gap = dk_point - sharp_point
        if market == "total":
            gap = gap if side == "under" else -gap
        if abs(gap) >= sc["stale_dk_points"]:
            sig.append({
                "name": "DK off-market",
                "detail": f"DK {dk_point:+g} vs sharp {sharp_point:+g}" if market == "spread" else f"DK {dk_point:g} vs sharp {sharp_point:g}",
                "direction": 1 if gap > 0 else -1,
            })
    # 2) Line movement since open (side perspective)
    if mv and mv.get("open_point") is not None and mv.get("cur_point") is not None:
        delta = mv["cur_point"] - mv["open_point"]
        if market == "total":
            delta = -delta if side == "over" else delta
            moved_toward = delta < 0  # number moved in favor of this side's backers => money on this side
        else:
            moved_toward = delta < 0  # side's number got worse => money came in on it
        if abs(delta) >= 0.5:
            sig.append({
                "name": "Line move",
                "detail": f"open {mv['open_point']:g} -> now {mv['cur_point']:g}",
                "direction": 1 if moved_toward else -1,
            })
            # 3) Reverse line movement: public on this side but the line moved away from it
            if split and split.get("bets_pct") is not None:
                if split["bets_pct"] >= sc["rlm_bets_pct"] and not moved_toward:
                    sig.append({"name": "RLM vs public", "detail": f"{split['bets_pct']:.0f}% of bets here but line moved away", "direction": -1})
                if split["bets_pct"] <= 100 - sc["rlm_bets_pct"] and moved_toward:
                    sig.append({"name": "RLM", "detail": f"only {split['bets_pct']:.0f}% of bets but line moved toward this side", "direction": 1})
    # 4) Money vs tickets
    if split and split.get("bets_pct") is not None and split.get("money_pct") is not None:
        gap = split["money_pct"] - split["bets_pct"]
        if gap >= sc["money_vs_bets_gap"]:
            sig.append({"name": "Big-money side", "detail": f"{split['money_pct']:.0f}% $ vs {split['bets_pct']:.0f}% bets", "direction": 1})
        elif gap <= -sc["money_vs_bets_gap"]:
            sig.append({"name": "Public-ticket side", "detail": f"{split['bets_pct']:.0f}% bets vs {split['money_pct']:.0f}% $", "direction": -1})
    # 5) Steam
    for st in steam[-2:]:
        toward = st["direction"] == "down"
        if market == "total":
            toward = (st["direction"] == "up") == (side == "over")
        sig.append({"name": "Steam", "detail": f"{st['books']} books moved together", "direction": 1 if toward else -1})
    # 6) Manual sharp plays
    for m in manual:
        sig.append({"name": f"Sharp: {m['source']}", "detail": m.get("detail", ""), "direction": m["direction"]})
    return sig


def net_signal(signals: list[dict]) -> int:
    return int(sum(s["direction"] for s in signals))
