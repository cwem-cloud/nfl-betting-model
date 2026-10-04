"""Odds conversions, de-vigging, EV and Kelly sizing."""
from __future__ import annotations

import math
from typing import Iterable


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)


def american_to_prob(odds: float) -> float:
    """Implied probability including vig."""
    return 1 / american_to_decimal(odds)


def prob_to_american(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def fmt_american(odds: float | None) -> str:
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return "—"
    odds = int(round(odds))
    return f"+{odds}" if odds > 0 else str(odds)


def devig(prices: Iterable[float]) -> list[float]:
    """Multiplicative de-vig of a set of mutually exclusive outcomes (American odds)."""
    raw = [american_to_prob(p) for p in prices]
    s = sum(raw)
    return [r / s for r in raw]


def devig_one_sided(price: float, typical_hold: float = 0.07) -> float:
    """Fair probability when only one side is offered (e.g. anytime TD 'Yes').

    One-sided props carry heavy juice; strip a typical hold proportionally.
    """
    return american_to_prob(price) / (1 + typical_hold)


def expected_value(p_win: float, american: float, p_push: float = 0.0) -> float:
    """EV per unit risked."""
    dec = american_to_decimal(american)
    p_loss = max(0.0, 1 - p_win - p_push)
    return p_win * (dec - 1) - p_loss


def kelly_fraction(p_win: float, american: float, p_push: float = 0.0) -> float:
    """Full Kelly stake as a fraction of bankroll (pushes return the stake)."""
    b = american_to_decimal(american) - 1
    p_loss = max(0.0, 1 - p_win - p_push)
    if b <= 0:
        return 0.0
    f = (b * p_win - p_loss) / b
    return max(0.0, f)


def size_units(p_win: float, american: float, p_push: float, cfg_units: dict, cap: float | None = None) -> float:
    """Fractional-Kelly units, rounded to 0.25U, clipped to [min, cap]."""
    f = kelly_fraction(p_win, american, p_push) * cfg_units["kelly_fraction"]
    units = f * 100 / cfg_units["unit_pct_bankroll"]
    cap = cap if cap is not None else cfg_units["max_units"]
    units = min(units, cap)
    if units < cfg_units["min_units"] * 0.5:
        return 0.0
    return max(cfg_units["min_units"], round(units * 4) / 4)


def profit_units(units: float, american: float, result: str) -> float:
    if result == "win":
        return units * (american_to_decimal(american) - 1)
    if result == "loss":
        return -units
    return 0.0
