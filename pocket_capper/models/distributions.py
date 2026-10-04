"""Discrete score distributions with football key numbers.

NFL margins cluster on 3, 7, 10, 14 ... The weights below are the ratio of the
empirical frequency of |final margin| = k (nflverse, 2002-2025) to a smoothed
version of the same curve. Multiplying a normal pmf by them puts the right mass
on key numbers, which is what makes -2.5 vs -3.5 worth real money.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

_NFL_MARGIN_KEY_WEIGHTS = [
    0.041, 0.728, 0.721, 2.543, 0.839, 0.614, 1.117, 1.741, 0.8, 0.358, 1.414,
    0.673, 0.527, 0.902, 1.746, 0.576, 0.81, 1.438, 1.029, 0.531, 1.099, 1.47,
    0.614, 0.721, 1.511, 0.86, 0.617, 1.053, 1.59, 0.594, 0.61,
]
# Totals keys are weaker; these are shrunk halfway toward 1 when applied.
_NFL_TOTAL_KEY_WEIGHTS = {
    30: 1.436, 31: 0.822, 32: 0.674, 33: 1.415, 34: 1.102, 35: 0.674, 36: 0.986,
    37: 1.449, 38: 0.764, 39: 0.758, 40: 1.228, 41: 1.317, 42: 0.527, 43: 1.218,
    44: 1.356, 45: 0.909, 46: 0.736, 47: 1.188, 48: 1.071, 49: 0.791, 50: 0.936,
    51: 1.58, 52: 0.943, 53: 0.651, 54: 1.006, 55: 1.261, 56: 0.495, 57: 1.242,
    58: 1.247, 59: 0.942,
}

_RANGE = np.arange(-80, 81)
_TOT_RANGE = np.arange(0, 141)


def _margin_weights(sport: str) -> np.ndarray:
    if sport != "nfl":
        w = np.ones_like(_RANGE, dtype=float)
        w[_RANGE == 0] = 0.08  # ties nearly impossible with OT in college too
        return w
    w = np.ones_like(_RANGE, dtype=float)
    for i, k in enumerate(np.abs(_RANGE)):
        if k < len(_NFL_MARGIN_KEY_WEIGHTS):
            kw = _NFL_MARGIN_KEY_WEIGHTS[k]
            w[i] = kw if k <= 21 else 1 + 0.5 * (kw - 1)
    return w


def margin_pmf(mu: float, sigma: float, sport: str = "nfl") -> tuple[np.ndarray, np.ndarray]:
    """pmf over integer home margins (home - away)."""
    base = norm.cdf(_RANGE + 0.5, mu, sigma) - norm.cdf(_RANGE - 0.5, mu, sigma)
    p = base * _margin_weights(sport)
    p /= p.sum()
    # re-centre so the reweighting does not drag the mean off the projection
    shift = mu - float((_RANGE * p).sum())
    if abs(shift) > 0.05:
        base = norm.cdf(_RANGE + 0.5, mu + shift, sigma) - norm.cdf(_RANGE - 0.5, mu + shift, sigma)
        p = base * _margin_weights(sport)
        p /= p.sum()
    return _RANGE, p


def total_pmf(mu: float, sigma: float, sport: str = "nfl") -> tuple[np.ndarray, np.ndarray]:
    p = norm.cdf(_TOT_RANGE + 0.5, mu, sigma) - norm.cdf(_TOT_RANGE - 0.5, mu, sigma)
    if sport == "nfl":
        w = np.array([1 + 0.5 * (_NFL_TOTAL_KEY_WEIGHTS.get(int(t), 1.0) - 1) for t in _TOT_RANGE])
        p = p * w
    p /= p.sum()
    return _TOT_RANGE, p


def spread_probs(mu_margin: float, sigma: float, home_line: float, sport: str = "nfl") -> dict:
    """Cover probabilities for a home spread line (home -3 => home_line = -3)."""
    x, p = margin_pmf(mu_margin, sigma, sport)
    adj = x + home_line
    return {
        "home": float(p[adj > 0].sum()),
        "away": float(p[adj < 0].sum()),
        "push": float(p[np.isclose(adj, 0)].sum()),
    }


def total_probs(mu_total: float, sigma: float, line: float, sport: str = "nfl") -> dict:
    x, p = total_pmf(mu_total, sigma, sport)
    return {
        "over": float(p[x > line].sum()),
        "under": float(p[x < line].sum()),
        "push": float(p[np.isclose(x, line)].sum()),
    }


def win_prob(mu_margin: float, sigma: float, sport: str = "nfl") -> float:
    """Moneyline win probability for home (ties split, ~OT resolved)."""
    x, p = margin_pmf(mu_margin, sigma, sport)
    return float(p[x > 0].sum() + 0.5 * p[x == 0].sum())


def margin_to_spread(mu_margin: float) -> float:
    """Home spread in betting convention (home favored by 3 => -3.0), half-point rounded."""
    return -round(mu_margin * 2) / 2
