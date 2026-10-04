"""Synthetic Odds API payloads for offline tests / demos."""
from __future__ import annotations

import numpy as np

from pocket_capper.data.teams import NFL_ABBR_TO_NAME


def _price(rng):
    return int(rng.choice([-105, -108, -110, -110, -112, -115]))


def game_payload(games, seed=0, books=("draftkings", "fanduel", "betmgm", "pinnacle")):
    """games: iterable of (event_id, home_abbr, away_abbr, home_margin_mkt, total_mkt, commence_iso)."""
    rng = np.random.default_rng(seed)
    out = []
    for ev_id, home, away, margin, total, commence in games:
        bks = []
        for bk in books:
            jitter = 0.0 if bk == "pinnacle" else rng.choice([0, 0, 0.5, -0.5, 1.0])
            hl = -round((margin + jitter) * 2) / 2
            tl = round((total + rng.choice([0, 0, 0.5, -0.5])) * 2) / 2
            from pocket_capper.market.odds_math import prob_to_american
            from pocket_capper.models.distributions import win_prob
            p_home = win_prob(margin, 13.2)
            ml_h = prob_to_american(min(0.97, p_home * 1.025))
            ml_a = prob_to_american(min(0.97, (1 - p_home) * 1.025))
            bks.append({"key": bk, "markets": [
                {"key": "h2h", "outcomes": [{"name": NFL_ABBR_TO_NAME[home], "price": ml_h},
                                            {"name": NFL_ABBR_TO_NAME[away], "price": ml_a}]},
                {"key": "spreads", "outcomes": [{"name": NFL_ABBR_TO_NAME[home], "price": _price(rng), "point": hl},
                                                {"name": NFL_ABBR_TO_NAME[away], "price": _price(rng), "point": -hl}]},
                {"key": "totals", "outcomes": [{"name": "Over", "price": _price(rng), "point": tl},
                                               {"name": "Under", "price": _price(rng), "point": tl}]},
            ]})
        out.append({"id": ev_id, "commence_time": commence, "home_team": NFL_ABBR_TO_NAME[home],
                    "away_team": NFL_ABBR_TO_NAME[away], "bookmakers": bks})
    return out


def props_payload(ev, players, seed=0, books=("draftkings", "fanduel")):
    """players: list of (name, market, point, fair_over_prob) ; anytime_td uses point None."""
    rng = np.random.default_rng(seed)
    bks = []
    for bk in books:
        mkts = {}
        for name, market, point, p in players:
            p = min(max(p + rng.normal(0, 0.03), 0.05), 0.95)
            if market == "player_anytime_td":
                price = int(100 * (1 - p) / p * 0.88) if p < 0.5 else int(-100 * p / (1 - p) * 1.1)
                mkts.setdefault(market, []).append({"name": "Yes", "description": name, "price": price})
            else:
                o = int(-100 * p / (1 - p) * 1.05) if p >= 0.5 else int(100 * (1 - p) / p * 0.95)
                u = int(-100 * (1 - p) / p * 1.05) if p <= 0.5 else int(100 * p / (1 - p) * 0.95)
                mkts.setdefault(market, []).extend([
                    {"name": "Over", "description": name, "price": o, "point": point},
                    {"name": "Under", "description": name, "price": u, "point": point},
                ])
        bks.append({"key": bk, "markets": [{"key": k, "outcomes": v} for k, v in mkts.items()]})
    return {**ev, "bookmakers": bks}
