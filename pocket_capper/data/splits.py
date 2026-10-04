"""Betting splits (% of bets vs % of money).

Sources:
  1. DraftKings Network public splits page (DK's own handle/bets). Best-effort HTML parse;
     page layout changes will make this return empty rather than fail the run.
  2. Manual import: paste/upload a CSV from any source you have (Action Pro, VSIN, etc.).
     Columns: sport, away, home, market (spread|total|ml), side (home|away|over|under),
     bets_pct, money_pct
"""
from __future__ import annotations

import io
import re
from datetime import datetime, timezone

import pandas as pd

from pocket_capper.data.http import SESSION

DK_SPLITS_URL = "https://dknetwork.draftkings.com/draftkings-sportsbook-betting-splits/"
DK_EVENT_GROUPS = {"nfl": 88808, "cfb": 87637}

SPLIT_COLS = ["sport", "away", "home", "market", "side", "bets_pct", "money_pct", "source", "fetched_at"]


def _pct(s) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", str(s))
    return float(m.group(1)) if m else None


def parse_dk_splits_html(html: str, sport: str) -> pd.DataFrame:
    """Parse the DK Network splits markup.

    The page renders one block per game with rows: market label, team/side label,
    handle %, bets %. We scan text rows in order and pair consecutive sides.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    rows = []
    now = datetime.now(timezone.utc).isoformat()
    for game in soup.select("div.tb-se, div[class*='splits-game'], div.tb-se-row-container"):
        title = game.find(class_=re.compile("tb-se-title|game-title|matchup"))
        teams = re.split(r"\s+@\s+|\s+at\s+", title.get_text(" ", strip=True)) if title else []
        if len(teams) != 2:
            continue
        away, home = teams[0].strip(), teams[1].strip()
        market = None
        for line in game.find_all(class_=re.compile("tb-sodd|split-row|tb-se-row")):
            txt = line.get_text(" ", strip=True)
            lower = txt.lower()
            if lower.startswith(("moneyline", "spread", "total")):
                market = {"moneyline": "ml", "spread": "spread", "total": "total"}[lower.split()[0]]
                continue
            pcts = re.findall(r"\d+(?:\.\d+)?\s*%", txt)
            if market and len(pcts) >= 2:
                handle, bets = _pct(pcts[0]), _pct(pcts[1])
                if market == "total":
                    side = "over" if "over" in lower else "under"
                else:
                    side = "away" if away.split()[-1].lower() in lower else "home"
                rows.append([sport, away, home, market, side, bets, handle, "DK Network", now])
    return pd.DataFrame(rows, columns=SPLIT_COLS)


def fetch_dk_splits(sport: str) -> pd.DataFrame:
    try:
        r = SESSION.get(
            DK_SPLITS_URL,
            params={"tb_eg": DK_EVENT_GROUPS[sport], "tb_edate": "n7days", "tb_emt": "0"},
            timeout=30,
        )
        r.raise_for_status()
        return parse_dk_splits_html(r.text, sport)
    except Exception:
        return pd.DataFrame(columns=SPLIT_COLS)


def parse_manual_csv(text: str, source: str = "manual") -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"sport", "away", "home", "market", "side", "bets_pct", "money_pct"} - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    df["source"] = df.get("source", source)
    df["fetched_at"] = datetime.now(timezone.utc).isoformat()
    for c in ("bets_pct", "money_pct"):
        df[c] = df[c].astype(str).str.rstrip("%").astype(float)
    return df[SPLIT_COLS]
