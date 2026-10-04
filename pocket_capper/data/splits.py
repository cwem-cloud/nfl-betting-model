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
    """Parse the DK Network splits page.

    Layout (verified live Oct 2026): div.tb-se per game; its h5 title reads "AWAY @ HOME";
    div.tb-market-wrap holds one block per market whose .tb-se-head names it (Moneyline / Spread /
    Total), and each .tb-sodd row has four cells: label, odds, % handle, % bets.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for game in soup.select("div.tb-se"):
        title = game.select_one(".tb-se-title h5") or game.select_one(".tb-se-title")
        if not title:
            continue
        parts = [t.strip() for t in title.get_text(" ", strip=True).split("@")]
        if len(parts) != 2:
            continue
        away, home = parts
        for block in game.select(".tb-market-wrap > div"):
            head = block.select_one(".tb-se-head")
            if not head:
                continue
            first = head.find("div")
            name = (first.get_text(strip=True) if first else head.get_text(" ", strip=True)).lower()
            market = "ml" if name.startswith("money") else "spread" if name.startswith("spread") else \
                "total" if name.startswith("total") else None
            if not market:
                continue
            for row in block.select(".tb-sodd"):
                cells = row.find_all("div", recursive=False)
                if len(cells) < 4:
                    continue
                label = cells[0].get_text(" ", strip=True)
                handle, bets = _pct(cells[2].get_text(" ", strip=True)), _pct(cells[3].get_text(" ", strip=True))
                if handle is None or bets is None:
                    continue
                low = label.lower()
                if market == "total":
                    side = "over" if low.startswith("over") else "under" if low.startswith("under") else None
                elif label.startswith(away):
                    side = "away"
                elif label.startswith(home):
                    side = "home"
                else:
                    side = None
                if side:
                    rows.append([sport, away, home, market, side, bets, handle, "DK Network", now])
    return pd.DataFrame(rows, columns=SPLIT_COLS)


def fetch_dk_splits(sport: str, max_pages: int = 8) -> pd.DataFrame:
    """All pages of the DK Network splits table (10 games per page, `tb_page` param)."""
    frames, seen = [], set()
    for page in range(1, max_pages + 1):
        try:
            r = SESSION.get(
                DK_SPLITS_URL,
                params={"tb_eg": DK_EVENT_GROUPS[sport], "tb_edate": "n7days", "tb_emt": "0", "tb_page": page},
                timeout=30,
            )
            r.raise_for_status()
        except Exception:
            break
        df = parse_dk_splits_html(r.text, sport)
        games = set(zip(df["away"], df["home"])) if len(df) else set()
        if not games or games <= seen:  # past the last page (some sites repeat the final page)
            break
        frames.append(df[[g not in seen for g in zip(df["away"], df["home"])]])
        seen |= games
        if f"tb_page={page + 1}" not in r.text.replace("&#038;", "&"):
            break
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SPLIT_COLS)


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
