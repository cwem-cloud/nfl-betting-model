"""Team name normalization + NFL stadium coordinates (for weather)."""
from __future__ import annotations

import re

NFL_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
NFL_ABBR_TO_NAME = {v: k for k, v in NFL_NAME_TO_ABBR.items()}

# lat, lon, roof ("dome" | "retractable" | "outdoors") keyed by home team
NFL_STADIUMS = {
    "ARI": (33.5276, -112.2626, "retractable"), "ATL": (33.7554, -84.4008, "retractable"),
    "BAL": (39.2780, -76.6227, "outdoors"), "BUF": (42.7738, -78.7870, "outdoors"),
    "CAR": (35.2258, -80.8528, "outdoors"), "CHI": (41.8623, -87.6167, "outdoors"),
    "CIN": (39.0955, -84.5161, "outdoors"), "CLE": (41.5061, -81.6995, "outdoors"),
    "DAL": (32.7473, -97.0945, "retractable"), "DEN": (39.7439, -105.0201, "outdoors"),
    "DET": (42.3400, -83.0456, "dome"), "GB": (44.5013, -88.0622, "outdoors"),
    "HOU": (29.6847, -95.4107, "retractable"), "IND": (39.7601, -86.1639, "retractable"),
    "JAX": (30.3239, -81.6373, "outdoors"), "KC": (39.0489, -94.4839, "outdoors"),
    "LV": (36.0909, -115.1833, "dome"), "LAC": (33.9535, -118.3392, "dome"),
    "LA": (33.9535, -118.3392, "dome"), "MIA": (25.9580, -80.2389, "outdoors"),
    "MIN": (44.9737, -93.2577, "dome"), "NE": (42.0909, -71.2643, "outdoors"),
    "NO": (29.9511, -90.0812, "dome"), "NYG": (40.8128, -74.0742, "outdoors"),
    "NYJ": (40.8128, -74.0742, "outdoors"), "PHI": (39.9008, -75.1675, "outdoors"),
    "PIT": (40.4468, -80.0158, "outdoors"), "SF": (37.4030, -121.9700, "outdoors"),
    "SEA": (47.5952, -122.3316, "outdoors"), "TB": (27.9759, -82.5033, "outdoors"),
    "TEN": (36.1665, -86.7713, "outdoors"), "WAS": (38.9077, -76.8645, "outdoors"),
}


def nfl_abbr(name: str) -> str | None:
    if name in NFL_ABBR_TO_NAME:
        return name
    return NFL_NAME_TO_ABBR.get(name)


def _norm(s: str) -> str:
    s = s.lower().replace("&", "and").replace("state", "st").replace("st.", "st")
    return re.sub(r"[^a-z0-9]", "", s)


# Odds API long names whose start differs from the CFBD school name (normalized prefixes).
_CFB_PREFIX_ALIASES = [
    ("louisianamonroe", "ulmonroe"), ("appalachianst", "appst"), ("southernmethodist", "smu"),
    ("centralflorida", "ucf"), ("brighamyoung", "byu"), ("louisianast", "lsu"), ("texaschristian", "tcu"),
    ("connecticut", "uconn"), ("texassanantonio", "utsa"), ("texaselpaso", "utep"),
    ("nevadalasvegas", "unlv"), ("alabamabirmingham", "uab"), ("floridainternational", "fiu"),
]


def match_cfb(odds_name: str, schools: list[str]) -> str | None:
    """Map an Odds API name like 'Alabama Crimson Tide' to a CFBD school 'Alabama'.

    Picks the longest school name that the odds name starts with, so 'Miami (OH) RedHawks'
    beats 'Miami' and 'Texas A&M Aggies' beats 'Texas'.
    """
    on = _norm(odds_name)
    for src, dst in _CFB_PREFIX_ALIASES:
        if on.startswith(src):
            on = dst + on[len(src):]
            break
    best, best_len = None, 0
    for s in schools:
        sn = _norm(s)
        if on.startswith(sn) and len(sn) > best_len:
            best, best_len = s, len(sn)
    return best


def norm_player(name: str) -> str:
    name = re.sub(r"\b(jr|sr|ii|iii|iv|v)\.?$", "", name.strip().lower())
    return re.sub(r"[^a-z]", "", name)
