"""
Build and cache an (N+1) × (N+1) travel-time matrix using the
OpenRouteService /v2/matrix/driving-car endpoint.

The extra node (index 0) is the SLP's home base.
Matrix values are in *minutes*.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import requests

_DEFAULT_CACHE = Path(__file__).parent.parent / "data" / "matrix_cache.json"

ORS_MATRIX_URL = "https://api.openrouteservice.org/v2/matrix/driving-car"

# ORS free tier: max 50 locations per matrix call
_ORS_MAX_LOCATIONS = 50


class MatrixError(Exception):
    pass


def _roster_hash(locations: list[tuple[float, float]]) -> str:
    """Stable hash of a location list so we can cache by roster."""
    serialised = json.dumps(locations, sort_keys=True)
    return hashlib.sha256(serialised.encode()).hexdigest()[:16]


def build_matrix(
    api_key: str,
    locations: list[tuple[float, float]],
    cache_path: Path = _DEFAULT_CACHE,
    retries: int = 3,
) -> list[list[float]]:
    """Return an N×N matrix of drive times in minutes.

    *locations* is ordered as [home_base, client_0, client_1, …].
    Results are cached by location roster hash.
    """
    n = len(locations)
    if n > _ORS_MAX_LOCATIONS:
        raise MatrixError(
            f"ORS free tier supports up to {_ORS_MAX_LOCATIONS} locations; "
            f"got {n}."
        )

    key = _roster_hash(locations)
    cached = _load_cache(cache_path)
    if key in cached:
        return cached[key]

    matrix = _fetch_matrix(api_key, locations, retries)
    cached[key] = matrix
    _save_cache(cache_path, cached)
    return matrix


# ── Cache helpers ────────────────────────────────────────────────────────────


def _load_cache(path: Path) -> dict[str, list[list[float]]]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(path: Path, data: dict[str, list[list[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


# ── ORS call ─────────────────────────────────────────────────────────────────


def _fetch_matrix(
    api_key: str,
    locations: list[tuple[float, float]],
    retries: int,
) -> list[list[float]]:
    """Call ORS matrix API and return travel times in minutes."""
    # ORS expects [[lon, lat], ...]
    coords = [[lon, lat] for lon, lat in locations]
    payload = {
        "locations": coords,
        "metrics": ["duration"],
        "units": "m",
    }
    headers = {
        "Authorization": api_key,
        "Content-Type": "application/json",
    }

    backoff = 2
    last_exc: Exception = MatrixError("no attempts")
    for attempt in range(retries):
        try:
            resp = requests.post(
                ORS_MATRIX_URL, json=payload, headers=headers, timeout=30
            )
            if resp.status_code == 429:
                time.sleep(backoff)
                backoff *= 2
                continue
            resp.raise_for_status()
            data = resp.json()
            durations = data["durations"]  # seconds
            # Convert to minutes, round to nearest minute
            return [
                [round(cell / 60, 1) if cell is not None else 9999.0 for cell in row]
                for row in durations
            ]
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(backoff)
                backoff *= 2

    raise MatrixError(
        f"Matrix API failed after {retries} attempts: {last_exc}"
    )
