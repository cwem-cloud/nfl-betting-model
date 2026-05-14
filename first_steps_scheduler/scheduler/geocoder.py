"""
OpenRouteService geocoding wrapper with persistent JSON cache.

Caches by normalized address string so the same intersection never
hits the API twice across restarts.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import requests

# Default cache location (next to the data/ dir)
_DEFAULT_CACHE = Path(__file__).parent.parent / "data" / "geocode_cache.json"

ORS_GEOCODE_URL = "https://api.openrouteservice.org/geocode/search"

# Focus bounding box: Indianapolis metro area
_FOCUS_POINT = (-86.1581, 39.7684)  # lon, lat


def _normalize(address: str) -> str:
    return " ".join(address.lower().split())


class GeocoderError(Exception):
    pass


class Geocoder:
    """Thin wrapper around the ORS geocoding API with persistent caching."""

    def __init__(
        self,
        api_key: str,
        cache_path: Path = _DEFAULT_CACHE,
    ) -> None:
        self.api_key = api_key
        self.cache_path = cache_path
        self._cache: dict[str, tuple[float, float]] = {}
        self._load_cache()

    # ── Cache I/O ────────────────────────────────────────────────────────────

    def _load_cache(self) -> None:
        if self.cache_path.exists():
            with open(self.cache_path) as f:
                raw = json.load(f)
            # stored as {address: [lon, lat]}
            self._cache = {k: tuple(v) for k, v in raw.items()}  # type: ignore[misc]

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump({k: list(v) for k, v in self._cache.items()}, f, indent=2)

    # ── Public API ───────────────────────────────────────────────────────────

    def geocode(self, address: str, retries: int = 3) -> tuple[float, float]:
        """Return (longitude, latitude) for *address*.

        Raises GeocoderError if the address cannot be resolved.
        """
        key = _normalize(address)
        if key in self._cache:
            return self._cache[key]

        coords = self._fetch(address, retries)
        self._cache[key] = coords
        self._save_cache()
        return coords

    def geocode_batch(
        self, addresses: list[str]
    ) -> dict[str, tuple[float, float] | Exception]:
        """Geocode a list of addresses, returning a dict of address → coords or error."""
        results: dict[str, tuple[float, float] | Exception] = {}
        for addr in addresses:
            try:
                results[addr] = self.geocode(addr)
            except GeocoderError as exc:
                results[addr] = exc
        return results

    # ── Internal ─────────────────────────────────────────────────────────────

    def _fetch(self, address: str, retries: int) -> tuple[float, float]:
        params = {
            "api_key": self.api_key,
            "text": address,
            "focus.point.lon": _FOCUS_POINT[0],
            "focus.point.lat": _FOCUS_POINT[1],
            "size": 1,
            "boundary.country": "US",
        }
        headers = {"Authorization": self.api_key}
        backoff = 2
        last_exc: Exception = GeocoderError("no attempts made")
        for attempt in range(retries):
            try:
                resp = requests.get(
                    ORS_GEOCODE_URL, params=params, headers=headers, timeout=10
                )
                if resp.status_code == 429:
                    # Rate limited – back off and retry
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                resp.raise_for_status()
                data = resp.json()
                features = data.get("features", [])
                if not features:
                    raise GeocoderError(f"No geocoding results for '{address}'")
                coords = features[0]["geometry"]["coordinates"]  # [lon, lat]
                return (float(coords[0]), float(coords[1]))
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
        raise GeocoderError(
            f"Geocoding failed for '{address}' after {retries} attempts: {last_exc}"
        )
