"""Kickoff-hour weather from Open-Meteo (free, no key)."""
from __future__ import annotations

import pandas as pd

from pocket_capper.data.http import cached_json

URL = "https://api.open-meteo.com/v1/forecast"


def kickoff_weather(lat: float, lon: float, kickoff_utc: pd.Timestamp) -> dict | None:
    try:
        data = cached_json(
            URL,
            {
                "latitude": round(lat, 3),
                "longitude": round(lon, 3),
                "hourly": "temperature_2m,wind_speed_10m,wind_gusts_10m,precipitation_probability",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "UTC",
                "forecast_days": 10,
            },
            None,
            max_age_hours=2,
        )
    except Exception:
        return None
    h = data.get("hourly") or {}
    times = pd.to_datetime(h.get("time", []), utc=True)
    if len(times) == 0:
        return None
    k = pd.Timestamp(kickoff_utc)
    k = k.tz_localize("UTC") if k.tzinfo is None else k.tz_convert("UTC")
    i = int(abs(times - k).argmin())
    if abs(times[i] - k) > pd.Timedelta(hours=3):
        return None  # outside the forecast window
    # average the 3 game hours
    sl = slice(i, i + 3)
    def avg(key):
        vals = [v for v in h.get(key, [])[sl] if v is not None]
        return sum(vals) / len(vals) if vals else 0.0
    return {
        "temp_f": avg("temperature_2m"),
        "wind_mph": avg("wind_speed_10m"),
        "gust_mph": avg("wind_gusts_10m"),
        "precip_prob": avg("precipitation_probability"),
    }
