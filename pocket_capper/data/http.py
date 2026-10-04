"""Small HTTP + on-disk cache helpers."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

from pocket_capper.settings import CACHE_DIR

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "pocket-capper/1.0"


def cached_download(url: str, filename: str, max_age_hours: float) -> Path:
    """Download url to the cache dir unless a fresh copy exists. Returns the path."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / filename
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
        return path
    try:
        r = SESSION.get(url, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
    except Exception:
        if path.exists():  # stale beats nothing
            return path
        raise
    return path


def cached_json(url: str, params: dict | None, headers: dict | None, max_age_hours: float) -> object:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(json.dumps([url, params], sort_keys=True).encode()).hexdigest()[:20]
    path = CACHE_DIR / f"json_{key}.json"
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600:
        return json.loads(path.read_text())
    try:
        r = SESSION.get(url, params=params, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception:
        if path.exists():
            return json.loads(path.read_text())
        raise
    path.write_text(json.dumps(data))
    return data
