"""Config + secrets loading. Secrets come from env vars or Streamlit secrets."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"


@lru_cache(maxsize=1)
def config() -> dict:
    with open(ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def secret(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    if val:
        return val
    try:  # Streamlit secrets, when running inside the app
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]) or default
    except Exception:
        pass
    return default
