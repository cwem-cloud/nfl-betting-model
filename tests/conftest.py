import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point the app at a throwaway SQLite file."""
    from pocket_capper import db

    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    db.engine.cache_clear()
    yield db
    db.engine.cache_clear()
