import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point the app at a throwaway SQLite file, or at a scratch Postgres when PC_TEST_PG is set."""
    import os

    from pocket_capper import db

    pg = os.environ.get("PC_TEST_PG")
    monkeypatch.setenv("DATABASE_URL", pg or f"sqlite:///{tmp_path / 'test.db'}")
    db.engine.cache_clear()
    if pg:
        db.meta.drop_all(db.engine())
        db.engine.cache_clear()
    yield db
    db.engine.cache_clear()
