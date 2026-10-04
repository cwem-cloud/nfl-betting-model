"""Persistence: SQLite by default (data/pocket_capper.db), Postgres via DATABASE_URL.

Tables
  runs            one row per Run click / scheduled run
  projections     blind (no-line) numbers per game per run
  odds_snapshots  every price we have seen, per book — the line-movement history
  splits          bets% / money% snapshots
  sharp_plays     manual entries from sharp sources (e.g. Bambino) + their grading
  picks           every official play, graded with closing line value
  boards          full slate output per run (game board, prop hub, TD zone) for the dashboard
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache

import pandas as pd
from sqlalchemy import (
    Column, Float, Integer, MetaData, String, Table, Text, create_engine, inspect, select, text, update,
)

from pocket_capper.settings import DATA_DIR, secret

meta = MetaData()

runs = Table(
    "runs", meta,
    Column("run_id", String, primary_key=True),
    Column("started_at", String), Column("finished_at", String),
    Column("sports", String), Column("status", String), Column("log", Text),
)
projections = Table(
    "projections", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String), Column("sport", String), Column("game_id", String),
    Column("event_id", String), Column("kickoff", String), Column("home", String), Column("away", String),
    Column("home_pts", Float), Column("away_pts", Float), Column("fair_spread_home", Float),
    Column("fair_total", Float), Column("home_win_prob", Float), Column("notes", Text),
)
odds_snapshots = Table(
    "odds_snapshots", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sport", String), Column("event_id", String), Column("commence_time", String),
    Column("home_name", String), Column("away_name", String), Column("book", String),
    Column("market", String), Column("side", String), Column("player", String),
    Column("point", Float), Column("price", Float), Column("fetched_at", String),
)
splits = Table(
    "splits", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("sport", String), Column("away", String), Column("home", String), Column("market", String),
    Column("side", String), Column("bets_pct", Float), Column("money_pct", Float),
    Column("source", String), Column("fetched_at", String),
)
sharp_plays = Table(
    "sharp_plays", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", String), Column("source", String), Column("sport", String),
    Column("event_id", String), Column("game_id", String), Column("matchup", String), Column("market", String),
    Column("selection", String), Column("line", Float), Column("price", Float), Column("units", Float),
    Column("notes", Text), Column("status", String), Column("profit_units", Float),
)
picks = Table(
    "picks", meta,
    Column("pick_id", String, primary_key=True),
    Column("created_at", String), Column("updated_at", String), Column("run_id", String),
    Column("sport", String), Column("category", String),  # side | total | ml | prop | attd
    Column("event_id", String), Column("game_id", String), Column("kickoff", String),
    Column("matchup", String), Column("market", String), Column("selection", String),
    Column("player", String), Column("player_id", String), Column("line", Float), Column("price", Float), Column("book", String),
    Column("units", Float), Column("model_prob", Float), Column("market_prob", Float),
    Column("final_prob", Float), Column("ev", Float), Column("fair_line", Float),
    Column("signals", Text), Column("rationale", Text),
    Column("latest_line", Float), Column("latest_price", Float), Column("latest_ev", Float),
    Column("close_line", Float), Column("close_price", Float), Column("clv_points", Float), Column("clv_prob", Float),
    Column("status", String), Column("result_value", Float), Column("profit_units", Float),
)

boards = Table(
    "boards", meta,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String), Column("sport", String), Column("kind", String),  # games | props | attd
    Column("created_at", String), Column("payload", Text),
)


def normalize_pg_url(url: str) -> str:
    """Accept any Postgres URL form (postgres://, postgresql://, postgresql+psycopg2://, a pasted
    `psql '...'` line) and route it to the psycopg 3 driver we ship."""
    url = url.strip().strip("'\"")
    if url.startswith("psql "):
        url = url[5:].strip().strip("'\"")
    scheme, sep, rest = url.partition("://")
    if sep and scheme.split("+")[0] in ("postgres", "postgresql"):
        return f"postgresql+psycopg://{rest}"
    return url


@lru_cache(maxsize=1)
def engine():
    url = secret("DATABASE_URL")
    if url:
        url = normalize_pg_url(url)
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{DATA_DIR / 'pocket_capper.db'}"
    eng = create_engine(url, future=True)
    meta.create_all(eng)
    _add_missing_columns(eng)
    return eng


def _add_missing_columns(eng) -> None:
    """Tiny forward-only migration: add columns introduced after a table was first created."""
    insp = inspect(eng)
    with eng.begin() as conn:
        for t in meta.sorted_tables:
            have = {c["name"] for c in insp.get_columns(t.name)}
            for c in t.columns:
                if c.name not in have:
                    conn.execute(text(f"ALTER TABLE {t.name} ADD COLUMN {c.name} {c.type.compile(eng.dialect)}"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def insert_df(table: Table, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    cols = [c.name for c in table.columns if c.name in df.columns]
    recs = df[cols].astype(object).where(df[cols].notna(), None).to_dict("records")
    with engine().begin() as conn:
        conn.execute(table.insert(), recs)


SNAP_KEY = ["event_id", "book", "market", "side", "player"]


def insert_snapshots(df: pd.DataFrame) -> int:
    """Store only prices that changed since the last stored snapshot of the same line.

    Movement, steam and closing lines only need the change points; storing every unchanged
    price every few hours would bloat the history (and the repo, in SQLite-commit mode).
    """
    if df is None or df.empty:
        return 0
    df = df.copy()
    if "player" not in df:
        df["player"] = None
    ids = list(df["event_id"].astype(str).unique())
    prev = pd.DataFrame()
    for i in range(0, len(ids), 200):
        chunk = ids[i:i + 200]
        marks = ",".join(f":e{j}" for j in range(len(chunk)))
        prev = pd.concat([prev, read(odds_snapshots, f"WHERE event_id IN ({marks})",
                                     {f"e{j}": e for j, e in enumerate(chunk)})])
    if not prev.empty:
        prev = prev.sort_values("fetched_at").groupby(
            [prev[k].fillna("") for k in SNAP_KEY])[["point", "price"]].last()
        key = pd.MultiIndex.from_frame(df[SNAP_KEY].fillna("").astype(str))
        last = prev.reindex(key)
        same = (
            (last["point"].values == df["point"].values) | (pd.isna(last["point"].values) & pd.isna(df["point"].values))
        ) & (last["price"].values == df["price"].values)
        df = df[~same]
    insert_df(odds_snapshots, df)
    return len(df)


def read(table: Table | str, where: str = "", params: dict | None = None) -> pd.DataFrame:
    name = table if isinstance(table, str) else table.name
    with engine().connect() as conn:
        df = pd.read_sql(text(f"SELECT * FROM {name} {where}"), conn, params=params or {})
    t = meta.tables.get(name)
    if t is not None:  # all-NULL float columns come back as object dtype
        for c in t.columns:
            if isinstance(c.type, Float) and c.name in df:
                df[c.name] = pd.to_numeric(df[c.name], errors="coerce")
    return df


def upsert_pick(p: dict) -> str:
    """Insert a new pick, or refresh the latest line on an existing pending one.

    A pick is locked the first time it is issued (that is the line you bet). Later runs
    only refresh latest_line/latest_price, which feeds CLV tracking.
    """
    with engine().begin() as conn:
        existing = conn.execute(select(picks.c.pick_id, picks.c.status).where(picks.c.pick_id == p["pick_id"])).first()
        if existing:
            conn.execute(
                update(picks)
                .where(picks.c.pick_id == p["pick_id"])
                .values(latest_line=p.get("line"), latest_price=p.get("price"), updated_at=now())
            )
            return "updated"
        rec = {c.name: p.get(c.name) for c in picks.columns}
        rec["signals"] = json.dumps(p.get("signals") or [])
        rec.update(created_at=now(), updated_at=now(), status="pending",
                   latest_line=p.get("line"), latest_price=p.get("price"), latest_ev=p.get("ev"))
        conn.execute(picks.insert(), [rec])
        return "inserted"


def refresh_pick(pick_id: str, line, price, ev) -> None:
    """Latest market + current EV for an open pick (feeds CLV and the 'edge still there?' check)."""
    with engine().begin() as conn:
        conn.execute(update(picks).where(picks.c.pick_id == pick_id, picks.c.status == "pending")
                     .values(latest_line=line, latest_price=price, latest_ev=ev, updated_at=now()))


def update_pick(pick_id: str, **values) -> None:
    with engine().begin() as conn:
        conn.execute(update(picks).where(picks.c.pick_id == pick_id).values(**values))


def update_sharp(play_id: int, **values) -> None:
    with engine().begin() as conn:
        conn.execute(update(sharp_plays).where(sharp_plays.c.id == play_id).values(**values))


def table_names() -> list[str]:
    return inspect(engine()).get_table_names()


BOARDS_KEPT = 3


def save_board(run_id: str, sport: str, kind: str, df: pd.DataFrame) -> None:
    """Store the latest board; only the last few per sport/kind are kept (the dashboard shows the newest)."""
    with engine().begin() as conn:
        conn.execute(boards.insert(), [{
            "run_id": run_id, "sport": sport, "kind": kind, "created_at": now(),
            "payload": df.to_json(orient="records", date_format="iso"),
        }])
        keep = [r[0] for r in conn.execute(
            select(boards.c.id).where(boards.c.sport == sport, boards.c.kind == kind)
            .order_by(boards.c.id.desc()).limit(BOARDS_KEPT))]
        conn.execute(boards.delete().where(boards.c.sport == sport, boards.c.kind == kind, boards.c.id.notin_(keep)))


def compact() -> None:
    """Reclaim space after pruning (SQLite only)."""
    if engine().dialect.name == "sqlite":
        with engine().connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT").execute(text("VACUUM"))


def latest_board(sport: str, kind: str) -> tuple[pd.DataFrame, str | None]:
    with engine().connect() as conn:
        row = conn.execute(
            select(boards.c.payload, boards.c.created_at)
            .where(boards.c.sport == sport, boards.c.kind == kind)
            .order_by(boards.c.id.desc()).limit(1)
        ).first()
    if not row:
        return pd.DataFrame(), None
    from io import StringIO
    return pd.read_json(StringIO(row[0]), orient="records"), row[1]
