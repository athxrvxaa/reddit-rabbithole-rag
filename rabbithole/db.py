"""SQLite helpers. One DB holds raw comments, thread metadata and extracted pairs."""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS comments(
    id TEXT PRIMARY KEY,
    link_id TEXT NOT NULL,
    parent_id TEXT,
    author TEXT,
    body TEXT,
    score INTEGER,
    created_utc INTEGER
);
CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS threads(
    thread_id TEXT PRIMARY KEY,
    found INTEGER NOT NULL,
    author TEXT,
    title TEXT,
    selftext TEXT,
    score INTEGER,
    created_utc INTEGER,
    num_comments INTEGER
);
CREATE TABLE IF NOT EXISTS pairs(
    pair_id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    op_author TEXT,
    op_source TEXT,
    title TEXT,
    context TEXT,
    category TEXT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    question_id TEXT,
    answer_id TEXT,
    question_score INTEGER,
    answer_score INTEGER,
    created_utc INTEGER,
    source_url TEXT,
    content_hash TEXT UNIQUE
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    return con


def get_state(con: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = con.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
    return row[0] if row else default


def set_state(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute("INSERT OR REPLACE INTO state(key, value) VALUES (?, ?)", (key, str(value)))
