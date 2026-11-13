"""One small SQLite database for everything that changes at runtime: runs, findings (memory), spans (traces)."""

from __future__ import annotations

import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, question TEXT, model TEXT, status TEXT, stage TEXT,
    state_json TEXT, created REAL, updated REAL
);
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, text TEXT, pmids_json TEXT,
    embedding BLOB, created REAL
);
CREATE TABLE IF NOT EXISTS spans (
    id TEXT PRIMARY KEY, run_id TEXT, parent_id TEXT, kind TEXT, name TEXT, model TEXT,
    start REAL, duration_ms REAL, status TEXT, error TEXT,
    tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, attrs_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_spans_run ON spans(run_id);
"""


class StateDB:
    def __init__(self, path: str):
        self.path = path
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)
            self._db.commit()

    def execute(self, sql: str, params: tuple = ()) -> int:
        with self._lock:
            cur = self._db.execute(sql, params)
            self._db.commit()
            return cur.lastrowid

    def query(self, sql: str, params: tuple = ()) -> list:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, params).fetchall()]
