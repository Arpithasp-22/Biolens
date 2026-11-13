"""Corpus store: papers + passages in SQLite, with FTS5 (BM25) and dense vectors side by side.

Brute-force cosine over a cached NumPy matrix is plenty for tens of thousands of passages. For more,
swap `dense()` for Qdrant/pgvector; nothing else in the project depends on how vectors are stored.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading

import numpy as np

from .embeddings import Embedder
from .models import Filters, Paper
from .textutil import chunk_text

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS papers (
    pmid TEXT PRIMARY KEY, title TEXT, abstract TEXT, journal TEXT,
    year INTEGER, pub_type TEXT, authors TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, pmid TEXT NOT NULL, idx INTEGER NOT NULL,
    text TEXT NOT NULL, embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_pmid ON chunks(pmid);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text, tokenize='porter unicode61');
"""


class CorpusStore:
    def __init__(self, path: str, embedder: Embedder):
        self.path, self.embedder = path, embedder
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._cache = None
        with self._lock:
            self._db.executescript(SCHEMA)
            row = self._db.execute("SELECT v FROM meta WHERE k='embedder'").fetchone()
            if row and row["v"] != embedder.name:
                raise ValueError(
                    f"{path} was built with embedder '{row['v']}', not '{embedder.name}'. "
                    "Re-ingest with the same embedder (or use a new corpus file)."
                )
            self._db.execute("INSERT OR REPLACE INTO meta VALUES ('embedder', ?)", (embedder.name,))
            self._db.commit()

    # ---------- writing ----------
    def add_papers(self, papers: list) -> int:
        papers = [p for p in papers if p.abstract or p.title]
        if not papers:
            return 0
        rows = [(p, i, t) for p in papers for i, t in enumerate(chunk_text(p.title, p.abstract))]
        vecs = self.embedder.embed([t for _, _, t in rows])
        with self._lock:
            for p in papers:
                self._delete_paper(p.pmid)
                self._db.execute(
                    "INSERT INTO papers VALUES (?,?,?,?,?,?,?)",
                    (p.pmid, p.title, p.abstract, p.journal, p.year, p.pub_type, json.dumps(p.authors)),
                )
            for (p, i, t), v in zip(rows, vecs):
                cur = self._db.execute(
                    "INSERT INTO chunks (pmid, idx, text, embedding) VALUES (?,?,?,?)",
                    (p.pmid, i, t, v.astype(np.float32).tobytes()),
                )
                self._db.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?,?)", (cur.lastrowid, t))
            self._db.commit()
            self._cache = None
        return len(papers)

    def _delete_paper(self, pmid: str):
        ids = [r["id"] for r in self._db.execute("SELECT id FROM chunks WHERE pmid=?", (pmid,))]
        for cid in ids:
            self._db.execute("DELETE FROM chunks_fts WHERE rowid=?", (cid,))
        self._db.execute("DELETE FROM chunks WHERE pmid=?", (pmid,))
        self._db.execute("DELETE FROM papers WHERE pmid=?", (pmid,))

    # ---------- reading ----------
    def count(self) -> dict:
        with self._lock:
            p = self._db.execute("SELECT COUNT(*) c FROM papers").fetchone()["c"]
            c = self._db.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]
        return {"papers": p, "chunks": c}

    def get_paper(self, pmid: str) -> Paper | None:
        with self._lock:
            r = self._db.execute("SELECT * FROM papers WHERE pmid=?", (str(pmid),)).fetchone()
        if not r:
            return None
        return Paper(r["pmid"], r["title"], r["abstract"], r["journal"], r["year"],
                     r["pub_type"], json.loads(r["authors"] or "[]"))

    def all_pmids(self) -> list:
        with self._lock:
            return [r["pmid"] for r in self._db.execute("SELECT pmid FROM papers ORDER BY pmid")]

    def get_chunks(self, ids: list) -> dict:
        if not ids:
            return {}
        q = ",".join("?" * len(ids))
        with self._lock:
            rows = self._db.execute(
                f"""SELECT c.id, c.pmid, c.text, p.title, p.year, p.journal, p.pub_type
                    FROM chunks c JOIN papers p ON p.pmid=c.pmid WHERE c.id IN ({q})""", ids
            ).fetchall()
        return {r["id"]: dict(r) for r in rows}

    def paper_chunks(self, pmid: str) -> list:
        with self._lock:
            return [r["text"] for r in self._db.execute(
                "SELECT text FROM chunks WHERE pmid=? ORDER BY idx", (str(pmid),))]

    @staticmethod
    def _where(f: Filters | None) -> tuple:
        clauses, params = [], []
        if f:
            if f.year_min:
                clauses.append("p.year >= ?"); params.append(f.year_min)
            if f.year_max:
                clauses.append("p.year <= ?"); params.append(f.year_max)
            if f.journal:
                clauses.append("p.journal LIKE ?"); params.append(f"%{f.journal}%")
            if f.pub_type:
                clauses.append("p.pub_type LIKE ?"); params.append(f"%{f.pub_type}%")
        return (" AND " + " AND ".join(clauses)) if clauses else "", params

    @staticmethod
    def _fts_query(q: str) -> str:
        words = [w for w in re.findall(r"[A-Za-z0-9]+", q) if len(w) > 1]
        return " OR ".join(f'"{w}"' for w in words)

    def bm25(self, query: str, k: int, filters: Filters | None = None) -> list:
        """[(chunk_id, score)] best first. Higher = better."""
        match = self._fts_query(query)
        if not match:
            return []
        where, params = self._where(filters)
        with self._lock:
            rows = self._db.execute(
                f"""SELECT c.id AS id, bm25(chunks_fts) AS s FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.rowid
                    JOIN papers p ON p.pmid = c.pmid
                    WHERE chunks_fts MATCH ? {where} ORDER BY s LIMIT ?""",
                [match, *params, k],
            ).fetchall()
        return [(r["id"], -r["s"]) for r in rows]

    def _matrix(self):
        if self._cache is None:
            with self._lock:
                rows = self._db.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
            ids = np.array([r["id"] for r in rows], dtype=np.int64)
            mat = (np.vstack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                   if rows else np.zeros((0, self.embedder.dim), dtype=np.float32))
            self._cache = (ids, mat)
        return self._cache

    def dense(self, query_vec: np.ndarray, k: int, filters: Filters | None = None) -> list:
        ids, mat = self._matrix()
        if len(ids) == 0:
            return []
        scores = mat @ query_vec.astype(np.float32)
        if filters and not filters.is_empty():
            where, params = self._where(filters)
            with self._lock:
                ok = {r["id"] for r in self._db.execute(
                    f"SELECT c.id FROM chunks c JOIN papers p ON p.pmid=c.pmid WHERE 1=1 {where}", params)}
            mask = np.array([i in ok for i in ids.tolist()])
            scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:k]
        return [(int(ids[i]), float(scores[i])) for i in order if np.isfinite(scores[i])]
