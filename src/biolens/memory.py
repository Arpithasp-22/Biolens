"""Persistent agent memory: run checkpoints (so research can be resumed) and findings (recalled later)."""

from __future__ import annotations

import json
import time
import uuid

import numpy as np

from .embeddings import Embedder
from .state import StateDB


class Memory:
    def __init__(self, db: StateDB, embedder: Embedder):
        self.db, self.embedder = db, embedder

    # ---------- runs / checkpoints ----------
    def create_run(self, question: str, model: str) -> str:
        run_id = uuid.uuid4().hex[:8]
        now = time.time()
        self.db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?)",
                        (run_id, question, model, "running", "start", "{}", now, now))
        return run_id

    def save_state(self, run_id: str, state: dict, status: str | None = None, stage: str | None = None):
        row = self.db.query("SELECT status, stage FROM runs WHERE id=?", (run_id,))[0]
        self.db.execute("UPDATE runs SET state_json=?, status=?, stage=?, updated=? WHERE id=?",
                        (json.dumps(state, default=str), status or row["status"], stage or row["stage"],
                         time.time(), run_id))

    def load_run(self, run_id: str) -> dict | None:
        rows = self.db.query("SELECT * FROM runs WHERE id=?", (run_id,))
        if not rows:
            return None
        r = rows[0]
        r["state"] = json.loads(r.pop("state_json") or "{}")
        return r

    def list_runs(self, limit: int = 20) -> list:
        return self.db.query(
            "SELECT id, question, model, status, stage, created FROM runs ORDER BY created DESC LIMIT ?", (limit,))

    # ---------- findings ----------
    def add_findings(self, run_id: str, findings: list):
        """findings: [{'text': str, 'pmids': [str]}]"""
        if not findings:
            return
        vecs = self.embedder.embed([f["text"] for f in findings])
        for f, v in zip(findings, vecs):
            self.db.execute("INSERT INTO findings (run_id, text, pmids_json, embedding, created) VALUES (?,?,?,?,?)",
                            (run_id, f["text"], json.dumps(f["pmids"]), v.astype(np.float32).tobytes(), time.time()))

    def recall(self, question: str, k: int = 3, min_sim: float = 0.3, exclude_run: str | None = None) -> list:
        rows = self.db.query("SELECT run_id, text, pmids_json, embedding FROM findings")
        rows = [r for r in rows if r["run_id"] != exclude_run]
        if not rows:
            return []
        q = self.embedder.embed([question])[0]
        sims = np.array([float(np.frombuffer(r["embedding"], dtype=np.float32) @ q) for r in rows])
        out = []
        for i in np.argsort(-sims)[:k]:
            if sims[i] >= min_sim:
                r = rows[i]
                out.append({"text": r["text"], "pmids": json.loads(r["pmids_json"]),
                            "run_id": r["run_id"], "similarity": round(float(sims[i]), 3)})
        return out
