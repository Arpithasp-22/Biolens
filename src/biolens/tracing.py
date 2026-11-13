"""Observability: every agent stage, LLM call and tool call becomes a span with latency, tokens, cost, errors."""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager

from .llm import LLM, LLMResult
from .pricing import cost_usd, load_pricing
from .state import StateDB


class SpanHandle:
    def __init__(self, span_id, kind, name, parent_id, run_id, attrs):
        self.id, self.kind, self.name, self.parent_id, self.run_id = span_id, kind, name, parent_id, run_id
        self.attrs = dict(attrs)
        self.model = None
        self.tokens_in = self.tokens_out = 0
        self.cost = None

    def set(self, **kw):
        self.attrs.update(kw)

    def add_usage(self, model: str, tokens_in: int, tokens_out: int, cost: float | None):
        self.model = model
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        if cost is not None:
            self.cost = (self.cost or 0.0) + cost


class Tracer:
    def __init__(self, db: StateDB):
        self.db = db
        self._local = threading.local()

    def _stack(self) -> list:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        return self._local.stack

    @property
    def run_id(self) -> str | None:
        return getattr(self._local, "run_id", None)

    @contextmanager
    def run(self, run_id: str):
        prev = self.run_id
        self._local.run_id = run_id
        try:
            yield
        finally:
            self._local.run_id = prev

    @contextmanager
    def span(self, kind: str, name: str, **attrs):
        stack = self._stack()
        h = SpanHandle(uuid.uuid4().hex[:12], kind, name, stack[-1].id if stack else None, self.run_id, attrs)
        stack.append(h)
        t0, status, err = time.time(), "ok", None
        try:
            yield h
        except BaseException as e:  # record then re-raise
            status, err = "error", f"{type(e).__name__}: {e}"[:500]
            raise
        finally:
            stack.pop()
            self.db.execute(
                "INSERT INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (h.id, h.run_id, h.parent_id, h.kind, h.name, h.model, t0, (time.time() - t0) * 1000,
                 status, err, h.tokens_in, h.tokens_out, h.cost, json.dumps(h.attrs, default=str)[:4000]))

    # ---------- reading ----------
    def spans(self, run_id: str) -> list:
        return self.db.query("SELECT * FROM spans WHERE run_id=? ORDER BY start", (run_id,))

    def summary(self, run_id: str) -> dict:
        rows = self.spans(run_id)
        llm = [r for r in rows if r["kind"] == "llm"]
        tools = [r for r in rows if r["kind"] == "tool"]
        costs = [r["cost_usd"] for r in llm]
        breakdown: dict = {}
        for r in tools:
            breakdown[r["name"]] = breakdown.get(r["name"], 0) + 1
        roots = [r for r in rows if r["parent_id"] is None]
        wall = sum(r["duration_ms"] for r in roots)
        return {
            "llm_calls": len(llm), "tool_calls": len(tools),
            "tokens_in": sum(r["tokens_in"] or 0 for r in llm),
            "tokens_out": sum(r["tokens_out"] or 0 for r in llm),
            "cost_usd": sum(c for c in costs if c is not None),
            "cost_known": all(c is not None for c in costs),
            "latency_ms": wall, "errors": sum(r["status"] == "error" for r in rows),
            "tool_breakdown": breakdown,
        }


class TracedLLM:
    """Wraps any LLM so every call is a span carrying tokens, cost and latency."""

    def __init__(self, llm: LLM, tracer: Tracer, pricing: dict | None = None):
        self._llm, self._tracer = llm, tracer
        self._pricing = pricing if pricing is not None else load_pricing()
        self.name = llm.name

    def complete(self, role, system, user, **kw) -> LLMResult:
        with self._tracer.span("llm", role, model=self.name) as sp:
            res = self._llm.complete(role, system, user, **kw)
            sp.add_usage(self.name, res.tokens_in, res.tokens_out,
                         cost_usd(self.name, res.tokens_in, res.tokens_out, self._pricing))
            return res


class TracedToolbox:
    def __init__(self, toolbox, tracer: Tracer):
        self._tb, self._tracer = toolbox, tracer

    def describe(self) -> str:
        return self._tb.describe()

    def names(self) -> list:
        return self._tb.names()

    def call(self, name: str, args: dict):
        with self._tracer.span("tool", name, args=args) as sp:
            out = self._tb.call(name, args)
            if isinstance(out, dict) and out.get("error"):
                sp.set(tool_error=str(out["error"])[:300])
                raise RuntimeError(f"tool '{name}' failed: {out['error']}")
            return out
