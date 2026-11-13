"""The tool set. Plain typed functions are the single source of truth: the agent calls them in-process
(LocalToolbox) or over the Model Context Protocol (McpToolbox), and mcp_server.py exposes the same functions."""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import threading
from typing import Any

from . import sandbox
from .models import Filters
from .pubmed import PubMedClient
from .retrieval import HybridRetriever
from .store import CorpusStore
from .verify import CitationVerifier

WEAK_COVERAGE = 0.3  # best passage covers < 30% of the query terms -> evidence is probably weak


def build_tools(retriever: HybridRetriever, store: CorpusStore, verifier: CitationVerifier,
                pubmed: PubMedClient | None = None) -> dict:
    """Returns {name: callable}. Every callable takes JSON-friendly args and returns a JSON-friendly dict."""

    def search_papers(query: str, k: int = 5, year_min: int | None = None, year_max: int | None = None,
                      journal: str | None = None, pub_type: str | None = None) -> dict:
        """Find the most relevant papers (one row per paper) with metadata filters. Use to survey a topic."""
        f = Filters.from_dict({"year_min": year_min, "year_max": year_max, "journal": journal, "pub_type": pub_type})
        hits = retriever.search_papers(query, k=max(1, min(k, 20)), filters=f)
        return {"papers": [{"pmid": h.pmid, "title": h.title, "year": h.year, "journal": h.journal,
                            "pub_type": h.pub_type, "best_passage": h.text[:300]} for h in hits]}

    def retrieve_passages(query: str, k: int = 5, year_min: int | None = None, year_max: int | None = None,
                          journal: str | None = None, pub_type: str | None = None, mode: str = "hybrid") -> dict:
        """Retrieve citable evidence passages (hybrid BM25+vector, reranked). Mode: hybrid|bm25|dense."""
        f = Filters.from_dict({"year_min": year_min, "year_max": year_max, "journal": journal, "pub_type": pub_type})
        hits = retriever.search(query, k=max(1, min(k, 20)), filters=f, mode=mode)
        cov = max((h.detail.get("coverage", 0.0) for h in hits), default=0.0)
        return {"passages": [h.to_dict() for h in hits], "max_coverage": round(cov, 3),
                "weak": cov < WEAK_COVERAGE}

    def get_document(pmid: str) -> dict:
        """Fetch one paper's full record (title, journal, year, type, abstract) by PMID/ID."""
        p = store.get_paper(pmid)
        if not p:
            return {"error": f"paper '{pmid}' not found in corpus"}
        return {"paper": p.to_dict(),
                "passages": [{"pmid": p.pmid, "title": p.title, "year": p.year, "text": t}
                             for t in store.paper_chunks(p.pmid)]}

    def run_python_analysis(code: str, data: Any = None) -> dict:
        """Run small stdlib-only Python (math, statistics, json, collections, itertools, re, ...) on `data`.
        Put the answer in a variable named `result`. Use for pooling effect sizes, percentages, unit conversions."""
        return sandbox.run_python(code, data)

    def verify_claim(claim: str, pmids: list) -> dict:
        """Check whether the cited papers support a claim. Returns supported, score and the best passage."""
        return verifier.verify(claim, pmids).to_dict()

    tools = {f.__name__: f for f in (search_papers, retrieve_passages, get_document,
                                     run_python_analysis, verify_claim)}

    if pubmed is not None:
        def fetch_pubmed(query: str, max_results: int = 10, year_min: int | None = None,
                         year_max: int | None = None) -> dict:
            """Search live PubMed and add the results to the local corpus so they can be retrieved."""
            papers = pubmed.search_and_fetch(query, retmax=max(1, min(max_results, 50)),
                                             year_min=year_min, year_max=year_max)
            added = store.add_papers(papers)
            return {"added": added, "pmids": [p.pmid for p in papers]}

        tools["fetch_pubmed"] = fetch_pubmed
    return tools


# --------------------------------------------------------------------------------------------
# Toolboxes: what the agent actually talks to
# --------------------------------------------------------------------------------------------
class LocalToolbox:
    def __init__(self, tools: dict):
        self.tools = tools

    def names(self) -> list:
        return list(self.tools)

    def describe(self) -> str:
        lines = []
        for name, fn in self.tools.items():
            sig = str(inspect.signature(fn)).replace("'", "")
            doc = (inspect.getdoc(fn) or "").split("\n")[0]
            lines.append(f"- {name}{sig}: {doc}")
        return "\n".join(lines)

    def call(self, name: str, args: dict):
        fn = self.tools.get(name)
        if fn is None:
            return {"error": f"unknown tool '{name}'. Available: {', '.join(self.tools)}"}
        try:
            return fn(**(args or {}))
        except TypeError as e:
            return {"error": f"bad arguments for {name}: {e}"}
        except Exception as e:  # tool failures become observations the agent can react to
            return {"error": f"{type(e).__name__}: {e}"}


class McpToolbox:
    """Discovers and calls tools on an MCP server (stdio command, URL, or in-process MCPServer)."""

    def __init__(self, target, timeout: float = 60.0):
        self._target, self._timeout = target, timeout
        self._loop = asyncio.new_event_loop()
        self._ready, self._error, self._tools, self._client = threading.Event(), None, [], None
        self._stop = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout) or self._error:
            raise RuntimeError(f"Could not connect to MCP server: {self._error or 'timeout'}")

    @classmethod
    def stdio(cls, *extra_args: str, **kw) -> "McpToolbox":
        import os

        from mcp import StdioServerParameters

        return cls(StdioServerParameters(command=sys.executable, env=dict(os.environ),
                                         args=["-m", "biolens.mcp_server", *extra_args]), **kw)

    def _run(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._main())

    async def _main(self):
        from mcp import Client

        self._stop = asyncio.Event()
        try:
            async with Client(self._target) as c:
                self._client = c
                self._tools = (await c.list_tools()).tools
                self._ready.set()
                await self._stop.wait()
        except Exception as e:
            self._error = e
            self._ready.set()

    def names(self) -> list:
        return [t.name for t in self._tools]

    def describe(self) -> str:
        lines = []
        for t in self._tools:
            props = (t.input_schema or {}).get("properties", {}) if hasattr(t, "input_schema") else {}
            args = ", ".join(f"{k}: {v.get('type', 'any')}" for k, v in props.items())
            lines.append(f"- {t.name}({args}): {(t.description or '').strip().splitlines()[0] if t.description else ''}")
        return "\n".join(lines)

    def call(self, name: str, args: dict):
        if name not in self.names():
            return {"error": f"unknown tool '{name}'. Available: {', '.join(self.names())}"}
        fut = asyncio.run_coroutine_threadsafe(self._client.call_tool(name, args or {}), self._loop)
        try:
            res = fut.result(self._timeout)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
        text = "".join(getattr(c, "text", "") for c in res.content)
        if res.is_error:
            return {"error": text or "tool error"}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"text": text}

    def close(self):
        if self._stop is not None:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=10)
