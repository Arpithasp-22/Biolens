"""Hybrid retrieval: BM25 + dense vectors -> Reciprocal Rank Fusion -> metadata filters -> reranking."""

from __future__ import annotations

import math
from typing import Protocol

from .embeddings import Embedder
from .models import Filters, Hit
from .store import CorpusStore
from .textutil import content_tokens


class Reranker(Protocol):
    def rerank(self, query: str, hits: list) -> list: ...


def rrf(rankings: list, k: int = 60) -> dict:
    """Reciprocal Rank Fusion over several best-first lists of (id, score). Returns {id: fused_score}."""
    fused: dict = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused


class LexicalReranker:
    """IDF-weighted query-term coverage blended with the fused retrieval score. No model needed.

    `coverage` (0-1) is also exposed on each hit so the agent can detect weak evidence.
    """

    def __init__(self, weight: float = 0.7):
        self.weight = weight

    def rerank(self, query: str, hits: list) -> list:
        q = list(dict.fromkeys(content_tokens(query)))
        if not hits or not q:
            return hits
        sets = [set(content_tokens(h.text)) for h in hits]
        n = len(hits)
        idf = {t: math.log(1 + n / (1 + sum(t in s for s in sets))) for t in q}
        total = sum(idf.values()) or 1.0
        top = max((h.score for h in hits), default=1.0) or 1.0
        for h, s in zip(hits, sets):
            cov = sum(idf[t] for t in q if t in s) / total
            h.detail["coverage"] = cov
            h.detail["fused"] = h.score
            h.score = self.weight * cov + (1 - self.weight) * (h.score / top)
        return sorted(hits, key=lambda h: h.score, reverse=True)


class CrossEncoderReranker:
    """Optional: sentence-transformers cross-encoder (pip install biolens[rerank])."""

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self._m = CrossEncoder(model)
        self._lex = LexicalReranker()

    def rerank(self, query: str, hits: list) -> list:
        if not hits:
            return hits
        hits = self._lex.rerank(query, hits)  # keeps `coverage` populated
        scores = self._m.predict([(query, h.text) for h in hits])
        for h, s in zip(hits, scores):
            h.detail["cross_encoder"] = float(s)
            h.score = float(s)
        return sorted(hits, key=lambda h: h.score, reverse=True)


class HybridRetriever:
    def __init__(self, store: CorpusStore, embedder: Embedder, reranker: Reranker | None = None,
                 candidates: int = 30, rrf_k: int = 60):
        self.store, self.embedder = store, embedder
        self.reranker = reranker if reranker is not None else LexicalReranker()
        self.candidates, self.rrf_k = candidates, rrf_k

    def search(self, query: str, k: int = 5, filters: Filters | None = None,
               mode: str = "hybrid", rerank: bool = True) -> list:
        """mode: 'hybrid' | 'bm25' | 'dense'."""
        n = max(self.candidates, k * 4)
        rankings = []
        if mode in ("hybrid", "bm25"):
            rankings.append(self.store.bm25(query, n, filters))
        if mode in ("hybrid", "dense"):
            qv = self.embedder.embed([query])[0]
            rankings.append(self.store.dense(qv, n, filters))
        fused = rrf(rankings, self.rrf_k)
        if not fused:
            return []
        top = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:n]
        chunks = self.store.get_chunks([cid for cid, _ in top])
        hits = []
        for cid, score in top:
            c = chunks.get(cid)
            if c:
                hits.append(Hit(cid, c["pmid"], c["text"], score, c["title"], c["year"],
                                c["journal"], c["pub_type"]))
        if rerank and self.reranker:
            hits = self.reranker.rerank(query, hits)
        return hits[:k]

    def search_papers(self, query: str, k: int = 5, filters: Filters | None = None) -> list:
        """Best passage per paper (paper-level view of the same ranking)."""
        seen, out = set(), []
        for h in self.search(query, k=k * 4, filters=filters):
            if h.pmid not in seen:
                seen.add(h.pmid)
                out.append(h)
            if len(out) >= k:
                break
        return out
