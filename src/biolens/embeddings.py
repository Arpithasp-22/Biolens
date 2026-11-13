"""Embedders. `hash` needs no model or network (good for tests/offline); swap in a real one for quality."""

from __future__ import annotations

import hashlib
import os
from typing import Protocol

import numpy as np

from .textutil import content_tokens


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list) -> np.ndarray: ...


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing over stemmed unigrams + bigrams. Lexical, deterministic, dependency-free."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _idx(self, feat: str) -> tuple:
        h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0

    def embed(self, texts: list) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            toks = content_tokens(text)
            for i, t in enumerate(toks):
                j, s = self._idx(t)
                out[r, j] += s
                if i + 1 < len(toks):
                    j, s = self._idx(t + "_" + toks[i + 1])
                    out[r, j] += 0.5 * s
        return _normalize(out)


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", dim: int = 1536):
        from openai import OpenAI  # lazy: optional dependency

        self._client = OpenAI()
        self.model, self.dim, self.name = model, dim, f"openai-{model}"

    def embed(self, texts: list) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 96):
            resp = self._client.embeddings.create(model=self.model, input=texts[i:i + 96])
            out.extend(d.embedding for d in resp.data)
        m = np.array(out, dtype=np.float32)
        self.dim = m.shape[1]
        return _normalize(m)


class SentenceTransformerEmbedder:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy: optional dependency

        self._m = SentenceTransformer(model)
        self.dim = self._m.get_sentence_embedding_dimension()
        self.name = f"st-{model.split('/')[-1]}"

    def embed(self, texts: list) -> np.ndarray:
        return _normalize(np.asarray(self._m.encode(texts, show_progress_bar=False), dtype=np.float32))


def get_embedder(spec: str | None = None) -> Embedder:
    """spec: 'hash' | 'openai[:model]' | 'st[:model]'. Defaults to $BIOLENS_EMBEDDER or 'hash'."""
    spec = spec or os.getenv("BIOLENS_EMBEDDER", "hash")
    kind, _, arg = spec.partition(":")
    if kind == "hash":
        return HashingEmbedder()
    if kind == "openai":
        return OpenAIEmbedder(arg or "text-embedding-3-small")
    if kind == "st":
        return SentenceTransformerEmbedder(arg or "sentence-transformers/all-MiniLM-L6-v2")
    raise ValueError(f"Unknown embedder '{spec}'")
