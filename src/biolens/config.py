"""Settings (environment variables, overridable per call) and the factory that wires everything together."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path


@dataclass
class Config:
    home: str = field(default_factory=lambda: os.getenv("BIOLENS_HOME", ".biolens"))
    model: str = field(default_factory=lambda: os.getenv("BIOLENS_MODEL", "mock"))
    embedder: str = field(default_factory=lambda: os.getenv("BIOLENS_EMBEDDER", "hash"))
    corpus_db: str | None = None
    state_db: str | None = None
    pubmed: bool = field(default_factory=lambda: os.getenv("BIOLENS_PUBMED", "0") == "1")
    judge: bool = False            # use the LLM as an entailment judge in citation verification
    use_mcp: bool = False          # agent reaches its tools through an MCP server (stdio subprocess)

    def __post_init__(self):
        Path(self.home).mkdir(parents=True, exist_ok=True)
        self.corpus_db = self.corpus_db or os.path.join(self.home, "corpus.db")
        self.state_db = self.state_db or os.path.join(self.home, "state.db")


def demo_path(name: str) -> str:
    return str(resources.files("biolens").joinpath("data", name))


def load_papers(path: str) -> list:
    from .models import Paper

    with open(path, encoding="utf-8") as f:
        return [Paper.from_dict(json.loads(line)) for line in f if line.strip()]


@dataclass
class Components:
    config: Config
    embedder: object
    store: object
    retriever: object
    verifier: object
    tools: dict
    state: object
    tracer: object
    memory: object


def build_components(config: Config | None = None) -> Components:
    from .embeddings import get_embedder
    from .memory import Memory
    from .pubmed import PubMedClient
    from .retrieval import HybridRetriever
    from .state import StateDB
    from .store import CorpusStore
    from .tools import build_tools
    from .tracing import Tracer
    from .verify import CitationVerifier

    config = config or Config()
    embedder = get_embedder(config.embedder)
    store = CorpusStore(config.corpus_db, embedder)
    retriever = HybridRetriever(store, embedder)
    verifier = CitationVerifier(store, embedder)
    tools = build_tools(retriever, store, verifier, PubMedClient() if config.pubmed else None)
    state = StateDB(config.state_db)
    return Components(config, embedder, store, retriever, verifier, tools, state, Tracer(state),
                      Memory(state, embedder))
