"""Build a ready-to-run agent from a Config (model, tools via local functions or MCP, tracing, memory)."""

from __future__ import annotations

from .agents import AgentConfig, ResearchAgent
from .config import Components, Config, build_components
from .llm import get_llm
from .tools import LocalToolbox, McpToolbox
from .tracing import TracedLLM, TracedToolbox
from .verify import CitationVerifier, LLMJudge


def build_agent(comp: Components | None = None, config: Config | None = None, llm=None,
                agent_config: AgentConfig | None = None, approve_plan=None) -> ResearchAgent:
    comp = comp or build_components(config)
    cfg = comp.config
    raw_llm = llm or get_llm(cfg.model)
    traced_llm = TracedLLM(raw_llm, comp.tracer)

    if cfg.use_mcp:
        args = ["--home", cfg.home, "--embedder", cfg.embedder] + (["--pubmed"] if cfg.pubmed else [])
        toolbox = McpToolbox.stdio(*args)
    else:
        toolbox = LocalToolbox(comp.tools)

    verifier = comp.verifier
    if cfg.judge:
        verifier = CitationVerifier(comp.store, comp.embedder, judge=LLMJudge(traced_llm))
    return ResearchAgent(traced_llm, TracedToolbox(toolbox, comp.tracer), verifier, comp.memory,
                         comp.tracer, comp.store, agent_config, approve_plan)
