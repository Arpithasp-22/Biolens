"""MCP server exposing the research tools (paper search, passage retrieval, document fetch, Python analysis,
claim verification) to any MCP-capable client: Claude Desktop, IDEs, or this project's own agent.

    python -m biolens.mcp_server --home .biolens            # stdio (default)
    python -m biolens.mcp_server --transport streamable-http

Claude Desktop config:
    {"mcpServers": {"biolens": {"command": "python", "args": ["-m", "biolens.mcp_server", "--home", "/path/.biolens"]}}}
"""

from __future__ import annotations

import argparse
import inspect

from mcp.server.mcpserver import MCPServer

from .config import Config, build_components

INSTRUCTIONS = (
    "Biomedical literature tools. Use retrieve_passages to gather citable evidence, search_papers to survey "
    "a topic, get_document for a full abstract, run_python_analysis for small calculations, and verify_claim "
    "to check that a cited paper really supports a statement. Always cite paper IDs."
)


def build_server(tools: dict, name: str = "biolens") -> MCPServer:
    server = MCPServer(name, instructions=INSTRUCTIONS)
    for tool_name, fn in tools.items():
        server.add_tool(fn, name=tool_name, description=inspect.getdoc(fn))
    return server


def main(argv: list | None = None):
    ap = argparse.ArgumentParser(description="biolens MCP server")
    ap.add_argument("--home", default=None, help="directory holding corpus.db / state.db")
    ap.add_argument("--embedder", default=None)
    ap.add_argument("--pubmed", action="store_true", help="also expose fetch_pubmed (live PubMed ingestion)")
    ap.add_argument("--transport", default="stdio", choices=["stdio", "sse", "streamable-http"])
    a = ap.parse_args(argv)
    cfg = Config(**{k: v for k, v in {"home": a.home, "embedder": a.embedder}.items() if v})
    cfg.pubmed = cfg.pubmed or a.pubmed
    build_server(build_components(cfg).tools).run(a.transport)


if __name__ == "__main__":
    main()
