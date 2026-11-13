"""Command line: ingest data, ask questions, inspect traces, run evals, serve MCP, open the dashboard."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .config import Config, build_components, demo_path, load_papers
from .factory import build_agent


def _cfg(a) -> Config:
    cfg = Config()
    for k in ("home", "model", "embedder"):
        if getattr(a, k, None):
            setattr(cfg, k, getattr(a, k))
    if getattr(a, "home", None):
        cfg = Config(home=a.home, model=cfg.model, embedder=cfg.embedder)
    cfg.use_mcp = getattr(a, "mcp", False)
    cfg.judge = getattr(a, "judge", False)
    cfg.pubmed = cfg.pubmed or getattr(a, "cmd", "") == "ingest"
    return cfg


def _interactive_approval(plan: dict):
    print("\n── Proposed research plan ──")
    for i, q in enumerate(plan["sub_questions"], 1):
        print(f"  {i}. {q}")
    if plan["filters"]:
        print(f"  filters: {plan['filters']}")
    ans = input("Approve? [Y]es / [n]o / [e]dit sub-questions: ").strip().lower()
    if ans in ("", "y", "yes"):
        return plan
    if ans.startswith("e"):
        subs = input("Sub-questions separated by ';': ")
        return {**plan, "sub_questions": [s.strip() for s in subs.split(";") if s.strip()]}
    return None


def _print_result(r, as_json: bool):
    if as_json:
        print(json.dumps(r.__dict__, indent=2, default=str))
        return
    print(f"\n[{r.status}] run {r.run_id}" + (f"  (self-corrected x{r.attempts})" if r.attempts else ""))
    print(r.answer)
    if r.dropped_claims:
        print(f"\n⚠ {len(r.dropped_claims)} claim(s) removed: not supported by their citations")
    m = r.metrics
    if m:
        cost = f"${m['cost_usd']:.4f}" if m.get("cost_known") else "cost unknown (set BIOLENS_PRICING)"
        print(f"\n{m['llm_calls']} LLM calls · {m['tool_calls']} tool calls · {m['tokens_in'] + m['tokens_out']} tokens "
              f"· {m['latency_ms'] / 1000:.1f}s · {cost} · {m['errors']} errors")
    if r.note and r.status != "completed":
        print(f"note: {r.note}")


def _print_trace(comp, run_id: str):
    spans = comp.tracer.spans(run_id)
    depth = {}
    for s in spans:
        depth[s["id"]] = depth.get(s["parent_id"], -1) + 1
        extra = f"{s['tokens_in']}→{s['tokens_out']} tok" if s["kind"] == "llm" else ""
        flag = f"  ✗ {s['error']}" if s["status"] == "error" else ""
        print(f"{'  ' * depth[s['id']]}{s['kind']:<5} {s['name']:<18} {s['duration_ms']:8.1f} ms  {extra}{flag}")


def main(argv: list | None = None):
    ap = argparse.ArgumentParser(prog="biolens", description=__doc__)
    ap.add_argument("--home", help="data directory (default $BIOLENS_HOME or ./.biolens)")
    ap.add_argument("--model", help="mock | openai:<m> | gemini:<m> | ollama:<m>")
    ap.add_argument("--embedder", help="hash | openai[:m] | st[:m]")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest-demo", help="load the bundled SYNTHETIC demo corpus")
    p = sub.add_parser("ingest", help="fetch papers from PubMed (needs internet)")
    p.add_argument("query"); p.add_argument("--max", type=int, default=50)
    p.add_argument("--year-min", type=int); p.add_argument("--year-max", type=int)
    p = sub.add_parser("ingest-file", help="load papers from a JSONL file"); p.add_argument("path")

    p = sub.add_parser("ask", help="research a question")
    p.add_argument("question"); p.add_argument("--interactive", action="store_true", help="approve the plan first")
    p.add_argument("--mcp", action="store_true", help="reach tools through an MCP server")
    p.add_argument("--judge", action="store_true", help="LLM entailment check for citations")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("resume", help="resume a run from its checkpoint"); p.add_argument("run_id")
    p = sub.add_parser("retrieve", help="search the corpus without the agent")
    p.add_argument("query"); p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", default="hybrid", choices=["hybrid", "bm25", "dense"])
    p.add_argument("--year-min", type=int); p.add_argument("--year-max", type=int)
    sub.add_parser("runs", help="list recent runs")
    p = sub.add_parser("trace", help="show the span tree of a run"); p.add_argument("run_id")
    p = sub.add_parser("eval", help="benchmark one or more models")
    p.add_argument("--golden", default=demo_path("demo_golden.jsonl"))
    p.add_argument("--models", default=None, help="comma-separated, e.g. openai:gpt-4o-mini,gemini:gemini-2.0-flash,ollama:llama3.1")
    p.add_argument("--retrieval-only", action="store_true")
    sub.add_parser("serve-mcp", help="run the MCP server over stdio")
    sub.add_parser("dashboard", help="open the observability dashboard")

    a = ap.parse_args(argv)
    cfg = _cfg(a)

    if a.cmd == "serve-mcp":
        from .mcp_server import main as serve
        return serve(["--home", cfg.home, "--embedder", cfg.embedder])
    if a.cmd == "dashboard":
        from . import dashboard
        return subprocess.call([sys.executable, "-m", "streamlit", "run", dashboard.__file__, "--", "--home", cfg.home])

    comp = build_components(cfg)
    if a.cmd == "ingest-demo":
        n = comp.store.add_papers(load_papers(demo_path("demo_corpus.jsonl")))
        print(f"Loaded {n} SYNTHETIC demo papers (fictional condition/drugs; not real medical evidence).")
    elif a.cmd == "ingest":
        from .pubmed import PubMedClient
        papers = PubMedClient().search_and_fetch(a.query, retmax=a.max, year_min=a.year_min, year_max=a.year_max)
        print(f"Fetched {len(papers)} papers with abstracts; indexed {comp.store.add_papers(papers)}.")
    elif a.cmd == "ingest-file":
        print(f"Indexed {comp.store.add_papers(load_papers(a.path))} papers.")
    elif a.cmd == "ask":
        agent = build_agent(comp, approve_plan=_interactive_approval if a.interactive else None)
        _print_result(agent.run(a.question), a.json)
    elif a.cmd == "resume":
        _print_result(build_agent(comp).run("", run_id=a.run_id), False)
    elif a.cmd == "retrieve":
        from .models import Filters
        f = Filters(year_min=a.year_min, year_max=a.year_max)
        for h in comp.retriever.search(a.query, k=a.k, filters=f, mode=a.mode):
            print(f"{h.score:.3f}  [{h.pmid}] ({h.year}, {h.pub_type})  {h.text[:110]}…")
    elif a.cmd == "runs":
        for r in comp.memory.list_runs():
            print(f"{r['id']}  {r['status']:<22} {r['model']:<22} {r['question'][:60]}")
    elif a.cmd == "trace":
        _print_trace(comp, a.run_id)
    elif a.cmd == "eval":
        from . import evaluation as ev
        items = ev.load_golden(a.golden)
        retr = ev.compare_retrieval(comp.retriever, items)
        print("Retrieval ablation\n" + ev.retrieval_markdown(retr))
        results = []
        if not a.retrieval_only:
            specs = (a.models or cfg.model).split(",")
            results = ev.run_comparison(specs, items, lambda s: build_agent(comp, llm=__import__("biolens.llm", fromlist=["x"]).get_llm(s)))
            print("\nAgent comparison\n" + ev.to_markdown(results))
        print("\nSaved:", ev.save_results(cfg.home, results, retr))


if __name__ == "__main__":
    main()
