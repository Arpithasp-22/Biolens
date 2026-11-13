import asyncio
import json
import sys

import pytest
from mcp import Client

from biolens import evaluation as ev
from biolens.config import demo_path
from biolens.factory import build_agent
from biolens.mcp_server import build_server
from biolens.mock_llm import MockLLM
from biolens.tools import LocalToolbox, McpToolbox

Q1 = "Does zorvalin reduce relapse in Marrow-Hartwell syndrome?"


# ---------- MCP ----------
def test_mcp_server_lists_and_calls_tools_in_process(comp):
    server = build_server(comp.tools)

    async def go():
        async with Client(server) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            res = await c.call_tool("retrieve_passages", {"query": "hepatic enzyme lumeral", "k": 2})
            calc = await c.call_tool("run_python_analysis", {"code": "result = sum(data)", "data": [1, 2, 3]})
            return names, json.loads(res.content[0].text), json.loads(calc.content[0].text)

    names, res, calc = asyncio.run(go())
    assert {"search_papers", "retrieve_passages", "get_document", "run_python_analysis", "verify_claim"} <= names
    assert res["passages"][0]["pmid"] == "DEMO0004"
    assert calc["result"] == 6


def test_agent_uses_tools_over_mcp_stdio(comp):
    """Full path: agent -> MCP client -> separate server process -> same corpus."""
    tb = McpToolbox.stdio("--home", comp.config.home)
    try:
        assert "retrieve_passages" in tb.names() and "retrieve_passages" in tb.describe()
        assert tb.call("get_document", {"pmid": "DEMO0001"})["paper"]["year"] == 2022
        assert "error" in tb.call("get_document", {"pmid": "nope"})
        assert "error" in tb.call("no_such_tool", {})
        comp.config.use_mcp = False
        from biolens.agents import ResearchAgent
        from biolens.tracing import TracedLLM, TracedToolbox
        agent = ResearchAgent(TracedLLM(MockLLM(), comp.tracer), TracedToolbox(tb, comp.tracer), comp.verifier,
                              comp.memory, comp.tracer, comp.store)
        r = agent.run(Q1)
        assert r.status == "completed" and "41%" in r.answer
        assert r.metrics["tool_breakdown"] == {"retrieve_passages": 1}
    finally:
        tb.close()


def test_local_toolbox_describes_and_rejects_bad_calls(comp):
    tb = LocalToolbox(comp.tools)
    assert "retrieve_passages(query: str" in tb.describe()
    assert "unknown tool" in tb.call("nope", {})["error"]
    assert "bad arguments" in tb.call("get_document", {"wrong": 1})["error"]


# ---------- evaluation ----------
def test_percentile():
    assert ev.percentile([], 50) == 0 and ev.percentile([10], 95) == 10
    assert ev.percentile([1, 2, 3, 4], 50) == 2.5 and ev.percentile(list(range(101)), 95) == 95


def test_retrieval_metrics_and_ablation(comp):
    items = ev.load_golden(demo_path("demo_golden.jsonl"))
    m = ev.retrieval_metrics(comp.retriever, items, k=5)
    assert m["recall@5"] >= 0.9 and m["n"] == 7
    rows = ev.compare_retrieval(comp.retriever, items)
    assert [r["retriever"] for r in rows] == ["bm25", "dense", "hybrid (RRF)", "hybrid + rerank"]
    assert "| hybrid + rerank |" in ev.retrieval_markdown(rows)


def test_agent_eval_scores_and_catches_hallucination(comp):
    items = ev.load_golden(demo_path("demo_golden.jsonl"))
    good = ev.evaluate_agent(build_agent(comp, llm=MockLLM()), items, "clean")["summary"]
    bad = ev.evaluate_agent(build_agent(comp, llm=MockLLM(hallucinate_first=True)), items, "hallucinating")["summary"]
    assert good["abstention_accuracy"] == 1.0                      # refuses the unanswerable question
    assert good["raw_hallucination_rate"] == 0.0 and bad["raw_hallucination_rate"] > 0.15
    assert bad["self_corrections"] > 0                             # safety net engaged
    assert bad["task_completion"] == good["task_completion"]       # ...and final answers are no worse for it
    assert good["cost_usd"] == 0.0 and good["latency_p95_ms"] >= good["latency_p50_ms"] > 0


def test_score_item_rules(comp):
    r = build_agent(comp, llm=MockLLM()).run(Q1)
    ok = ev.score_item(ev.GoldenItem("a", Q1, ["DEMO0001"], [["41%"]], True), r)
    miss = ev.score_item(ev.GoldenItem("b", Q1, ["DEMO0001"], [["99%"]], True), r)
    wrong = ev.score_item(ev.GoldenItem("c", Q1, [], [], False), r)       # should have abstained
    assert ok["success"] and not miss["success"] and not wrong["success"]
    assert 0 < ok["citation_precision"] <= 1 and ok["citation_recall"] == 1.0


def test_comparison_markdown(comp):
    items = ev.load_golden(demo_path("demo_golden.jsonl"))[:2]
    res = ev.run_comparison(["mock", "mock"], items, lambda s: build_agent(comp, llm=MockLLM()))
    md = ev.to_markdown(res)
    assert md.count("\n") == 3 and "Task completion" in md and "| mock |" in md


# ---------- CLI ----------
def test_cli_flow(tmp_path, capsys):
    from biolens.cli import main
    home = str(tmp_path / "h")
    main(["--home", home, "ingest-demo"])
    main(["--home", home, "ask", Q1])
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out and "[completed]" in out and "41%" in out
    main(["--home", home, "runs"])
    run_id = capsys.readouterr().out.split()[0]
    main(["--home", home, "trace", run_id])
    assert "retrieve_passages" in capsys.readouterr().out
    main(["--home", home, "retrieve", "hepatic lumeral", "-k", "1"])
    assert "DEMO0004" in capsys.readouterr().out
    main(["--home", home, "eval", "--retrieval-only"])
    assert "Retrieval ablation" in capsys.readouterr().out


# ---------- dashboard ----------
def test_dashboard_renders(comp, make_agent):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    make_agent().run(Q1)
    make_agent().run("What is the effect of zorvalin on Parkinson's disease?")

    def app(db_path):
        from biolens import dashboard
        runs, spans = dashboard.load(db_path)
        dashboard.render(runs, spans)

    at = AppTest.from_function(app, args=(comp.config.state_db,)).run(timeout=60)
    assert not at.exception
    assert [m.label for m in at.metric][:2] == ["Runs", "LLM calls"] and at.metric[0].value == "2"
    assert at.selectbox  # run drill-down present
