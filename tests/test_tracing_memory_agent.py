import pytest

from biolens.agents import ResearchAgent
from biolens.mock_llm import MockLLM
from biolens.tools import LocalToolbox
from biolens.tracing import TracedToolbox

Q1 = "Does zorvalin reduce relapse in Marrow-Hartwell syndrome?"
QUNK = "What is the effect of zorvalin on Parkinson's disease?"


# ---------- tracing ----------
def test_spans_nest_and_record_errors(comp):
    t = comp.tracer
    with t.run("r1"):
        with t.span("agent", "outer"):
            with t.span("tool", "inner"):
                pass
            with pytest.raises(ValueError):
                with t.span("tool", "bad"):
                    raise ValueError("boom")
    rows = {r["name"]: r for r in t.spans("r1")}
    assert rows["inner"]["parent_id"] == rows["outer"]["id"] and rows["outer"]["parent_id"] is None
    assert rows["bad"]["status"] == "error" and "boom" in rows["bad"]["error"]
    s = t.summary("r1")
    assert s["tool_calls"] == 2 and s["errors"] == 1


def test_llm_spans_carry_tokens_and_cost(comp, make_agent):
    r = make_agent().run(Q1)
    m = r.metrics
    assert m["llm_calls"] == 4 and m["tokens_in"] > 0 and m["tokens_out"] > 0
    assert m["cost_known"] and m["cost_usd"] == 0.0           # mock is free
    assert m["tool_breakdown"] == {"retrieve_passages": 1} and m["errors"] == 0 and m["latency_ms"] > 0


def test_unknown_model_cost_is_unknown_not_guessed(comp):
    from biolens.tracing import TracedLLM
    class Fake(MockLLM):
        name = "openai:gpt-unpriced"
    with comp.tracer.run("rx"):
        TracedLLM(Fake(), comp.tracer, pricing={}).complete("planner", "s", "QUESTION: q")
    assert comp.tracer.summary("rx")["cost_known"] is False
    with comp.tracer.run("ry"):
        TracedLLM(Fake(), comp.tracer, pricing={"mock": {"in": 1, "out": 1}, "openai:gpt-unpriced": {"in": 1.0, "out": 2.0}}) \
            .complete("planner", "s", "QUESTION: q")
    assert comp.tracer.summary("ry")["cost_usd"] > 0


# ---------- agent ----------
def test_agent_answers_with_verified_citations(make_agent):
    r = make_agent().run(Q1)
    assert r.status == "completed" and "41%" in r.answer
    assert "DEMO0001" in {p for c in r.claims for p in c["citations"]}
    assert all(c["supported"] for c in r.claims) and r.references


def test_agent_applies_planner_filters(make_agent):
    r = make_agent().run("Did exercise improve walking distance in Marrow-Hartwell syndrome trials since 2020?")
    assert r.plan["filters"] == {"year_min": 2020} and "38 meters" in r.answer


def test_agent_abstains_when_corpus_cannot_answer(make_agent):
    r = make_agent().run(QUNK)
    assert r.status == "insufficient_evidence" and "could not find sufficient evidence" in r.answer
    assert r.claims == []


def test_self_correction_removes_unsupported_claim(make_agent):
    r = make_agent(llm=MockLLM(hallucinate_first=True)).run(Q1)
    assert r.first_draft["unsupported"] == 1 and r.attempts == 1
    assert r.status == "completed" and "73%" not in r.answer
    assert [c for c in r.claims if not c["supported"]] == []


def test_self_correction_disabled_drops_but_reports(comp):
    from biolens.agents import AgentConfig
    from biolens.factory import build_agent
    r = build_agent(comp, llm=MockLLM(hallucinate_first=True), agent_config=AgentConfig(max_retries=0)).run(Q1)
    assert r.attempts == 0 and len(r.dropped_claims) == 1 and "73%" not in r.answer


def test_human_in_the_loop_reject_and_edit(make_agent, comp):
    assert make_agent(approve_plan=lambda plan: None).run(Q1).status == "rejected"
    seen = {}
    def edit(plan):
        seen["plan"] = plan
        return {**plan, "sub_questions": ["Which genetic marker is linked to severity of Marrow-Hartwell syndrome?"]}
    r = make_agent(approve_plan=edit).run(Q1)
    assert seen["plan"]["sub_questions"] == [Q1]
    assert r.plan["sub_questions"][0].startswith("Which genetic marker")          # the edit was applied
    assert "DEMO0006" in r.evidence_pmids                                          # ...and actually researched
    assert r.question == Q1                                                        # the answer still targets the user's question


def test_memory_recalls_findings_in_later_runs(make_agent, comp):
    make_agent().run(Q1)
    r2 = make_agent().run("Is zorvalin effective at lowering relapse?")
    rec = comp.memory.load_run(r2.run_id)["state"]["recalled"]
    assert rec and "zorvalin" in rec[0]["text"].lower()


def test_resume_continues_from_checkpoint(comp):
    from biolens.factory import build_agent
    flaky = MockLLM(fail_roles=("writer",))
    r = build_agent(comp, llm=flaky).run(Q1)
    assert r.status == "failed" and "writer" in r.note
    assert comp.memory.load_run(r.run_id)["stage"] == "researched"      # plan + evidence were saved
    good = MockLLM()
    r2 = build_agent(comp, llm=good).run("", run_id=r.run_id)
    assert r2.status == "completed" and r2.run_id == r.run_id and "41%" in r2.answer
    assert "planner" not in good.calls and "researcher" not in good.calls  # skipped finished stages
    assert r2.question == Q1


def test_tool_errors_become_observations_not_crashes(comp):
    class Broken(LocalToolbox):
        def call(self, name, args):
            return {"error": "backend down"}
    agent = ResearchAgent(MockLLM(), TracedToolbox(Broken({}), comp.tracer), comp.verifier, comp.memory,
                          comp.tracer, comp.store)
    r = agent.run(Q1)
    assert r.status == "insufficient_evidence"                       # no evidence, but no crash
    assert r.metrics["errors"] >= 1


def test_run_ids_are_listed(make_agent, comp):
    r = make_agent().run(Q1)
    assert r.run_id in [x["id"] for x in comp.memory.list_runs()]
