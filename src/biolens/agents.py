"""The research agent.

Design: a *reliable workflow* with autonomy only where it earns its keep.

    recall memory -> PLANNER -> [human approval] -> RESEARCHER (tool loop) -> WRITER -> VERIFIER
                                                         ^                                 |
                                                         +---- self-correction (retry) ----+

- Planner / Researcher / Writer are separate LLM roles with their own prompts and JSON contracts.
- The Researcher is the only autonomous part: it decides which tool to call and when it has enough.
- Writer output is never trusted: every claim is checked against the cited paper; unsupported claims trigger
  one targeted re-research + rewrite, and anything still unsupported is dropped (and reported).
- After every stage the state is checkpointed, so a crashed or interrupted run can be resumed by id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

from .llm import llm_json
from .memory import Memory
from .tracing import Tracer
from .verify import CitationVerifier

PLANNER_SYS = (
    "You are the PLANNER of a biomedical literature research assistant. Break the user's question into 1-3 "
    "focused sub-questions and extract metadata filters ONLY if the user states them (year_min, year_max, "
    "journal, pub_type). Prior findings from earlier research may be provided; use them to avoid repeating work. "
    'Reply JSON: {"sub_questions": [str], "filters": {}, "notes": str}'
)

RESEARCHER_SYS = (
    "You are the RESEARCHER. Gather citable evidence for ONE sub-question using these tools:\n{tools}\n\n"
    "Prefer retrieve_passages for evidence. Use the filters you are given. If results look weak (weak=true), "
    "reformulate the query or relax filters. Call at most one tool per turn. When you have enough evidence, or "
    'after {max_steps} tool calls, finish. Reply JSON: {{"action":"tool","tool":str,"args":{{}},"reason":str}} '
    'or {{"action":"finish","reason":str}}'
)

WRITER_SYS = (
    "You are the WRITER. Answer the question using ONLY the numbered EVIDENCE passages. Write 1-4 claims; each "
    "claim is ONE sentence that the cited passage directly supports. Copy numbers exactly; never add facts that "
    "are not in the evidence; cite only IDs shown in EVIDENCE. If the evidence does not answer the question, "
    'return no claims and set insufficient=true. Reply JSON: {"claims":[{"text":str,"citations":[str]}],'
    '"insufficient":bool,"note":str}'
)

NO_EVIDENCE_ANSWER = "I could not find sufficient evidence in the indexed literature to answer this question."


@dataclass
class AgentConfig:
    max_subquestions: int = 3
    max_steps: int = 3          # tool calls per sub-question
    max_retries: int = 1        # self-correction rounds
    k: int = 5
    max_evidence: int = 12      # passages shown to the writer


@dataclass
class AgentResult:
    run_id: str
    question: str
    status: str                 # completed | insufficient_evidence | rejected | failed
    answer: str = ""
    claims: list = field(default_factory=list)       # verified claims (dicts)
    dropped_claims: list = field(default_factory=list)  # claims removed after failing verification
    references: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    attempts: int = 0
    first_draft: dict = field(default_factory=dict)  # stats before self-correction (for evals)
    evidence_pmids: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    note: str = ""


class ResearchAgent:
    def __init__(self, llm, toolbox, verifier: CitationVerifier, memory: Memory, tracer: Tracer, store,
                 config: AgentConfig | None = None, approve_plan: Callable | None = None):
        self.llm, self.toolbox, self.verifier = llm, toolbox, verifier
        self.memory, self.tracer, self.store = memory, tracer, store
        self.cfg = config or AgentConfig()
        self.approve_plan = approve_plan  # human-in-the-loop: plan -> plan | None (None = reject)

    # ------------------------------------------------------------------ public
    def run(self, question: str, run_id: str | None = None) -> AgentResult:
        """Start a new run, or resume `run_id` from its last checkpoint."""
        if run_id:
            rec = self.memory.load_run(run_id)
            if not rec:
                raise ValueError(f"No run '{run_id}'")
            question, state = rec["question"], rec["state"]
            if rec["status"] == "completed":
                return self._result_from_state(run_id, question, state, "completed")
        else:
            run_id = self.memory.create_run(question, self.llm.name)
            state = {"question": question}
        result = None
        with self.tracer.run(run_id):
            try:
                with self.tracer.span("agent", "research", question=question):
                    result = self._execute(run_id, question, state)
            except Exception as e:  # span already recorded the error; checkpoint stays for resume
                self.memory.save_state(run_id, state, status="failed")
                result = AgentResult(run_id, question, "failed", note=f"{type(e).__name__}: {e}",
                                     plan=state.get("plan", {}))
        result.metrics = self.tracer.summary(run_id)
        return result

    # ------------------------------------------------------------------ workflow
    def _execute(self, run_id: str, question: str, state: dict) -> AgentResult:
        if "plan" not in state:
            with self.tracer.span("agent", "planner"):
                state["recalled"] = self.memory.recall(question, exclude_run=run_id)
                prior = "\n".join(f"- {r['text']} [{', '.join(r['pmids'])}]" for r in state["recalled"]) or "none"
                plan = llm_json(self.llm, "planner", PLANNER_SYS,
                                f"QUESTION: {question}\nPRIOR_FINDINGS:\n{prior}")
                plan = self._clean_plan(plan, question)
            if self.approve_plan:
                with self.tracer.span("agent", "human_approval") as sp:
                    approved = self.approve_plan(plan)
                    sp.set(approved=approved is not None)
                if approved is None:
                    self.memory.save_state(run_id, {**state, "plan": plan}, status="rejected", stage="plan")
                    return AgentResult(run_id, question, "rejected", plan=plan, note="plan rejected by user")
                plan = self._clean_plan(approved, question)
            state["plan"] = plan
            self.memory.save_state(run_id, state, stage="planned")

        plan = state["plan"]
        if "evidence" not in state:
            evidence: list = []
            for sq in plan["sub_questions"]:
                self._research(sq, plan["filters"], evidence)
            state["evidence"] = evidence
            self.memory.save_state(run_id, state, stage="researched")

        evidence = state["evidence"]
        if "verdicts" not in state:
            draft, verdicts = self._write_and_verify(question, evidence, feedback=None)
            state.update(draft=draft, verdicts=verdicts, attempt=0,
                         first_draft={"claims": len(verdicts),
                                      "unsupported": sum(not v["supported"] for v in verdicts),
                                      "fabricated_citations": sum(len(v["fabricated"]) for v in verdicts),
                                      "insufficient": bool(draft.get("insufficient"))})
            self.memory.save_state(run_id, state, stage="drafted")

        # self-correction: re-research and rewrite when claims fail verification (or evidence looked empty)
        while state["attempt"] < self.cfg.max_retries and self._needs_retry(state):
            state["attempt"] += 1
            with self.tracer.span("agent", "self_correct", attempt=state["attempt"]):
                bad = [v["claim"] for v in state["verdicts"] if not v["supported"]]
                if bad:
                    for claim in bad:
                        self._research(question, {}, evidence, hint=claim)
                else:  # nothing usable found: broaden (drop filters, query by the raw question)
                    self._research(question, {}, evidence, hint=question)
                feedback = ("These claims were NOT supported by their citations; fix or remove them: "
                            + " | ".join(bad)) if bad else "Previous evidence was insufficient; use the new evidence."
                draft, verdicts = self._write_and_verify(question, evidence, feedback=feedback)
                state.update(draft=draft, verdicts=verdicts, evidence=evidence)
            self.memory.save_state(run_id, state, stage=f"corrected-{state['attempt']}")

        return self._finalize(run_id, question, state)

    # ------------------------------------------------------------------ stages
    def _clean_plan(self, plan: dict, question: str) -> dict:
        subs = [str(s).strip() for s in (plan.get("sub_questions") or []) if str(s).strip()]
        filters = {k: v for k, v in (plan.get("filters") or {}).items()
                   if k in ("year_min", "year_max", "journal", "pub_type") and v}
        return {"sub_questions": (subs or [question])[: self.cfg.max_subquestions], "filters": filters,
                "notes": str(plan.get("notes", ""))}

    def _research(self, sub_question: str, filters: dict, evidence: list, hint: str | None = None):
        steps, last = 0, "none"
        with self.tracer.span("agent", "researcher", sub_question=sub_question):
            system = RESEARCHER_SYS.format(tools=self.toolbox.describe(), max_steps=self.cfg.max_steps)
            while steps < self.cfg.max_steps:
                user = (f"SUB_QUESTION: {sub_question}\nFILTERS: {json.dumps(filters)}\nSTEPS_TAKEN: {steps}\n"
                        f"EVIDENCE_COLLECTED: {len(evidence)} passages from "
                        f"{len({e['pmid'] for e in evidence})} papers\nLAST_RESULT: {last}\n")
                if hint:
                    user += f"QUERY_HINT: {hint}\n"
                action = llm_json(self.llm, "researcher", system, user)
                if action.get("action") != "tool":
                    break
                steps += 1
                try:
                    out = self.toolbox.call(str(action.get("tool")), action.get("args") or {})
                except RuntimeError as e:  # tool failed: tell the agent, let it adapt
                    last = f"ERROR {e}"
                    continue
                self._collect(out, evidence)
                last = self._summarize(out)

    @staticmethod
    def _collect(out, evidence: list):
        if not isinstance(out, dict):
            return
        have = {(e["pmid"], e["text"]) for e in evidence}
        for p in out.get("passages") or []:
            if p.get("pmid") and p.get("text") and (p["pmid"], p["text"]) not in have:
                evidence.append({k: p.get(k) for k in ("pmid", "text", "title", "year", "journal")})

    @staticmethod
    def _summarize(out) -> str:
        if not isinstance(out, dict):
            return str(out)[:200]
        if "passages" in out:
            return f"{len(out['passages'])} passages, max_coverage={out.get('max_coverage')}, weak={out.get('weak')}"
        return json.dumps(out)[:300]

    def _write_and_verify(self, question: str, evidence: list, feedback: str | None) -> tuple:
        shown = evidence[: self.cfg.max_evidence]
        with self.tracer.span("agent", "writer"):
            lines = "\n".join(f"[{e['pmid']}] {e['text']}" for e in shown) or "(no evidence found)"
            user = f"QUESTION: {question}\nEVIDENCE:\n{lines}\n"
            if feedback:
                user += f"FEEDBACK: {feedback}\n"
            draft = llm_json(self.llm, "writer", WRITER_SYS, user)
        claims = [{"text": str(c.get("text", "")).strip(), "citations": [str(x) for x in c.get("citations", [])]}
                  for c in (draft.get("claims") or []) if str(c.get("text", "")).strip()]
        allowed = {e["pmid"] for e in evidence}
        with self.tracer.span("agent", "verifier", claims=len(claims)) as sp:
            verdicts = [self.verifier.verify(c["text"], c["citations"], allowed).to_dict() for c in claims]
            sp.set(unsupported=sum(not v["supported"] for v in verdicts))
        draft["claims"] = claims
        return draft, verdicts

    @staticmethod
    def _needs_retry(state: dict) -> bool:
        if any(not v["supported"] for v in state["verdicts"]):
            return True
        return bool(state["draft"].get("insufficient")) and not state["verdicts"]

    # ------------------------------------------------------------------ output
    def _finalize(self, run_id: str, question: str, state: dict) -> AgentResult:
        status = "completed" if any(v["supported"] for v in state["verdicts"]) else "insufficient_evidence"
        self.memory.save_state(run_id, state, status=status, stage="done")
        if status == "completed":
            good = [v for v in state["verdicts"] if v["supported"]]
            self.memory.add_findings(run_id, [{"text": v["claim"], "pmids": v["citations"]} for v in good])
        return self._result_from_state(run_id, question, state, status)

    def _result_from_state(self, run_id: str, question: str, state: dict, status: str) -> AgentResult:
        verdicts = state.get("verdicts", [])
        good = [v for v in verdicts if v["supported"]]
        dropped = [v for v in verdicts if not v["supported"]]
        pmids = list(dict.fromkeys(p for v in good for p in v["citations"]))
        refs = []
        for p in pmids:
            paper = self.store.get_paper(p)
            if paper:
                refs.append({"pmid": p, "title": paper.title, "journal": paper.journal, "year": paper.year})
        if good:
            body = "\n".join(f"• {v['claim']} [{', '.join(v['citations'])}]" for v in good)
            body += "\n\nSources:\n" + "\n".join(
                f"[{r['pmid']}] {r['title']} ({r['journal']}, {r['year']})" for r in refs)
        else:
            body = NO_EVIDENCE_ANSWER
        return AgentResult(
            run_id, question, status, body, good, dropped, refs, state.get("plan", {}), state.get("attempt", 0),
            state.get("first_draft", {}), sorted({e["pmid"] for e in state.get("evidence", [])}),
            note=state.get("draft", {}).get("note", ""))
