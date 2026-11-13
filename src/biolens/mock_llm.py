"""Deterministic stand-in LLM so the whole pipeline (agent, tracing, memory, evals, MCP) runs offline and in CI.

It is NOT a quality benchmark: it parses the prompts and answers extractively. Use it to test plumbing;
use real models (openai:/gemini:/ollama:) for any number you intend to report.
"""

from __future__ import annotations

import json
import re
import time

from .llm import LLMResult
from .textutil import content_tokens, sentences, stem

# tiny paraphrase table so "side effects" can match "adverse events", etc. (keys are stemmed on load)
_SYN = {"side": ["adverse"], "effective": ["efficacy", "reduced", "reduction"], "effect": ["adverse", "efficacy"],
        "safe": ["safety", "malformations", "adverse"], "reduce": ["reduction", "reduced"],
        "compare": ["versus", "pooled", "than"], "link": ["associated", "association"], "severity": ["severe"],
        "marker": ["variant"], "genetic": ["variant", "genetics"], "improve": ["improved", "reduced"]}
SYN = {stem(k): {stem(v) for v in vs} for k, vs in _SYN.items()}


def _tok(n: float) -> int:
    return max(1, int(n * 1.3))


class MockLLM:
    name = "mock"

    def __init__(self, hallucinate_first: bool = False, fail_roles: tuple = ()):
        self.hallucinate_first = hallucinate_first
        self.fail_roles = fail_roles
        self.calls = []

    def complete(self, role, system, user, *, json_mode=False, temperature=0.0, max_tokens=1500):
        t0 = time.time()
        self.calls.append(role)
        if role in self.fail_roles:
            raise RuntimeError(f"mock failure injected for role '{role}'")
        out = getattr(self, f"_{role}", self._unknown)(user)
        text = json.dumps(out)
        return LLMResult(text, _tok(len((system + user).split())), _tok(len(text.split())), "mock",
                         (time.time() - t0) * 1000)

    def _unknown(self, user):
        return {}

    # ---- roles ----
    def _planner(self, user):
        q = re.search(r"QUESTION: (.*)", user).group(1)
        filters = {}
        m = re.search(r"\b(?:since|after|from)\s+(\d{4})", q, re.I)
        if m:
            filters["year_min"] = int(m.group(1))
        return {"sub_questions": [q], "filters": filters, "notes": "single-step question"}

    def _researcher(self, user):
        steps = int(re.search(r"STEPS_TAKEN: (\d+)", user).group(1))
        if steps > 0:
            return {"action": "finish", "reason": "evidence collected"}
        sub = re.search(r"SUB_QUESTION: (.*)", user).group(1)
        hint = re.search(r"QUERY_HINT: (.*)", user)
        filters = json.loads(re.search(r"FILTERS: (.*)", user).group(1))
        args = {"query": hint.group(1) if hint else sub, "k": 5, **filters}
        return {"action": "tool", "tool": "retrieve_passages", "args": args, "reason": "gather evidence"}

    def _writer(self, user):
        q = re.search(r"QUESTION: (.*)", user).group(1)
        ev = re.findall(r"^\[(\S+)\] (.*)$", user, re.M)
        qt = list(dict.fromkeys(content_tokens(q)))
        claims, seen = [], set()
        scored = []
        for pmid, text in ev:
            toks = set(content_tokens(text))
            cov = sum(1 for t in qt if t in toks or SYN.get(t, set()) & toks) / max(1, len(qt))
            scored.append((cov, pmid, text))
        for cov, pmid, text in sorted(scored, key=lambda x: -x[0]):
            if cov < 0.6 or pmid in seen or len(claims) >= 3:
                continue
            seen.add(pmid)
            body = [x for x in sentences(text) if not x.startswith("[SYNTHETIC DEMO]")] or [text]
            best = max(body, key=lambda x: len(set(content_tokens(x)) & set(qt)) + 0.5 * bool(re.search(r"\d", x)))
            claims.append({"text": best, "citations": [pmid]})
        if not claims:
            return {"claims": [], "insufficient": True, "note": "retrieved passages do not address the question"}
        if self.hallucinate_first and "FEEDBACK:" not in user:
            claims.append({"text": "The treatment improved outcomes by 73% in a trial of 900 patients.",
                           "citations": [claims[0]["citations"][0]]})
        return {"claims": claims, "insufficient": False, "note": ""}

    def _judge(self, user):
        claim = re.search(r"CLAIM: (.*)", user).group(1)
        passage = re.search(r"PASSAGE: (.*)", user, re.S).group(1)
        ct, pt = set(content_tokens(claim)), set(content_tokens(passage))
        return {"supported": len(ct & pt) / max(1, len(ct)) >= 0.7, "reason": "token overlap"}
