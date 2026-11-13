"""Citation verification: does the cited paper actually support the claim?

Default scorer is cheap and deterministic (lexical coverage + embedding similarity + a numeric guard that
catches invented statistics). For real LLM outputs, pass an `LLMJudge` for entailment-style checking;
the heuristic then only picks the passage the judge reads.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .embeddings import Embedder
from .llm import LLM, llm_json
from .store import CorpusStore
from .textutil import content_tokens, numbers, sentences


@dataclass
class ClaimVerdict:
    claim: str
    citations: list
    supported: bool
    score: float
    best_pmid: str | None = None
    best_passage: str = ""
    reason: str = ""
    fabricated: list = field(default_factory=list)  # cited ids that were never retrieved / don't exist

    def to_dict(self) -> dict:
        return asdict(self)


JUDGE_SYSTEM = (
    "You check whether a passage from a research paper supports a claim. Answer strictly from the passage. "
    'Reply JSON: {"supported": true|false, "reason": "<one short sentence>"}. '
    "A claim is supported only if the passage states or clearly implies it, including any numbers."
)


class LLMJudge:
    def __init__(self, llm: LLM):
        self.llm = llm

    def supported(self, claim: str, passage: str) -> tuple:
        out = llm_json(self.llm, "judge", JUDGE_SYSTEM, f"CLAIM: {claim}\n\nPASSAGE: {passage}")
        return bool(out.get("supported")), str(out.get("reason", ""))


class CitationVerifier:
    def __init__(self, store: CorpusStore, embedder: Embedder, threshold: float = 0.5,
                 judge: LLMJudge | None = None, lexical_weight: float = 0.6):
        self.store, self.embedder = store, embedder
        self.threshold, self.judge, self.lw = threshold, judge, lexical_weight

    def _windows(self, pmid: str) -> list:
        paper = self.store.get_paper(pmid)
        if not paper:
            return []
        sents = sentences(paper.abstract)
        wins = list(sents) + [f"{a} {b}" for a, b in zip(sents, sents[1:])]
        if sents:
            wins.append(" ".join(sents))
        return wins or [paper.title]

    def _score_windows(self, claim: str, windows: list) -> tuple:
        ct = set(content_tokens(claim))
        cn = numbers(claim)
        vecs = self.embedder.embed([claim] + windows)
        best, best_i = -1.0, 0
        for i, w in enumerate(windows):
            lex = len(ct & set(content_tokens(w))) / len(ct) if ct else 0.0
            cos = max(0.0, float(vecs[0] @ vecs[i + 1]))
            s = self.lw * lex + (1 - self.lw) * cos
            if cn and not cn <= numbers(w):  # an invented number is the classic hallucination
                s *= 0.5
            if s > best:
                best, best_i = s, i
        return best, windows[best_i]

    def verify(self, claim: str, pmids: list, allowed: set | None = None) -> ClaimVerdict:
        pmids = [str(p) for p in pmids]
        if not pmids:
            return ClaimVerdict(claim, [], False, 0.0, reason="no citation given")
        fabricated = [p for p in pmids if (allowed is not None and p not in allowed)
                      or self.store.get_paper(p) is None]
        valid = [p for p in pmids if p not in fabricated]
        if not valid:
            return ClaimVerdict(claim, pmids, False, 0.0, reason="cited papers were not retrieved/do not exist",
                                fabricated=fabricated)
        best_s, best_p, best_w = -1.0, None, ""
        for p in valid:
            s, w = self._score_windows(claim, self._windows(p))
            if s > best_s:
                best_s, best_p, best_w = s, p, w
        if self.judge:
            ok, why = self.judge.supported(claim, best_w)
            reason = f"judge: {why}"
        else:
            ok = best_s >= self.threshold
            reason = f"score {best_s:.2f} {'>=' if ok else '<'} threshold {self.threshold}"
        return ClaimVerdict(claim, valid, ok, round(best_s, 3), best_p, best_w, reason, fabricated)
