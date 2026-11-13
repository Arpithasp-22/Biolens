"""Evaluation: retrieval quality, answer correctness, citation accuracy, hallucination rate, task completion,
latency/cost, and side-by-side model comparison on identical questions.

Golden file (JSONL), one object per line:
  {"id": "q1", "question": "...", "relevant_pmids": ["123"], "must_include": [["41%", "41 percent"], ["placebo"]],
   "answerable": true, "filters": {"year_min": 2020}}
`must_include` is a list of groups; a group is satisfied if ANY of its strings appears (case-insensitive) in the answer.
`answerable: false` marks questions the corpus cannot answer: the right behaviour is to abstain.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

from .models import Filters


@dataclass
class GoldenItem:
    id: str
    question: str
    relevant_pmids: list = field(default_factory=list)
    must_include: list = field(default_factory=list)
    answerable: bool = True
    filters: dict = field(default_factory=dict)


def load_golden(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [GoldenItem(**json.loads(line)) for line in f if line.strip()]


def percentile(values: list, p: float) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    i = (len(v) - 1) * p / 100
    lo, hi = math.floor(i), math.ceil(i)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def _mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else 0.0


# ------------------------------------------------------------------ retrieval
def retrieval_metrics(retriever, items: list, k: int = 5, mode: str = "hybrid", rerank: bool = True) -> dict:
    """Paper-level recall@k, MRR and nDCG@k over answerable items that have gold papers."""
    rec, mrr, ndcg = [], [], []
    for it in items:
        rel = set(it.relevant_pmids)
        if not it.answerable or not rel:
            continue
        hits = retriever.search(it.question, k=k * 4, filters=Filters.from_dict(it.filters), mode=mode, rerank=rerank)
        ranked = list(dict.fromkeys(h.pmid for h in hits))[:k]
        rec.append(len(rel & set(ranked)) / len(rel))
        mrr.append(next((1 / (i + 1) for i, p in enumerate(ranked) if p in rel), 0.0))
        dcg = sum(1 / math.log2(i + 2) for i, p in enumerate(ranked) if p in rel)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(len(rel), k)))
        ndcg.append(dcg / ideal if ideal else 0.0)
    return {f"recall@{k}": _mean(rec), "mrr": _mean(mrr), f"ndcg@{k}": _mean(ndcg), "n": len(rec)}


def compare_retrieval(retriever, items: list, k: int = 5) -> list:
    """Ablation: does hybrid + reranking actually beat BM25 or dense alone?"""
    rows = []
    for label, mode, rr in [("bm25", "bm25", False), ("dense", "dense", False),
                            ("hybrid (RRF)", "hybrid", False), ("hybrid + rerank", "hybrid", True)]:
        rows.append({"retriever": label, **retrieval_metrics(retriever, items, k, mode, rr)})
    return rows


# ------------------------------------------------------------------ agent
def _answer_text(result) -> str:
    return " ".join(c["claim"] for c in result.claims).lower()


def score_item(item: GoldenItem, result) -> dict:
    text = _answer_text(result)
    abstained = result.status == "insufficient_evidence"
    cited = {p for c in result.claims for p in c["citations"]}
    rel = set(item.relevant_pmids)
    fd = result.first_draft or {}
    claims_raw = fd.get("claims", 0)

    if item.answerable:
        groups = item.must_include
        correctness = (sum(any(s.lower() in text for s in g) for g in groups) / len(groups)) if groups \
            else float(result.status == "completed")
        success = result.status == "completed" and correctness == 1.0
    else:
        correctness = float(abstained)
        success = abstained

    return {
        "id": item.id, "status": result.status, "success": success, "correctness": correctness,
        "abstained": abstained,
        "citation_precision": (len(cited & rel) / len(cited)) if cited and rel else None,   # vs gold papers
        "citation_recall": (len(cited & rel) / len(rel)) if rel and item.answerable else None,
        "raw_hallucination_rate": (fd.get("unsupported", 0) / claims_raw) if claims_raw else None,
        "fabricated_citations": fd.get("fabricated_citations", 0),
        "self_corrected": result.attempts > 0 and result.status == "completed",
        "latency_ms": result.metrics.get("latency_ms", 0.0),
        "tokens": result.metrics.get("tokens_in", 0) + result.metrics.get("tokens_out", 0),
        "cost_usd": result.metrics.get("cost_usd", 0.0), "cost_known": result.metrics.get("cost_known", True),
        "tool_calls": result.metrics.get("tool_calls", 0), "errors": result.metrics.get("errors", 0),
    }


def _avg(rows: list, key: str):
    vals = [r[key] for r in rows if r[key] is not None]
    return _mean(vals) if vals else None


def evaluate_agent(agent, items: list, label: str = "") -> dict:
    rows = [score_item(it, agent.run(it.question)) for it in items]
    ans = [r for r, it in zip(rows, items) if it.answerable]
    unans = [r for r, it in zip(rows, items) if not it.answerable]
    lat = [r["latency_ms"] for r in rows]
    summary = {
        "model": label or agent.llm.name, "n": len(rows),
        "task_completion": _avg(rows, "success"),
        "answer_correctness": _avg(ans, "correctness"),
        "citation_precision": _avg(rows, "citation_precision"),
        "citation_recall": _avg(rows, "citation_recall"),
        "raw_hallucination_rate": _avg(rows, "raw_hallucination_rate"),
        "fabricated_citations": sum(r["fabricated_citations"] for r in rows),
        "abstention_accuracy": _avg(unans, "correctness") if unans else None,
        "self_corrections": sum(r["self_corrected"] for r in rows),
        "latency_p50_ms": percentile(lat, 50), "latency_p95_ms": percentile(lat, 95),
        "tokens": sum(r["tokens"] for r in rows),
        "cost_usd": sum(r["cost_usd"] for r in rows) if all(r["cost_known"] for r in rows) else None,
        "errors": sum(r["errors"] for r in rows),
    }
    return {"summary": summary, "items": rows}


def run_comparison(model_specs: list, items: list, make_agent) -> list:
    """make_agent(model_spec) -> ResearchAgent. Same questions, same corpus, one row per model."""
    return [evaluate_agent(make_agent(spec), items, spec) for spec in model_specs]


# ------------------------------------------------------------------ reporting
def _fmt(v, pct=False, money=False):
    if v is None:
        return "n/a"
    if money:
        return f"${v:.4f}"
    return f"{v:.0%}" if pct else (f"{v:,.0f}" if isinstance(v, float) and v > 50 else f"{v:.2f}" if isinstance(v, float) else str(v))


def to_markdown(results: list) -> str:
    cols = [("model", "Model", {}), ("task_completion", "Task completion", {"pct": 1}),
            ("answer_correctness", "Correctness", {"pct": 1}), ("citation_precision", "Citation precision", {"pct": 1}),
            ("citation_recall", "Citation recall", {"pct": 1}), ("raw_hallucination_rate", "Raw hallucination", {"pct": 1}),
            ("abstention_accuracy", "Abstains correctly", {"pct": 1}), ("self_corrections", "Self-corrections", {}),
            ("latency_p50_ms", "p50 ms", {}), ("latency_p95_ms", "p95 ms", {}), ("tokens", "Tokens", {}),
            ("cost_usd", "Cost", {"money": 1})]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    for r in results:
        s = r["summary"]
        lines.append("| " + " | ".join(str(s[k]) if k == "model" else _fmt(s[k], **kw) for k, _, kw in cols) + " |")
    return "\n".join(lines)


def retrieval_markdown(rows: list) -> str:
    keys = [k for k in rows[0] if k not in ("retriever", "n")]
    lines = ["| Retriever | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    for r in rows:
        lines.append(f"| {r['retriever']} | " + " | ".join(f"{r[k]:.2f}" for k in keys) + " |")
    return "\n".join(lines)


def save_results(home: str, results: list, retrieval: list | None = None) -> str:
    os.makedirs(os.path.join(home, "evals"), exist_ok=True)
    path = os.path.join(home, "evals", time.strftime("eval-%Y%m%d-%H%M%S.json"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "retrieval": retrieval}, f, indent=2, default=str)
    return path
