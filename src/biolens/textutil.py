"""Small, dependency-free text helpers shared by retrieval, verification and the mock LLM."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be been being by can could did do does for from had has have how in into is it its
    of on or our than that the their then there these they this those to was were what when where which who
    whom why will with would about after again against all also any because before between both but each few
    more most other over same should so some such than too very we you your i he she them his her not no nor
    only own off out up down s t just""".split()
)

_TOKEN = re.compile(r"[a-z][a-z0-9\-]*|\d+(?:\.\d+)?")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")


def stem(tok: str) -> str:
    """A deliberately tiny suffix stripper; consistent matters more than linguistically right."""
    if tok.isdigit() or len(tok) <= 3:
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        tok = tok[:-3] + "y"
    elif tok.endswith("ing") and len(tok) > 5:
        tok = tok[:-3]
    elif tok.endswith("ed") and len(tok) > 4:
        tok = tok[:-2]
    elif tok.endswith("es") and len(tok) > 4:
        tok = tok[:-2]
    elif tok.endswith("s") and not tok.endswith("ss"):
        tok = tok[:-1]
    if tok.endswith("e") and len(tok) > 3:
        tok = tok[:-1]
    return tok


def tokens(text: str) -> list:
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list:
    """Stemmed non-stopword tokens (numbers kept)."""
    return [stem(t) for t in tokens(text) if t not in STOPWORDS and len(t) > 1 or t.isdigit()]


def numbers(text: str) -> set:
    return set(_NUMBER.findall(text))


def sentences(text: str) -> list:
    text = re.sub(r"\s+", " ", text or "").strip()
    return [s for s in _SENT_SPLIT.split(text) if s]


def chunk_text(title: str, abstract: str, max_words: int = 70) -> list:
    """Split an abstract into passages of whole sentences (~max_words). Title leads the first one."""
    sents = sentences(abstract)
    if not sents:
        return [title] if title else []
    chunks, cur, n = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and n + w > max_words:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(s)
        n += w
    if cur:
        chunks.append(" ".join(cur))
    if title:
        chunks[0] = f"{title}. {chunks[0]}"
    return chunks
