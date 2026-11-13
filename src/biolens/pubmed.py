"""Minimal NCBI E-utilities client (esearch + efetch). Free; an API key raises the rate limit.

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25497/
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from .models import Paper

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def _text(el) -> str:
    """All text inside an element, including inline markup like <i> or <sub>."""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def parse_pubmed_xml(xml_text: str) -> list:
    """Parse an efetch PubmedArticleSet into Papers. Articles without an abstract are skipped."""
    papers = []
    root = ET.fromstring(xml_text)
    for art in root.iter("PubmedArticle"):
        cit = art.find("MedlineCitation")
        if cit is None:
            continue
        pmid = _text(cit.find("PMID"))
        article = cit.find("Article")
        if not pmid or article is None:
            continue
        parts = []
        for ab in article.findall("Abstract/AbstractText"):
            label = ab.get("Label")
            parts.append(f"{label}: {_text(ab)}" if label else _text(ab))
        abstract = " ".join(p for p in parts if p)
        if not abstract:
            continue
        year = 0
        pd = article.find("Journal/JournalIssue/PubDate")
        if pd is not None:
            m = re.search(r"\d{4}", _text(pd.find("Year")) or _text(pd.find("MedlineDate")))
            year = int(m.group()) if m else 0
        authors = []
        for a in article.findall("AuthorList/Author"):
            last, ini = _text(a.find("LastName")), _text(a.find("Initials"))
            if last:
                authors.append(f"{last} {ini}".strip())
        ptypes = [_text(p) for p in article.findall("PublicationTypeList/PublicationType")]
        papers.append(Paper(
            pmid=pmid, title=_text(article.find("ArticleTitle")).rstrip("."), abstract=abstract,
            journal=_text(article.find("Journal/Title")), year=year,
            pub_type="; ".join(ptypes), authors=authors[:10],
        ))
    return papers


class PubMedClient:
    def __init__(self, email: str | None = None, api_key: str | None = None, tool: str = "biolens"):
        self.email = email or os.getenv("NCBI_EMAIL")
        self.api_key = api_key or os.getenv("NCBI_API_KEY")
        self.tool = tool
        self._min_gap = 0.11 if self.api_key else 0.34  # NCBI: 10 req/s with a key, 3 without
        self._last = 0.0

    def _get(self, endpoint: str, **params) -> str:
        params.update(tool=self.tool)
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        url = f"{BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
        for attempt in range(3):
            wait = self._min_gap - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    return r.read().decode("utf-8")
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError("unreachable")

    def search(self, query: str, retmax: int = 20, year_min: int | None = None,
               year_max: int | None = None) -> list:
        params = dict(db="pubmed", term=query, retmax=retmax, retmode="json", sort="relevance")
        if year_min or year_max:
            params.update(datetype="pdat", mindate=year_min or 1900, maxdate=year_max or 3000)
        data = json.loads(self._get("esearch.fcgi", **params))
        return data["esearchresult"].get("idlist", [])

    def fetch(self, pmids: list) -> list:
        papers = []
        for i in range(0, len(pmids), 100):
            xml = self._get("efetch.fcgi", db="pubmed", id=",".join(pmids[i:i + 100]), retmode="xml")
            papers.extend(parse_pubmed_xml(xml))
        return papers

    def search_and_fetch(self, query: str, retmax: int = 20, **kw) -> list:
        ids = self.search(query, retmax=retmax, **kw)
        return self.fetch(ids) if ids else []
