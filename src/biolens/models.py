"""Plain data containers used across the project."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Paper:
    pmid: str
    title: str
    abstract: str
    journal: str = ""
    year: int = 0
    pub_type: str = ""
    authors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Paper":
        return Paper(
            pmid=str(d["pmid"]), title=d.get("title", ""), abstract=d.get("abstract", ""),
            journal=d.get("journal", ""), year=int(d.get("year") or 0),
            pub_type=d.get("pub_type", ""), authors=list(d.get("authors", [])),
        )


@dataclass
class Filters:
    year_min: int | None = None
    year_max: int | None = None
    journal: str | None = None       # substring match
    pub_type: str | None = None      # substring match, e.g. "Randomized"

    @staticmethod
    def from_dict(d: dict | None) -> "Filters":
        d = d or {}
        return Filters(
            year_min=int(d["year_min"]) if d.get("year_min") else None,
            year_max=int(d["year_max"]) if d.get("year_max") else None,
            journal=d.get("journal") or None, pub_type=d.get("pub_type") or None,
        )

    def is_empty(self) -> bool:
        return not any((self.year_min, self.year_max, self.journal, self.pub_type))


@dataclass
class Hit:
    chunk_id: int
    pmid: str
    text: str
    score: float
    title: str = ""
    year: int = 0
    journal: str = ""
    pub_type: str = ""
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pmid": self.pmid, "title": self.title, "year": self.year, "journal": self.journal,
            "pub_type": self.pub_type, "text": self.text, "score": round(self.score, 4),
            "coverage": round(self.detail.get("coverage", 0.0), 3),
        }
