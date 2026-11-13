import pytest

from biolens.embeddings import HashingEmbedder
from biolens.models import Filters, Paper
from biolens.retrieval import rrf
from biolens.store import CorpusStore
from biolens.textutil import chunk_text, content_tokens, numbers, sentences, stem


def test_stemming_is_consistent():
    assert stem("reduced") == stem("reduce") == stem("reduces")
    assert content_tokens("The relapse rates were reduced by 41%") == [stem("relapse"), stem("rates"), stem("reduce"), "41"]
    assert stem("rates") == stem("rate")


def test_numbers_and_sentences_and_chunks():
    assert numbers("by 41% and 0.61, n=312") == {"41", "0.61", "312"}
    assert len(sentences("One. Two! Three?")) == 3
    chunks = chunk_text("Title", " ".join(f"Sentence number {i} has several words in it." for i in range(30)))
    assert len(chunks) > 1 and chunks[0].startswith("Title.")


def test_rrf_rewards_agreement():
    fused = rrf([[(1, 0), (2, 0), (3, 0)], [(3, 0), (2, 0), (9, 0)]], k=60)
    assert fused[2] > fused[1] and fused[3] > fused[1]
    assert max(fused, key=fused.get) in (2, 3)


def test_corpus_counts_and_upsert(comp):
    assert comp.store.count()["papers"] == 15
    p = comp.store.get_paper("DEMO0004")
    comp.store.add_papers([Paper(p.pmid, "New title", "Replaced abstract about nothing.", p.journal, 2030, p.pub_type)])
    assert comp.store.count()["papers"] == 15
    assert comp.store.get_paper("DEMO0004").year == 2030
    assert not [h for h in comp.retriever.search("hepatic enzyme elevation", k=3) if "hepatic" in h.text]


def test_embedder_mismatch_is_rejected(comp):
    other = HashingEmbedder(dim=64)          # different embedder -> incompatible vectors
    with pytest.raises(ValueError, match="embedder"):
        CorpusStore(comp.config.corpus_db, other)


@pytest.mark.parametrize("mode", ["hybrid", "bm25", "dense"])
def test_retrieval_modes_find_the_right_paper(comp, mode):
    hits = comp.retriever.search("hepatic enzyme elevation lumeral", k=3, mode=mode)
    assert hits[0].pmid == "DEMO0004"


def test_metadata_filters(comp):
    hits = comp.retriever.search("Marrow-Hartwell syndrome", k=15, filters=Filters(year_min=2024))
    assert hits and all(h.year >= 2024 for h in hits)
    hits = comp.retriever.search("Marrow-Hartwell syndrome", k=15, filters=Filters(pub_type="Randomized"))
    assert hits and all("Randomized" in h.pub_type for h in hits)
    assert comp.retriever.search("zorvalin", filters=Filters(year_min=2100)) == []


def test_reranker_exposes_coverage(comp):
    good = comp.retriever.search("zorvalin relapse placebo", k=1)[0]
    bad = comp.retriever.search("quantum chromodynamics", k=1)
    assert good.detail["coverage"] > 0.9
    assert not bad or bad[0].detail["coverage"] < 0.3


def test_search_papers_dedupes(comp):
    pmids = [h.pmid for h in comp.retriever.search_papers("zorvalin", k=5)]
    assert len(pmids) == len(set(pmids))
