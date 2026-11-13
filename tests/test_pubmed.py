from biolens.pubmed import parse_pubmed_xml

XML = """<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle><MedlineCitation><PMID Version="1">111</PMID><Article>
  <Journal><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue><Title>Test Journal</Title></Journal>
  <ArticleTitle>Effect of <i>X</i> on Y.</ArticleTitle>
  <Abstract><AbstractText Label="BACKGROUND">Y is common.</AbstractText>
   <AbstractText Label="RESULTS">X reduced Y by 20% (CO<sub>2</sub> adjusted).</AbstractText></Abstract>
  <AuthorList><Author><LastName>Smith</LastName><Initials>J</Initials></Author></AuthorList>
  <PublicationTypeList><PublicationType>Randomized Controlled Trial</PublicationType><PublicationType>Journal Article</PublicationType></PublicationTypeList>
 </Article></MedlineCitation></PubmedArticle>
 <PubmedArticle><MedlineCitation><PMID>222</PMID><Article>
  <Journal><JournalIssue><PubDate><MedlineDate>2019 Jan-Feb</MedlineDate></PubDate></JournalIssue><Title>Other</Title></Journal>
  <ArticleTitle>No abstract here</ArticleTitle></Article></MedlineCitation></PubmedArticle>
 <PubmedArticle><MedlineCitation><PMID>333</PMID><Article>
  <Journal><JournalIssue><PubDate><MedlineDate>2019 Jan-Feb</MedlineDate></PubDate></JournalIssue><Title>Other</Title></Journal>
  <ArticleTitle>Plain abstract</ArticleTitle><Abstract><AbstractText>Just text.</AbstractText></Abstract>
 </Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def test_parse_pubmed_xml():
    papers = parse_pubmed_xml(XML)
    assert [p.pmid for p in papers] == ["111", "333"]          # article without abstract skipped
    p = papers[0]
    assert p.title == "Effect of X on Y"                         # inline markup flattened, trailing dot removed
    assert p.abstract.startswith("BACKGROUND: Y is common. RESULTS: X reduced Y by 20% (CO2 adjusted)")
    assert (p.year, p.journal, p.authors) == (2021, "Test Journal", ["Smith J"])
    assert "Randomized Controlled Trial" in p.pub_type
    assert papers[1].year == 2019                                # MedlineDate fallback
