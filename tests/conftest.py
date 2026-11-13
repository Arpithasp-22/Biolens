import pytest

from biolens.config import Config, build_components, demo_path, load_papers
from biolens.factory import build_agent
from biolens.mock_llm import MockLLM


@pytest.fixture()
def comp(tmp_path):
    c = build_components(Config(home=str(tmp_path)))
    c.store.add_papers(load_papers(demo_path("demo_corpus.jsonl")))
    return c


@pytest.fixture()
def make_agent(comp):
    def _make(llm=None, **kw):
        return build_agent(comp, llm=llm or MockLLM(), **kw)
    return _make
