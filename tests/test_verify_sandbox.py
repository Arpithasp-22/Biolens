import pytest

from biolens import sandbox
from biolens.mock_llm import MockLLM
from biolens.verify import CitationVerifier, LLMJudge


def test_verifier_accepts_supported_claim(comp):
    v = comp.verifier.verify("Zorvalin reduced relapse rates by 41% compared with placebo.", ["DEMO0001"])
    assert v.supported and v.best_pmid == "DEMO0001" and v.score > 0.9


def test_verifier_catches_invented_numbers(comp):
    v = comp.verifier.verify("Zorvalin reduced relapse rates by 73% compared with placebo.", ["DEMO0001"])
    assert not v.supported


def test_verifier_catches_wrong_paper_and_fabricated_ids(comp):
    assert not comp.verifier.verify("Lumeral raised hepatic enzymes in 8% of patients.", ["DEMO0001"]).supported
    v = comp.verifier.verify("Anything at all.", ["PMID999"])
    assert not v.supported and v.fabricated == ["PMID999"]
    v = comp.verifier.verify("Zorvalin reduced relapse rates by 41%.", ["DEMO0001"], allowed={"DEMO0002"})
    assert not v.supported and v.fabricated == ["DEMO0001"]       # real paper, but never retrieved this run
    assert not comp.verifier.verify("Claim without a citation.", []).supported


def test_verifier_with_llm_judge(comp):
    judged = CitationVerifier(comp.store, comp.embedder, judge=LLMJudge(MockLLM()))
    assert judged.verify("Zorvalin reduced relapse rates by 41% compared with placebo.", ["DEMO0001"]).supported
    assert not judged.verify("Lumeral cures every disease.", ["DEMO0001"]).supported


def test_sandbox_runs_analysis():
    out = sandbox.run_python("import statistics\nresult = statistics.mean(data)", [1, 2, 3, 6])
    assert out["ok"] and out["result"] == 3
    out = sandbox.run_python("print('hi')\nresult = {'rr': round(0.61 / 0.82, 3)}")
    assert out["ok"] and out["stdout"].strip() == "hi" and out["result"] == {"rr": 0.744}


@pytest.mark.parametrize("code", [
    "import os\nresult = os.listdir('/')",
    "import subprocess",
    "from socket import socket",
    "result = open('/etc/passwd').read()",
    "result = eval('1+1')",
    "result = ().__class__.__bases__[0].__subclasses__()",
    "result = getattr(data, 'x')",
    "__import__('os')",
])
def test_sandbox_blocks_dangerous_code(code):
    out = sandbox.run_python(code, {})
    assert not out["ok"] and out["error"]


def test_sandbox_limits_and_errors():
    assert "timed out" in sandbox.run_python("while True:\n    pass", timeout=1.5)["error"]
    assert "ZeroDivisionError" in sandbox.run_python("result = 1/0")["error"]
    assert "SyntaxError" in sandbox.run_python("def (")["error"]
    big = sandbox.run_python("print('x' * 100000)\nresult = 1", max_output=100)
    assert len(big["stdout"]) <= 100
