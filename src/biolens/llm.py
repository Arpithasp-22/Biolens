"""Provider-agnostic LLM interface. Specs look like 'openai:gpt-4o-mini', 'gemini:gemini-2.0-flash',
'ollama:llama3.1' (any open-weight model served by Ollama/vLLM via an OpenAI-compatible API), or 'mock'.

The agent talks to every provider through one JSON-in/JSON-out protocol instead of each vendor's native
function-calling format, so GPT, Gemini and open-weight models are compared on identical prompts.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResult:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""
    latency_ms: float = 0.0


class LLM(Protocol):
    name: str

    def complete(self, role: str, system: str, user: str, *, json_mode: bool = False,
                 temperature: float = 0.0, max_tokens: int = 1500) -> LLMResult: ...


class OpenAILLM:
    """OpenAI, or any OpenAI-compatible server (Ollama, vLLM, LM Studio) via base_url."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 prefix: str = "openai"):
        from openai import OpenAI  # lazy: optional dependency

        self._client = OpenAI(base_url=base_url, api_key=api_key) if (base_url or api_key) else OpenAI()
        self.model, self.name = model, f"{prefix}:{model}"

    def complete(self, role, system, user, *, json_mode=False, temperature=0.0, max_tokens=1500):
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        t0 = time.time()
        r = self._client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
        u = r.usage
        return LLMResult(r.choices[0].message.content or "", getattr(u, "prompt_tokens", 0) or 0,
                         getattr(u, "completion_tokens", 0) or 0, self.model, (time.time() - t0) * 1000)


class GeminiLLM:
    def __init__(self, model: str, api_key: str | None = None):
        from google import genai  # lazy: optional dependency

        self._genai = genai
        self._client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        self.model, self.name = model, f"gemini:{model}"

    def complete(self, role, system, user, *, json_mode=False, temperature=0.0, max_tokens=1500):
        from google.genai import types

        cfg = types.GenerateContentConfig(
            system_instruction=system, temperature=temperature, max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_mode else None)
        t0 = time.time()
        r = self._client.models.generate_content(model=self.model, contents=user, config=cfg)
        u = getattr(r, "usage_metadata", None)
        return LLMResult(r.text or "", getattr(u, "prompt_token_count", 0) or 0,
                         getattr(u, "candidates_token_count", 0) or 0, self.model, (time.time() - t0) * 1000)


def get_llm(spec: str | None = None) -> LLM:
    spec = spec or os.getenv("BIOLENS_MODEL", "mock")
    kind, _, model = spec.partition(":")
    if kind == "mock":
        from .mock_llm import MockLLM

        return MockLLM()
    if kind == "openai":
        return OpenAILLM(model or "gpt-4o-mini")
    if kind == "gemini":
        return GeminiLLM(model or "gemini-2.0-flash")
    if kind == "ollama":
        return OpenAILLM(model or "llama3.1", base_url=os.getenv("OLLAMA_URL", "http://localhost:11434/v1"),
                         api_key="ollama", prefix="ollama")
    raise ValueError(f"Unknown model spec '{spec}'. Use mock | openai:<m> | gemini:<m> | ollama:<m>")


def extract_json(text: str) -> dict:
    """Parse a JSON object out of an LLM reply (tolerates code fences and surrounding prose)."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.IGNORECASE)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        a, b = t.find("{"), t.rfind("}")
        if a != -1 and b > a:
            return json.loads(t[a:b + 1])
        raise ValueError(f"No JSON object in reply: {text[:120]!r}")


def llm_json(llm: LLM, role: str, system: str, user: str, **kw) -> dict:
    """Ask for JSON; if the reply doesn't parse, ask once more before giving up."""
    last = ""
    for attempt in range(2):
        prompt = user if attempt == 0 else (
            user + "\n\nYour previous reply was not valid JSON. Reply with ONLY one JSON object.")
        last = llm.complete(role, system, prompt, json_mode=True, **kw).text
        try:
            return extract_json(last)
        except ValueError:
            continue
    raise ValueError(f"{llm.name} did not return valid JSON for role '{role}': {last[:200]!r}")
