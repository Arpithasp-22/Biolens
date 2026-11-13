"""Cost estimation. Prices change often, so none are hard-coded: provide your own table.

Set BIOLENS_PRICING to a JSON file like:
    {"openai:gpt-4o-mini": {"in": 0.15, "out": 0.60}, "gemini:gemini-2.0-flash": {"in": 0.10, "out": 0.40}}
Values are USD per 1M tokens. Models without an entry report cost = None (shown as "unknown"),
never a guess. `mock` and `ollama:*` (local) are free.
"""

from __future__ import annotations

import json
import os

_FREE_PREFIXES = ("mock", "ollama:")


def load_pricing(path: str | None = None) -> dict:
    path = path or os.getenv("BIOLENS_PRICING")
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def cost_usd(model_name: str, tokens_in: int, tokens_out: int, table: dict) -> float | None:
    if model_name.startswith(_FREE_PREFIXES):
        return 0.0
    p = table.get(model_name)
    if not p:
        return None
    return (tokens_in * p["in"] + tokens_out * p["out"]) / 1_000_000
