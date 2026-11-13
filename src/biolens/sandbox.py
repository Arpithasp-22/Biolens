"""Best-effort sandbox for the `run_python_analysis` tool.

Layers: AST allow-list (no dunder access, no open/eval/exec/getattr, stdlib-only imports) -> separate
interpreter in isolated mode with an empty environment -> restricted builtins and import hook ->
CPU/memory rlimits and a wall-clock timeout.

This blocks the common mistakes and prompt-injected one-liners, but it is NOT a hard security
boundary. For untrusted multi-tenant use, run it in a container/gVisor/Firecracker instead.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys

ALLOWED_IMPORTS = ("math", "statistics", "json", "collections", "itertools", "re", "fractions",
                   "decimal", "datetime", "random")
SAFE_BUILTINS = ("abs all any bin bool chr dict divmod enumerate filter float format frozenset hex int "
                 "isinstance iter len list map max min next oct ord pow print range repr reversed round "
                 "set slice sorted str sum tuple zip Exception ValueError TypeError KeyError IndexError "
                 "ZeroDivisionError").split()
_BANNED_NAMES = {"eval", "exec", "compile", "open", "input", "__import__", "globals", "locals", "vars",
                 "getattr", "setattr", "delattr", "breakpoint", "memoryview", "exit", "quit", "dir"}

_WRAPPER = r'''
import sys, json, builtins
req = json.loads(sys.stdin.read())
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (req["cpu"], req["cpu"]))
    resource.setrlimit(resource.RLIMIT_AS, (req["mem"], req["mem"]))
except Exception:
    pass
allowed = set(req["allowed"])
real_import = builtins.__import__
def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in allowed:
        raise ImportError("import of '%s' is not allowed" % name)
    return real_import(name, globals, locals, fromlist, level)
safe = {n: getattr(builtins, n) for n in req["builtins"]}
safe["__import__"] = safe_import
env = {"__builtins__": safe, "data": req["data"]}
exec(compile(req["code"], "<analysis>", "exec"), env)
sys.stdout.write("\n__RESULT__" + json.dumps(env.get("result"), default=str))
'''


def validate(code: str) -> str | None:
    """Return an error string if the code uses anything outside the allow-list."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import of '{a.name}' is not allowed (allowed: {', '.join(ALLOWED_IMPORTS)})"
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"import from '{node.module}' is not allowed"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"access to '{node.attr}' is not allowed"
        elif isinstance(node, ast.Name) and (node.id in _BANNED_NAMES or node.id.startswith("__")):
            return f"use of '{node.id}' is not allowed"
    return None


def run_python(code: str, data=None, timeout: float = 10.0, max_output: int = 4000) -> dict:
    """Run `code` with `data` in scope; set a variable named `result` to return a value."""
    err = validate(code)
    if err:
        return {"ok": False, "error": err, "stdout": "", "result": None}
    payload = json.dumps({"code": code, "data": data, "allowed": list(ALLOWED_IMPORTS),
                          "builtins": SAFE_BUILTINS, "cpu": int(timeout) + 1, "mem": 768 * 1024 * 1024})
    try:
        p = subprocess.run([sys.executable, "-I", "-c", _WRAPPER], input=payload, capture_output=True,
                           text=True, timeout=timeout, env={})
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timed out after {timeout}s", "stdout": "", "result": None}
    if p.returncode != 0:
        msg = (p.stderr.strip().splitlines() or ["unknown error"])[-1]
        return {"ok": False, "error": msg[:500], "stdout": p.stdout[:max_output], "result": None}
    out, _, res = p.stdout.rpartition("\n__RESULT__")
    try:
        result = json.loads(res)
    except json.JSONDecodeError:
        result = None
    return {"ok": True, "error": None, "stdout": out[:max_output], "result": result}
