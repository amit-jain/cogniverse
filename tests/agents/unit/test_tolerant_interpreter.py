"""Wire-contract tests for TolerantPythonInterpreter._send_request.

The Deno sandbox channel can carry unsolicited ``id: null`` messages (late
async rejections from a prior execute). dspy's stock ``_send_request`` reads
one line and raises ``Response ID mismatch ... got None``; the tolerant
reader must skip those lines, return the matching response, and still raise
on genuine id mismatches. The stub process speaks the exact JSON-RPC frames
observed on the real channel.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from cogniverse_agents.inference.tolerant_interpreter import (
    CodeInterpreterError,
    TolerantPythonInterpreter,
)

pytestmark = [pytest.mark.unit]


class _StubProcess:
    """Pipes-only stand-in for the Deno subprocess (never exits)."""

    def __init__(self, stdout_lines: list[str]):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO("\n".join(stdout_lines) + "\n")
        self.stderr = io.StringIO()

    def poll(self):
        return None


def _interpreter_with(stdout_lines: list[str]) -> TolerantPythonInterpreter:
    interp = TolerantPythonInterpreter.__new__(TolerantPythonInterpreter)
    interp._request_id = 0
    interp.deno_process = _StubProcess(stdout_lines)
    return interp


def test_returns_response_matching_request_id():
    interp = _interpreter_with(
        ['{"jsonrpc":"2.0","id":1,"result":{"status":"registered"}}']
    )
    response = interp._send_request("register", {"outputs": ["answer"]}, "registering")
    assert response == {"jsonrpc": "2.0", "id": 1, "result": {"status": "registered"}}
    sent = json.loads(interp.deno_process.stdin.getvalue())
    assert sent == {
        "jsonrpc": "2.0",
        "method": "register",
        "params": {"outputs": ["answer"]},
        "id": 1,
    }


def test_skips_unsolicited_id_null_error_then_returns_match():
    # The exact failure from the agents sweep: a late unhandled-rejection
    # notification (no id) lands before the registration response.
    interp = _interpreter_with(
        [
            '{"jsonrpc":"2.0","error":{"message":"Unhandled async rejection"}}',
            "Pyodide package loading: micropip",
            '{"jsonrpc":"2.0","id":1,"result":{"status":"registered"}}',
        ]
    )
    response = interp._send_request("register", {}, "registering tools/outputs")
    assert response["id"] == 1
    assert response["result"] == {"status": "registered"}


def test_mismatched_nonnull_id_raises():
    interp = _interpreter_with(['{"jsonrpc":"2.0","id":99,"result":{}}'])
    with pytest.raises(CodeInterpreterError, match="expected 1, got 99"):
        interp._send_request("register", {}, "registering tools/outputs")


def test_error_response_with_matching_id_raises():
    interp = _interpreter_with(
        ['{"jsonrpc":"2.0","id":1,"error":{"message":"register failed"}}']
    )
    with pytest.raises(CodeInterpreterError, match="register failed"):
        interp._send_request("register", {}, "registering tools/outputs")


def test_deno_cache_dir_in_allow_read_even_when_probe_poisoned(monkeypatch):
    # dspy resolves the Deno cache dir via a once-per-process
    # ``deno info --json`` probe whose result is lru_cached; one transient
    # failure leaves every later interpreter without read access to
    # pyodide.asm.wasm. The tolerant subclass must pin the cache dir into
    # --allow-read regardless of that probe's outcome.
    from dspy.primitives.python_interpreter import PythonInterpreter

    monkeypatch.setattr(PythonInterpreter, "_get_deno_dir", staticmethod(lambda: None))
    monkeypatch.delenv("DENO_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

    interp = TolerantPythonInterpreter()
    allow_read = next(a for a in interp.deno_command if a.startswith("--allow-read="))
    expected = str(Path.home() / ".cache" / "deno")
    assert expected in allow_read.split("=", 1)[1].split(",")


def test_deno_dir_env_wins(monkeypatch):
    monkeypatch.setenv("DENO_DIR", "/custom/deno-cache")
    interp = TolerantPythonInterpreter()
    allow_read = next(a for a in interp.deno_command if a.startswith("--allow-read="))
    assert "/custom/deno-cache" in allow_read.split("=", 1)[1].split(",")
