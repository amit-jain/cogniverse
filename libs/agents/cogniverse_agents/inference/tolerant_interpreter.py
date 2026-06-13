"""PythonInterpreter that survives unsolicited JSON-RPC messages.

dspy's ``PythonInterpreter.execute`` loop tolerates non-JSON lines, tool-call
notifications, and unsolicited ``id: null`` error messages on the Deno
channel. ``_send_request`` (used for tool/output registration) does not — it
reads exactly one line and demands the matching response id. When the sandbox
emits a late message after a completed execute (e.g. an unhandled async
rejection from a tool promise), the next registration reads that stale line
and raises ``Response ID mismatch ... got None``, which dspy.RLM records as a
failed step and resolves via fallback extraction.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import dspy
from dspy.primitives.code_interpreter import CodeInterpreter
from dspy.primitives.python_interpreter import (
    CodeInterpreterError,
    PythonInterpreter,
)

logger = logging.getLogger(__name__)


def _deno_cache_dir() -> str:
    """The Deno cache dir Pyodide loads its wasm from.

    dspy resolves this via a once-per-process ``deno info --json`` probe
    whose result is lru_cached — a single transient subprocess failure
    poisons every later interpreter in the process with a missing
    ``--allow-read`` entry ("NotCapable: Requires read access to ...
    pyodide.asm.wasm"). Resolve it from the documented locations instead.
    """
    env = os.environ.get("DENO_DIR")
    if env:
        return env
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return str(base / "deno")


class TolerantPythonInterpreter(PythonInterpreter):
    """Skips stale channel messages instead of failing the request."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        deno_dir = _deno_cache_dir()
        for i, arg in enumerate(self.deno_command):
            if not arg.startswith("--allow-read="):
                continue
            paths = arg.split("=", 1)[1].split(",")
            if deno_dir not in paths:
                self.deno_command[i] = f"{arg},{deno_dir}"
            break

    def _send_request(self, method: str, params: dict, context: str) -> dict:
        self._request_id += 1
        request_id = self._request_id
        msg = json.dumps(
            {"jsonrpc": "2.0", "method": method, "params": params, "id": request_id}
        )
        self.deno_process.stdin.write(msg + "\n")
        self.deno_process.stdin.flush()

        while True:
            response_line = self.deno_process.stdout.readline().strip()
            if not response_line:
                exit_code = self.deno_process.poll()
                if exit_code is not None:
                    stderr = (
                        self.deno_process.stderr.read()
                        if self.deno_process.stderr
                        else ""
                    )
                    raise CodeInterpreterError(
                        f"Deno exited (code {exit_code}) {context}: {stderr}"
                    )
                raise CodeInterpreterError(f"No response {context}")

            if not response_line.startswith("{"):
                logger.debug("Skipping non-JSON line %s: %s", context, response_line)
                continue
            try:
                response = json.loads(response_line)
            except json.JSONDecodeError:
                logger.debug("Skipping malformed JSON %s: %s", context, response_line)
                continue

            if response.get("id") != request_id:
                if response.get("id") is None:
                    # Unsolicited notification from a previous request (e.g.
                    # late async rejection) — the execute loop skips these too.
                    logger.debug(
                        "Skipping unsolicited message %s: %s",
                        context,
                        response_line[:200],
                    )
                    continue
                raise CodeInterpreterError(
                    f"Response ID mismatch {context}: "
                    f"expected {request_id}, got {response.get('id')}"
                )
            if "error" in response:
                raise CodeInterpreterError(
                    f"Error {context}: "
                    f"{response['error'].get('message', 'Unknown error')}"
                )
            return response


class TolerantRLM(dspy.RLM):
    """dspy.RLM whose default REPL is :class:`TolerantPythonInterpreter`.

    Mirrors the parent's per-forward interpreter lifecycle (fresh instance,
    shutdown on exit) so thread ownership stays bound to the executing
    thread; an explicitly injected interpreter is honored unchanged.
    """

    @contextmanager
    def _interpreter_context(
        self, execution_tools: dict[str, Callable]
    ) -> Iterator[CodeInterpreter]:
        if self._interpreter is not None:
            with super()._interpreter_context(execution_tools) as repl:
                yield repl
            return
        repl = TolerantPythonInterpreter(
            tools=execution_tools,
            output_fields=self._get_output_fields_info(),
        )
        try:
            yield repl
        finally:
            repl.shutdown()
