"""An OpenAI-compatible endpoint on a local port, for LM wire tests.

Serves ``GET /v1/models`` (with ``max_model_len`` when ``context_window`` is
set), ``POST /tokenize`` (four template tokens per message plus one per
whitespace-separated word) and ``POST /v1/chat/completions`` (answers
``reply``, or ``chat_error`` as ``(status, error body)``). Every request body
is appended to ``received`` as ``(path, body)``. With ``hold_until`` set, each
chat completion is held until that many are in flight at once (or
``hold_timeout_s`` passes); ``max_in_flight`` records the most seen together.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class _ConcurrentServer(ThreadingHTTPServer):
    # socketserver's listen backlog of 5 resets connections when concurrency
    # tests connect several clients at once.
    request_queue_size = 64


@dataclass
class LLMEndpointStub:
    server: ThreadingHTTPServer
    api_base: str
    received: list = field(default_factory=list)
    in_flight: dict = field(default_factory=lambda: {"now": 0, "max": 0})

    @property
    def max_in_flight(self) -> int:
        return self.in_flight["max"]

    def posted(self, path: str) -> list[dict[str, Any]]:
        return [body for seen, body in self.received if seen == path]

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def serve_llm_endpoint(
    *,
    model: str = "google/gemma-4-e4b-it",
    context_window: int | None = None,
    reply: str = "ok",
    chat_error: tuple[int, dict[str, Any]] | None = None,
    hold_until: int | None = None,
    hold_timeout_s: float = 3.0,
) -> LLMEndpointStub:
    received: list = []
    in_flight = {"now": 0, "max": 0}
    arrivals = threading.Condition()

    class Handler(BaseHTTPRequestHandler):
        def _answer(self, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler contract
            received.append((self.path, None))
            entry: dict[str, Any] = {"id": model, "object": "model"}
            if context_window is not None:
                entry["max_model_len"] = context_window
            self._answer(200, {"object": "list", "data": [entry]})

        def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler contract
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length))
            received.append((self.path, body))
            if self.path == "/tokenize":
                count = sum(
                    4 + len(str(message["content"]).split())
                    for message in body["messages"]
                )
                self._answer(200, {"count": count, "tokens": []})
                return
            if chat_error is not None:
                self._answer(*chat_error)
                return
            with arrivals:
                in_flight["now"] += 1
                in_flight["max"] = max(in_flight["max"], in_flight["now"])
                arrivals.notify_all()
                if hold_until is not None:
                    arrivals.wait_for(
                        lambda: in_flight["now"] >= hold_until, timeout=hold_timeout_s
                    )
            try:
                self._answer_completion()
            finally:
                with arrivals:
                    in_flight["now"] -= 1

        def _answer_completion(self) -> None:
            self._answer(
                200,
                {
                    "id": "chatcmpl-stub",
                    "object": "chat.completion",
                    "created": 0,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": reply},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                },
            )

        def log_message(self, *args):
            return

    server = _ConcurrentServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return LLMEndpointStub(
        server=server,
        api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
        received=received,
        in_flight=in_flight,
    )


@contextmanager
def served_dspy_lm(context_window: int = 8192) -> Iterator[LLMEndpointStub]:
    """Bind a direct LM on a stub endpoint publishing ``context_window``.

    For a test that drives an agent's ``call_dspy`` with the module's
    ``forward`` replaced: the call still fits its inputs to the window the
    bound LM's endpoint publishes and counts on its ``/tokenize``.
    """
    import dspy

    from cogniverse_foundation.config.llm_factory import create_dspy_lm
    from cogniverse_foundation.config.unified_config import LLMEndpointConfig

    stub = serve_llm_endpoint(context_window=context_window)
    lm = create_dspy_lm(
        LLMEndpointConfig(model="openai/test-model", api_base=stub.api_base)
    )
    try:
        with dspy.context(lm=lm):
            yield stub
    finally:
        stub.close()
