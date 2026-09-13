"""Minimal OpenAI-compatible stub backend for the local semantic-router stack.

Stands in for a real vLLM chat backend so the stack's assertions can be
exact and offline. It answers ``POST /v1/chat/completions`` and reflects
back, in the assistant message content (as JSON), everything the semantic
router decided and forwarded:

  - ``served_model``   — the ``model`` field the router sent (proves the
                         router rewrote ``auto``/the request into a concrete
                         catalog model)
  - ``reasoning``      — whether the router asked for reasoning, read from
                         ``chat_template_kwargs.enable_thinking`` /
                         ``.thinking`` / a top-level ``reasoning_effort``
  - ``routing_headers``— the ``x-vsr-*`` / tier headers that reached the
                         backend (proves the router forwarded them)
  - ``response_format``— the ``response_format`` object verbatim (proves a
                         ``json_schema`` payload survived the router's
                         request re-serialization)
  - ``echo``           — the last user message (proves the round trip)
  - ``temperature``    — the ``temperature`` the router forwarded
  - ``call_index``     — a per-process counter, incremented once per request
                         that reaches this backend. A router cache hit replays
                         a stored body verbatim, so a repeated ``call_index``
                         is a hit and a fresh one is an upstream call. Counting
                         calls is what separates "the cache answered" from "the
                         backend answered the same way twice".

When reasoning is requested it also fills ``message.reasoning_content`` and
``usage.completion_tokens_details.reasoning_tokens`` so a client can assert
the reasoning path was taken.

A user message beginning with ``FAULT:`` makes this backend fail in a named
way instead of answering, so the fault contract of everything in front of it
is exercised against one running stack:

  - ``FAULT:status:<code>`` — answer ``<code>`` with an OpenAI-shaped error
    body (the shape a real provider sends for 401/403/429/503)
  - ``FAULT:reset``         — close the connection with no response
  - ``FAULT:trickle:<s>``   — hold the request open ``<s>`` seconds, then answer

The sentinel travels in the request body, so it survives the router's request
re-serialization without depending on any header being forwarded.

Pure standard library — the container needs only ``python:3.12-slim`` with
this file mounted; no pip install, nothing to break on first run.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BACKEND_TAG = os.environ.get("BACKEND_TAG", "stub")
PORT = int(os.environ.get("PORT", "8000"))

# Every extension header that reaches the backend: the router's own
# ``x-vsr-*`` and destination headers, the tier/tenant headers cogniverse
# sends, and whatever Envoy adds on the way.
_ROUTING_HEADER_PREFIXES = ("x-",)

_CALLS_LOCK = threading.Lock()
_CALLS = 0
_REQUESTS: list[dict] = []


_FAULT_PREFIX = "FAULT:"

# The error body shape a real OpenAI-compatible provider sends on a refusal:
# a single ``error`` object carrying message/type/code. The per-status values
# mirror what an upstream that rejects the key, the tenant or the rate sends.
_REFUSALS = {
    400: ("Malformed completion", "invalid_request_error", "invalid_request"),
    422: ("Invalid completion fields", "invalid_request_error", "invalid_fields"),
    502: ("Bad gateway", "server_error", "bad_gateway"),
    504: ("Gateway timeout", "server_error", "gateway_timeout"),
    401: ("Incorrect API key provided", "invalid_request_error", "invalid_api_key"),
    403: (
        "You are not allowed to access this model",
        "invalid_request_error",
        "model_not_permitted",
    ),
    429: (
        "Rate limit reached for this model",
        "rate_limit_error",
        "rate_limit_exceeded",
    ),
    503: ("The engine is currently overloaded", "server_error", "engine_overloaded"),
}


def refusal_body(status: int) -> dict:
    """The OpenAI-shaped error body this backend answers ``status`` with."""
    message, kind, code = _REFUSALS[status]
    return {"error": {"message": message, "type": kind, "param": None, "code": code}}


def parse_fault(text: str) -> tuple[str, str] | None:
    """``(kind, argument)`` for a ``FAULT:`` sentinel message, else ``None``."""
    if not text.startswith(_FAULT_PREFIX):
        return None
    parts = text[len(_FAULT_PREFIX) :].split(":", 1)
    return (parts[0], parts[1] if len(parts) > 1 else "")


def _next_call_index() -> int:
    global _CALLS
    with _CALLS_LOCK:
        _CALLS += 1
        return _CALLS


def _reasoning_requested(body: dict) -> bool:
    ctk = body.get("chat_template_kwargs") or {}
    if ctk.get("enable_thinking") is True or ctk.get("thinking") is True:
        return True
    effort = body.get("reasoning_effort")
    return isinstance(effort, str) and effort.lower() in {"low", "medium", "high"}


def _last_user_message(body: dict) -> str:
    for msg in reversed(body.get("messages") or []):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
    return ""


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 (http.server API)
        if self.path.rstrip("/") in ("/health", "/healthz", ""):
            self._send_json(200, {"status": "ok", "backend_tag": BACKEND_TAG})
        elif self.path.rstrip("/") == "/requests":
            with _CALLS_LOCK:
                records = list(_REQUESTS)
            self._send_json(200, {"requests": records})
        elif self.path.rstrip("/").endswith("/models"):
            self._send_json(
                200,
                {"object": "list", "data": [{"id": BACKEND_TAG, "object": "model"}]},
            )
        else:
            self._send_json(404, {"error": {"message": f"no route {self.path}"}})

    def do_POST(self):  # noqa: N802 (http.server API)
        if not self.path.endswith("/chat/completions"):
            self._send_json(404, {"error": {"message": f"no route {self.path}"}})
            return

        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send_json(400, {"error": {"message": "invalid JSON body"}})
            return

        prompt = _last_user_message(body)
        with _CALLS_LOCK:
            _REQUESTS.append({"prompt": prompt, "model": body.get("model")})
        fault = parse_fault(prompt)
        if BACKEND_TAG == "teacher" and prompt.startswith("TEACHER_FAULT:"):
            fault = parse_fault(prompt.removeprefix("TEACHER_").split("|", 1)[0])
        if fault is not None:
            # Counted like any other request that reached this backend, so a
            # caller can measure how many attempts a failure consumed.
            _next_call_index()
            self._serve_fault(*fault)
            return

        routing_headers = {
            k.lower(): v
            for k, v in self.headers.items()
            if k.lower().startswith(_ROUTING_HEADER_PREFIXES)
        }
        reasoning = _reasoning_requested(body)

        reflection = {
            "backend_tag": BACKEND_TAG,
            "served_model": body.get("model"),
            "reasoning": reasoning,
            "routing_headers": routing_headers,
            "response_format": body.get("response_format"),
            "echo": _last_user_message(body),
            "temperature": body.get("temperature"),
            "call_index": _next_call_index(),
        }
        message = {"role": "assistant", "content": json.dumps(reflection)}
        if reasoning:
            message["reasoning_content"] = f"[{BACKEND_TAG}] thinking about the request"

        usage = {"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20}
        if reasoning:
            usage["completion_tokens_details"] = {"reasoning_tokens": 7}

        self._send_json(
            200,
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model") or BACKEND_TAG,
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": usage,
            },
        )

    def _serve_fault(self, kind: str, argument: str) -> None:
        """Fail the request in the named way instead of answering it."""
        if kind == "misleading":
            status = int(argument)
            self._send_json(
                status,
                {
                    "error": {
                        "message": "503 timeout no healthy upstream connection refused",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                },
            )
            return
        if kind == "status":
            status = int(argument)
            self._send_json(status, refusal_body(status))
            return
        if kind == "reset":
            self.close_connection = True
            self.wfile.close()
            return
        if kind == "trickle":
            time.sleep(float(argument))
            self._send_json(
                200,
                {
                    "id": "chatcmpl-stub-trickle",
                    "object": "chat.completion",
                    "created": 0,
                    "model": BACKEND_TAG,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "late"},
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
            return
        self._send_json(400, {"error": {"message": f"unknown fault {kind!r}"}})

    def log_message(self, *args):  # silence per-request logging noise
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), _Handler)
    print(f"stub-upstream[{BACKEND_TAG}] listening on :{PORT}", flush=True)
    server.serve_forever()
