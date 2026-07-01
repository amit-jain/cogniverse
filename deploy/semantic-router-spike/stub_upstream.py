"""Minimal OpenAI-compatible stub backend for the semantic-router spike.

Stands in for a real vLLM chat backend so the spike's assertions can be
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
  - ``echo``           — the last user message (proves the round trip)

When reasoning is requested it also fills ``message.reasoning_content`` and
``usage.completion_tokens_details.reasoning_tokens`` so a client can assert
the reasoning path was taken.

Pure standard library — the container needs only ``python:3.12-slim`` with
this file mounted; no pip install, nothing to break on first run.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BACKEND_TAG = os.environ.get("BACKEND_TAG", "stub")
PORT = int(os.environ.get("PORT", "8000"))

# Header names the spike uses to carry routing metadata to the backend.
_ROUTING_HEADER_PREFIXES = ("x-vsr-", "x-authz-", "x-tenant-", "x-task")


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
            "echo": _last_user_message(body),
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

    def log_message(self, *args):  # silence per-request logging noise
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", PORT), _Handler)
    print(f"stub-upstream[{BACKEND_TAG}] listening on :{PORT}", flush=True)
    server.serve_forever()
