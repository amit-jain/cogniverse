"""Real service endpoints that replay one recorded answer.

Lets a test drive the production path -- the production client class, the same
adapter, the same HTTP round trip -- over a response body captured from a
served model. The client is the concrete type production builds; only the
answer is recorded, so what the test exercises is how cogniverse handles that
answer.
"""

from __future__ import annotations

import json
import socket
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import dspy


def _handler(content: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802 (http.server API)
            self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            body = json.dumps(
                {
                    "id": "chatcmpl-recorded",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "recorded",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": content},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    return Handler


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextmanager
def recorded_completion_lm(content: str):
    """Yield a ``dspy.LM`` whose endpoint answers every call with ``content``."""
    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(content))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield dspy.LM(
            model="openai/recorded",
            api_base=f"http://127.0.0.1:{port}/v1",
            api_key="not-required",
            cache=False,
            num_retries=0,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)


def _gliner_handler(entities: list[dict]):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802 (http.server API)
            self.rfile.read(int(self.headers.get("Content-Length", 0) or 0))
            body = json.dumps({"entities": entities}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    return Handler


@contextmanager
def recorded_gliner_extractor(entities: list[dict], *, model_name: str):
    """Yield the production ``GLiNERRelationshipExtractor`` over a recorded answer.

    The extractor and its remote client are the classes the served agent
    builds; the inference service answers every prediction with ``entities``,
    so a caller can steer the analysis path without a loaded model.
    """
    from cogniverse_agents.routing.relationship_extraction_tools import (
        GLiNERRelationshipExtractor,
    )
    from cogniverse_core.common.models.model_loaders import RemoteGlinerClient

    port = _free_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), _gliner_handler(entities))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    try:
        extractor = GLiNERRelationshipExtractor(
            model_name=model_name, inference_url=url
        )
        extractor.gliner_model = RemoteGlinerClient(
            url=url, model_name=model_name, timeout=10.0
        )
        yield extractor
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
