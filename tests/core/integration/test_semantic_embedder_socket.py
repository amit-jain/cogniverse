from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests
from cogniverse_cli.modal_inference_config import get_inference_service_spec

from cogniverse_core.common.models.semantic_embedder import RemoteOpenAIEmbedder

pytestmark = pytest.mark.integration


@contextmanager
def _hung_embedding_server():
    requests_seen: list[tuple[str, str | None, dict[str, object]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            requests_seen.append(
                (self.path, self.headers.get("Authorization"), payload)
            )
            time.sleep(1.0)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}", requests_seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_remote_timeout_keeps_request_context_and_redacts_secret():
    spec = get_inference_service_spec("denseon")
    token = "socket-timeout-secret"

    with _hung_embedding_server() as (base_url, requests_seen):
        embedder = RemoteOpenAIEmbedder(
            base_url,
            spec.model_id,
            timeout=0.1,
            headers={"Authorization": f"Bearer {token}"},
        )
        with pytest.raises(requests.ReadTimeout) as caught:
            embedder.encode("Marie Curie discovered radium.", is_query=True)

    assert requests_seen == [
        (
            "/v1/embeddings",
            f"Bearer {token}",
            {
                "model": "lightonai/DenseOn",
                "input": ["query: Marie Curie discovered radium."],
            },
        )
    ]
    assert "127.0.0.1" in str(caught.value)
    assert "/v1/embeddings" in str(caught.value)
    assert token not in str(caught.value)
