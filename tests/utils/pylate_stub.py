"""A stand-in for the PyLate pooling service ingestion's ColBERT loader calls.

``/pooling`` answers each input with one 128-dim vector per whitespace token,
derived from the token, and ``/windows`` with one span covering each input.
"""

import hashlib
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

DIM = 128


def token_vector(token: str) -> list[float]:
    """The unit-scale vector ``token`` is encoded as."""
    digest = hashlib.sha256(token.encode()).digest() * (DIM // 32)
    return [(byte - 127.5) / 127.5 for byte in digest[:DIM]]


@contextmanager
def serve_pylate_stub() -> Iterator[str]:
    """Serve the stand-in; yields its base URL."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):  # noqa: N802 (http.server API)
            request = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
            )
            texts = request["input"]
            if self.path == "/pooling":
                data = [
                    {
                        "data": [token_vector(t) for t in text.split()]
                        or [token_vector("")]
                    }
                    for text in texts
                ]
            elif self.path == "/windows":
                data = [{"spans": [[0, len(text)]]} for text in texts]
            else:
                data = None
            body = json.dumps({"data": data}).encode()
            self.send_response(200 if data is not None else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
