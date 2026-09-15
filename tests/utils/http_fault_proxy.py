"""Forward real HTTP requests, with a test-controlled fault at the socket seam."""

from __future__ import annotations

import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx


class HTTPFaultProxy:
    """A real upstream connection with optional request barriers and failures."""

    def __init__(
        self,
        upstream: str,
        intercept: Callable[[str, str, bytes], tuple[int, bytes] | None],
    ) -> None:
        self.requests: list[tuple[str, str, bytes]] = []
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=30, trust_env=False)
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def forward(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                request = (self.command, self.path, body)
                with proxy._lock:
                    proxy.requests.append(request)
                fault = intercept(*request)
                if fault is None:
                    response = proxy._client.request(
                        self.command,
                        upstream.rstrip("/") + self.path,
                        content=body,
                        headers={
                            key: value
                            for key, value in self.headers.items()
                            if key.lower()
                            not in {"host", "content-length", "connection"}
                        },
                    )
                    status, payload = response.status_code, response.content
                    content_type = response.headers.get(
                        "content-type", "application/json"
                    )
                else:
                    status, payload = fault
                    content_type = "application/json"
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = forward
            do_POST = forward
            do_PUT = forward
            do_PATCH = forward
            do_DELETE = forward

            def log_message(self, *_args):
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self._server.server_port}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)
        self._client.close()
