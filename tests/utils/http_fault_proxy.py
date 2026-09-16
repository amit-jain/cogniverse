"""A real HTTP forwarding boundary with one controllable request gate."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import requests


class HTTPFaultProxy:
    def __init__(self, upstream):
        self.upstream = upstream
        self.entered = threading.Event()
        self.release = threading.Event()
        self.expired = threading.Event()
        self.requests = []
        self._lock = threading.Lock()
        self._predicate = None
        self._failure = False
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self._forward()

            def do_POST(self):
                self._forward()

            def do_PUT(self):
                self._forward()

            def do_DELETE(self):
                self._forward()

            def _forward(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                with proxy._lock:
                    proxy.requests.append((self.command, self.path, body))
                    gated = proxy._predicate and proxy._predicate(
                        self.command, self.path, body
                    )
                    if gated:
                        proxy._predicate = None
                if gated:
                    proxy.entered.set()
                    if not proxy.release.wait(3):
                        proxy.expired.set()
                    if proxy._failure:
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(b'{"error":"injected storage refusal"}')
                        return
                upstream = proxy.upstream(self.path)
                response = requests.request(
                    self.command,
                    upstream + self.path,
                    data=body,
                    headers={
                        name: value
                        for name, value in self.headers.items()
                        if name.lower() not in {"host", "content-length", "connection"}
                    },
                    timeout=120,
                )
                self.send_response(response.status_code)
                self.send_header(
                    "Content-Type",
                    response.headers.get("Content-Type", "application/json"),
                )
                self.send_header("Content-Length", str(len(response.content)))
                self.end_headers()
                self.wfile.write(response.content)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.port = self.server.server_address[1]

    def arm(self, predicate, *, failure=False):
        self.entered.clear()
        self.release.clear()
        self.expired.clear()
        self._predicate = predicate
        self._failure = failure

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class InterceptFaultProxy:
    """Forward every request upstream unless ``intercept`` returns a fault.

    ``intercept(method, path, body)`` returns ``None`` to forward, or
    ``(status, payload)`` to answer without contacting upstream. ``payload`` is
    the response body as bytes, or any JSON-serialisable value, which is sent
    as ``application/json``. Every request seen is recorded in ``requests``.
    """

    def __init__(
        self,
        upstream: str,
        intercept: Callable[[str, str, bytes], tuple[int, object] | None] | None = None,
    ) -> None:
        self.upstream_url = upstream.rstrip("/")
        self.intercept = intercept
        self.requests: list[tuple[str, str, bytes]] = []
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=60, trust_env=False)
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def forward(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                request = (self.command, self.path, body)
                with proxy._lock:
                    proxy.requests.append(request)
                    intercept = proxy.intercept
                fault = intercept(*request) if intercept else None
                if fault is None:
                    response = proxy._client.request(
                        self.command,
                        proxy.upstream_url + self.path,
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
                    status, value = fault
                    payload = (
                        value
                        if isinstance(value, bytes)
                        else json.dumps(value).encode()
                    )
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
        self.port = self._server.server_port
        self.url = f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_args):
        self.intercept = None
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)
        self._client.close()
