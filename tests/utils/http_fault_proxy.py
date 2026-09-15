"""A real HTTP forwarding boundary with one controllable request gate."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
