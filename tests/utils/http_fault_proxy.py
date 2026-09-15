"""HTTP forwarding with controlled barriers and faults at a real service boundary."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests


class HttpFaultProxy:
    def __init__(self, upstream_url: str):
        self.upstream_url = upstream_url.rstrip("/")
        self.intercept = None
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def forward(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                fault = (
                    owner.intercept(self.command, self.path, body)
                    if owner.intercept
                    else None
                )
                if fault is None:
                    response = requests.request(
                        self.command,
                        owner.upstream_url + self.path,
                        data=body,
                        headers={
                            key: value
                            for key, value in self.headers.items()
                            if key.lower()
                            not in {"host", "connection", "content-length"}
                        },
                        timeout=60,
                    )
                    status, payload = response.status_code, response.content
                    content_type = response.headers.get(
                        "Content-Type", "application/json"
                    )
                else:
                    status, value = fault
                    payload = json.dumps(value).encode()
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

            def log_message(self, format, *args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.intercept = None
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)
        assert self.thread.is_alive() is False
