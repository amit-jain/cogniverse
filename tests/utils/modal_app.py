"""A Modal chat-completions web endpoint on a real socket.

Answers what Modal answers in each deployment state, so a test exercises the
real HTTP boundary an LM call meets when the app it addresses is undeployed,
cold or serving.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tests.foundation.integration._sr_stack.stub_upstream import UNDEPLOYED_BODY


class ModalApp:
    """A Modal web endpoint serving an OpenAI-compatible chat completion.

    ``deployed=False`` answers what Modal's edge answers for an app that is
    not deployed. ``deploy(cold_start_s=...)`` makes the next request wait
    that long, as the first request to a scaled-to-zero app does.
    """

    def __init__(self) -> None:
        self.deployed = False
        self.cold_start_s = 0.0
        self.requests: list[str] = []
        self._lock = threading.Lock()
        app = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):  # noqa: N802 (http.server API)
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt = body["messages"][-1]["content"]
                with app._lock:
                    app.requests.append(prompt)
                    deployed = app.deployed
                    cold_start_s, app.cold_start_s = app.cold_start_s, 0.0
                if not deployed:
                    self.send_response(404)
                    self.send_header("Content-Type", "text/plain; charset=utf-8")
                    self.send_header("Content-Length", str(len(UNDEPLOYED_BODY)))
                    self.end_headers()
                    self.wfile.write(UNDEPLOYED_BODY)
                    return
                time.sleep(cold_start_s)
                payload = json.dumps(
                    {
                        "id": "chatcmpl-modal",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "google/gemma-4-e4b-it",
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": f"served:{prompt}",
                                },
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
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def deploy(self, cold_start_s: float = 0.0) -> None:
        with self._lock:
            self.deployed = True
            self.cold_start_s = cold_start_s

    def count(self, prompts) -> int:
        with self._lock:
            return sum(1 for prompt in self.requests if prompt in prompts)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
