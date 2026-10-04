"""The runtime image's environment keeps the first LM call off GitHub.

litellm's import fetches ``model_prices_and_context_window.json`` from
raw.githubusercontent.com unless ``LITELLM_LOCAL_MODEL_COST_MAP`` is set. A
fresh interpreter, given the value the runtime image sets, makes its first
LM call to a real chat-completions endpoint and records every host it
resolves; GitHub must not be among them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
VARIABLE = "LITELLM_LOCAL_MODEL_COST_MAP"
COST_MAP_HOST = "raw.githubusercontent.com"
ANSWER = "bundled cost map answer"

pytestmark = pytest.mark.unit

FIRST_CALL = r"""
import json, socket, sys

resolved = []
real_getaddrinfo = socket.getaddrinfo

def recording_getaddrinfo(host, *args, **kwargs):
    resolved.append(str(host))
    if str(host) == "raw.githubusercontent.com":
        raise OSError("the test does not let the cost map fetch leave the host")
    return real_getaddrinfo(host, *args, **kwargs)

socket.getaddrinfo = recording_getaddrinfo

from cogniverse_foundation.config.body_bounded_lm import BodyBoundedLM

lm = BodyBoundedLM("openai/stub", api_base=sys.argv[1], api_key="k", num_retries=0)
answer = lm(messages=[{"role": "user", "content": "hello"}])
print(json.dumps({"resolved": sorted(set(resolved)), "answer": answer}))
"""


def _runtime_image_value() -> str:
    text = (REPO_ROOT / "libs" / "runtime" / "Dockerfile").read_text()
    final = text[text.rindex("\nFROM ") :]
    values = [
        line.split("=", 1)[1]
        for line in final.splitlines()
        if line.startswith(f"ENV {VARIABLE}=")
    ]
    assert len(values) == 1, values
    return values[0]


class _ChatCompletions:
    def __init__(self) -> None:
        self.requests = 0
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                stub.requests += 1
                body = json.dumps(
                    {
                        "id": "stub",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "stub",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": ANSWER},
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

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)


def _first_lm_call(cost_map_value: str | None) -> tuple[dict, int]:
    env = {k: v for k, v in os.environ.items() if k != VARIABLE}
    if cost_map_value is not None:
        env[VARIABLE] = cost_map_value
    stub = _ChatCompletions()
    with stub as api_base:
        finished = subprocess.run(
            [sys.executable, "-c", FIRST_CALL, api_base],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            cwd=REPO_ROOT,
        )
    assert finished.returncode == 0, finished.stderr[-4000:]
    return json.loads(finished.stdout.strip().splitlines()[-1]), stub.requests


def test_with_the_runtime_images_value_the_first_call_resolves_only_its_lm():
    outcome, requests = _first_lm_call(_runtime_image_value())

    assert outcome == {"resolved": ["127.0.0.1"], "answer": [ANSWER]}
    assert requests == 1


def test_without_it_the_first_call_reaches_for_github():
    """The recorder sees the fetch the image setting prevents."""
    outcome, requests = _first_lm_call(None)

    assert outcome == {"resolved": ["127.0.0.1", COST_MAP_HOST], "answer": [ANSWER]}
    assert requests == 1
