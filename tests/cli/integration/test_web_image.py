"""The web client image ``cogniverse up`` builds serves the UI against a runtime.

The image is built from the Dockerfile and context the CLI's image tooling
declares and run under the chart's security context, against a recording
stand-in for the runtime's HTTP surface.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import httpx
import pytest

from tests.utils.web_client import free_port
from tests.utils.web_image import build_web_image, run_web_container

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

KEY = "sk-web-image-test"
AGENTS = ["search_agent", "coding_agent"]
TIERS = {"tiers": ["basic", "pro"]}


@contextmanager
def recording_runtime() -> Iterator[tuple[str, list[tuple[str, str, str | None]]]]:
    """The runtime routes the server calls; records (method, path, auth)."""
    seen: list[tuple[str, str, str | None]] = []

    class Runtime(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(("GET", self.path, self.headers.get("Authorization")))
            body = {"/agents/": {"agents": AGENTS}, "/admin/router-tiers": TIERS}.get(
                self.path
            )
            self.send_response(200 if body is not None else 404)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body or {"detail": "Not Found"}).encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Runtime)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", seen
    finally:
        server.shutdown()
        thread.join(timeout=10)


@pytest.fixture(scope="module")
def web_image() -> str:
    return build_web_image()


def test_the_image_serves_the_client_health_and_runtime_routes(web_image):
    with recording_runtime() as (runtime_url, seen):
        with run_web_container(runtime_url, KEY, free_port(), tag=web_image) as (
            url,
            _,
        ):
            health = httpx.get(f"{url}/healthz", timeout=10)
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}

            page = httpx.get(f"{url}/", timeout=10)
            assert page.status_code == 200
            assert page.headers["content-type"] == "text/html; charset=utf-8"
            assert "<title>Cogniverse</title>" in page.text
            assert '<div id="root"></div>' in page.text
            scripts = re.findall(
                r'<script type="module" crossorigin src="([^"]+)"', page.text
            )
            assert len(scripts) == 1 and scripts[0].startswith("/assets/index-")
            bundle = httpx.get(f"{url}{scripts[0]}", timeout=10)
            assert bundle.status_code == 200
            assert bundle.headers["content-type"] == "text/javascript; charset=utf-8"
            assert "/ui-api/copilotkit" in bundle.text

            deep_link = httpx.get(f"{url}/tenants", timeout=10)
            assert deep_link.text == page.text

            agents = httpx.get(f"{url}/ui-api/agents", timeout=10)
            assert agents.status_code == 200
            assert agents.json() == {"agents": AGENTS}

            tiers = httpx.get(f"{url}/ui-api/runtime/admin/router-tiers", timeout=10)
            assert tiers.status_code == 200
            assert tiers.json() == TIERS

    assert seen == [
        ("GET", "/agents/", None),
        ("GET", "/admin/router-tiers", f"Bearer {KEY}"),
    ]


def test_concurrent_clients_each_get_their_own_relayed_answer(web_image):
    """Sixteen browsers loading at once: every agent list and every proxied
    call is answered from the runtime, each proxied call carries the key."""
    with recording_runtime() as (runtime_url, seen):
        with run_web_container(runtime_url, KEY, free_port(), tag=web_image) as (
            url,
            _,
        ):
            barrier = threading.Barrier(16)

            def call(index: int) -> tuple[int, dict]:
                path = (
                    "/ui-api/agents"
                    if index % 2
                    else "/ui-api/runtime/admin/router-tiers"
                )
                barrier.wait(timeout=10)
                response = httpx.get(f"{url}{path}", timeout=20)
                return response.status_code, response.json()

            with ThreadPoolExecutor(max_workers=16) as pool:
                results = list(pool.map(call, range(16)))

    assert results == [
        (200, {"agents": AGENTS} if index % 2 else TIERS) for index in range(16)
    ]
    assert sorted(seen) == sorted(
        [("GET", "/agents/", None)] * 8
        + [("GET", "/admin/router-tiers", f"Bearer {KEY}")] * 8
    )


def test_a_dead_runtime_leaves_health_up_and_names_itself_on_api_calls(web_image):
    dead_port = free_port()
    runtime_url = f"http://127.0.0.1:{dead_port}"
    with run_web_container(runtime_url, KEY, free_port(), tag=web_image) as (url, _):
        health = httpx.get(f"{url}/healthz", timeout=10)
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        agents = httpx.get(f"{url}/ui-api/agents", timeout=20)
        assert agents.status_code == 502
        assert agents.json() == {
            "error": f"The Cogniverse runtime at {runtime_url} did not answer (TypeError)."
        }

        page = httpx.get(f"{url}/", timeout=10)
        assert page.status_code == 200
        assert "<title>Cogniverse</title>" in page.text
