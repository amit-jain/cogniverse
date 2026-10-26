"""Post-deploy image pruning against the host's real Docker build cache.

Each test builds throwaway ``cogniverse/build-cache-probe-*`` generations from
busybox, prunes with the deploy's own function, and reads BuildKit's cache
records back through ``docker buildx du``.
"""

from __future__ import annotations

import subprocess
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from cogniverse_cli.images import prune_superseded_images

pytestmark = pytest.mark.integration

_BASE = "busybox:1.36"


def _docker(*argv: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ["docker", *argv],
        input=stdin,
        capture_output=True,
        text=True,
        check=True,
        timeout=600,
    ).stdout


def _build(tag: str, *steps: str, network_host: bool = False) -> None:
    dockerfile = "\n".join([f"FROM {_BASE}", *(f"RUN {s}" for s in steps)]) + "\n"
    argv = ["build", "-t", tag]
    if network_host:
        argv += ["--network", "host"]
    _docker(*argv, "-", stdin=dockerfile)


def _cached_commands() -> list[str]:
    prefix = "Description:  mount / from exec /bin/sh -c "
    return [
        line[len(prefix) :]
        for line in _docker("buildx", "du", "--verbose").splitlines()
        if line.startswith(prefix)
    ]


def _host_tags(repo: str) -> set[str]:
    out = _docker("images", "--format", "{{.Repository}}:{{.Tag}}", repo)
    return set(out.split())


@pytest.fixture
def probe_repo():
    _docker("pull", "-q", _BASE)
    repo = f"cogniverse/build-cache-probe-{uuid.uuid4().hex[:10]}"
    yield repo
    for tag in _host_tags(repo):
        subprocess.run(["docker", "rmi", tag], capture_output=True, check=False)
    subprocess.run(
        ["docker", "builder", "prune", "-f"], capture_output=True, check=False
    )


def test_prune_drops_the_removed_generations_cache_and_keeps_the_rest(probe_repo):
    marker = probe_repo.rsplit("-", 1)[-1]
    for n in (1, 2, 3):
        _build(f"{probe_repo}:0.1.dev{n}", f"echo {marker}-gen{n} > /marker")
    commands = {f"echo {marker}-gen{n} > /marker" for n in (1, 2, 3)}
    assert commands <= set(_cached_commands())

    removed = prune_superseded_images([f"{probe_repo}:0.1.dev3"])

    assert removed == [f"{probe_repo}:0.1.dev1"]
    assert _host_tags(probe_repo) == {
        f"{probe_repo}:0.1.dev2",
        f"{probe_repo}:0.1.dev3",
    }
    mine = [c for c in _cached_commands() if marker in c]
    assert sorted(mine) == [
        f"echo {marker}-gen2 > /marker",
        f"echo {marker}-gen3 > /marker",
    ]


def test_prune_during_an_in_flight_build_keeps_that_builds_cache(probe_repo):
    marker = probe_repo.rsplit("-", 1)[-1]
    body = f"{marker}-fetched"
    reached = threading.Event()
    release = threading.Event()

    class _Gate(BaseHTTPRequestHandler):
        def do_GET(self):
            reached.set()
            release.wait(timeout=300)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Gate)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]

    for n in (1, 2, 3):
        _build(f"{probe_repo}:0.1.dev{n}", f"echo {marker}-gen{n} > /marker")

    pre = f"echo {marker}-pre > /pre"
    fetch = f"wget -qO /marker http://127.0.0.1:{port}/"
    build_error: list[Exception] = []

    def in_flight_build():
        try:
            _build(f"{probe_repo}:0.1.dev4", pre, fetch, network_host=True)
        except Exception as exc:
            build_error.append(exc)

    builder = threading.Thread(target=in_flight_build)
    builder.start()
    try:
        assert reached.wait(timeout=300), "the dev4 build never reached its RUN step"
        removed = prune_superseded_images([f"{probe_repo}:0.1.dev3"])
    finally:
        release.set()
        builder.join(timeout=600)
        server.shutdown()

    assert build_error == []
    assert removed == [f"{probe_repo}:0.1.dev1"]
    assert _docker("run", "--rm", f"{probe_repo}:0.1.dev4", "cat", "/marker") == body
    mine = [c for c in _cached_commands() if marker in c or str(port) in c]
    assert sorted(mine) == sorted(
        [
            f"echo {marker}-gen2 > /marker",
            f"echo {marker}-gen3 > /marker",
            pre,
            fetch,
        ]
    )
