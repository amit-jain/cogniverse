"""Install, build and serve ``clients/web`` for tests, and serve an app on uvicorn.

The client is installed from its own lockfile into a scratch directory, so a
test runs exactly the packages CI and a deployment resolve. Node 22 and npm
are required; their absence fails the test.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, List, Tuple

import uvicorn

from tests.utils.node_env import node_env

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIENT_DIR = REPO_ROOT / "clients" / "web"
MIN_NODE_MAJOR = 22
# Starts of the web server tried before a lost port race fails the test.
PORT_ATTEMPTS = 5
CLIENT_FILES = (
    "package.json",
    "package-lock.json",
    "tsconfig.json",
    "tsconfig.server.json",
    "vite.config.ts",
    "index.html",
)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _node() -> str:
    node = shutil.which("node")
    npm = shutil.which("npm")
    assert node is not None and npm is not None, (
        "node and npm are required to run the web client; the runtime CI job "
        "installs node 22"
    )
    version = subprocess.run(
        [node, "--version"], capture_output=True, text=True, timeout=30
    ).stdout.strip()
    assert int(version.lstrip("v").split(".")[0]) >= MIN_NODE_MAJOR, (
        f"the web client requires node >= {MIN_NODE_MAJOR}, got {version}"
    )
    return node


def _npm(root: Path, *args: str, timeout: int) -> subprocess.CompletedProcess:
    node = _node()
    return subprocess.run(
        [shutil.which("npm"), *args],
        cwd=root,
        env=node_env(node),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def install_web_client(root: Path) -> Path:
    """Copy the client's sources into ``root`` and ``npm ci`` its lockfile."""
    for name in CLIENT_FILES:
        shutil.copy(CLIENT_DIR / name, root / name)
    shutil.copytree(CLIENT_DIR / "src", root / "src")
    install = _npm(root, "ci", "--no-fund", "--no-audit", timeout=600)
    assert install.returncode == 0, (
        f"npm ci from the client lockfile failed:\n{install.stdout}\n{install.stderr}"
    )
    return root


def build_web_client(root: Path) -> Path:
    """Build the installed client the way a deployment does (``npm run build``)."""
    build = _npm(root, "run", "build", timeout=600)
    assert build.returncode == 0, (
        f"npm run build failed:\n{build.stdout}\n{build.stderr}"
    )
    return root


@contextmanager
def serve_web(
    client_dir: Path,
    runtime_url: str,
    *,
    telemetry_url: str,
    built: bool = False,
) -> Iterator[str]:
    """Run the client's Node server: the built ``dist`` entry when ``built``,
    otherwise the sources through tsx. Yields its base URL. The server acts
    for each tenant with a harness key it mints through the runtime.

    ``telemetry_url`` points CopilotKit's telemetry sink at a local recorder
    so a test never reports to CopilotKit's servers.
    """
    node = _node()
    entry = (
        ["dist/server/index.js"]
        if built
        else ["--import", "tsx", "src/server/index.ts"]
    )
    # A free port can be taken by another process before the server binds
    # it; the server then exits with EADDRINUSE and starts on another port.
    for _ in range(PORT_ATTEMPTS):
        port = free_port()
        proc = subprocess.Popen(
            [node, *entry],
            cwd=client_dir,
            env=node_env(
                node,
                COGNIVERSE_RUNTIME_URL=runtime_url,
                PORT=str(port),
                COPILOTKIT_TELEMETRY_URL=telemetry_url,
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Drained on a thread so a chatty server never blocks on a full pipe.
        output: List[str] = []
        reader = threading.Thread(
            target=lambda proc=proc, output=output: output.extend(
                iter(proc.stdout.readline, "")
            ),
            daemon=True,
        )
        reader.start()
        ready = f"cogniverse-web listening on http://127.0.0.1:{port}\n"
        deadline = time.monotonic() + 60
        while ready not in output and proc.poll() is None:
            assert time.monotonic() < deadline, (
                f"the web server did not start: {output}"
            )
            time.sleep(0.05)
        if ready in output:
            break
        reader.join(timeout=10)
        if not any("EADDRINUSE" in line for line in output):
            break
    assert ready in output, f"the web server exited: {output}"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        proc.terminate()
        assert proc.wait(timeout=20) == 0, (
            f"the web server did not exit cleanly: {output}"
        )
        reader.join(timeout=10)


def browse_as(context, tenant: str) -> None:
    """Open every page of the Playwright ``context`` with ``tenant`` as the
    web client's active tenant."""
    context.add_init_script(
        f"localStorage.setItem('cogniverse.tenant', {json.dumps(tenant)})"
    )


@contextmanager
def serve_app(app) -> Iterator[str]:
    """Serve an ASGI app on a real uvicorn socket; yields its base URL."""
    port = free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert server.started, "uvicorn did not start"
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=20)
        assert not thread.is_alive()


@contextmanager
def recording_telemetry_sink() -> Iterator[Tuple[str, List[str]]]:
    """A local stand-in for CopilotKit's telemetry endpoint.

    Yields its URL and the list of request paths it has received.
    """
    received: List[str] = []

    class Recorder(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(self.path)
            self.send_response(204)
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/ingest", received
    finally:
        server.shutdown()
        thread.join(timeout=10)
