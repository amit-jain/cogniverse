"""Process-level tests for dependency-gated runtime commands."""

from __future__ import annotations

import contextlib
import http.server
import os
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
]


def test_process_boundary_keeps_the_isolated_backend_sentinel() -> None:
    assert os.environ["BACKEND_URL"] == os.environ.get(
        "TEST_BACKEND_URL", "http://localhost"
    )
    assert os.environ["BACKEND_PORT"] == os.environ.get("TEST_BACKEND_PORT", "29071")


class _StatusHandler(http.server.BaseHTTPRequestHandler):
    status = 200

    def do_GET(self) -> None:
        self.send_response(self.status)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        return


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextlib.contextmanager
def _http_listener(port: int, status: int = 200) -> Iterator[None]:
    handler = type("StatusHandler", (_StatusHandler,), {"status": status})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@contextlib.contextmanager
def _tcp_listener(port: int) -> Iterator[None]:
    stopped = threading.Event()
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen()
    listener.settimeout(0.1)

    def accept_connections() -> None:
        while not stopped.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            with connection:
                pass

    thread = threading.Thread(target=accept_connections, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=2)
        listener.close()


_WRITE_MARKER = """
import os
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(
    f"{sys.argv[2]}:{os.getpid()}", encoding="utf-8"
)
"""


def _start_wrapper(
    *,
    http_url: str,
    tcp_endpoint: str,
    marker: Path,
    value: str,
    timeout: str = "5",
    http_statuses: str | None = None,
) -> subprocess.Popen[str]:
    http_args = (
        ["--http-status", http_url, http_statuses]
        if http_statuses is not None
        else ["--http", http_url]
    )
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cogniverse_runtime.startup_wait",
            "--timeout-seconds",
            timeout,
            *http_args,
            "--tcp",
            tcp_endpoint,
            "--",
            sys.executable,
            "-c",
            _WRITE_MARKER,
            str(marker),
            value,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_configured_http_status_launches_child_on_real_404_response(tmp_path):
    http_port = _free_port()
    tcp_port = _free_port()
    marker = tmp_path / "child"
    http_url = f"http://127.0.0.1:{http_port}/document/v1/probe"
    process = _start_wrapper(
        http_url=http_url,
        http_statuses="200,404",
        tcp_endpoint=f"127.0.0.1:{tcp_port}",
        marker=marker,
        value="feed-plane-ready",
    )

    with _http_listener(http_port, status=404), _tcp_listener(tcp_port):
        stdout, stderr = process.communicate(timeout=5)

    assert process.returncode == 0
    assert stderr == ""
    assert stdout.splitlines() == [
        f"waiting for http dependency {http_url} (statuses: 200,404)",
        f"http dependency ready: {http_url} (statuses: 200,404)",
        f"waiting for tcp dependency 127.0.0.1:{tcp_port}",
        f"tcp dependency ready: 127.0.0.1:{tcp_port}",
    ]
    assert marker.read_text(encoding="utf-8") == f"feed-plane-ready:{process.pid}"


def test_unconfigured_http_status_never_launches_child(tmp_path):
    http_port = _free_port()
    tcp_port = _free_port()
    marker = tmp_path / "child"
    http_url = f"http://127.0.0.1:{http_port}/document/v1/probe"
    process = _start_wrapper(
        http_url=http_url,
        http_statuses="200,404",
        tcp_endpoint=f"127.0.0.1:{tcp_port}",
        marker=marker,
        value="must-not-run",
        timeout="0.3",
    )

    with _http_listener(http_port, status=503), _tcp_listener(tcp_port):
        stdout, stderr = process.communicate(timeout=3)

    dependency = f"{http_url} (statuses: 200,404)"
    assert process.returncode == 1
    assert stdout.splitlines() == [f"waiting for http dependency {dependency}"]
    assert stderr.strip() == (
        f"timed out waiting for http dependency {dependency} after 0.30 seconds"
    )
    assert not marker.exists()


@pytest.mark.parametrize("statuses", ["", "200,", "200,200", "99", "600", "ok"])
def test_malformed_http_statuses_are_rejected_before_child_execution(
    tmp_path, statuses
):
    marker = tmp_path / "child"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cogniverse_runtime.startup_wait",
            "--timeout-seconds",
            "1",
            "--http-status",
            "http://127.0.0.1:1/probe",
            statuses,
            "--",
            sys.executable,
            "-c",
            _WRITE_MARKER,
            str(marker),
            "must-not-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=3,
    )

    assert result.returncode == 2
    assert "invalid HTTP status list" in result.stderr
    assert not marker.exists()


def test_concurrent_commands_wait_for_real_http_and_tcp_dependencies(tmp_path):
    http_port = _free_port()
    tcp_port = _free_port()
    http_url = f"http://127.0.0.1:{http_port}/ApplicationStatus"
    tcp_endpoint = f"127.0.0.1:{tcp_port}"
    markers = [tmp_path / "first", tmp_path / "second"]
    processes = [
        _start_wrapper(
            http_url=http_url,
            tcp_endpoint=tcp_endpoint,
            marker=marker,
            value=value,
        )
        for marker, value in zip(markers, ("alpha", "beta"), strict=True)
    ]

    time.sleep(0.2)
    assert [process.poll() for process in processes] == [None, None]
    assert [marker.exists() for marker in markers] == [False, False]

    with _tcp_listener(tcp_port), _http_listener(http_port):
        results = [process.communicate(timeout=5) for process in processes]

    assert [process.returncode for process in processes] == [0, 0]
    assert [stderr for _, stderr in results] == ["", ""]
    assert [marker.read_text(encoding="utf-8") for marker in markers] == [
        f"alpha:{processes[0].pid}",
        f"beta:{processes[1].pid}",
    ]
    expected_output = [
        f"waiting for http dependency {http_url}",
        f"http dependency ready: {http_url}",
        f"waiting for tcp dependency {tcp_endpoint}",
        f"tcp dependency ready: {tcp_endpoint}",
    ]
    assert [stdout.splitlines() for stdout, _ in results] == [
        expected_output,
        expected_output,
    ]


def test_http_non_200_never_executes_child(tmp_path):
    http_port = _free_port()
    tcp_port = _free_port()
    marker = tmp_path / "child"
    http_url = f"http://127.0.0.1:{http_port}/ApplicationStatus"
    process = _start_wrapper(
        http_url=http_url,
        tcp_endpoint=f"127.0.0.1:{tcp_port}",
        marker=marker,
        value="must-not-run",
        timeout="0.3",
    )

    with _http_listener(http_port, status=503):
        stdout, stderr = process.communicate(timeout=3)

    assert process.returncode == 1
    assert stdout.splitlines() == [f"waiting for http dependency {http_url}"]
    assert stderr.strip() == (
        f"timed out waiting for http dependency {http_url} after 0.30 seconds"
    )
    assert not marker.exists()


def test_refused_tcp_dependency_never_executes_child(tmp_path):
    http_port = _free_port()
    tcp_port = _free_port()
    marker = tmp_path / "child"
    http_url = f"http://127.0.0.1:{http_port}/ApplicationStatus"
    process = _start_wrapper(
        http_url=http_url,
        tcp_endpoint=f"127.0.0.1:{tcp_port}",
        marker=marker,
        value="must-not-run",
        timeout="0.3",
    )

    with _http_listener(http_port):
        stdout, stderr = process.communicate(timeout=3)

    assert process.returncode == 1
    assert stdout.splitlines() == [
        f"waiting for http dependency {http_url}",
        f"http dependency ready: {http_url}",
        f"waiting for tcp dependency 127.0.0.1:{tcp_port}",
    ]
    assert stderr.strip() == (
        f"timed out waiting for tcp dependency 127.0.0.1:{tcp_port} after 0.30 seconds"
    )
    assert not marker.exists()


def test_malformed_tcp_dependency_is_rejected_before_child_execution(tmp_path):
    marker = tmp_path / "child"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cogniverse_runtime.startup_wait",
            "--timeout-seconds",
            "1",
            "--tcp",
            "redis:6379:legacy",
            "--",
            sys.executable,
            "-c",
            _WRITE_MARKER,
            str(marker),
            "must-not-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=3,
    )

    assert result.returncode == 2
    assert "invalid TCP dependency 'redis:6379:legacy'; expected HOST:PORT" in (
        result.stderr
    )
    assert not marker.exists()
