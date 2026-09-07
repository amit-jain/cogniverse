"""Real runtime processes waiting on local HTTP and signal boundaries."""

from __future__ import annotations

import contextlib
import errno
import http.server
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest
import requests

from cogniverse_runtime.backend_startup import BACKEND_STARTUP_RETRY_INTERVAL_S

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast, pytest.mark.no_shared_vespa]

ROOT = Path(__file__).resolve().parents[3]
POLL = BACKEND_STARTUP_RETRY_INTERVAL_S
SLACK = 1.0
LOGGER = "cogniverse_runtime.runtime_cli"
PROBE_LOGGER = "cogniverse_runtime.backend_startup"


def _records(path, level):
    return [
        fields[3]
        for line in path.read_text().splitlines()
        if len(fields := line.split(" - ", 3)) == 4 and fields[1:3] == [LOGGER, level]
    ]


def _probe_calls(path):
    """Backend probe calls recorded by the runtime process itself.

    ``_wait_for_backend_startup`` is invoked once per readiness attempt with a
    zero budget, so it emits exactly one line per attempt and its own counter
    restarts each call. Counting the lines gives the attempt total the abort
    message must report, derived from the log rather than restated.
    """
    return [
        fields[3]
        for line in path.read_text().splitlines()
        if len(fields := line.split(" - ", 3)) == 4
        and fields[1:3] == [PROBE_LOGGER, "INFO"]
        and fields[3].startswith("Backend not ready, retrying (attempt 1,")
    ]


def _until(predicate, process, log, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        if process.poll() is not None:
            pytest.fail(f"runtime exited with {process.returncode}:\n{log.read_text()}")
        time.sleep(0.02)
    pytest.fail(f"runtime did not reach the expected state:\n{log.read_text()}")


@contextlib.contextmanager
def _reserved_ports():
    with socket.socket() as backend, socket.socket() as config, socket.socket() as api:
        # Keep both backend ports reserved but not listening throughout the outage.
        while True:
            backend.bind(("127.0.0.1", 0))
            port = backend.getsockname()[1]
            if port + 10991 < 65536:
                try:
                    config.bind(("127.0.0.1", port + 10991))
                    break
                except OSError:
                    pass
            backend.close()
            backend = socket.socket()
        api.bind(("127.0.0.1", 0))
        yield backend, port, api


@contextlib.contextmanager
def _runtime(tmp_path, port, api, *, grace="300", name="runtime"):
    log = tmp_path / f"{name}.log"
    api_port = api.getsockname()[1]
    api.close()
    command = json.loads(
        next(
            line[4:]
            for line in (ROOT / "libs/runtime/Dockerfile").read_text().splitlines()
            if line.startswith("CMD ")
        )
    )
    command[0] = sys.executable
    command[command.index("--port") + 1] = str(api_port)
    command += ["--lifespan", "off"]
    env = dict(
        os.environ,
        BACKEND_URL="http://127.0.0.1",
        BACKEND_PORT=str(port),
        RUNTIME_STARTUP_GRACE_SECONDS=grace,
        LOG_LEVEL="INFO",
        PYTHONUNBUFFERED="1",
    )
    with log.open("w") as output:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT
        )
        try:
            yield process, log, api_port
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=POLL + SLACK)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def _failure(port):
    return f"Backend data and config planes did not become ready at http://127.0.0.1:{port}"


def _waiting(process, log):
    _until(lambda: len(_records(log, "WARNING")) == 1, process, log)


@pytest.mark.parametrize("backend_mode", ["refused", "hung"])
def test_sigterm_aborts_runtime_wait_within_one_poll_interval(tmp_path, backend_mode):
    with (
        _reserved_ports() as (backend, port, api),
        contextlib.ExitStack() as stack,
    ):
        if backend_mode == "hung":
            backend.listen()
            backend.settimeout(30)
        process, log, _ = stack.enter_context(_runtime(tmp_path, port, api))
        if backend_mode == "hung":
            connection, _ = backend.accept()
            stack.enter_context(connection)
            connection.settimeout(5)
            assert (
                connection.recv(4096).split(b"\r\n", 1)[0]
                == b"GET /ApplicationStatus HTTP/1.1"
            )
        else:
            _waiting(process, log)
        started = time.monotonic()
        process.send_signal(signal.SIGTERM)
        code = process.wait(timeout=POLL + SLACK)
        elapsed = time.monotonic() - started
        assert code == 0
        assert elapsed < POLL + SLACK
        attempts = len(_probe_calls(log))
        assert _records(log, "INFO") == [
            f"Waiting for backend startup readiness at http://127.0.0.1:{port}...",
            "Runtime stopping before startup: Backend startup dependency wait aborted "
            f"by shutdown request after {attempts} "
            f"{'attempt' if attempts == 1 else 'attempts'}: "
            "ConfigStoreUnavailableError: Backend data and config planes did not "
            f"become ready at http://127.0.0.1:{port}",
        ]
        assert _records(log, "ERROR") == []
        print(f"SIGTERM elapsed={elapsed:.3f}s limit={POLL + SLACK:.3f}s exit={code}")


def test_runtime_logs_one_grace_error_and_keeps_waiting(tmp_path):
    with (
        _reserved_ports() as (_, port, api),
        _runtime(tmp_path, port, api, grace="1") as (process, log, _),
    ):
        _waiting(process, log)
        started = time.monotonic()
        _until(lambda: len(_records(log, "ERROR")) == 1, process, log)
        elapsed = time.monotonic() - started
        assert elapsed < 1 + SLACK
        boundary_records = [
            line.split(" - ", 3)
            for line in log.read_text().splitlines()
            if line.split(" - ", 3)[1:3] == [LOGGER, "INFO"]
            or line.split(" - ", 3)[1:3] == [LOGGER, "ERROR"]
        ]
        timestamps = [
            datetime.strptime(fields[0], "%Y-%m-%d %H:%M:%S,%f")
            for fields in boundary_records
        ]
        assert 1.0 <= (timestamps[1] - timestamps[0]).total_seconds() < 1.0 + SLACK
        _until(
            lambda: len(_records(log, "WARNING")) == 3,
            process,
            log,
            timeout=2 * POLL + SLACK,
        )
        assert process.poll() is None
        assert _records(log, "ERROR") == [
            "Backend startup dependency was not ready after 2 attempts within 1.0s: "
            "ConfigStoreUnavailableError: "
            + _failure(port)
            + "; keeping the runtime alive and retrying"
        ]
        assert len(_records(log, "ERROR")) == 1


class _BackendHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        statuses = {
            "/ApplicationStatus": 200,
            "/document/v1/config_metadata/config_metadata/docid/probe": 404,
        }
        self.send_response(statuses[self.path])
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        pass


def test_concurrent_runtime_entrypoints_serve_when_backend_starts(tmp_path):
    with _reserved_ports() as (backend, port, api), socket.socket() as second_api:
        second_api.bind(("127.0.0.1", 0))
        with (
            _runtime(tmp_path, port, api, name="first") as first,
            _runtime(tmp_path, port, second_api, name="second") as second,
        ):
            for process, log, api_port in (first, second):
                _waiting(process, log)
                with socket.socket() as probe:
                    assert (
                        probe.connect_ex(("127.0.0.1", api_port)) == errno.ECONNREFUSED
                    )
            backend.close()
            server = http.server.ThreadingHTTPServer(
                ("127.0.0.1", port), _BackendHandler
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                for process, log, api_port in (first, second):
                    _until(
                        lambda: (
                            _records(log, "INFO")[-1:]
                            == ["Backend feed endpoint is ready"]
                        ),
                        process,
                        log,
                    )

                    def serves():
                        try:
                            return (
                                requests.get(
                                    f"http://127.0.0.1:{api_port}/health/live",
                                    timeout=0.5,
                                ).status_code
                                == 200
                            )
                        except requests.ConnectionError:
                            return False

                    _until(serves, process, log)
                    response = requests.get(
                        f"http://127.0.0.1:{api_port}/health/live", timeout=1
                    )
                    assert response.status_code == 200
                    assert response.json() == {"status": "alive"}
                    assert process.poll() is None
                    assert _records(log, "INFO") == [
                        f"Waiting for backend startup readiness at http://127.0.0.1:{port}...",
                        "Backend feed endpoint is ready",
                    ]
                    assert _records(log, "ERROR") == []
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
