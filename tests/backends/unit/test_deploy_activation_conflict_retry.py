"""A retried Vespa deploy must resend the whole application package.

``_deploy_package`` builds the zip once with ``app_package.to_zip()``, which
returns a ``BytesIO``. ``requests`` reads that stream to EOF on the first
POST, so an ACTIVATION_CONFLICT retry posted an empty body and Vespa
answered ``400 services.xml does not exist in application package``. The
409 retry could therefore never succeed; it only converted a retriable
conflict into a confusing hard failure.

These tests drive a real HTTP server on a real socket. Patching
``requests.post`` cannot catch this class of bug: a mock never reads the
request body, so the stream is never consumed and every retry looks fine.
"""

from __future__ import annotations

import json
import logging
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

import pytest
from vespa.package import ApplicationPackage

from cogniverse_vespa.backend import VespaBackend
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

DEPLOY_PATH = "/application/v2/tenant/default/prepareandactivate"

CONFLICT_BODY = {
    "error-code": "ACTIVATION_CONFLICT",
    "message": (
        "This session 2002 was prepared when session 2000 was active, but "
        "session 2001 has since become active: refusing to activate this"
    ),
}

EXPECTED_PACKAGE_ENTRIES = {
    "schemas/conflictprobe.sd",
    "search/query-profiles/default.xml",
    "search/query-profiles/types/root.xml",
    "services.xml",
    "validation-overrides.xml",
}


class _ConfigServer:
    """Real config-server stand-in recording every request body."""

    def __init__(self, statuses):
        self.bodies: list[bytes] = []
        self.paths: list[str] = []
        statuses = list(statuses)
        recorder = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                recorder.bodies.append(self.rfile.read(length))
                recorder.paths.append(self.path)
                idx = len(recorder.bodies) - 1
                status = statuses[min(idx, len(statuses) - 1)]
                payload = (
                    json.dumps(CONFLICT_BODY).encode()
                    if status == 409
                    else json.dumps({"message": "ok"}).encode()
                )
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]

    def __enter__(self):
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def _make_backend(port: int) -> VespaBackend:
    backend = object.__new__(VespaBackend)
    backend._url = "http://127.0.0.1"
    backend._port = 8080
    backend._config_port = port
    return backend


def _make_schema_manager(port: int) -> VespaSchemaManager:
    manager = object.__new__(VespaSchemaManager)
    manager.backend_endpoint = "http://127.0.0.1"
    manager.backend_port = port
    manager._logger = logging.getLogger("test.vsm")
    return manager


def _entries(body: bytes) -> set[str]:
    return set(zipfile.ZipFile(BytesIO(body)).namelist())


def test_activation_conflict_retry_resends_the_complete_package():
    """The retry after a 409 must carry the same complete zip, not an
    empty body left over from the consumed stream."""
    app_package = ApplicationPackage(name="conflictprobe")

    with _ConfigServer([409, 200]) as server:
        backend = _make_backend(server.port)
        backend._deploy_package(app_package)

    assert len(server.bodies) == 2, (
        f"Expected one conflict then one retry, got {len(server.bodies)} requests"
    )
    assert server.paths == [DEPLOY_PATH, DEPLOY_PATH]
    assert server.bodies[0] == server.bodies[1], (
        "Retry must resend the identical package; retry body was "
        f"{len(server.bodies[1])} bytes vs {len(server.bodies[0])} on the "
        "first attempt"
    )
    assert _entries(server.bodies[1]) == EXPECTED_PACKAGE_ENTRIES


def test_successful_deploy_sends_the_package_once():
    """Pin the happy path: exactly one POST carrying the full package."""
    app_package = ApplicationPackage(name="conflictprobe")

    with _ConfigServer([200]) as server:
        backend = _make_backend(server.port)
        backend._deploy_package(app_package)

    assert len(server.bodies) == 1
    assert _entries(server.bodies[0]) == EXPECTED_PACKAGE_ENTRIES


def test_non_retriable_status_raises_after_one_attempt():
    """A 400 is not a conflict: fail immediately, do not burn retries."""
    app_package = ApplicationPackage(name="conflictprobe")

    with _ConfigServer([400]) as server:
        backend = _make_backend(server.port)
        with pytest.raises(RuntimeError) as exc_info:
            backend._deploy_package(app_package)

    assert len(server.bodies) == 1
    assert "400" in str(exc_info.value)


def test_conflict_on_every_attempt_exhausts_retries_and_raises():
    """All five attempts carry the full package, then the deploy raises."""
    app_package = ApplicationPackage(name="conflictprobe")

    with _ConfigServer([409]) as server:
        backend = _make_backend(server.port)
        with pytest.raises(RuntimeError) as exc_info:
            backend._deploy_package(app_package)

    assert len(server.bodies) == 5, f"Expected 5 attempts, got {len(server.bodies)}"
    for i, body in enumerate(server.bodies):
        assert _entries(body) == EXPECTED_PACKAGE_ENTRIES, (
            f"Attempt {i + 1} sent an incomplete package: {_entries(body)}"
        )
    assert "ACTIVATION_CONFLICT" in str(exc_info.value)


def test_schema_manager_retry_resends_the_complete_package():
    """VespaSchemaManager._deploy_package retries any 409 while holding the
    process-wide deploy lock; each attempt must carry the full package."""
    app_package = ApplicationPackage(name="conflictprobe")

    with _ConfigServer([409, 200]) as server:
        manager = _make_schema_manager(server.port)
        manager._deploy_package(app_package)

    assert len(server.bodies) == 2
    assert server.bodies[0] == server.bodies[1]
    assert _entries(server.bodies[1]) == EXPECTED_PACKAGE_ENTRIES
