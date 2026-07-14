"""Deploys must fail loudly when the schema never activates or Vespa stalls.

``_wait_for_schema_convergence`` used to log "proceeding anyway" on timeout
and return normally, so ``deploy_schemas`` returned True for a schema Vespa
never activated and callers fed/searched a nonexistent doctype. The two
deploy POSTs were also the only requests calls in the package without a
timeout — a config server that accepts the connection but never responds
wedged the call forever, one of them while holding the process-wide deploy
lock.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
import requests
from vespa.package import ApplicationPackage

from cogniverse_vespa import vespa_schema_manager as vsm_module
from cogniverse_vespa.backend import VespaBackend
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager


def _probe_response(errors: list) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"root": {"errors": errors, "children": []}}
    return response


def _make_backend() -> VespaBackend:
    backend = object.__new__(VespaBackend)
    backend._url = "http://localhost"
    backend._port = 8080
    return backend


def test_convergence_timeout_raises():
    """A schema that never becomes query-visible must raise, naming it."""
    backend = _make_backend()
    not_found = _probe_response(
        [{"code": 8, "summary": "Schema 'video_x_acme' not found"}]
    )

    with patch("requests.post", return_value=not_found), patch("time.sleep"):
        with pytest.raises(RuntimeError) as exc_info:
            backend._wait_for_schema_convergence(["video_x_acme"], timeout=2)

    assert "video_x_acme" in str(exc_info.value)


def test_convergence_success_returns():
    """Pin the happy path: a clean probe converges without raising."""
    backend = _make_backend()
    converged = _probe_response([])

    with patch("requests.post", return_value=converged), patch("time.sleep"):
        backend._wait_for_schema_convergence(["video_ok_acme"], timeout=2)


def test_convergence_partial_raises_and_names_only_missing():
    """Only the schemas that failed to converge appear in the error."""
    backend = _make_backend()

    def probe(_url, json=None, timeout=None):
        name = json["yql"]
        if "video_ok_acme" in name:
            return _probe_response([])
        return _probe_response([{"code": 8, "summary": "not found"}])

    with patch("requests.post", side_effect=probe), patch("time.sleep"):
        with pytest.raises(RuntimeError) as exc_info:
            backend._wait_for_schema_convergence(
                ["video_ok_acme", "video_missing_acme"], timeout=2
            )

    message = str(exc_info.value)
    assert "video_missing_acme" in message
    assert "video_ok_acme" not in message


@pytest.fixture()
def stalled_server():
    """A real socket that accepts connections and never responds."""
    server = socket.create_server(("127.0.0.1", 0))
    server.settimeout(0.2)
    port = server.getsockname()[1]
    stop = threading.Event()
    held: list[socket.socket] = []

    def _accept_loop():
        while not stop.is_set():
            try:
                conn, _ = server.accept()
                held.append(conn)
            except TimeoutError:
                continue
            except OSError:
                return

    thread = threading.Thread(target=_accept_loop, daemon=True)
    thread.start()
    yield port
    stop.set()
    for conn in held:
        conn.close()
    server.close()
    thread.join(timeout=2)


def _assert_deploy_times_out(deploy_fn) -> None:
    outcome: dict = {}

    def _run():
        try:
            deploy_fn()
            outcome["result"] = "returned"
        except requests.exceptions.Timeout:
            outcome["result"] = "timeout"
        except Exception as exc:
            outcome["result"] = f"other:{exc!r}"

    worker = threading.Thread(target=_run, daemon=True)
    started = time.monotonic()
    worker.start()
    worker.join(timeout=8)
    elapsed = time.monotonic() - started

    assert outcome.get("result") == "timeout", (
        f"deploy did not time out within {elapsed:.1f}s: "
        f"{outcome.get('result', 'STILL HANGING')}"
    )


def test_backend_deploy_post_times_out_instead_of_hanging(stalled_server, monkeypatch):
    from cogniverse_vespa import backend as backend_module

    monkeypatch.setattr(backend_module, "DEPLOY_REQUEST_TIMEOUT_S", (1, 1))

    backend = object.__new__(VespaBackend)
    backend._url = "http://127.0.0.1"
    backend._config_port = stalled_server

    _assert_deploy_times_out(
        lambda: backend._deploy_package(ApplicationPackage(name="testapp"))
    )


def test_schema_manager_deploy_post_times_out_instead_of_hanging(
    stalled_server, monkeypatch
):
    monkeypatch.setattr(vsm_module, "DEPLOY_REQUEST_TIMEOUT_S", (1, 1))

    manager = object.__new__(VespaSchemaManager)
    manager.backend_endpoint = "http://127.0.0.1"
    manager.backend_port = stalled_server
    manager._logger = logging.getLogger("test_schema_manager")

    _assert_deploy_times_out(
        lambda: manager._deploy_package(ApplicationPackage(name="testapp"))
    )
