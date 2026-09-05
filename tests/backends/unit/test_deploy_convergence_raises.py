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
from collections import Counter
from unittest.mock import call, patch

import pytest
import requests
from vespa.package import ApplicationPackage

from cogniverse_vespa import vespa_schema_manager as vsm_module
from cogniverse_vespa.backend import VespaBackend
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager


def _probe_response(status_code: int) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    return response


def _make_backend() -> VespaBackend:
    backend = object.__new__(VespaBackend)
    backend._url = "http://localhost"
    backend._port = 8080
    return backend


@pytest.fixture
def probe_clock(monkeypatch):
    elapsed = [0.0]

    def sleep(seconds):
        elapsed[0] += seconds

    monkeypatch.setattr(time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(time, "sleep", sleep)
    return elapsed


@pytest.mark.parametrize("status", [200, 400, 404, 429, 503, 504, 599])
def test_convergence_timeout_raises(status, probe_clock):
    """Only a condition failure proves the conditional feed reached storage."""
    backend = _make_backend()
    with patch("requests.Session.post", return_value=_probe_response(status)):
        with pytest.raises(RuntimeError) as exc_info:
            backend._wait_for_schema_convergence(["video_x_acme"], timeout=2)

    assert str(exc_info.value) == (
        "Schema convergence not confirmed after 2s — deploy was "
        "accepted by the config server but these schemas never became "
        "feed-ready: ['video_x_acme']"
    )
    assert probe_clock == [2.0]


def test_convergence_success_returns(probe_clock):
    backend = _make_backend()
    with patch("requests.Session.post", return_value=_probe_response(412)) as post:
        result = backend._wait_for_schema_convergence(["video_ok_acme"], timeout=2)

    assert result is None
    assert probe_clock == [0.0]
    assert post.call_args_list == [
        call(
            "http://localhost:8080/document/v1/video_ok_acme/video_ok_acme/"
            "docid/convergence_probe",
            params={"condition": "false", "timeout": "2000ms"},
            json={"fields": {}},
            timeout=2.0,
        )
    ]


def test_convergence_partial_raises_and_names_only_missing(probe_clock):
    backend = _make_backend()
    calls = []

    def probe(url, **kwargs):
        name = url.split("/")[5]
        calls.append(name)
        return _probe_response(412 if name == "video_ok_acme" else 400)

    with patch("requests.Session.post", side_effect=probe):
        with pytest.raises(RuntimeError) as exc_info:
            backend._wait_for_schema_convergence(
                ["video_ok_acme", "video_missing_acme"], timeout=2
            )

    assert str(exc_info.value) == (
        "Schema convergence not confirmed after 2s — deploy was "
        "accepted by the config server but these schemas never became "
        "feed-ready: ['video_missing_acme']"
    )
    assert calls == ["video_missing_acme", "video_ok_acme", "video_missing_acme"]


def test_convergence_retries_only_pending_schemas(probe_clock):
    backend = _make_backend()
    ready = [f"video_{i:03d}_acme" for i in range(130)]
    calls = Counter()

    def probe(url, **kwargs):
        name = url.split("/")[5]
        calls[name] += 1
        return _probe_response(599 if name == "video_late_acme" else 412)

    with patch("requests.Session.post", side_effect=probe):
        with pytest.raises(RuntimeError) as exc_info:
            backend._wait_for_schema_convergence(
                [*ready, "video_late_acme", *ready], timeout=3
            )
    assert str(exc_info.value) == (
        "Schema convergence not confirmed after 3s — deploy was "
        "accepted by the config server but these schemas never became "
        "feed-ready: ['video_late_acme']"
    )
    assert calls == Counter({**dict.fromkeys(ready, 1), "video_late_acme": 3})


def test_convergence_empty_schema_list_does_not_probe(probe_clock):
    with patch("requests.Session.post") as post:
        result = _make_backend()._wait_for_schema_convergence([])
    assert result is None
    assert post.call_args_list == []
    assert probe_clock == [0.0]


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


def test_convergence_hung_http_raises_with_schema_names(stalled_server):
    backend = _make_backend()
    backend._url = "http://127.0.0.1"
    backend._port = stalled_server
    with pytest.raises(RuntimeError) as exc_info:
        backend._wait_for_schema_convergence(["video_hung_acme"], timeout=1)
    assert str(exc_info.value) == (
        "Schema convergence not confirmed after 1s — deploy was "
        "accepted by the config server but these schemas never became "
        "feed-ready: ['video_hung_acme']"
    )
