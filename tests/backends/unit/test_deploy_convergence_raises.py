"""A deploy returns only once its generation runs on every service and each
new schema accepts a real feed.

``_wait_for_schema_convergence`` no longer trusts a conditional-POST 412 from
the data path: the distributor answers that 412 ("document does not exist")
from its bucket-space mapping while the content node's DocumentDB is still
initializing, so a real feed to the same doctype is rejected with
APP_FATAL_ERROR "No handler for document type". The gate now waits for the
config-server ``serviceconverge`` report to show every service running the
activated generation, then writes and removes one probe document per new
schema over the same document/v1 path a real feed uses.

The two deploy POSTs are also the only requests calls in the package without a
timeout — a config server that accepts the connection but never responds
wedged the call forever, one of them while holding the process-wide deploy
lock.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from unittest.mock import call, patch

import pytest
import requests
from vespa.package import ApplicationPackage

from cogniverse_vespa import vespa_schema_manager as vsm_module
from cogniverse_vespa.backend import VespaBackend
from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

CONVERGE_PATH = (
    "http://localhost:19071/application/v2/tenant/default/application/default/"
    "environment/prod/region/default/instance/default/serviceconverge"
)
DOCUMENT_URL = "http://localhost:8080/document/v1"


def _response(status_code: int, *, payload=None, text: str | None = None):
    response = requests.Response()
    response.status_code = status_code
    if payload is not None:
        response._content = json.dumps(payload).encode()
        response.headers["Content-Type"] = "application/json"
    elif text is not None:
        response._content = text.encode()
    return response


def _make_backend() -> VespaBackend:
    backend = object.__new__(VespaBackend)
    backend._url = "http://localhost"
    backend._port = 8080
    backend._config_port = 19071
    return backend


def _services(generation: int, *, behind=None):
    behind = behind or {}
    types = [
        "container",
        "container-clustercontroller",
        "distributor",
        "logserver-container",
        "metricsproxy-container",
        "searchnode",
        "storagenode",
    ]
    return {
        "services": [
            {
                "type": t,
                "host": "h",
                "port": 9000 + i,
                "currentGeneration": behind.get(t, generation),
            }
            for i, t in enumerate(types)
        ]
    }


@pytest.fixture
def probe_clock(monkeypatch):
    elapsed = [0.0]

    def sleep(seconds):
        elapsed[0] += seconds

    monkeypatch.setattr(time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(time, "sleep", sleep)
    return elapsed


def test_gate_waits_for_generation_then_feeds_each_new_schema(probe_clock):
    backend = _make_backend()
    get_calls = []
    feed_calls = []

    def fake_get(url, **kwargs):
        get_calls.append(kwargs["params"])
        # First poll: searchnode still one generation behind; second: caught up.
        if len(get_calls) == 1:
            return _response(200, payload=_services(9, behind={"searchnode": 8}))
        return _response(200, payload=_services(9))

    def fake_post(url, **kwargs):
        feed_calls.append(("POST", url))
        return _response(200)

    def fake_delete(url, **kwargs):
        feed_calls.append(("DELETE", url))
        return _response(200)

    with (
        patch("requests.Session.get", side_effect=fake_get),
        patch("requests.Session.post", side_effect=fake_post),
        patch("requests.Session.delete", side_effect=fake_delete),
    ):
        result = backend._wait_for_schema_convergence(9, ["video_new_acme"], timeout=5)

    assert result is None
    assert get_calls == [{"timeout": "5"}, {"timeout": "4"}]
    probe = f"{DOCUMENT_URL}/video_new_acme/video_new_acme/docid/convergence_probe"
    assert feed_calls == [("POST", probe), ("DELETE", probe)]
    assert probe_clock == [1.0]


def test_gate_raises_naming_the_service_that_never_runs_the_generation(probe_clock):
    backend = _make_backend()

    def fake_get(url, **kwargs):
        return _response(200, payload=_services(9, behind={"searchnode": -1}))

    with (
        patch("requests.Session.get", side_effect=fake_get),
        patch("requests.Session.post") as post,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            backend._wait_for_schema_convergence(9, ["video_new_acme"], timeout=3)

    assert str(exc_info.value) == (
        "Schema convergence not confirmed after 3s — generation 9 was activated "
        "by the config server but is not live on every service: services behind "
        "generation 9: ['searchnode@h:9005=-1']"
    )
    assert post.call_args_list == []
    assert probe_clock == [3.0]


def test_gate_raises_naming_the_schema_the_feed_path_rejects(probe_clock):
    backend = _make_backend()

    def fake_get(url, **kwargs):
        return _response(200, payload=_services(9))

    def fake_post(url, **kwargs):
        return _response(
            400,
            text='{"message":"Document type video_missing_acme does not exist"}',
        )

    with (
        patch("requests.Session.get", side_effect=fake_get),
        patch("requests.Session.post", side_effect=fake_post),
        patch("requests.Session.delete") as delete,
    ):
        with pytest.raises(RuntimeError) as exc_info:
            backend._wait_for_schema_convergence(9, ["video_missing_acme"], timeout=2)

    assert str(exc_info.value) == (
        "Schema convergence not confirmed after 2s — generation 9 is live on "
        "every service but these schemas never accepted a feed: "
        "{'video_missing_acme': 'feed HTTP 400: {\"message\":\"Document type "
        "video_missing_acme does not exist\"}'}"
    )
    assert delete.call_args_list == []
    assert probe_clock == [2.0]


def test_gate_feeds_only_new_schemas_never_the_whole_package(probe_clock):
    backend = _make_backend()
    feed_targets = []

    def fake_get(url, **kwargs):
        return _response(200, payload=_services(9))

    def fake_feed(url, **kwargs):
        feed_targets.append(url.split("/document/v1/")[1].split("/")[0])
        return _response(200)

    package_schemas = [f"video_{i:03d}_acme" for i in range(130)]
    with (
        patch("requests.Session.get", side_effect=fake_get),
        patch("requests.Session.post", side_effect=fake_feed),
        patch("requests.Session.delete", side_effect=fake_feed),
    ):
        backend._wait_for_schema_convergence(9, package_schemas[-1:], timeout=5)

    assert feed_targets == ["video_129_acme", "video_129_acme"]


def test_gate_empty_new_schema_list_still_waits_for_the_generation(probe_clock):
    backend = _make_backend()

    def fake_get(url, **kwargs):
        return _response(200, payload=_services(9))

    with (
        patch("requests.Session.get", side_effect=fake_get) as get,
        patch("requests.Session.post") as post,
    ):
        result = backend._wait_for_schema_convergence(9, [], timeout=5)

    assert result is None
    assert get.call_args_list == [
        call(CONVERGE_PATH, params={"timeout": "5"}, timeout=10.0)
    ]
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


def test_gate_hung_config_server_raises_with_generation(stalled_server):
    backend = _make_backend()
    backend._url = "http://127.0.0.1"
    backend._config_port = stalled_server
    with pytest.raises(RuntimeError) as exc_info:
        backend._wait_for_schema_convergence(9, ["video_hung_acme"], timeout=1)
    message = str(exc_info.value)
    assert message.startswith(
        "Schema convergence not confirmed after 1s — generation 9 was activated "
        "by the config server but is not live on every service: serviceconverge "
        "request failed: "
    ), message
