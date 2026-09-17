"""The admin ArtifactManager targets the Phoenix endpoints wired at startup.

The endpoints are injected once at the entrypoint via set_phoenix_endpoints
and read from module state here, never from the process environment and never
from a built-in default.
"""

from __future__ import annotations

import threading

import pytest

from cogniverse_runtime.routers import admin

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.fixture
def recorded_initialize(monkeypatch):
    from cogniverse_telemetry_phoenix.provider import PhoenixProvider

    configs: list[dict] = []
    real_initialize = PhoenixProvider.initialize

    def _recording_initialize(self, config):
        configs.append(dict(config))
        return real_initialize(self, config)

    monkeypatch.setattr(PhoenixProvider, "initialize", _recording_initialize)
    monkeypatch.setattr(admin, "_phoenix_endpoints", admin._phoenix_endpoints)
    return configs


def test_build_artifact_manager_uses_wired_endpoints_not_env(
    monkeypatch, recorded_initialize
):
    monkeypatch.setenv("TELEMETRY_HTTP_ENDPOINT", "http://env-should-not-win:6006")
    monkeypatch.setenv("TELEMETRY_OTLP_ENDPOINT", "env-should-not-win:4317")

    admin.set_phoenix_endpoints("http://wired-phoenix:6006", "wired-phoenix:4317")
    manager = admin._build_artifact_manager("acme:acme")

    assert recorded_initialize == [
        {
            "tenant_id": "acme:acme",
            "http_endpoint": "http://wired-phoenix:6006",
            "grpc_endpoint": "wired-phoenix:4317",
        }
    ]
    assert manager._tenant_id == "acme:acme"
    assert manager._provider._http_endpoint == "http://wired-phoenix:6006"


def test_an_unwired_router_refuses_to_build_an_artifact_manager(
    monkeypatch, recorded_initialize
):
    monkeypatch.setattr(admin, "_phoenix_endpoints", {})

    with pytest.raises(RuntimeError) as excinfo:
        admin._build_artifact_manager("acme:acme")

    assert str(excinfo.value) == (
        "admin artifact store for tenant 'acme:acme' has no Phoenix endpoints: "
        "set_phoenix_endpoints was never called"
    )
    assert recorded_initialize == []


@pytest.mark.parametrize(
    ("http_endpoint", "grpc_endpoint", "message"),
    [
        (
            "",
            "wired-phoenix:4317",
            "admin Phoenix wiring needs http_endpoint; got http_endpoint='', "
            "grpc_endpoint='wired-phoenix:4317'",
        ),
        (
            "http://wired-phoenix:6006",
            "",
            "admin Phoenix wiring needs grpc_endpoint; got "
            "http_endpoint='http://wired-phoenix:6006', grpc_endpoint=''",
        ),
    ],
)
def test_wiring_an_empty_endpoint_is_refused_and_keeps_the_previous_pair(
    monkeypatch, http_endpoint, grpc_endpoint, message
):
    monkeypatch.setattr(
        admin,
        "_phoenix_endpoints",
        {"http_endpoint": "http://before:6006", "grpc_endpoint": "before:4317"},
    )

    with pytest.raises(ValueError) as excinfo:
        admin.set_phoenix_endpoints(http_endpoint, grpc_endpoint)

    assert str(excinfo.value) == message
    assert admin._phoenix_endpoints == {
        "http_endpoint": "http://before:6006",
        "grpc_endpoint": "before:4317",
    }


class _PausingEndpoints(dict):
    """Endpoint state that parks a writer after its first item assignment."""

    def __init__(self, initial, half_written, reader_done):
        super().__init__(initial)
        self._half_written = half_written
        self._reader_done = reader_done

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._half_written.set()
        self._reader_done.wait(timeout=5)


def test_a_build_racing_a_rewire_sees_one_complete_pair(
    monkeypatch, recorded_initialize
):
    before = {"http_endpoint": "http://before:6006", "grpc_endpoint": "before:4317"}
    after = {"http_endpoint": "http://after:6006", "grpc_endpoint": "after:4317"}
    half_written = threading.Event()
    reader_done = threading.Event()
    monkeypatch.setattr(
        admin,
        "_phoenix_endpoints",
        _PausingEndpoints(before, half_written, reader_done),
    )

    writer = threading.Thread(
        target=admin.set_phoenix_endpoints,
        args=(after["http_endpoint"], after["grpc_endpoint"]),
    )
    writer.start()
    # A writer that assigns item by item parks here with one endpoint written;
    # one that replaces the pair finishes without ever touching the old state.
    while writer.is_alive() and not half_written.wait(timeout=0.01):
        pass
    admin._build_artifact_manager("acme:acme")
    reader_done.set()
    writer.join(timeout=10)

    assert writer.is_alive() is False
    assert recorded_initialize == [{"tenant_id": "acme:acme", **after}]
    assert dict(admin._phoenix_endpoints) == after
