"""The admin canary ArtifactManager targets the Phoenix endpoints wired at
startup, not the process environment read at request time.

Reading os.environ inside a request handler couples the router to the process
environment; the endpoints are now injected once at the entrypoint via
set_phoenix_endpoints and read from module state here.
"""

from __future__ import annotations

import pytest

from cogniverse_runtime.routers import admin

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def test_build_artifact_manager_uses_wired_endpoints_not_env(monkeypatch):
    captured = {}

    class _FakeProvider:
        def initialize(self, cfg):
            captured["cfg"] = cfg

    class _FakeArtifactManager:
        def __init__(self, telemetry_provider=None, tenant_id=None):
            captured["tenant_id"] = tenant_id
            captured["provider"] = telemetry_provider

    monkeypatch.setattr(
        "cogniverse_telemetry_phoenix.provider.PhoenixProvider", _FakeProvider
    )
    monkeypatch.setattr(
        "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
        _FakeArtifactManager,
    )
    # Env is set but must be IGNORED — the helper reads the wired value.
    monkeypatch.setenv("PHOENIX_HTTP_ENDPOINT", "http://env-should-not-win:6006")
    monkeypatch.setenv("PHOENIX_GRPC_ENDPOINT", "env-should-not-win:4317")

    admin.set_phoenix_endpoints("http://wired-phoenix:6006", "wired-phoenix:4317")
    try:
        admin._build_artifact_manager("acme:acme")
    finally:
        # Restore module defaults so other tests are unaffected.
        admin.set_phoenix_endpoints("http://localhost:6006", "localhost:4317")

    assert captured["cfg"]["http_endpoint"] == "http://wired-phoenix:6006"
    assert captured["cfg"]["grpc_endpoint"] == "wired-phoenix:4317"
    assert captured["cfg"]["tenant_id"] == "acme:acme"
    assert captured["tenant_id"] == "acme:acme"
