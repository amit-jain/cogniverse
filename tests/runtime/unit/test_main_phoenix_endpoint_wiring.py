"""main.py wires the admin Phoenix endpoints from telemetry config.

The admin canary/signature-variant path builds its ArtifactManager through the
startup-wired module state, so the runtime must feed that state from
TELEMETRY_HTTP_ENDPOINT / TELEMETRY_OTLP_ENDPOINT rather than an unrelated
PHOENIX_* env fallback.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime import main as runtime_main
from cogniverse_runtime.routers import admin
from cogniverse_telemetry_phoenix.provider import PhoenixProvider
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _AbortStartup(RuntimeError):
    pass


class _FakeSchemaManager:
    def upload_metadata_schemas(self, *args, **kwargs) -> None:
        return None


class _FakeBackend:
    def __init__(self) -> None:
        self.schema_manager = _FakeSchemaManager()


class _FakeBackendRegistry:
    def __init__(self) -> None:
        self._backend_instances = {}

    def list_backends(self):
        return []

    def get_ingestion_backend(self, *args, **kwargs):
        return _FakeBackend()


class _FakeConfigLoader:
    def load_backends(self) -> None:
        return None

    def load_agents(self, agent_registry=None) -> None:
        return None


@pytest.mark.asyncio
async def test_lifespan_wires_admin_phoenix_endpoints_from_telemetry_env(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}
    saved = dict(admin._phoenix_endpoints)
    original_set = admin.set_phoenix_endpoints
    original_get_system_config = ConfigManager.get_system_config
    config_manager = ConfigManager(store=InMemoryConfigStore())

    def _spy_set_phoenix_endpoints(http_endpoint: str, grpc_endpoint: str) -> None:
        captured["http_endpoint"] = http_endpoint
        captured["grpc_endpoint"] = grpc_endpoint
        raise _AbortStartup("stop after endpoint wiring")

    def _fake_registry():
        return _FakeBackendRegistry()

    def _fake_loader():
        return _FakeConfigLoader()

    async def _fake_wait_for_backend_startup(*args, **kwargs):
        return runtime_main.BackendStartupState.FEED_READY

    def _spy_get_system_config(self):
        config = original_get_system_config(self)
        captured["system_config"] = config
        return config

    monkeypatch.setenv("TELEMETRY_HTTP_ENDPOINT", "http://wired-phoenix:6006")
    monkeypatch.setenv("TELEMETRY_OTLP_ENDPOINT", "wired-phoenix:4317")
    monkeypatch.setenv("PHOENIX_HTTP_ENDPOINT", "http://wrong-phoenix:6006")
    monkeypatch.setenv("PHOENIX_GRPC_ENDPOINT", "wrong-phoenix:4317")
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        lambda: config_manager,
    )
    monkeypatch.setattr(runtime_main.BackendRegistry, "get_instance", _fake_registry)
    monkeypatch.setattr(runtime_main, "get_config_loader", _fake_loader)
    monkeypatch.setattr(ConfigManager, "get_system_config", _spy_get_system_config)
    monkeypatch.setattr(admin, "set_phoenix_endpoints", _spy_set_phoenix_endpoints)
    monkeypatch.setattr(PhoenixProvider, "initialize", lambda self, config: None)
    monkeypatch.setattr(
        runtime_main, "_wait_for_backend_startup", _fake_wait_for_backend_startup
    )

    try:
        with pytest.raises(_AbortStartup, match="stop after endpoint wiring"):
            async with runtime_main.lifespan(FastAPI()):
                pass

        assert captured["http_endpoint"] == "http://wired-phoenix:6006"
        assert captured["grpc_endpoint"] == "wired-phoenix:4317"
        assert captured["system_config"].telemetry_url == "http://wired-phoenix:6006"
        assert (
            captured["system_config"].telemetry_collector_endpoint
            == "wired-phoenix:4317"
        )
    finally:
        original_set(saved["http_endpoint"], saved["grpc_endpoint"])
