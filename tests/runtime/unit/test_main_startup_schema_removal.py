"""The runtime's startup metadata deploy never overrides schema removal.

A prepare-and-activate replaces the WHOLE Vespa application. With the
``contentTypeRemoval`` override, a startup package that missed a peer
tenant's schema — one activated between this process's enumeration and its
post, or registered in another process — deleted that schema and every
document in it. Without the override Vespa refuses the same package with
``INVALID_APPLICATION_PACKAGE`` and the data survives, which is why the
lifespan call site must keep passing ``allow_schema_removal=False``.
Reaping the schemas of deleted tenants belongs to
``POST /admin/reconcile-orphans``.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime import main as runtime_main
from cogniverse_telemetry_phoenix.provider import PhoenixProvider
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _AbortStartup(RuntimeError):
    pass


class _RecordingSchemaManager:
    def __init__(self, recorded: dict) -> None:
        self._recorded = recorded

    def upload_metadata_schemas(self, *args, **kwargs) -> None:
        self._recorded["args"] = args
        self._recorded["kwargs"] = kwargs
        raise _AbortStartup("stop after the startup metadata deploy")


class _FakeBackend:
    def __init__(self, recorded: dict) -> None:
        self.schema_manager = _RecordingSchemaManager(recorded)


class _FakeBackendRegistry:
    def __init__(self, recorded: dict) -> None:
        self._recorded = recorded
        self._backend_instances: dict = {}

    def list_backends(self):
        return []

    def get_ingestion_backend(self, *args, **kwargs):
        return _FakeBackend(self._recorded)

    def get_search_backend(self, *args, **kwargs):
        return _FakeBackend(self._recorded)

    def clear_instances(self) -> None:
        self._backend_instances.clear()


class _FakeConfigLoader:
    def load_backends(self) -> None:
        return None

    def load_agents(self, agent_registry=None) -> None:
        return None


@pytest.mark.asyncio
async def test_startup_metadata_deploy_disables_schema_removal(
    monkeypatch: pytest.MonkeyPatch,
):
    recorded: dict = {}
    config_manager = ConfigManager(store=InMemoryConfigStore())

    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        lambda: config_manager,
    )
    monkeypatch.setattr(
        runtime_main.BackendRegistry,
        "get_instance",
        lambda: _FakeBackendRegistry(recorded),
    )
    monkeypatch.setattr(runtime_main, "get_config_loader", lambda: _FakeConfigLoader())
    monkeypatch.setattr(PhoenixProvider, "initialize", lambda self, config: None)

    with pytest.raises(_AbortStartup, match="stop after the startup metadata deploy"):
        async with runtime_main.lifespan(FastAPI()):
            pass

    assert recorded["args"] == ()
    assert recorded["kwargs"]["allow_schema_removal"] is False
    assert set(recorded["kwargs"]) == {"app_name", "allow_schema_removal"}
