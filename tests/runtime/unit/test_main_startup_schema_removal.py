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
    monkeypatch: pytest.MonkeyPatch, workflow_state_redis_url
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
    monkeypatch.setenv("REDIS_URL", workflow_state_redis_url)
    monkeypatch.setattr(
        "cogniverse_runtime.backend_startup.metadata_schemas_current", lambda _: False
    )

    with pytest.raises(_AbortStartup, match="stop after the startup metadata deploy"):
        async with runtime_main.lifespan(FastAPI()):
            pass

    assert recorded["args"] == ()
    assert recorded["kwargs"]["allow_schema_removal"] is False
    assert set(recorded["kwargs"]) == {"app_name", "allow_schema_removal"}


@pytest.mark.asyncio
async def test_an_unusable_redis_stops_startup_before_any_side_effect(
    monkeypatch: pytest.MonkeyPatch,
):
    """A pod crashlooping on Redis must not redeploy schemas, probe Phoenix or
    store SystemConfig on every restart: Redis is refused first."""
    import socket

    from cogniverse_runtime.a2a_task_store import A2ATaskStoreError

    recorded: dict = {}
    side_effects: list = []
    config_manager = ConfigManager(store=InMemoryConfigStore())
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]

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
    monkeypatch.setenv("REDIS_URL", f"redis://:startup-pw@127.0.0.1:{closed_port}/0")
    monkeypatch.setattr(
        "cogniverse_runtime.backend_startup.metadata_schemas_current",
        lambda _: side_effects.append("metadata-check") or False,
    )
    monkeypatch.setattr(
        runtime_main,
        "_probe_phoenix_reachability",
        lambda: side_effects.append("phoenix-probe"),
    )
    monkeypatch.setattr(
        config_manager,
        "set_system_config",
        lambda config: side_effects.append("system-config-write"),
    )

    with pytest.raises(A2ATaskStoreError) as refused:
        async with runtime_main.lifespan(FastAPI()):
            pass

    assert str(refused.value) == (
        "shared A2A task store unavailable: connect to "
        f"redis://127.0.0.1:{closed_port}/0"
    )
    assert recorded == {}
    assert side_effects == []


_LEASE_HELD = (
    "Vespa deployment lease still held by 'peer-host:1:9' after 120s; refusing "
    "to replace the application package concurrently with another deployer"
)


class _ContendedSchemaManager:
    """Answers each startup deploy with the next outcome, recording the call."""

    def __init__(self, outcomes: list) -> None:
        import threading

        self._outcomes = outcomes
        self.calls: list = []
        self.deployed = threading.Event()

    def upload_metadata_schemas(self, **kwargs) -> None:
        import threading

        self.calls.append((kwargs, threading.get_ident()))
        outcome = self._outcomes.pop(0)
        if outcome is not None:
            raise outcome
        self.deployed.set()


@pytest.mark.asyncio
async def test_current_live_metadata_schemas_are_not_redeployed(monkeypatch):
    manager = _ContendedSchemaManager([])
    monkeypatch.setattr(
        "cogniverse_runtime.backend_startup.metadata_schemas_current",
        lambda schema_manager: schema_manager is manager,
    )

    retry = await runtime_main._deploy_metadata_schemas_at_startup(
        manager, "cogniverse"
    )

    assert (retry, manager.calls) == (None, [])


@pytest.mark.asyncio
async def test_a_held_deployment_lease_is_retried_in_the_background_not_fatal(
    monkeypatch, caplog
):
    import asyncio
    import threading

    manager = _ContendedSchemaManager(
        [TimeoutError(_LEASE_HELD), TimeoutError(_LEASE_HELD), None]
    )
    monkeypatch.setattr(
        "cogniverse_runtime.backend_startup.metadata_schemas_current", lambda _: False
    )
    monkeypatch.setattr(runtime_main, "METADATA_DEPLOY_RETRY_SECONDS", 0.05)

    with caplog.at_level("INFO", logger=runtime_main.logger.name):
        retry = await runtime_main._deploy_metadata_schemas_at_startup(
            manager, "cogniverse"
        )
        calls_at_startup = len(manager.calls)
        await asyncio.wait_for(retry, timeout=5)

    loop_thread = threading.get_ident()
    assert calls_at_startup == 1
    assert [kwargs for kwargs, _ in manager.calls] == [
        {"app_name": "cogniverse", "allow_schema_removal": False}
    ] * 3
    assert all(thread != loop_thread for _, thread in manager.calls)
    assert manager.deployed.is_set() is True
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == runtime_main.logger.name
    ] == [
        f"Metadata schema deploy did not get the deployment lease ({_LEASE_HELD}); "
        "retrying every 0s in the background",
        f"Metadata schema deploy still waiting: {_LEASE_HELD}",
        "Metadata schemas deployed via system backend in the background",
    ]


@pytest.mark.asyncio
async def test_a_refused_startup_metadata_deploy_still_fails_startup(monkeypatch):
    refusal = RuntimeError("Vespa refused the application package")
    manager = _ContendedSchemaManager([refusal])
    monkeypatch.setattr(
        "cogniverse_runtime.backend_startup.metadata_schemas_current", lambda _: False
    )

    with pytest.raises(RuntimeError) as raised:
        await runtime_main._deploy_metadata_schemas_at_startup(manager, "cogniverse")

    assert raised.value is refusal
    assert len(manager.calls) == 1


class _MigratingRegistry:
    """Answers each migration run with the next outcome, recording the call."""

    def __init__(self, outcomes: list) -> None:
        self._outcomes = outcomes
        self.calls: list = []

    def redeploy_drifted_schemas(self, base_schema_name: str):
        import threading

        self.calls.append((base_schema_name, threading.get_ident()))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@pytest.mark.asyncio
async def test_the_schema_migration_waits_out_a_held_lease_off_the_loop(
    monkeypatch, caplog
):
    import threading

    registry = _MigratingRegistry([TimeoutError(_LEASE_HELD), ["provenance_acme_acme"]])
    monkeypatch.setattr(runtime_main, "METADATA_DEPLOY_RETRY_SECONDS", 0.05)

    with caplog.at_level("INFO", logger=runtime_main.logger.name):
        await runtime_main._migrate_drifted_schemas(registry, "provenance")

    assert [base for base, _ in registry.calls] == ["provenance", "provenance"]
    assert all(thread != threading.get_ident() for _, thread in registry.calls)
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == runtime_main.logger.name
    ] == [
        "Migration of drifted provenance schemas did not get the deployment lease "
        f"({_LEASE_HELD}); retrying in 0s",
        "Migration of drifted provenance schemas redeployed ['provenance_acme_acme']",
    ]


def _migration_failures():
    from cogniverse_core.registries.exceptions import (
        BackendDeploymentError,
        SchemaRevisionConflictError,
    )

    return [
        RuntimeError("Vespa refused the application package"),
        BackendDeploymentError("Vespa refused the application package"),
        SchemaRevisionConflictError(
            "provenance_acme_acme", "tombstone", activated=True
        ),
    ]


@pytest.mark.parametrize(
    "failure",
    _migration_failures(),
    ids=["runtime", "backend-deployment", "tombstone-after-activation"],
)
@pytest.mark.asyncio
async def test_a_failed_schema_migration_is_logged_not_raised(caplog, failure):
    registry = _MigratingRegistry([failure])

    with caplog.at_level("ERROR", logger=runtime_main.logger.name):
        await runtime_main._migrate_drifted_schemas(registry, "provenance")

    assert len(registry.calls) == 1
    assert [
        (record.getMessage(), record.exc_info[1])
        for record in caplog.records
        if record.name == runtime_main.logger.name
    ] == [
        (
            "Migration of drifted provenance schemas failed; the next runtime "
            "start runs it again",
            failure,
        )
    ]
