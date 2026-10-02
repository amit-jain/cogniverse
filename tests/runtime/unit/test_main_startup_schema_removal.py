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

from cogniverse_core.registries.schema_registry import (
    DriftedSchemaRedeploy,
    SchemaRefusal,
)
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

    def redeploy_drifted_schemas(self, should_stop=None):
        import threading

        self.calls.append(threading.get_ident())
        self.should_stop = should_stop
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _refusal(tenant_id: str, schema_name: str, error: str) -> SchemaRefusal:
    return SchemaRefusal(
        tenant_id=tenant_id,
        base_schema_name=schema_name.rsplit("_", 2)[0],
        schema_name=schema_name,
        error=error,
        refused_at="2026-10-02T00:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_the_schema_migration_waits_out_a_held_lease_off_the_loop(
    monkeypatch, caplog
):
    import threading

    registry = _MigratingRegistry(
        [
            TimeoutError(_LEASE_HELD),
            DriftedSchemaRedeploy(
                redeployed=[
                    "provenance_acme_acme",
                    "video_colpali_smol500_mv_frame_acme_acme",
                ],
                refused=[
                    _refusal(
                        "globex:globex",
                        "document_text_globex_globex",
                        "Vespa refused the application package: field-type-change",
                    )
                ],
            ),
        ]
    )
    monkeypatch.setattr(runtime_main, "SCHEMA_MIGRATION_RETRY_SECONDS", 0.05)

    with caplog.at_level("INFO", logger=runtime_main.logger.name):
        await runtime_main._migrate_drifted_schemas(lambda: registry)

    assert len(registry.calls) == 2
    assert all(thread != threading.get_ident() for thread in registry.calls)
    assert [
        (record.levelname, record.getMessage())
        for record in caplog.records
        if record.name == runtime_main.logger.name
    ] == [
        (
            "WARNING",
            "Migration of drifted schemas did not get the deployment lease "
            f"({_LEASE_HELD}); retrying in 0s",
        ),
        (
            "INFO",
            "Migration of drifted schemas redeployed ['provenance_acme_acme', "
            "'video_colpali_smol500_mv_frame_acme_acme']",
        ),
        (
            "ERROR",
            "Migration of drifted schemas could not redeploy "
            "document_text_globex_globex for tenant globex:globex: Vespa refused "
            "the application package: field-type-change",
        ),
    ]


def _migration_failures():
    from cogniverse_core.registries.exceptions import (
        BackendDeploymentError,
        SchemaRegistryInitializationError,
        SchemaRevisionConflictError,
    )

    return [
        RuntimeError("Vespa refused the application package"),
        BackendDeploymentError(
            "Cannot enumerate Vespa-deployed schemas before deploy: connection refused"
        ),
        SchemaRevisionConflictError(
            "provenance_acme_acme", "tombstone", activated=True
        ),
        SchemaRegistryInitializationError("failed to read schema storage"),
    ]


@pytest.mark.parametrize(
    "failure",
    _migration_failures(),
    ids=[
        "runtime",
        "backend-deployment",
        "tombstone-after-activation",
        "registry-storage",
    ],
)
@pytest.mark.asyncio
async def test_a_failed_schema_migration_is_run_again_until_one_completes(
    monkeypatch, caplog, failure
):
    """A run that did not complete is logged with its error and run again;
    the runtime's task never ends with the failure."""
    registry = _MigratingRegistry(
        [failure, failure, DriftedSchemaRedeploy(redeployed=[], refused=[])]
    )
    monkeypatch.setattr(runtime_main, "SCHEMA_MIGRATION_RETRY_SECONDS", 0.05)

    with caplog.at_level("INFO", logger=runtime_main.logger.name):
        await runtime_main._migrate_drifted_schemas(lambda: registry)

    assert len(registry.calls) == 3
    retried = (
        "WARNING",
        "Migration of drifted schemas did not complete "
        f"({type(failure).__name__}: {failure}); retrying in 0s",
        failure,
    )
    assert [
        (
            record.levelname,
            record.getMessage(),
            record.exc_info[1] if record.exc_info else None,
        )
        for record in caplog.records
        if record.name == runtime_main.logger.name
    ] == [
        retried,
        retried,
        ("INFO", "Migration of drifted schemas redeployed []", None),
    ]


@pytest.mark.asyncio
async def test_a_registry_that_cannot_be_resolved_is_resolved_again(
    monkeypatch, caplog
):
    """Resolving the system backend runs inside the background migration,
    off the loop, so its failure never reaches startup: the next run resolves
    it again."""
    import threading

    failure = RuntimeError("backend config store unreachable")
    registry = _MigratingRegistry(
        [DriftedSchemaRedeploy(redeployed=["provenance_acme_acme"], refused=[])]
    )
    threads = []

    def resolvable_second_time():
        threads.append(threading.get_ident())
        if len(threads) == 1:
            raise failure
        return registry

    monkeypatch.setattr(runtime_main, "SCHEMA_MIGRATION_RETRY_SECONDS", 0.05)

    with caplog.at_level("INFO", logger=runtime_main.logger.name):
        await runtime_main._migrate_drifted_schemas(resolvable_second_time)

    assert threading.get_ident() not in threads
    assert len(threads) == 2
    assert len(registry.calls) == 1
    assert [
        (
            record.getMessage(),
            record.exc_info[1] if record.exc_info else None,
        )
        for record in caplog.records
        if record.name == runtime_main.logger.name
    ] == [
        (
            "Migration of drifted schemas did not complete (RuntimeError: backend "
            "config store unreachable); retrying in 0s",
            failure,
        ),
        ("Migration of drifted schemas redeployed ['provenance_acme_acme']", None),
    ]


@pytest.mark.asyncio
async def test_a_stop_set_while_a_run_is_retried_starts_no_further_run(
    monkeypatch, caplog
):
    """Shutdown sets the stop between attempts: the attempt that failed is
    logged and no further run starts."""
    import threading

    stop = threading.Event()
    failure = RuntimeError("config server unreachable")

    class _StoppedMeanwhile(_MigratingRegistry):
        def redeploy_drifted_schemas(self, should_stop=None):
            stop.set()
            return super().redeploy_drifted_schemas(should_stop=should_stop)

    registry = _StoppedMeanwhile([failure])
    monkeypatch.setattr(runtime_main, "SCHEMA_MIGRATION_RETRY_SECONDS", 0.05)

    with caplog.at_level("INFO", logger=runtime_main.logger.name):
        await runtime_main._migrate_drifted_schemas(lambda: registry, stop)

    assert len(registry.calls) == 1
    assert [
        (record.levelname, record.getMessage())
        for record in caplog.records
        if record.name == runtime_main.logger.name
    ] == [
        (
            "WARNING",
            "Migration of drifted schemas did not complete (RuntimeError: config "
            "server unreachable); retrying in 0s",
        )
    ]


@pytest.mark.asyncio
async def test_the_background_retry_skips_a_deploy_a_peer_already_made(
    monkeypatch, caplog
):
    import asyncio

    manager = _ContendedSchemaManager([TimeoutError(_LEASE_HELD)])
    checks = iter([False, True])
    monkeypatch.setattr(
        "cogniverse_runtime.backend_startup.metadata_schemas_current",
        lambda _: next(checks),
    )
    monkeypatch.setattr(runtime_main, "METADATA_DEPLOY_RETRY_SECONDS", 0.05)

    with caplog.at_level("INFO", logger=runtime_main.logger.name):
        retry = await runtime_main._deploy_metadata_schemas_at_startup(
            manager, "cogniverse"
        )
        await asyncio.wait_for(retry, timeout=5)

    assert len(manager.calls) == 1
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == runtime_main.logger.name
    ] == [
        f"Metadata schema deploy did not get the deployment lease ({_LEASE_HELD}); "
        "retrying every 0s in the background",
        "Metadata schemas are live and current; background deploy skipped",
    ]


@pytest.mark.parametrize(
    "variable,value,message",
    [
        ("A2A_MAX_TASKS", "0", "A2A_MAX_TASKS (max_tasks) must be >= 1, got 0"),
        (
            "A2A_TASK_LEASE_SECONDS",
            "0",
            "A2A_TASK_LEASE_SECONDS (lease_seconds) must be > 0, got 0.0",
        ),
        (
            "A2A_CANCEL_TIMEOUT_SECONDS",
            "-1",
            "A2A_CANCEL_TIMEOUT_SECONDS (cancel_timeout_seconds) must be > 0, got -1.0",
        ),
        (
            "A2A_DRAIN_TIMEOUT_SECONDS",
            "nan",
            "A2A_DRAIN_TIMEOUT_SECONDS (drain_timeout_seconds) must be > 0, got nan",
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_bad_a2a_setting_stops_startup_before_any_side_effect(
    monkeypatch, workflow_state_redis_url, variable, value, message
):
    recorded: dict = {}
    side_effects: list = []
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
    monkeypatch.setenv(variable, value)
    monkeypatch.setattr(
        "cogniverse_runtime.backend_startup.metadata_schemas_current",
        lambda _: side_effects.append("metadata-check") or False,
    )
    monkeypatch.setattr(
        runtime_main,
        "_probe_phoenix_reachability",
        lambda: side_effects.append("phoenix-probe"),
    )

    with pytest.raises(ValueError) as refused:
        async with runtime_main.lifespan(FastAPI()):
            pass

    assert str(refused.value) == message
    assert recorded == {}
    assert side_effects == []


@pytest.mark.asyncio
async def test_a_stopped_migration_reports_the_schemas_it_left(caplog):
    import threading

    stop = threading.Event()
    registry = _MigratingRegistry(
        [
            DriftedSchemaRedeploy(
                redeployed=["provenance_acme_acme"],
                refused=[],
                skipped=["video_colpali_smol500_mv_frame_globex_globex"],
            )
        ]
    )

    with caplog.at_level("INFO", logger=runtime_main.logger.name):
        await runtime_main._migrate_drifted_schemas(lambda: registry, stop)

    assert registry.should_stop == stop.is_set
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == runtime_main.logger.name
    ] == [
        "Migration of drifted schemas redeployed ['provenance_acme_acme']",
        "Migration of drifted schemas stopped before redeploying "
        "['video_colpali_smol500_mv_frame_globex_globex']; the next runtime "
        "start redeploys them",
    ]
