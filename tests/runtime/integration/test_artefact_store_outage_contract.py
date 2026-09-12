"""A dataset-store outage is a named state, never "nothing promoted yet".

Every read here runs against a real Phoenix (a container this module owns, so
it may be paused) or a real socket that refuses, through the real
``phoenix.client`` — the outage shapes production sees.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

from cogniverse_agents.optimizer.artifact_manager import (
    ARTIFACT_LOAD_LOADED,
    ARTIFACT_LOAD_NO_ARTIFACT,
    ARTIFACT_LOAD_STORE_UNAVAILABLE,
    ArtifactManager,
)
from cogniverse_agents.optimizer.signature_variants import DEFAULT_VARIANT_ID
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.telemetry.providers.base import (
    DatasetNotFoundError,
    DatasetStoreUnavailableError,
)
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import admin as admin_router
from cogniverse_telemetry_phoenix import provider as phoenix_provider
from cogniverse_telemetry_phoenix.provider import PhoenixProvider

pytestmark = pytest.mark.integration

# Nothing listens here: the repo-wide dead-port convention.
DEAD_ENDPOINT = "http://127.0.0.1:29071"

# The shipped per-op budget (``_DATASET_OP_TIMEOUT_S``) sizes a read against a
# loaded store. The paused-container case only needs the client to give up, so
# these tests shorten it and assert what the give-up produces.
PAUSED_READ_TIMEOUT_S = 5.0


@pytest.fixture(scope="module")
def owned_phoenix():
    """A Phoenix container this module owns, so it may be paused safely."""
    from tests.utils.vllm_sidecar import OWNER_LABEL

    port_offset = (os.getpid() % 1000) * 10
    http_port = 26006 + port_offset
    grpc_port = 24317 + port_offset
    http_endpoint = f"http://localhost:{http_port}"
    name = f"phoenix_outage_pid{os.getpid()}_{uuid.uuid4().hex[:8]}"

    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            f"{OWNER_LABEL}={os.getpid()}",
            "-p",
            f"{http_port}:6006",
            "-p",
            f"{grpc_port}:4317",
            "-e",
            "PHOENIX_WORKING_DIR=/phoenix",
            "arizephoenix/phoenix:14.2.1",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    try:
        deadline = time.monotonic() + 120
        ready = False
        while time.monotonic() < deadline:
            try:
                if requests.get(http_endpoint, timeout=2).status_code == 200:
                    ready = True
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        if not ready:
            logs = subprocess.run(
                ["docker", "logs", name], capture_output=True, text=True, timeout=10
            )
            raise RuntimeError(f"Phoenix not ready:\n{logs.stdout}\n{logs.stderr}")
        yield {
            "container_name": name,
            "http_endpoint": http_endpoint,
            "grpc_endpoint": f"http://localhost:{grpc_port}",
        }
    finally:
        subprocess.run(["docker", "unpause", name], capture_output=True, timeout=30)
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)


def _manager(http_endpoint: str, grpc_endpoint: str, tenant_id: str) -> ArtifactManager:
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": http_endpoint,
            "grpc_endpoint": grpc_endpoint,
        }
    )
    return ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)


@pytest.fixture
def tenant_id() -> str:
    return f"outage{uuid.uuid4().hex[:8]}"


@pytest.fixture
def dead_manager(tenant_id) -> ArtifactManager:
    return _manager(DEAD_ENDPOINT, DEAD_ENDPOINT, tenant_id)


@pytest.fixture
def live_manager(owned_phoenix, tenant_id) -> ArtifactManager:
    return _manager(
        owned_phoenix["http_endpoint"], owned_phoenix["grpc_endpoint"], tenant_id
    )


class TestStorePrimitiveNamesTheOutage:
    """``load_blob`` distinguishes "no dataset" from "no answer"."""

    @pytest.mark.asyncio
    async def test_refused_port_raises_typed_error_naming_endpoint(
        self, dead_manager, tenant_id
    ):
        with pytest.raises(DatasetStoreUnavailableError) as excinfo:
            await dead_manager.load_blob("model", "entity_extraction")

        err = excinfo.value
        assert err.endpoint == DEAD_ENDPOINT
        assert err.dataset == f"dspy-model-{tenant_id}:{tenant_id}-entity_extraction"
        assert DEAD_ENDPOINT in str(err)

    @pytest.mark.asyncio
    async def test_paused_store_raises_typed_error_not_absent_blob(
        self, live_manager, owned_phoenix, tenant_id, monkeypatch
    ):
        monkeypatch.setattr(
            phoenix_provider, "_DATASET_OP_TIMEOUT_S", PAUSED_READ_TIMEOUT_S
        )
        subprocess.run(
            ["docker", "pause", owned_phoenix["container_name"]],
            check=True,
            capture_output=True,
            timeout=30,
        )
        try:
            with pytest.raises(DatasetStoreUnavailableError) as excinfo:
                await live_manager.load_blob("model", "entity_extraction")
        finally:
            subprocess.run(
                ["docker", "unpause", owned_phoenix["container_name"]],
                check=True,
                capture_output=True,
                timeout=30,
            )

        assert excinfo.value.endpoint == owned_phoenix["http_endpoint"]
        assert (
            excinfo.value.dataset
            == f"dspy-model-{tenant_id}:{tenant_id}-entity_extraction"
        )

    @pytest.mark.asyncio
    async def test_live_store_without_the_blob_returns_none(self, live_manager):
        assert await live_manager.load_blob("model", "entity_extraction") is None

    @pytest.mark.asyncio
    async def test_live_store_missing_dataset_raises_not_found(self, live_manager):
        with pytest.raises(DatasetNotFoundError):
            await live_manager._provider.datasets.get_dataset(
                name=f"dspy-prompts-absent-{uuid.uuid4().hex[:8]}"
            )


def _dispatcher(factory) -> AgentDispatcher:
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    config_manager = ConfigManager(store=store)
    return AgentDispatcher(
        agent_registry=AgentRegistry(
            tenant_id="test:unit", config_manager=config_manager
        ),
        config_manager=config_manager,
        schema_loader=FilesystemSchemaLoader(base_path=Path("configs/schemas")),
        artifact_manager_factory=factory,
    )


def _telemetry_manager(http_endpoint: str, grpc_endpoint: str):
    """A manager whose providers reach ``http_endpoint``.

    The manager is a process singleton and the provider registry caches per
    tenant, so both are reset before the build; ``_reset_telemetry_singletons``
    restores them for the rest of the session.
    """
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
    return TelemetryManager(
        config=TelemetryConfig(
            otlp_endpoint=grpc_endpoint,
            provider_config={
                "http_endpoint": http_endpoint,
                "grpc_endpoint": grpc_endpoint,
            },
            batch_config=BatchExportConfig(use_sync_export=True),
        )
    )


@pytest.fixture
def _reset_telemetry_singletons():
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry

    yield
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()


@pytest.fixture(autouse=True)
def _clean_admin_overrides():
    """Own the admin module state these tests read and write."""
    endpoints = dict(admin_router._phoenix_endpoints)
    admin_router._reset_admin_overrides_for_tests()
    yield
    admin_router._reset_admin_overrides_for_tests()
    admin_router._phoenix_endpoints.clear()
    admin_router._phoenix_endpoints.update(endpoints)


class TestDispatchOverlayNamesTheOutage:
    """The per-request overlay says which state it is serving default on."""

    @pytest.mark.asyncio
    async def test_outage_overlay_is_store_unavailable_with_no_variant(
        self, dead_manager, tenant_id
    ):
        admin_router.set_phoenix_endpoints(DEAD_ENDPOINT, DEAD_ENDPOINT)
        dispatcher = _dispatcher(lambda t: dead_manager)

        overlay = await dispatcher.resolve_artefact_for_request(
            "entity_extraction_agent", tenant_id, "seed-1"
        )

        assert overlay["artifact_load_status"] == ARTIFACT_LOAD_STORE_UNAVAILABLE
        assert overlay["served_from"] == "default"
        assert overlay["prompts"] is None
        assert overlay["version"] is None
        assert overlay["variant_id"] is None
        assert overlay["error"].startswith("DatasetStoreUnavailableError:")
        assert DEAD_ENDPOINT in overlay["error"]

    @pytest.mark.asyncio
    async def test_live_store_without_artefacts_is_loaded_on_the_default_variant(
        self, live_manager, owned_phoenix, tenant_id
    ):
        admin_router.set_phoenix_endpoints(
            owned_phoenix["http_endpoint"], owned_phoenix["grpc_endpoint"]
        )
        dispatcher = _dispatcher(lambda t: live_manager)

        overlay = await dispatcher.resolve_artefact_for_request(
            "entity_extraction_agent", tenant_id, "seed-1"
        )

        assert overlay == {
            "prompts": None,
            "served_from": "default",
            "version": None,
            "variant_id": DEFAULT_VARIANT_ID,
            "artifact_load_status": ARTIFACT_LOAD_LOADED,
        }

    @pytest.mark.asyncio
    async def test_promoted_prompts_reach_the_overlay_with_loaded_status(
        self, live_manager, owned_phoenix, tenant_id
    ):
        admin_router.set_phoenix_endpoints(
            owned_phoenix["http_endpoint"], owned_phoenix["grpc_endpoint"]
        )
        await live_manager.save_prompts(
            "entity_extraction_agent", {"system": "PROMOTED"}
        )
        dispatcher = _dispatcher(lambda t: live_manager)

        overlay = await dispatcher.resolve_artefact_for_request(
            "entity_extraction_agent", tenant_id, "seed-1"
        )

        assert overlay == {
            "prompts": {"system": "PROMOTED"},
            "served_from": "default",
            "version": None,
            "variant_id": DEFAULT_VARIANT_ID,
            "artifact_load_status": ARTIFACT_LOAD_LOADED,
        }


class TestAgentLoaderNamesTheOutage:
    """``load_optimized_module`` separates an outage from an absent artifact."""

    def _agent(self, http_endpoint: str, grpc_endpoint: str, tenant_id: str):
        from cogniverse_agents.query_enhancement_agent import (
            QueryEnhancementAgent,
            QueryEnhancementDeps,
        )

        agent = QueryEnhancementAgent(deps=QueryEnhancementDeps(), port=18011)
        agent.telemetry_manager = _telemetry_manager(http_endpoint, grpc_endpoint)
        agent._artifact_tenant_id = tenant_id
        # Without this the manager can hand back a provider built from the
        # process default, and every assertion below would hold against the
        # wrong endpoint.
        assert (
            agent.telemetry_manager.get_provider(
                tenant_id=tenant_id
            ).datasets.http_endpoint
            == http_endpoint
        )
        return agent

    def test_refused_store_records_store_unavailable(
        self, tenant_id, _reset_telemetry_singletons
    ):
        agent = self._agent(DEAD_ENDPOINT, DEAD_ENDPOINT, tenant_id)

        agent._load_artifact()

        assert agent.artifact_load_status == ARTIFACT_LOAD_STORE_UNAVAILABLE

    def test_live_store_without_artifact_records_no_artifact(
        self, owned_phoenix, tenant_id, _reset_telemetry_singletons
    ):
        agent = self._agent(
            owned_phoenix["http_endpoint"], owned_phoenix["grpc_endpoint"], tenant_id
        )

        agent._load_artifact()

        assert agent.artifact_load_status == ARTIFACT_LOAD_NO_ARTIFACT


class TestArtefactManagerCacheUnderConcurrency:
    """The per-tenant manager cache builds once and never crosses tenants."""

    def test_concurrent_cold_resolution_builds_one_manager_per_tenant(
        self, telemetry_manager_with_phoenix, monkeypatch
    ):
        from cogniverse_runtime.routers import agents as agents_router

        builds: list[str] = []
        builds_lock = threading.Lock()
        original_init = ArtifactManager.__init__

        def counting_init(self, telemetry_provider, tenant_id):
            with builds_lock:
                builds.append(tenant_id)
            original_init(self, telemetry_provider, tenant_id)

        monkeypatch.setattr(ArtifactManager, "__init__", counting_init)
        monkeypatch.setattr(
            "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
            lambda *a, **k: telemetry_manager_with_phoenix,
        )

        factory = agents_router._build_artifact_manager_factory()
        tenants = [f"cc{uuid.uuid4().hex[:6]}", f"cc{uuid.uuid4().hex[:6]}"]
        concurrency = 16
        barrier = threading.Barrier(concurrency)

        def resolve(i: int):
            tenant = tenants[i % len(tenants)]
            barrier.wait(timeout=30)
            return tenant, factory(tenant)

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            resolved = list(pool.map(resolve, range(concurrency)))

        assert sorted(builds) == sorted(tenants)
        per_tenant = {t: set() for t in tenants}
        for tenant, manager in resolved:
            per_tenant[tenant].add(id(manager))
            assert manager._tenant_id == f"{tenant}:{tenant}"
        assert [len(ids) for ids in per_tenant.values()] == [1, 1]

    @pytest.mark.asyncio
    async def test_concurrent_outage_resolutions_all_report_store_unavailable(
        self, tenant_id
    ):
        admin_router.set_phoenix_endpoints(DEAD_ENDPOINT, DEAD_ENDPOINT)
        managers = {
            t: _manager(DEAD_ENDPOINT, DEAD_ENDPOINT, t)
            for t in (tenant_id, f"{tenant_id}b")
        }
        dispatcher = _dispatcher(lambda t: managers[t])

        overlays = await asyncio.gather(
            *[
                dispatcher.resolve_artefact_for_request(
                    "entity_extraction_agent", t, f"seed-{i}"
                )
                for i in range(8)
                for t in managers
            ]
        )

        assert [o["artifact_load_status"] for o in overlays] == [
            ARTIFACT_LOAD_STORE_UNAVAILABLE
        ] * 16
        assert {o["variant_id"] for o in overlays} == {None}
