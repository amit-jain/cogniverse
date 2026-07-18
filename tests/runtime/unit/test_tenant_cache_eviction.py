"""Per-tenant runtime caches are bounded and evicted on tenant delete.

Three long-lived per-tenant caches — the dispatcher's gateway agents, the
graph-manager factory, and the per-tenant ArtifactManager factory — held a
plain dict entry per tenant forever. Each is a registered TenantLRUCache
now: the LRU bound caps the working set, and delete_tenant_internal drops
the deleted tenant's entries from every registered cache.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cogniverse_foundation.caching import (
    TenantLRUCache,
    evict_tenant_from_registered_caches,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


@pytest.fixture
def gateway_dispatcher(monkeypatch):
    """A fully constructed AgentDispatcher with a stubbed GatewayAgent."""
    import cogniverse_agents.gateway_agent as gw
    import cogniverse_foundation.telemetry.manager as tm_mod
    from cogniverse_foundation.config.unified_config import RoutingConfigUnified
    from cogniverse_runtime.agent_dispatcher import AgentDispatcher

    monkeypatch.setattr(
        tm_mod,
        "get_telemetry_manager",
        lambda *a, **k: SimpleNamespace(name="tm"),
        raising=True,
    )

    def _init(self, deps=None, **k):
        self.deps = deps

    monkeypatch.setattr(gw.GatewayAgent, "__init__", _init)
    monkeypatch.setattr(
        gw.GatewayAgent, "_load_artifact", lambda self: None, raising=False
    )

    config_manager = SimpleNamespace(
        get_system_config=lambda: SimpleNamespace(inference_service_urls={}),
        get_routing_config=lambda tid: RoutingConfigUnified(tenant_id=tid),
    )
    return AgentDispatcher(
        agent_registry=SimpleNamespace(),
        config_manager=config_manager,
        schema_loader=SimpleNamespace(),
    )


class TestGatewayAgentCache:
    @pytest.mark.asyncio
    async def test_tenant_delete_evicts_only_that_tenants_gateway(
        self, gateway_dispatcher
    ):
        d = gateway_dispatcher
        agent_a = await d._get_or_build_gateway_agent("gwevict:a")
        agent_b = await d._get_or_build_gateway_agent("gwevict:b")

        evict_tenant_from_registered_caches("gwevict:a")

        assert "gwevict:a" not in d._gateway_agents
        assert "gwevict:b" in d._gateway_agents
        rebuilt = await d._get_or_build_gateway_agent("gwevict:a")
        assert rebuilt is not agent_a
        assert await d._get_or_build_gateway_agent("gwevict:b") is agent_b

    @pytest.mark.asyncio
    async def test_gateway_cache_is_lru_bounded_at_the_module_capacity(
        self, gateway_dispatcher
    ):
        from cogniverse_runtime.agent_dispatcher import GATEWAY_AGENT_CACHE_CAPACITY

        d = gateway_dispatcher
        assert GATEWAY_AGENT_CACHE_CAPACITY == 64
        assert d._gateway_agents.capacity == GATEWAY_AGENT_CACHE_CAPACITY

        for i in range(GATEWAY_AGENT_CACHE_CAPACITY + 1):
            await d._get_or_build_gateway_agent(f"gwbound:t{i}")

        assert len(d._gateway_agents) == GATEWAY_AGENT_CACHE_CAPACITY
        assert "gwbound:t0" not in d._gateway_agents
        assert "gwbound:t1" in d._gateway_agents
        assert f"gwbound:t{GATEWAY_AGENT_CACHE_CAPACITY}" in d._gateway_agents


class _StubGraphManager:
    def __init__(self, backend=None, tenant_id=None, schema_name=None, **kwargs):
        self.tenant_id = tenant_id
        self.schema_name = schema_name


def _graph_factory():
    import cogniverse_runtime.main as main

    backend = MagicMock()
    backend.get_tenant_schema_name.side_effect = lambda tid, base: (
        f"{base}_{tid.replace(':', '_')}"
    )
    config_manager = SimpleNamespace(
        get_system_config=lambda: SimpleNamespace(
            inference_service_urls={"colbert_pylate": "http://colbert:9000"}
        )
    )
    with patch("cogniverse_agents.graph.graph_manager.GraphManager", _StubGraphManager):
        factory = main._build_graph_manager_factory(backend, config_manager)
    return factory


class TestGraphManagerCache:
    def test_tenant_delete_evicts_only_that_tenants_graph_manager(self):
        factory = _graph_factory()
        mgr_a = factory("graphevict:a")
        mgr_b = factory("graphevict:b")
        assert factory("graphevict:a") is mgr_a

        evict_tenant_from_registered_caches("graphevict:a")

        assert factory("graphevict:a") is not mgr_a
        assert factory("graphevict:b") is mgr_b

    def test_graph_cache_is_lru_bounded_at_the_module_capacity(self):
        import cogniverse_runtime.main as main

        assert main.GRAPH_MANAGER_CACHE_CAPACITY == 64
        factory = _graph_factory()

        first = factory("graphbound:t0")
        for i in range(1, main.GRAPH_MANAGER_CACHE_CAPACITY + 1):
            factory(f"graphbound:t{i}")

        last = factory(f"graphbound:t{main.GRAPH_MANAGER_CACHE_CAPACITY}")
        assert factory("graphbound:t0") is not first  # oldest was evicted
        assert factory(f"graphbound:t{main.GRAPH_MANAGER_CACHE_CAPACITY}") is last


class _StubArtifactManager:
    def __init__(self, provider, tenant_id):
        self.provider = provider
        self.tenant_id = tenant_id


def _artifact_factory():
    from cogniverse_runtime.routers import agents as agents_router

    tm = SimpleNamespace(get_provider=lambda tenant_id: SimpleNamespace(t=tenant_id))
    with (
        patch(
            "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
            return_value=tm,
        ),
        patch(
            "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
            _StubArtifactManager,
        ),
    ):
        return agents_router._build_artifact_manager_factory()


class TestArtifactManagerCache:
    def test_tenant_delete_evicts_only_that_tenants_artifact_manager(self):
        factory = _artifact_factory()
        am_a = factory("amevict:a")
        am_b = factory("amevict:b")
        assert factory("amevict:a") is am_a

        evict_tenant_from_registered_caches("amevict:a")

        assert factory("amevict:a") is not am_a
        assert factory("amevict:b") is am_b

    def test_artifact_cache_is_lru_bounded_at_the_module_capacity(self):
        from cogniverse_runtime.routers import agents as agents_router

        assert agents_router.ARTIFACT_MANAGER_CACHE_CAPACITY == 64
        factory = _artifact_factory()

        first = factory("ambound:t0")
        for i in range(1, agents_router.ARTIFACT_MANAGER_CACHE_CAPACITY + 1):
            factory(f"ambound:t{i}")

        assert factory("ambound:t0") is not first  # oldest was evicted


@pytest.mark.asyncio
async def test_delete_tenant_internal_evicts_registered_tenant_caches(monkeypatch):
    """The tenant-delete path drops the deleted tenant from every registered
    per-tenant cache, not just the existence cache."""
    from cogniverse_foundation.caching import register_tenant_cache
    from cogniverse_runtime.admin import tenant_manager as tm

    backend = MagicMock()
    backend.schema_manager.delete_tenant_schemas.return_value = []
    monkeypatch.setattr(tm, "get_backend", lambda: backend)

    async def _tenant(_tid):
        return MagicMock()

    async def _org(_org_id):
        return None

    async def _remaining(_org_id):
        return [MagicMock()]

    monkeypatch.setattr(tm, "get_tenant_internal", _tenant)
    monkeypatch.setattr(tm, "get_organization_internal", _org)
    monkeypatch.setattr(tm, "list_tenants_for_org_internal", _remaining)

    cache: TenantLRUCache[str] = register_tenant_cache(TenantLRUCache(capacity=4))
    cache.set("delwire:a", "gateway-agent-a")
    cache.set("delwire:b", "gateway-agent-b")

    result = await tm.delete_tenant_internal("delwire:a")

    assert result["status"] == "deleted"
    assert "delwire:a" not in cache
    assert cache.get("delwire:b") == "gateway-agent-b"
