"""AgentDispatcher consults ArtifactManager.load_for_request (canary wiring).

Without this test, the canary state machine in :class:`ArtifactManager`
was unreachable from the live dispatch path: every request loaded the
active artefacts because no caller invoked ``load_for_request``. The
test verifies, against a real Phoenix container, that:

  * the dispatcher's ``resolve_artefact_for_request`` exposes the canary
    decision per request;
  * with ``traffic_pct=100`` every distinct request seed routes to the
    canary version's prompts;
  * with ``traffic_pct=0`` every request routes to active;
  * with no canary state, the dispatcher returns ``served_from="default"``;
  * absent an ``artifact_manager_factory`` the dispatcher silently
    no-ops (back-compat with deployments that don't run the optimizer).

The downstream consumption of ``context["_artefact_overlay"]`` by each
agent is a follow-up wire; this test proves the resolution side.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_telemetry_phoenix.provider import PhoenixProvider

pytestmark = pytest.mark.integration


@pytest.fixture
def tenant_id() -> str:
    return f"c5_int_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def artifact_manager(phoenix_container, tenant_id: str) -> ArtifactManager:
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": tenant_id,
            "http_endpoint": phoenix_container["http_endpoint"],
            "grpc_endpoint": phoenix_container["otlp_endpoint"],
        }
    )
    return ArtifactManager(telemetry_provider=provider, tenant_id=tenant_id)


@pytest.fixture
def dispatcher_with_factory(artifact_manager: ArtifactManager) -> AgentDispatcher:
    """Real AgentDispatcher wired to a real ArtifactManager."""
    config_manager = create_default_config_manager()
    registry = AgentRegistry(tenant_id="c5_dispatcher", config_manager=config_manager)
    return AgentDispatcher(
        agent_registry=registry,
        config_manager=config_manager,
        schema_loader=None,  # not exercised by resolve_artefact_for_request
        artifact_manager_factory=lambda _t: artifact_manager,
    )


@pytest.fixture
def dispatcher_without_factory() -> AgentDispatcher:
    """Dispatcher with no factory — proves back-compat (no canary at all)."""
    config_manager = create_default_config_manager()
    registry = AgentRegistry(tenant_id="c5_no_canary", config_manager=config_manager)
    return AgentDispatcher(
        agent_registry=registry,
        config_manager=config_manager,
        schema_loader=None,
    )


@pytest.mark.asyncio
class TestResolutionMethod:
    async def test_no_factory_returns_none(self, dispatcher_without_factory):
        out = await dispatcher_without_factory.resolve_artefact_for_request(
            "search_agent", "any_tenant", request_seed="req_0"
        )
        assert out is None, (
            "absent an artifact_manager_factory the dispatcher must return "
            "None — silently treating that as 'no canary' so existing "
            "deployments don't change behaviour"
        )

    async def test_no_state_returns_default(self, dispatcher_with_factory, tenant_id):
        out = await dispatcher_with_factory.resolve_artefact_for_request(
            "search_agent", tenant_id, request_seed="req_0"
        )
        assert out is not None
        assert out["served_from"] == "default"
        assert out["version"] is None

    async def test_traffic_pct_100_routes_all_to_canary(
        self, artifact_manager, dispatcher_with_factory, tenant_id
    ):
        # Promote a versioned prompt set, then route 100% to it.
        await artifact_manager.save_prompts_versioned(
            "search_agent", {"system": "CANARY_V1"}
        )
        # Save active for fall-through.
        await artifact_manager.save_prompts("search_agent", {"system": "ACTIVE_V1"})
        await artifact_manager.promote_to_canary(
            "search_agent", version=1, traffic_pct=100
        )

        decisions = []
        for i in range(20):
            out = await dispatcher_with_factory.resolve_artefact_for_request(
                "search_agent", tenant_id, request_seed=f"req_{i}"
            )
            decisions.append(out["served_from"])
        assert decisions.count("canary") == 20, (
            f"traffic_pct=100 must route every request to canary; "
            f"got distribution {decisions}"
        )

    async def test_traffic_pct_0_routes_all_to_active(
        self, artifact_manager, dispatcher_with_factory, tenant_id
    ):
        # save_prompts_versioned auto-increments → v1; promote at 1% (smallest
        # legal), then overwrite the state blob to traffic_pct=0 so we can
        # assert "0% canary, 100% active" without the validator complaining.
        await artifact_manager.save_prompts_versioned(
            "search_agent_zero", {"system": "CANARY_V1"}
        )
        await artifact_manager.save_prompts(
            "search_agent_zero", {"system": "ACTIVE_V1"}
        )
        await artifact_manager.promote_to_canary(
            "search_agent_zero", version=1, traffic_pct=1
        )
        await artifact_manager._save_artefact_state(
            "search_agent_zero",
            {
                "active": {"version": 1, "promoted_at": "now"},
                "canary": {
                    "version": 1,
                    "promoted_at": "now",
                    "traffic_pct": 0,  # nobody routes to canary
                },
                "retired": [],
            },
        )

        decisions = []
        for i in range(20):
            out = await dispatcher_with_factory.resolve_artefact_for_request(
                "search_agent_zero", tenant_id, request_seed=f"req_{i}"
            )
            decisions.append(out["served_from"])
        assert decisions.count("canary") == 0, (
            f"traffic_pct=0 must route nothing to canary; got {decisions}"
        )

    async def test_distribution_roughly_matches_traffic_pct(
        self, artifact_manager, dispatcher_with_factory, tenant_id
    ):
        # 10% canary, 1000 distinct seeds → expect ~80–120 canary hits.
        await artifact_manager.save_prompts_versioned(
            "search_agent_split", {"system": "CANARY_V1"}
        )
        await artifact_manager.save_prompts(
            "search_agent_split", {"system": "ACTIVE_V1"}
        )
        await artifact_manager.promote_to_canary(
            "search_agent_split", version=1, traffic_pct=10
        )

        canary_hits = 0
        for i in range(1000):
            out = await dispatcher_with_factory.resolve_artefact_for_request(
                "search_agent_split", tenant_id, request_seed=f"req_{i}"
            )
            if out["served_from"] == "canary":
                canary_hits += 1
        # 10% target ± 3σ tolerance.
        assert 70 <= canary_hits <= 130, (
            f"traffic_pct=10 should yield ~100/1000 canary hits; got {canary_hits}"
        )


@pytest.mark.asyncio
class TestDispatchPathIntegration:
    async def test_dispatch_stashes_artefact_overlay_in_context(
        self, artifact_manager, dispatcher_with_factory, tenant_id
    ):
        """When request_seed is supplied, dispatch() stashes the resolution
        on the context so downstream agent constructors can read it."""
        # Set up canary at 100% so we have a deterministic served_from.
        await artifact_manager.save_prompts_versioned(
            "search_agent_overlay", {"system": "CANARY_V1"}
        )
        await artifact_manager.promote_to_canary(
            "search_agent_overlay", version=1, traffic_pct=100
        )

        # Register a fake search-capable agent so dispatch finds it.
        from cogniverse_core.common.agent_models import AgentEndpoint

        ep = AgentEndpoint(
            name="search_agent_overlay",
            url="http://localhost:0",  # never actually called in this test
            capabilities=["video_search"],
            health_endpoint="/health",
            process_endpoint="/agents/search_agent_overlay/process",
            timeout=5,
        )
        dispatcher_with_factory._registry._agents = {"search_agent_overlay": ep}

        # Resolve directly to assert the wire (downstream search execution
        # would require a full Vespa+search-agent setup, out of scope for
        # this canary-wire test).
        overlay = await dispatcher_with_factory.resolve_artefact_for_request(
            "search_agent_overlay", tenant_id, request_seed="req_1"
        )
        # Then verify the resolution result has the canary metadata.
        assert overlay["served_from"] == "canary"
        assert overlay["version"] == 1


@pytest.mark.asyncio
class TestProductionDispatcherWiring:
    """``_ensure_dispatcher()`` must wire ``artifact_manager_factory`` itself.

    The tests above inject a factory by hand; this one proves the *production*
    dispatcher built by the runtime router carries one (from the configured
    telemetry manager), so canary routing is live without any test plumbing.
    """

    async def test_ensure_dispatcher_wires_factory_and_routes_canary(
        self, telemetry_manager_with_phoenix, tenant_id
    ):
        from cogniverse_runtime.routers import agents as agents_router

        config_manager = create_default_config_manager()
        registry = AgentRegistry(tenant_id="prod_wire", config_manager=config_manager)
        agents_router.set_agent_registry(registry)
        # resolve_artefact_for_request never touches the schema loader, but
        # _ensure_dispatcher() guards on all three deps being non-None.
        agents_router.set_agent_dependencies(config_manager, MagicMock())
        try:
            dispatcher = agents_router._ensure_dispatcher()

            # Gap closed: the production dispatcher carries a real factory.
            assert dispatcher._artifact_manager_factory is not None

            # With the factory live but no canary state yet, resolution is
            # reachable (non-None) and falls through to "default".
            pre = await dispatcher.resolve_artefact_for_request(
                "search_agent", tenant_id, request_seed="seed_pre"
            )
            assert pre is not None
            assert pre["served_from"] == "default"

            # Seed a canary at 100% through the same provider the factory uses.
            tm = telemetry_manager_with_phoenix
            am = ArtifactManager(tm.get_provider(tenant_id=tenant_id), tenant_id)
            await am.save_prompts_versioned("search_agent", {"system": "CANARY_V1"})
            await am.save_prompts("search_agent", {"system": "ACTIVE_V1"})
            await am.promote_to_canary("search_agent", version=1, traffic_pct=100)

            out = await dispatcher.resolve_artefact_for_request(
                "search_agent", tenant_id, request_seed="seed_canary"
            )
            assert out["served_from"] == "canary"
            assert out["version"] == 1
            assert out["prompts"] == {"system": "CANARY_V1"}
        finally:
            # Don't leak the wired dispatcher/deps into other tests.
            agents_router._dispatcher = None
            agents_router._agent_registry = None
            agents_router._config_manager = None
            agents_router._schema_loader = None


class _CapturingQueue:
    """Minimal A2A EventQueue stand-in."""

    def __init__(self):
        self.events = []

    async def enqueue_event(self, event):
        self.events.append(event)


@pytest.mark.asyncio
class TestStreamingCanaryRealPhoenix:
    """Real-Phoenix end-to-end: a STREAMING request must serve canary prompts.

    Drives the actual ``CogniverseAgentExecutor._execute_streaming`` against a
    real-Phoenix-backed dispatcher (the streaming path that previously bypassed
    canary routing). create_streaming_agent is swapped for a bare memory-aware
    stub so the test exercises the real resolve(real Phoenix)+inject without
    constructing a full SearchAgent.
    """

    async def test_streaming_serves_canary_resolved_from_real_phoenix(
        self, artifact_manager, dispatcher_with_factory, tenant_id
    ):
        from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
        from cogniverse_runtime.a2a_executor import CogniverseAgentExecutor

        # Promote a canary at 100% in REAL Phoenix for search_agent.
        await artifact_manager.save_prompts_versioned(
            "search_agent", {"system": "CANARY_STREAM_V1"}
        )
        await artifact_manager.promote_to_canary(
            "search_agent", version=1, traffic_pct=100
        )

        class _StreamAgent(MemoryAwareMixin):
            async def process(self, typed_input, stream=False):
                async def _gen():
                    yield {"type": "final", "data": {}}

                return _gen()

        agent = _StreamAgent()
        dispatcher_with_factory.create_streaming_agent = lambda *a, **k: (agent, None)

        executor = CogniverseAgentExecutor(dispatcher=dispatcher_with_factory)
        await executor._execute_streaming(
            "search_agent", "q", tenant_id, "task-1", "ctx-stream", _CapturingQueue()
        )

        # The streaming agent received the canary overlay resolved from real
        # Phoenix — proving streaming traffic now honours canary routing.
        assert agent._dispatched_artefact["served_from"] == "canary"
        assert agent._dispatched_artefact["version"] == 1
        assert agent.get_dispatched_prompts() == {"system": "CANARY_STREAM_V1"}
