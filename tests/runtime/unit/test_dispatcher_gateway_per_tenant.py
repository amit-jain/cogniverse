"""The GatewayAgent's routing thresholds are per-tenant, so the dispatcher
must build one instance per tenant, not share a single instance.

The gateway's ``fast_path_confidence_threshold`` / ``gliner_threshold`` are
loaded per tenant from the artifact store into ``deps``. A single shared
instance baked in whichever tenant hit it first and served every other
tenant those thresholds; the streaming path built its own instance and never
loaded the artifact at all. Both paths now resolve through
``_get_or_build_gateway_agent`` which caches per tenant and always loads.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cogniverse_runtime.agent_dispatcher import AgentDispatcher

# Per-tenant thresholds the fake artifact store hands back.
_TENANT_THRESHOLD = {"acme:acme": 0.11, "globex:globex": 0.87}


@pytest.fixture
def dispatcher(monkeypatch):
    import cogniverse_agents.gateway_agent as gw
    import cogniverse_foundation.telemetry.manager as tm_mod

    sentinel_tm = SimpleNamespace(name="telemetry-manager")
    monkeypatch.setattr(
        tm_mod, "get_telemetry_manager", lambda *a, **k: sentinel_tm, raising=True
    )

    def _init(self, deps=None, **k):
        self.deps = deps

    def _load(self):
        # Real _load_artifact reads gateway_thresholds for _artifact_tenant_id;
        # here it applies the per-tenant value so the instance is distinguishable.
        self.deps.fast_path_confidence_threshold = _TENANT_THRESHOLD[
            self._artifact_tenant_id
        ]

    monkeypatch.setattr(gw.GatewayAgent, "__init__", _init)
    monkeypatch.setattr(gw.GatewayAgent, "_load_artifact", _load, raising=False)

    d = object.__new__(AgentDispatcher)
    d._resolve_gliner_url = lambda *a, **k: None
    return d, sentinel_tm


@pytest.mark.asyncio
async def test_each_tenant_gets_its_own_gateway_with_its_own_thresholds(dispatcher):
    d, _ = dispatcher

    acme = await d._get_or_build_gateway_agent("acme:acme")
    globex = await d._get_or_build_gateway_agent("globex:globex")

    assert acme is not globex
    assert acme._artifact_tenant_id == "acme:acme"
    assert globex._artifact_tenant_id == "globex:globex"
    assert acme.deps.fast_path_confidence_threshold == 0.11
    assert globex.deps.fast_path_confidence_threshold == 0.87


@pytest.mark.asyncio
async def test_same_tenant_reuses_the_cached_instance(dispatcher):
    d, _ = dispatcher

    first = await d._get_or_build_gateway_agent("acme:acme")
    second = await d._get_or_build_gateway_agent("acme:acme")

    assert first is second


@pytest.mark.asyncio
async def test_streaming_gateway_loads_the_tenant_artifact_and_telemetry(dispatcher):
    d, sentinel_tm = dispatcher
    d._registry = SimpleNamespace(
        get_agent=lambda name: SimpleNamespace(capabilities=["gateway"])
    )

    agent, typed_input = await d.create_streaming_agent(
        "gateway_agent", "find the clip", "globex:globex"
    )

    # Pre-fix the streaming path built a bare GatewayAgent and never set these.
    assert agent._artifact_tenant_id == "globex:globex"
    assert agent.telemetry_manager is sentinel_tm
    assert agent.deps.fast_path_confidence_threshold == 0.87
    assert typed_input.tenant_id == "globex:globex"


@pytest.mark.asyncio
async def test_streaming_and_dispatch_share_the_same_per_tenant_instance(dispatcher):
    d, _ = dispatcher
    d._registry = SimpleNamespace(
        get_agent=lambda name: SimpleNamespace(capabilities=["gateway"])
    )

    dispatched = await d._get_or_build_gateway_agent("acme:acme")
    streamed, _ = await d.create_streaming_agent("gateway_agent", "q", "acme:acme")

    assert dispatched is streamed
