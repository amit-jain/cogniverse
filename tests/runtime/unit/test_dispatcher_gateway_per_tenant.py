"""The GatewayAgent's routing thresholds are per-tenant, so the dispatcher
must build one instance per tenant, not share a single instance.

The gateway's ``fast_path_confidence_threshold`` / ``gliner_threshold`` are
loaded per tenant from the artifact store into ``deps``. A single shared
instance baked in whichever tenant hit it first and served every other
tenant those thresholds; the streaming path built its own instance and never
loaded the artifact at all. Both paths now resolve through
``_get_or_build_gateway_agent`` which caches per tenant, loads on build, and
re-loads the artifact on cache hits once the reload interval elapses so a
warm pod serves recalibrated thresholds without a restart.
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

    from cogniverse_foundation.config.unified_config import RoutingConfigUnified

    d = object.__new__(AgentDispatcher)
    d._resolve_gliner_url = lambda *a, **k: None
    # Default routing config (its GLiNER defaults match GatewayDeps, so seeding
    # leaves these tenants' behavior unchanged); a test may override this.
    d._config_manager = SimpleNamespace(
        get_routing_config=lambda tid: RoutingConfigUnified(tenant_id=tid)
    )
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
async def test_gateway_seeds_gliner_config_from_routing_config(dispatcher):
    """The tenant's dashboard-editable RoutingConfigUnified GLiNER model and
    threshold reach the live GatewayDeps (they were ignored before — the gateway
    only ever saw its own defaults plus the optimization artifact)."""
    from cogniverse_foundation.config.unified_config import RoutingConfigUnified

    d, _ = dispatcher
    d._config_manager = SimpleNamespace(
        get_routing_config=lambda tid: RoutingConfigUnified(
            tenant_id=tid,
            gliner_model="acme/custom-gliner",
            gliner_threshold=0.55,
            gliner_device="cuda",
            enable_fast_path=False,
        )
    )

    agent = await d._get_or_build_gateway_agent("acme:acme")

    assert agent.deps.gliner_model_name == "acme/custom-gliner"
    assert agent.deps.gliner_threshold == 0.55
    assert agent.deps.gliner_device == "cuda"
    assert agent.deps.enable_fast_path is False


@pytest.mark.asyncio
async def test_fast_path_threshold_seeded_from_config_without_artifact(
    dispatcher,
    monkeypatch,
):
    """With no optimization artifact, the tenant's configured fast-path
    threshold reaches the gateway (the artifact would override it if present)."""
    import cogniverse_agents.gateway_agent as gw
    from cogniverse_foundation.config.unified_config import RoutingConfigUnified

    d, _ = dispatcher
    monkeypatch.setattr(
        gw.GatewayAgent, "_load_artifact", lambda self: None, raising=False
    )
    d._config_manager = SimpleNamespace(
        get_routing_config=lambda tid: RoutingConfigUnified(
            tenant_id=tid, fast_path_confidence_threshold=0.65
        )
    )

    agent = await d._get_or_build_gateway_agent("noartifact:noartifact")

    assert agent.deps.fast_path_confidence_threshold == 0.65


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


@pytest.mark.asyncio
async def test_cached_gateway_reloads_artifact_after_reload_interval(dispatcher):
    """A warm pod must pick up recalibrated thresholds without a restart.

    Once the reload interval has elapsed, a cache hit re-runs _load_artifact
    on the SAME instance, so values persisted by the optimization loop start
    serving; previously the artifact was read only on the cache-miss build
    and a warm pod kept the stale thresholds until restart."""
    d, _ = dispatcher
    d._gateway_artifact_ttl_s = 0.0

    first = await d._get_or_build_gateway_agent("acme:acme")
    assert first.deps.fast_path_confidence_threshold == 0.11

    _TENANT_THRESHOLD["acme:acme"] = 0.44
    try:
        second = await d._get_or_build_gateway_agent("acme:acme")
    finally:
        _TENANT_THRESHOLD["acme:acme"] = 0.11

    assert second is first
    assert second.deps.fast_path_confidence_threshold == 0.44


@pytest.mark.asyncio
async def test_cached_gateway_does_not_reload_inside_the_interval(dispatcher):
    """Within the reload interval a cache hit serves the loaded values with
    no artifact re-read — the refresh is bounded, not per-request."""
    d, _ = dispatcher

    first = await d._get_or_build_gateway_agent("acme:acme")
    assert first.deps.fast_path_confidence_threshold == 0.11

    _TENANT_THRESHOLD["acme:acme"] = 0.44
    try:
        second = await d._get_or_build_gateway_agent("acme:acme")
    finally:
        _TENANT_THRESHOLD["acme:acme"] = 0.11

    assert second is first
    assert second.deps.fast_path_confidence_threshold == 0.11
