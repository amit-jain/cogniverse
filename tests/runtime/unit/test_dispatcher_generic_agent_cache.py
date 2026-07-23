"""Generic A2A agents are cached per (tenant, agent_name), TTL-refreshing only
the artifact — mirroring the gateway cache.

``_execute_generic_agent`` rebuilt the agent on every dispatch: class resolution,
deps + agent construction, ``_init_agent_memory`` (to_thread), ``_bind_graph_manager``
(to_thread), telemetry injection, and a Phoenix ``_load_artifact`` read (to_thread) —
per request, for entity_extraction / query_enhancement / profile_selection. The
build now happens once per (tenant, agent_name); a cache hit past the reload
interval re-reads only the artifact, stamping the reload time before the await so
concurrent dispatches never stampede duplicate reloads. Per-request state (the
artefact overlay, session id) is NOT cached — it rides Task-isolated ContextVars
and ``_apply_artefact_overlay`` runs on every dispatch, cache hit or miss.
"""

from __future__ import annotations

import sys
import time
import types
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _install_overlay_spy(dispatcher, sink: list) -> None:
    """Shadow ``_apply_artefact_overlay`` with a counting wrapper (it is a
    staticmethod, so an instance attribute overrides it for ``self.`` calls)."""
    orig = type(dispatcher)._apply_artefact_overlay

    def _spy(agent, context):
        sink.append(agent)
        return orig(agent, context)

    dispatcher._apply_artefact_overlay = _spy


# --- A self-contained importable agent module, driven through the REAL
# _execute_generic_agent path (class resolution -> cache -> overlay -> input
# -> process), so the test exercises the whole dispatch surface, not a stub seam.


class FakeDeps(BaseModel):
    model_config = ConfigDict(extra="allow")


class FakeInput(BaseModel):
    query: str
    tenant_id: str | None = None


class FakeOutput(BaseModel):
    echo: str
    tenant_id: str | None = None


class _FakeAgentMixin:
    """Records construction + artifact-load counts on the concrete class."""

    load_delay = 0.0

    def __init__(self, deps=None, **_kwargs):
        type(self).build_count = getattr(type(self), "build_count", 0) + 1
        self.deps = deps
        self.telemetry_manager = None
        self._config_manager = None

    def _load_artifact(self) -> None:
        type(self).load_count = getattr(type(self), "load_count", 0) + 1
        if self.load_delay:
            time.sleep(self.load_delay)

    async def process(self, typed_input):
        return FakeOutput(echo=typed_input.query, tenant_id=typed_input.tenant_id)


class FakeAgentA(_FakeAgentMixin):
    build_count = 0
    load_count = 0


class FakeAgentB(_FakeAgentMixin):
    build_count = 0
    load_count = 0


_FAKE_MODULE_NAME = "cogniverse_fake_generic_agent_mod"
_fake_module = types.ModuleType(_FAKE_MODULE_NAME)
_fake_module.FakeAgentA = FakeAgentA
_fake_module.FakeAgentB = FakeAgentB
_fake_module.FakeDeps = FakeDeps
_fake_module.FakeInput = FakeInput
_fake_module.FakeOutput = FakeOutput
sys.modules[_FAKE_MODULE_NAME] = _fake_module


@pytest.fixture
def dispatcher(monkeypatch):
    """A real AgentDispatcher wired to the fake importable agent module.

    Registers ``fakea``/``fakeb`` in AGENT_CLASSES, resets the class-level
    build/load counters, clears the class-resolution memo, and stubs the
    GLiNER lookup + telemetry manager so the build path runs without infra.
    """
    import cogniverse_foundation.telemetry.manager as tm_mod
    import cogniverse_runtime.agent_dispatcher as ad
    from cogniverse_runtime.agent_dispatcher import AgentDispatcher
    from cogniverse_runtime.config_loader import ConfigLoader

    for cls in (FakeAgentA, FakeAgentB):
        cls.build_count = 0
        cls.load_count = 0
        cls.load_delay = 0.0

    # _execute_generic_agent now binds the tenant-routed LM, which reads the
    # semantic-router config via get_config. Default it to router-disabled so
    # these cache tests take the plain ambient-LM path; the routing test below
    # overrides it to enabled.
    from unittest.mock import MagicMock

    from cogniverse_foundation.config.semantic_router import SemanticRouterConfig

    _disabled_cfg = MagicMock()
    _disabled_cfg.get_semantic_router.return_value = SemanticRouterConfig(enabled=False)
    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.get_config",
        lambda tenant_id, config_manager: _disabled_cfg,
    )

    ad._GENERIC_AGENT_CLASSES.clear()
    monkeypatch.setitem(
        ConfigLoader.AGENT_CLASSES, "fakea", f"{_FAKE_MODULE_NAME}:FakeAgentA"
    )
    monkeypatch.setitem(
        ConfigLoader.AGENT_CLASSES, "fakeb", f"{_FAKE_MODULE_NAME}:FakeAgentB"
    )
    monkeypatch.setattr(
        tm_mod, "get_telemetry_manager", lambda *a, **k: SimpleNamespace(name="tm")
    )

    d = AgentDispatcher(
        agent_registry=SimpleNamespace(),
        config_manager=SimpleNamespace(),
        schema_loader=SimpleNamespace(),
    )
    monkeypatch.setattr(d, "_resolve_gliner_url", lambda: None)
    return d


@pytest.mark.asyncio
async def test_two_dispatches_build_the_agent_once(dispatcher):
    d = dispatcher

    r1 = await d._execute_generic_agent("fakea", "q1", {}, "acme:acme")
    r2 = await d._execute_generic_agent("fakea", "q2", {}, "acme:acme")

    assert FakeAgentA.build_count == 1, "agent rebuilt on the second dispatch"
    assert r1 == {
        "status": "success",
        "agent": "fakea",
        "echo": "q1",
        "tenant_id": "acme:acme",
    }
    assert r2["echo"] == "q2"


@pytest.mark.asyncio
async def test_first_build_loads_artifact_once(dispatcher):
    d = dispatcher

    await d._execute_generic_agent("fakea", "q", {}, "acme:acme")
    await d._execute_generic_agent("fakea", "q", {}, "acme:acme")

    # First dispatch builds + loads; second is a within-interval cache hit.
    assert FakeAgentA.build_count == 1
    assert FakeAgentA.load_count == 1


@pytest.mark.asyncio
async def test_distinct_agent_names_get_distinct_cached_entries(dispatcher):
    d = dispatcher

    ra = await d._execute_generic_agent("fakea", "q", {}, "acme:acme")
    rb = await d._execute_generic_agent("fakeb", "q", {}, "acme:acme")

    assert ra["agent"] == "fakea"
    assert rb["agent"] == "fakeb"
    assert FakeAgentA.build_count == 1
    assert FakeAgentB.build_count == 1

    per_tenant = d._generic_agents.get("acme:acme")
    assert set(per_tenant.keys()) == {"fakea", "fakeb"}
    assert isinstance(per_tenant["fakea"].agent, FakeAgentA)
    assert isinstance(per_tenant["fakeb"].agent, FakeAgentB)
    assert per_tenant["fakea"].agent is not per_tenant["fakeb"].agent


@pytest.mark.asyncio
async def test_cache_hit_past_ttl_reloads_artifact(dispatcher):
    d = dispatcher
    d._generic_agent_ttl_s = 0.0

    await d._execute_generic_agent("fakea", "q", {}, "acme:acme")
    assert FakeAgentA.load_count == 1

    await d._execute_generic_agent("fakea", "q", {}, "acme:acme")

    # ttl=0 => every cache hit re-reads the artifact on the SAME instance.
    assert FakeAgentA.build_count == 1
    assert FakeAgentA.load_count == 2


@pytest.mark.asyncio
async def test_concurrent_reload_is_stamped_before_await_no_double_reload(dispatcher):
    import asyncio

    d = dispatcher
    FakeAgentA.load_delay = 0.03

    # Build the entry, then force it stale so the next dispatch reloads.
    await d._execute_generic_agent("fakea", "q", {}, "acme:acme")
    assert FakeAgentA.load_count == 1
    entry = d._generic_agents.get("acme:acme")["fakea"]
    entry.loaded_at = time.monotonic() - (d._generic_agent_ttl_s + 1000.0)

    await asyncio.gather(
        d._execute_generic_agent("fakea", "c1", {}, "acme:acme"),
        d._execute_generic_agent("fakea", "c2", {}, "acme:acme"),
    )

    # Stamp-before-await: the first dispatch refreshes loaded_at before yielding
    # to the reload thread, so the second concurrent dispatch sees a fresh stamp
    # and skips the reload. Exactly one reload beyond the initial build-load.
    assert FakeAgentA.build_count == 1
    assert FakeAgentA.load_count == 2


@pytest.mark.asyncio
async def test_concurrent_cold_dispatches_build_the_agent_once(dispatcher):
    import asyncio

    d = dispatcher
    # Widen the build window so cold first-touches overlap on the event loop.
    FakeAgentA.load_delay = 0.05

    await asyncio.gather(
        d._execute_generic_agent("fakea", "c1", {}, "acme:acme"),
        d._execute_generic_agent("fakea", "c2", {}, "acme:acme"),
        d._execute_generic_agent("fakea", "c3", {}, "acme:acme"),
    )

    # Without an in-flight guard, each concurrent first-touch runs a full build
    # (class wiring, memory init, schema-deploying artifact read) and all but one
    # are discarded. The guard funnels them onto a single build.
    assert FakeAgentA.build_count == 1
    assert FakeAgentA.load_count == 1


@pytest.mark.asyncio
async def test_reload_failure_serves_cached_agent_and_reschedules_retry(dispatcher):
    from cogniverse_runtime.agent_dispatcher import RELOAD_RETRY_COOLDOWN_S

    d = dispatcher
    d._generic_agent_ttl_s = 100.0

    r1 = await d._execute_generic_agent("fakea", "q", {}, "acme:acme")
    assert r1["echo"] == "q"
    entry = d._generic_agents.get("acme:acme")["fakea"]

    def _boom():
        raise RuntimeError("artifact store down")

    entry.agent._load_artifact = _boom
    entry.loaded_at = time.monotonic() - (d._generic_agent_ttl_s + 1000.0)

    # A transient reload failure must NOT fail the dispatch — the cached agent is
    # still valid, so the request is served; the failed reload is retried a short
    # cooldown out, not suppressed for a full TTL and not hammered every request.
    r2 = await d._execute_generic_agent("fakea", "q2", {}, "acme:acme")
    assert r2["echo"] == "q2"

    age = time.monotonic() - entry.loaded_at
    expected = d._generic_agent_ttl_s - RELOAD_RETRY_COOLDOWN_S
    assert expected - 2.0 <= age <= expected + 2.0, (
        f"reload failure should reschedule ~{expected}s in, got age {age}"
    )


@pytest.mark.asyncio
async def test_per_request_overlay_applied_on_cache_hit(dispatcher):
    d = dispatcher

    seen: list = []
    _install_overlay_spy(d, seen)

    await d._execute_generic_agent("fakea", "q1", {}, "acme:acme")
    await d._execute_generic_agent("fakea", "q2", {}, "acme:acme")

    # Overlay injection runs on EVERY dispatch, cache hit included — it writes a
    # Task-isolated ContextVar, never the shared cached instance.
    assert len(seen) == 2
    assert FakeAgentA.build_count == 1


@pytest.mark.asyncio
async def test_cached_agent_receives_dispatcher_injected_config_manager(dispatcher):
    """The build path injects telemetry + config_manager once; the cached
    instance keeps them across dispatches."""
    d = dispatcher

    await d._execute_generic_agent("fakea", "q", {}, "acme:acme")
    agent = d._generic_agents.get("acme:acme")["fakea"].agent

    assert agent._config_manager is d._config_manager
    assert agent.telemetry_manager == SimpleNamespace(name="tm")


@pytest.mark.asyncio
async def test_unknown_agent_still_raises(dispatcher):
    d = dispatcher
    with pytest.raises(ValueError, match="no supported execution path"):
        await d._execute_generic_agent("nope", "q", {}, "acme:acme")


@pytest.mark.asyncio
async def test_generic_dispatch_binds_tenant_routed_lm(dispatcher, monkeypatch):
    """A direct dispatch of a generic agent (entity_extraction /
    query_enhancement fall here) must run under the tenant's SEMANTIC_ROUTER
    tier LM, not the process-global default — the same routing the orchestrated
    path and answer agents apply."""
    from unittest.mock import MagicMock

    import dspy

    from cogniverse_foundation.config.semantic_router import SemanticRouterConfig

    d = dispatcher

    seen: dict = {}

    async def _capturing_process(self, typed_input):
        seen["lm"] = dspy.settings.lm
        return FakeOutput(echo=typed_input.query, tenant_id=typed_input.tenant_id)

    monkeypatch.setattr(FakeAgentA, "process", _capturing_process)

    sentinel_lm = MagicMock(name="tenant_routed_lm")
    ambient_lm = MagicMock(name="ambient_default_lm")

    cfg = MagicMock()
    cfg.get_semantic_router.return_value = SemanticRouterConfig(enabled=True)
    endpoint = MagicMock(name="endpoint")
    cfg.get_llm_config.return_value.resolve.return_value = endpoint

    monkeypatch.setattr(
        "cogniverse_foundation.config.utils.get_config",
        lambda tenant_id, config_manager: cfg,
    )

    captured: dict = {}

    def _fake_create_routed_lm(ep, router, tenant_id):
        captured["tenant_id"] = tenant_id
        return sentinel_lm

    monkeypatch.setattr(
        "cogniverse_foundation.config.semantic_router.create_routed_lm",
        _fake_create_routed_lm,
    )

    with dspy.context(lm=ambient_lm):
        await d._execute_generic_agent("fakea", "q", {}, "acme:acme")

    assert seen["lm"] is sentinel_lm, (
        f"generic dispatch ran on {seen['lm']!r}, not the tenant-routed LM"
    )
    assert captured["tenant_id"] == "acme:acme"
