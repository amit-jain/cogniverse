"""Dispatcher artifact loads must run off the event loop.

``_load_artifact()`` does a synchronous Phoenix round trip. Called inline from
the async dispatch path it froze the loop for the whole fetch. Each site is now
``await asyncio.to_thread(agent._load_artifact)``. The proof is deterministic:
the offloaded load records ``threading.get_ident()``, which must differ from the
event loop's thread.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cogniverse_runtime.agent_dispatcher import AgentDispatcher


class _StopHere(Exception):
    pass


@pytest.mark.asyncio
async def test_gateway_load_artifact_runs_off_the_event_loop(monkeypatch):
    import cogniverse_agents.gateway_agent as gw
    import cogniverse_foundation.telemetry.manager as tm_mod

    # Avoid building a real telemetry manager (dead-sentinel backend).
    monkeypatch.setattr(tm_mod, "_telemetry_manager", SimpleNamespace(), raising=False)

    recorded = {}

    def _rec_load(self):
        recorded["thread"] = threading.get_ident()

    async def _stub_process(self, inp):
        raise _StopHere()

    monkeypatch.setattr(gw.GatewayAgent, "__init__", lambda self, deps=None, **k: None)
    monkeypatch.setattr(gw.GatewayAgent, "_load_artifact", _rec_load, raising=False)
    monkeypatch.setattr(gw.GatewayAgent, "_process_impl", _stub_process, raising=False)

    d = object.__new__(AgentDispatcher)
    d._gateway_agent = None
    d.consult_egress_policy = lambda *a, **k: None
    d._verify_routing_egress = lambda *a, **k: None
    d._get_rail_chains = lambda *a, **k: None
    d._resolve_gliner_url = lambda *a, **k: None
    # The gateway build seeds GLiNER deps from the tenant routing config.
    d._config_manager = SimpleNamespace(
        get_routing_config=lambda tenant_id: SimpleNamespace(
            gliner_model="urchade/gliner_multi-v2.1",
            gliner_threshold=0.3,
            gliner_device="cpu",
            fast_path_confidence_threshold=0.7,
            enable_fast_path=True,
        )
    )

    loop_thread = threading.get_ident()
    with pytest.raises(_StopHere):
        await d._execute_gateway_task("q", {}, "acme:acme")

    assert recorded.get("thread") is not None
    # to_thread offload => _load_artifact ran on a worker thread, not the loop.
    assert recorded["thread"] != loop_thread


@pytest.mark.unit
@pytest.mark.ci_fast
def test_artifact_manager_factory_reuses_manager_per_tenant(monkeypatch):
    """The dispatcher's artefact factory reuses one ArtifactManager per tenant.

    ArtifactManager carries a 5s-TTL request cache for artefact state + prompts
    (invalidated on promote/retire). Building a fresh manager on every dispatch
    threw that cache away, so each request re-read state + prompts from Phoenix.
    Caching the manager per tenant lets the TTL cache span dispatches; distinct
    tenants still get distinct managers (and providers)."""
    from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
    from cogniverse_runtime.routers import agents as agents_router

    tm = MagicMock()
    tm.get_provider = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
        lambda: tm,
    )

    factory = agents_router._build_artifact_manager_factory()
    assert factory is not None

    a1 = factory("acme:acme")
    a2 = factory("acme:acme")
    b1 = factory("beta:beta")

    assert isinstance(a1, ArtifactManager)
    assert a1 is a2, "same tenant must reuse one manager so its TTL cache survives"
    assert a1 is not b1, "different tenants get distinct managers"
    # Provider fetched once per tenant, not once per dispatch.
    assert tm.get_provider.call_count == 2
