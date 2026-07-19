"""The orchestration policy http client lives with the cached agent.

The orchestration path used to build a fresh ``httpx.AsyncClient`` per request
via ``sandbox_manager.make_http_client`` and ``aclose()`` it in a finally. The
per-tenant OrchestratorAgent is now cached, so the policy client is built ONCE
per tenant and reused across dispatches (one pooled client per tenant) instead
of a build-and-teardown per complex query. These pin that contract:
make_http_client runs once, the same client stays on the cached agent, and a
failing request does not tear it down.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cogniverse_runtime.agent_dispatcher import AgentDispatcher


class _StubOrchestrator:
    def __init__(self, **kwargs):
        self.http_client = kwargs.get("http_client")
        self.telemetry_manager = None
        self._artifact_tenant_id = None

    def _load_artifact(self):
        pass

    async def _process_impl(self, input_data):
        return SimpleNamespace(model_dump=lambda: {"result": "ok"})


def _dispatcher_with_spy_client():
    spy_client = MagicMock()
    spy_client.aclose = AsyncMock()
    sandbox = MagicMock()
    sandbox.make_http_client = MagicMock(return_value=spy_client)
    dispatcher = AgentDispatcher(
        agent_registry=MagicMock(),
        config_manager=MagicMock(),
        schema_loader=None,
        sandbox_manager=sandbox,
    )
    return dispatcher, spy_client, sandbox


def _patches(process_impl):
    return [
        patch(
            "cogniverse_agents.orchestrator_agent.OrchestratorAgent", _StubOrchestrator
        ),
        patch.object(AgentDispatcher, "_init_agent_memory", lambda *a, **k: None),
        patch.object(AgentDispatcher, "_apply_artefact_overlay", lambda *a, **k: None),
        patch(
            "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
            return_value=None,
        ),
        patch.object(_StubOrchestrator, "_process_impl", process_impl),
    ]


@pytest.mark.asyncio
async def test_policy_client_built_once_and_reused_across_requests():
    dispatcher, spy_client, sandbox = _dispatcher_with_spy_client()

    async def ok(self, input_data):
        return SimpleNamespace(model_dump=lambda: {"result": "ok"})

    with contextlib.ExitStack() as stack:
        for p in _patches(ok):
            stack.enter_context(p)
        first = await dispatcher._execute_orchestration_task(
            query="q1", context={"tenant_id": "acme:prod"}, tenant_id="acme:prod"
        )
        second = await dispatcher._execute_orchestration_task(
            query="q2", context={"tenant_id": "acme:prod"}, tenant_id="acme:prod"
        )

    assert first["status"] == "success"
    assert second["status"] == "success"
    # One pooled client per tenant, built on the cache-miss build and reused.
    sandbox.make_http_client.assert_called_once_with("orchestrator_agent")
    # Reused across requests, never torn down per dispatch.
    spy_client.aclose.assert_not_awaited()
    cached = dispatcher._orchestrator_agents.get("acme:prod").agent
    assert cached.http_client is spy_client


@pytest.mark.asyncio
async def test_failing_request_does_not_tear_down_cached_client():
    dispatcher, spy_client, sandbox = _dispatcher_with_spy_client()

    async def boom(self, input_data):
        raise RuntimeError("orchestration blew up")

    with contextlib.ExitStack() as stack:
        for p in _patches(boom):
            stack.enter_context(p)
        with pytest.raises(RuntimeError, match="blew up"):
            await dispatcher._execute_orchestration_task(
                query="q", context={"tenant_id": "acme:prod"}, tenant_id="acme:prod"
            )

    # The cached client must survive a failed request for the next one to reuse.
    spy_client.aclose.assert_not_awaited()
    assert dispatcher._orchestrator_agents.get("acme:prod") is not None
