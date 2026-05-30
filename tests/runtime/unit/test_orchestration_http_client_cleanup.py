"""_execute_orchestration_task must close the per-request policy client.

Regression (PERF/leak): the orchestration path built a fresh
``httpx.AsyncClient`` (own connection pool + transport) via
``sandbox_manager.make_http_client`` and never ``aclose()``d it, leaking a
client per orchestration request. These assert it is closed on BOTH the
success and the error path.
"""

from __future__ import annotations

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
    return dispatcher, spy_client


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
async def test_client_closed_on_success():
    dispatcher, spy_client = _dispatcher_with_spy_client()

    async def ok(self, input_data):
        return SimpleNamespace(model_dump=lambda: {"result": "ok"})

    import contextlib

    with contextlib.ExitStack() as stack:
        for p in _patches(ok):
            stack.enter_context(p)
        result = await dispatcher._execute_orchestration_task(
            query="q", context={"tenant_id": "acme:prod"}, tenant_id="acme:prod"
        )

    assert result["status"] == "success"
    spy_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_client_closed_on_error():
    dispatcher, spy_client = _dispatcher_with_spy_client()

    async def boom(self, input_data):
        raise RuntimeError("orchestration blew up")

    import contextlib

    with contextlib.ExitStack() as stack:
        for p in _patches(boom):
            stack.enter_context(p)
        with pytest.raises(RuntimeError, match="blew up"):
            await dispatcher._execute_orchestration_task(
                query="q", context={"tenant_id": "acme:prod"}, tenant_id="acme:prod"
            )

    # finally must still close the leaked client.
    spy_client.aclose.assert_awaited_once()
