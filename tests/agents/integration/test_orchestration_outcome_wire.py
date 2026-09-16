"""A failed orchestration reaches its consumers as a failure.

Real OrchestratorAgent, AgentRegistry, AgentDispatcher and the
``/agents/{name}/process`` route, all over ``httpx.ASGITransport``: the
orchestrator's child calls are real HTTP against an ASGI child service, and
the request itself is real HTTP against the runtime router. The plan is
prepared so the assertions pin the outcome, not an LM's step choice.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext

import httpx
import pytest
from fastapi import FastAPI, HTTPException, Request

from cogniverse_agents.orchestrator_agent import (
    AccumulatedEvidence,
    AgentStep,
    OrchestrationPlan,
    OrchestratorAgent,
    OrchestratorDeps,
)
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_foundation.telemetry.config import TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import agents as agents_router

pytestmark = pytest.mark.integration

TENANT = "prodfixagents:outcome"
CHILD_ERROR = (
    "HTTPStatusError: Server error '503 Service Unavailable' for url "
    "'http://children/child/second'\nFor more information check: "
    "https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/503"
)


class _Runtime:
    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []
        self.memory_writes: list[tuple[str, str]] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.client: httpx.AsyncClient | None = None
        self.dispatcher: AgentDispatcher | None = None


@pytest.fixture
async def runtime(monkeypatch):
    state = _Runtime()
    child_app = FastAPI()

    @child_app.post("/child/{name}")
    async def child(name: str, request: Request):
        body = await request.json()
        query = body["query"]
        state.calls.append((name, query, body["context"]["tenant_id"]))
        if query == "held-failure":
            state.entered.set()
            await asyncio.wait_for(state.release.wait(), 10)
            raise HTTPException(503, "child interrupted after acceptance")
        if query == "all-failed" or (query == "mixed" and name == "second"):
            raise HTTPException(503, "child unavailable")
        return {"status": "success", "answer": f"{name}:{query}"}

    config_manager = create_default_config_manager()
    registry = AgentRegistry(tenant_id=TENANT, config_manager=config_manager)
    for name in ("first", "second"):
        registry.register_agent(
            AgentEndpoint(
                name=name,
                url="http://children",
                process_endpoint=f"/child/{name}",
                capabilities=["text_generation"],
            )
        )

    class PreparedOrchestrator(OrchestratorAgent):
        async def _create_plan(self, query, conversation_context, gateway_context):
            return OrchestrationPlan(
                query=query,
                reasoning="two required sources",
                steps=[
                    AgentStep(
                        agent_name=name, input_data={"query": query}, reasoning=name
                    )
                    for name in ("first", "second")
                ],
            )

        async def _iterative_retrieval_loop(self, query, plan, **kwargs):
            executed = await self._execute_plan(
                plan,
                tenant_id=kwargs["tenant_id"],
                workflow_id=kwargs["workflow_id"],
                execution_order_sink=kwargs["execution_order_sink"],
                agent_observations_sink=kwargs["agent_observations_sink"],
            )
            kwargs["agent_results_sink"].update(executed)
            return AccumulatedEvidence(iterations_executed=1, exit_reason="max_iter")

        def _ensure_memory_for_tenant(self, tenant_id):
            return None

        def get_relevant_context(self, query):
            return ""

        def remember_success(self, query, summary):
            state.memory_writes.append((query, summary))

        def _semantic_router_lm_context(self, tenant_id):
            return nullcontext()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=child_app)
    ) as children:
        agent = PreparedOrchestrator(
            deps=OrchestratorDeps(), registry=registry, config_manager=config_manager
        )
        agent.telemetry_manager = TelemetryManager(TelemetryConfig(enabled=False))
        agent.workflow_intelligence = None
        agent._http_client_override = children

        class PreparedDispatcher(AgentDispatcher):
            async def _get_or_build_orchestrator(self, tenant_id):
                return agent

            def consult_egress_policy(self, agent_name):
                return None

            def _verify_egress(self, agent_name, tenant_id):
                return None

            async def dispatch(self, agent_name, query, context, top_k=10):
                result = await self._execute_orchestration_task(
                    query, context, context["tenant_id"]
                )
                self._stamp_answer(result)
                return result

            def _init_agent_memory(self, *args, **kwargs):
                return None

            def _bind_graph_manager(self, *args, **kwargs):
                return None

        dispatcher = PreparedDispatcher(registry, config_manager, schema_loader=None)
        state.dispatcher = dispatcher
        monkeypatch.setattr(agents_router, "_ensure_dispatcher", lambda: dispatcher)
        app = FastAPI()
        app.include_router(agents_router.router, prefix="/agents")
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://runtime"
        ) as client:
            state.client = client
            yield state


async def _process(state: _Runtime, query: str, tenant: str = TENANT) -> dict:
    response = await state.client.post(
        "/agents/orchestrator_agent/process",
        json={
            "agent_name": "orchestrator_agent",
            "query": query,
            "context": {"tenant_id": tenant},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.asyncio
async def test_every_step_failing_is_not_an_answer(runtime):
    result = await _process(runtime, "all-failed")

    assert result["status"] == "failed"
    assert result["message"] == "No orchestration step completed successfully"
    assert "answer" not in result
    output = result["orchestration_result"]["final_output"]
    assert output["status"] == "failed"
    assert output["aggregated_content"] == ""
    assert sorted(output["results"]) == ["first", "second"]
    assert output["results"]["second"] == {
        "status": "error",
        "message": CHILD_ERROR,
    }
    assert runtime.memory_writes == []
    assert runtime.calls == [
        ("first", "all-failed", TENANT),
        ("second", "all-failed", TENANT),
    ]


@pytest.mark.asyncio
async def test_one_step_failing_keeps_the_other_step_s_answer(runtime):
    result = await _process(runtime, "mixed")

    assert result["status"] == "partial"
    assert result["message"] == "Some orchestration steps did not complete successfully"
    assert result["answer"] == "{'status': 'success', 'answer': 'first:mixed'}"
    output = result["orchestration_result"]["final_output"]
    assert output["status"] == "partial"
    assert output["results"]["first"] == {"status": "success", "answer": "first:mixed"}
    assert output["results"]["second"] == {"status": "error", "message": CHILD_ERROR}
    assert runtime.memory_writes == []


@pytest.mark.asyncio
async def test_every_step_succeeding_is_a_success_and_is_remembered(runtime):
    result = await _process(runtime, "fine")

    assert result["status"] == "success"
    assert result["answer"] == (
        "{'status': 'success', 'answer': 'first:fine'}\n\n"
        "{'status': 'success', 'answer': 'second:fine'}"
    )
    assert result["orchestration_result"]["final_output"]["status"] == "success"
    assert runtime.memory_writes == [
        ("fine", "Executed 2/2 steps (2 successful). Plan: two required sources")
    ]


@pytest.mark.asyncio
async def test_a_failing_request_does_not_take_a_concurrent_one_with_it(runtime):
    pending = asyncio.create_task(
        _process(runtime, "held-failure", "prodfixagents:held")
    )
    try:
        await asyncio.wait_for(runtime.entered.wait(), 10)
        succeeded = await asyncio.wait_for(_process(runtime, "fine"), 10)
        assert succeeded["status"] == "success"
        assert runtime.memory_writes == [
            ("fine", "Executed 2/2 steps (2 successful). Plan: two required sources")
        ]
    finally:
        runtime.release.set()

    failed = await asyncio.wait_for(pending, 10)
    assert failed["status"] == "failed"
    assert "answer" not in failed
    assert runtime.memory_writes == [
        ("fine", "Executed 2/2 steps (2 successful). Plan: two required sources")
    ]
    assert sorted(runtime.calls) == [
        ("first", "fine", TENANT),
        ("first", "held-failure", "prodfixagents:held"),
        ("second", "fine", TENANT),
        ("second", "held-failure", "prodfixagents:held"),
    ]


async def _a2a_final_event(runtime, query: str):
    """Drive the A2A executor and return the task's final status event."""
    from a2a.server.events import EventQueue

    from cogniverse_runtime.a2a_executor import CogniverseAgentExecutor

    executor = CogniverseAgentExecutor(runtime.dispatcher)
    queue = EventQueue()
    await executor._execute_non_streaming(
        "orchestrator_agent",
        query,
        {"tenant_id": TENANT},
        10,
        "task-1",
        "context-1",
        queue,
    )
    event = await queue.dequeue_event()
    while not event.final:
        event = await queue.dequeue_event()
    return event


@pytest.mark.asyncio
async def test_a_failed_orchestration_is_a_failed_a2a_task(runtime):
    """No step answered, so the task ends failed and carries no answer."""
    from a2a.types import TaskState
    from a2a.utils import get_message_text

    event = await _a2a_final_event(runtime, "all-failed")

    assert event.status.state is TaskState.failed
    assert json.loads(get_message_text(event.status.message)) == {
        "type": "error",
        "agent": "orchestrator_agent",
        "error_type": "NoAnswerError",
        "message": (
            "Agent 'orchestrator_agent' failed with NoAnswerError. "
            "See runtime logs for detail."
        ),
    }
    assert runtime.memory_writes == []


@pytest.mark.asyncio
async def test_a_partial_orchestration_carries_its_answer_to_a2a(runtime):
    """A partial run answered, so it ends in the state an answered run ends in."""
    from a2a.types import TaskState
    from a2a.utils import get_message_text

    event = await _a2a_final_event(runtime, "mixed")

    assert event.status.state is TaskState.input_required
    payload = json.loads(get_message_text(event.status.message))
    assert payload["status"] == "partial"
    assert payload["orchestration_result"]["final_output"]["status"] == "partial"
    assert payload["answer"] == "{'status': 'success', 'answer': 'first:mixed'}"
    assert runtime.memory_writes == []
