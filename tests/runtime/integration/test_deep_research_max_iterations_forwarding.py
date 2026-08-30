"""``context["max_iterations"]`` reaches ``DeepResearchInput`` through the
runtime dispatch path.

``AgentDispatcher._execute_deep_research_task`` builds the ``DeepResearchInput``
the deep research agent runs on. The forwarding tests mount the real agents
router on the real dispatcher and record the typed input the agent receives;
the real-LM tests let the run continue through the real LM, encoder and
Vespa and pin what the research loop guarantees for the forwarded bound.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_agents.deep_research_agent import (
    DeepResearchAgent,
    DeepResearchInput,
    DeepResearchOutput,
)
from cogniverse_core.registries.agent_registry import AgentEndpoint, AgentRegistry
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import agents
from tests.runtime.integration.conftest import skip_if_no_lm

pytestmark = pytest.mark.integration

TENANT_ID = "test:unit"
MAX_ITERATIONS_DEFAULT = DeepResearchInput.model_fields["max_iterations"].default


@pytest.fixture(scope="module")
def deep_research_dispatcher(config_manager, schema_loader):
    registry = AgentRegistry(tenant_id=TENANT_ID, config_manager=config_manager)
    registry.register_agent(
        AgentEndpoint(
            name="deep_research_agent",
            url="http://localhost:8010",
            capabilities=["deep_research", "analysis"],
            health_endpoint="/health",
        )
    )
    return AgentDispatcher(
        agent_registry=registry,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )


@pytest.fixture
def route_client(deep_research_dispatcher):
    agents._dispatcher = deep_research_dispatcher
    app = FastAPI()
    app.include_router(agents.router, prefix="/agents")
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    agents._dispatcher = None


@pytest.fixture
def recorded_inputs(monkeypatch):
    """Replace the research loop with a recorder of the typed input it was
    handed; the dispatcher, router and agent construction stay real."""
    captured: list[DeepResearchInput] = []

    async def _record(self, input_data):
        captured.append(input_data)
        return DeepResearchOutput(summary="recorded")

    monkeypatch.setattr(DeepResearchAgent, "_process_impl", _record)
    return captured


@pytest.fixture
def recorded_live_inputs(monkeypatch):
    """Record the typed input and call through to the real research loop."""
    captured: list[DeepResearchInput] = []
    original = DeepResearchAgent.process

    async def _record_then_run(self, input, stream=False):
        captured.append(input)
        return await original(self, input, stream)

    monkeypatch.setattr(DeepResearchAgent, "process", _record_then_run)
    return captured


def _post(client: TestClient, query: str, context: dict):
    return client.post(
        "/agents/deep_research_agent/process",
        json={"agent_name": "deep_research_agent", "query": query, "context": context},
    )


class TestMaxIterationsForwarding:
    def test_request_context_bound_reaches_input(self, route_client, recorded_inputs):
        query = "what changed in checkout latency after the outage?"
        resp = _post(route_client, query, {"tenant_id": TENANT_ID, "max_iterations": 1})

        assert resp.status_code == 200, resp.text
        assert len(recorded_inputs) == 1
        received = recorded_inputs[0]
        assert received.max_iterations == 1
        assert received.query == query
        assert received.tenant_id == TENANT_ID
        assert received.rlm is None

        body = resp.json()
        assert body["status"] == "success"
        assert body["agent"] == "deep_research_agent"
        assert body["message"] == f"Research complete for '{query}'"
        assert body["result"]["summary"] == "recorded"

    def test_absent_bound_uses_input_default(self, route_client, recorded_inputs):
        resp = _post(route_client, "no bound supplied", {"tenant_id": TENANT_ID})

        assert resp.status_code == 200, resp.text
        assert len(recorded_inputs) == 1
        assert recorded_inputs[0].max_iterations == MAX_ITERATIONS_DEFAULT

    @pytest.mark.asyncio
    async def test_direct_dispatch_forwards_bound(
        self, deep_research_dispatcher, recorded_inputs
    ):
        result = await deep_research_dispatcher.dispatch(
            agent_name="deep_research_agent",
            query="direct dispatch",
            context={"tenant_id": TENANT_ID, "max_iterations": 2},
        )

        assert len(recorded_inputs) == 1
        assert recorded_inputs[0].max_iterations == 2
        assert result["status"] == "success"
        assert result["result"]["summary"] == "recorded"


@skip_if_no_lm
class TestMaxIterationsLive:
    """Real LM, real Tomoro query encoder, real Vespa behind the dispatcher's
    own ``_execute_search_task`` search leg."""

    @pytest.mark.asyncio
    async def test_bound_of_one_runs_exactly_one_iteration(
        self,
        deep_research_dispatcher,
        tomoro_search_url,
        dspy_lm_planning,
        vespa_instance,
        recorded_live_inputs,
    ):
        # The research loop advances ``iteration`` before it can stop, and a
        # run that never iterates raises instead of returning, so a returned
        # result with bound 1 always reports exactly one iteration.
        result = await deep_research_dispatcher.dispatch(
            agent_name="deep_research_agent",
            query="What visual patterns appear in outdoor activity videos?",
            context={"tenant_id": TENANT_ID, "max_iterations": 1},
        )

        assert len(recorded_live_inputs) == 1
        assert recorded_live_inputs[0].max_iterations == 1
        assert result["status"] == "success"
        assert result["agent"] == "deep_research_agent"
        assert result["result"]["iterations_used"] == 1

    @pytest.mark.asyncio
    async def test_absent_bound_runs_on_input_default(
        self,
        deep_research_dispatcher,
        tomoro_search_url,
        dspy_lm_planning,
        vespa_instance,
        recorded_live_inputs,
    ):
        # ``iterations_used`` is not pinned here: with the default bound the
        # evaluator LM decides whether the loop stops early.
        result = await deep_research_dispatcher.dispatch(
            agent_name="deep_research_agent",
            query="What visual patterns appear in outdoor activity videos?",
            context={"tenant_id": TENANT_ID},
        )

        assert len(recorded_live_inputs) == 1
        assert recorded_live_inputs[0].max_iterations == MAX_ITERATIONS_DEFAULT
        assert result["status"] == "success"
        assert result["agent"] == "deep_research_agent"
