"""synthesis_depth field reaches OrchestratorInput end-to-end.

synthesis_depth unreachable: ``synthesis_depth`` was a field on
OrchestratorInput but no upstream caller (gateway, A2A,
``_execute_orchestration_task``) ever copied it into the input. Deep
synthesis was reachable only by direct in-process construction.

This test mounts the real agents router on a TestClient, registers
a stub orchestrator that records the OrchestratorInput it receives,
and asserts:

  * a top-level ``synthesis_depth: "deep"`` on the AgentTask body
    arrives in the dispatched OrchestratorInput;
  * a value nested under ``context.synthesis_depth`` also arrives
    (admin / programmatic callers can use either shape);
  * omitting the field leaves OrchestratorInput.synthesis_depth as
    None (legacy callers unchanged);
  * a gateway-classified shape (``context.gateway_context.synthesis_depth``)
    wins over a loose top-level entry — gateway is the trusted classifier
    and a manual override should not weaken its decision.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.registries.agent_registry import (
    AgentEndpoint,
    AgentRegistry,
)
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import agents

pytestmark = pytest.mark.integration


@pytest.fixture
def captured_inputs():
    """Mutable bucket the patched OrchestratorAgent._process_impl writes to."""
    return []


@pytest.fixture
def synthesis_client(captured_inputs):
    """TestClient with a stub orchestrator that records its dispatched input."""
    cm = create_default_config_manager()
    registry = AgentRegistry(tenant_id="h10_test", config_manager=cm)
    registry.register_agent(
        AgentEndpoint(
            name="orchestrator_agent",
            url="http://localhost:8011",
            capabilities=["orchestration"],
            health_endpoint="/health",
        )
    )

    dispatcher = AgentDispatcher(
        agent_registry=registry,
        config_manager=cm,
        schema_loader=None,
    )
    agents._dispatcher = dispatcher  # bypass _ensure_dispatcher's bootstrap

    async def _stub_process_impl(self, input_data):
        # Capture the typed input the dispatcher built and return a
        # minimal valid OrchestratorOutput so the endpoint can
        # serialize without errors.
        from cogniverse_agents.orchestrator_agent import OrchestratorOutput

        captured_inputs.append(input_data)
        return OrchestratorOutput(
            query=input_data.query,
            workflow_id="stub",
            plan_steps=[],
            plan_reasoning="stub",
            agent_results={},
            final_output={"answer": "stub"},
        )

    # Patch the orchestrator's _process_impl so we don't need a real
    # DeepSynthesisWorkflow run for this forwarding test. Memory init
    # also gets stubbed out — the orchestrator's tenant-aware memory
    # bootstrap requires a denseon endpoint that isn't part of this
    # test's scope.
    with (
        patch(
            "cogniverse_agents.orchestrator_agent.OrchestratorAgent._process_impl",
            new=_stub_process_impl,
        ),
        patch(
            "cogniverse_runtime.agent_dispatcher.AgentDispatcher._init_agent_memory",
            new=lambda *a, **kw: None,
        ),
        patch(
            "cogniverse_agents.orchestrator_agent.OrchestratorAgent._load_artifact",
            new=lambda self: None,
        ),
    ):
        app = FastAPI()
        app.include_router(agents.router, prefix="/agents")
        yield TestClient(app, raise_server_exceptions=False)

    agents._dispatcher = None


def _post(client: TestClient, body: dict):
    return client.post("/agents/orchestrator_agent/process", json=body)


class TestSynthesisDepthForwarding:
    def test_top_level_field_reaches_orchestrator_input(
        self, synthesis_client, captured_inputs
    ):
        resp = _post(
            synthesis_client,
            {
                "agent_name": "orchestrator_agent",
                "query": "deep audit query",
                "context": {"tenant_id": "h10_test"},
                "synthesis_depth": "deep",
            },
        )
        assert resp.status_code == 200, resp.text
        assert captured_inputs, "stubbed orchestrator was never invoked"
        assert captured_inputs[-1].synthesis_depth == "deep", (
            "top-level synthesis_depth on AgentTask must propagate to "
            f"OrchestratorInput; got {captured_inputs[-1].synthesis_depth!r}"
        )

    def test_nested_context_field_reaches_orchestrator_input(
        self, synthesis_client, captured_inputs
    ):
        resp = _post(
            synthesis_client,
            {
                "agent_name": "orchestrator_agent",
                "query": "nested context",
                "context": {
                    "tenant_id": "h10_test",
                    "synthesis_depth": "deep",
                },
            },
        )
        assert resp.status_code == 200, resp.text
        assert captured_inputs[-1].synthesis_depth == "deep", (
            f"context['synthesis_depth'] should reach OrchestratorInput; "
            f"got {captured_inputs[-1].synthesis_depth!r}"
        )

    def test_default_omission_leaves_field_none(
        self, synthesis_client, captured_inputs
    ):
        resp = _post(
            synthesis_client,
            {
                "agent_name": "orchestrator_agent",
                "query": "no flag",
                "context": {"tenant_id": "h10_test"},
            },
        )
        assert resp.status_code == 200, resp.text
        assert captured_inputs[-1].synthesis_depth is None, (
            "absent synthesis_depth must stay None on OrchestratorInput; "
            f"got {captured_inputs[-1].synthesis_depth!r}"
        )

    def test_gateway_context_wins_over_loose_context(
        self, synthesis_client, captured_inputs
    ):
        # Gateway already classified this as deep; an out-of-band loose
        # context["synthesis_depth"]="standard" must NOT override the
        # trusted gateway classifier.
        resp = _post(
            synthesis_client,
            {
                "agent_name": "orchestrator_agent",
                "query": "gateway-classified",
                "context": {
                    "tenant_id": "h10_test",
                    "synthesis_depth": "standard",
                    "gateway_context": {"synthesis_depth": "deep"},
                },
            },
        )
        assert resp.status_code == 200, resp.text
        assert captured_inputs[-1].synthesis_depth == "deep", (
            "gateway_context.synthesis_depth must win over a loose "
            f"context entry; got {captured_inputs[-1].synthesis_depth!r}"
        )
