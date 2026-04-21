"""
Unit tests for agent router endpoints.

Tests the gateway→orchestration handoff via AgentDispatcher, and HTTP-level
round-trip tests for the annotation queue endpoints.
"""

from contextlib import contextmanager
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_agents.routing.annotation_agent import (
    AnnotationPriority,
    AnnotationRequest,
    AnnotationStatus,
)
from cogniverse_agents.routing.annotation_queue import AnnotationQueue
from cogniverse_evaluation.evaluators.routing_evaluator import RoutingOutcome
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import agents as agents_router


@pytest.fixture
def mock_telemetry_manager():
    """Create mock telemetry manager."""
    manager = MagicMock()

    @contextmanager
    def fake_span(*args, **kwargs):
        yield MagicMock()

    manager.span = fake_span
    return manager


@pytest.fixture
def dispatcher():
    """Create an AgentDispatcher with mock dependencies."""
    registry = MagicMock()
    config_manager = MagicMock()
    schema_loader = MagicMock()
    return AgentDispatcher(
        agent_registry=registry,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )


def _make_gateway_output(*, complexity="simple", modality="video",
                          generation_type="raw_results", routed_to="search_agent",
                          confidence=0.9):
    """Build a mock GatewayOutput for tests."""
    output = Mock()
    output.complexity = complexity
    output.modality = modality
    output.generation_type = generation_type
    output.routed_to = routed_to
    output.confidence = confidence
    output.reasoning = "test reasoning"
    return output


@pytest.mark.unit
class TestGatewayOrchestrationHandoff:
    """Test that AgentDispatcher routes through GatewayAgent for triage."""

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_simple_query_routes_to_downstream(self, dispatcher):
        """Simple query via GatewayAgent dispatches directly to execution agent."""
        gateway_output = _make_gateway_output(
            complexity="simple", routed_to="search_agent", modality="video",
        )

        # Registry: gateway_agent has ["gateway"],
        # search_agent has ["search"] for downstream
        gateway_ep = MagicMock()
        gateway_ep.capabilities = ["gateway"]
        search_ep = MagicMock()
        search_ep.capabilities = ["search"]

        def get_agent_by_name(name):
            if name == "gateway_agent":
                return gateway_ep
            if name == "search_agent":
                return search_ep
            return None

        dispatcher._registry.get_agent.side_effect = get_agent_by_name

        mock_downstream = {
            "status": "success",
            "agent": "search_agent",
            "message": "Found 3 results",
            "results_count": 3,
            "results": [],
            "profile": "test_profile",
        }

        with (
            patch(
                "cogniverse_agents.gateway_agent.GatewayAgent._process_impl",
                new_callable=AsyncMock,
                return_value=gateway_output,
            ),
            patch(
                "cogniverse_agents.gateway_agent.GatewayAgent.__init__",
                return_value=None,
            ),
            patch.object(
                dispatcher,
                "_execute_downstream_agent",
                new_callable=AsyncMock,
                return_value=mock_downstream,
            ),
        ):
            result = await dispatcher.dispatch(
                agent_name="gateway_agent",
                query="find videos of cats",
                context={"tenant_id": "test_tenant"},
            )

        assert result["status"] == "success"
        assert result["agent"] == "gateway_agent"
        assert result["gateway"]["complexity"] == "simple"
        assert result["gateway"]["routed_to"] == "search_agent"
        assert result["downstream_result"] == mock_downstream

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_complex_query_routes_to_orchestrator(self, dispatcher):
        """Complex query via GatewayAgent forwards to OrchestratorAgent."""
        gateway_output = _make_gateway_output(
            complexity="complex", routed_to="orchestrator_agent",
            modality="both", confidence=0.4,
        )

        gateway_ep = MagicMock()
        gateway_ep.capabilities = ["gateway"]
        dispatcher._registry.get_agent.return_value = gateway_ep

        mock_orch_result = {
            "status": "success",
            "agent": "orchestrator_agent",
            "message": "Orchestrated 'find robots' via A2A pipeline",
            "orchestration_result": {"workflow_id": "wf_test"},
            "gateway_context": {
                "modality": "both",
                "generation_type": "raw_results",
                "confidence": 0.4,
            },
        }

        with (
            patch(
                "cogniverse_agents.gateway_agent.GatewayAgent._process_impl",
                new_callable=AsyncMock,
                return_value=gateway_output,
            ),
            patch(
                "cogniverse_agents.gateway_agent.GatewayAgent.__init__",
                return_value=None,
            ),
            patch.object(
                dispatcher,
                "_execute_orchestration_task",
                new_callable=AsyncMock,
                return_value=mock_orch_result,
            ),
        ):
            result = await dispatcher.dispatch(
                agent_name="gateway_agent",
                query="find robots then summarize and create report",
                context={"tenant_id": "test_tenant"},
            )

        assert result["status"] == "success"
        assert result["agent"] == "orchestrator_agent"
        assert result["gateway_context"]["modality"] == "both"

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_routing_capability_triggers_gateway(self, dispatcher):
        """Agent with 'routing' capability also routes through gateway (backward compat)."""
        routing_ep = MagicMock()
        routing_ep.capabilities = ["routing"]
        dispatcher._registry.get_agent.return_value = routing_ep

        mock_gateway_result = {
            "status": "success",
            "agent": "gateway_agent",
            "message": "Routed 'test' to search_agent (simple)",
            "gateway": {"complexity": "simple"},
            "downstream_result": {},
        }

        with patch.object(
            dispatcher,
            "_execute_gateway_task",
            new_callable=AsyncMock,
            return_value=mock_gateway_result,
        ) as mock_gw:
            result = await dispatcher.dispatch(
                agent_name="routing_agent",
                query="test",
                context={"tenant_id": "t1"},
            )

        mock_gw.assert_called_once()
        assert result["status"] == "success"

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_orchestration_capability_dispatches_directly(self, dispatcher):
        """Agent with 'orchestration' capability dispatches to orchestration task."""
        orch_ep = MagicMock()
        orch_ep.capabilities = ["orchestration"]
        dispatcher._registry.get_agent.return_value = orch_ep

        mock_orch_result = {
            "status": "success",
            "agent": "orchestrator_agent",
            "message": "Orchestrated 'complex q' via A2A pipeline",
            "orchestration_result": {},
            "gateway_context": None,
        }

        with patch.object(
            dispatcher,
            "_execute_orchestration_task",
            new_callable=AsyncMock,
            return_value=mock_orch_result,
        ) as mock_orch:
            result = await dispatcher.dispatch(
                agent_name="orchestrator_agent",
                query="complex q",
                context={"tenant_id": "t1"},
            )

        mock_orch.assert_called_once()
        assert result["agent"] == "orchestrator_agent"


@pytest.mark.unit
class TestAgentDispatcherCapabilityRouting:
    """Test that dispatch routes to the correct _execute_* method by capability."""

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_dispatch_unknown_agent_raises(self, dispatcher):
        """Unknown agent name raises ValueError."""
        dispatcher._registry.get_agent.return_value = None

        with pytest.raises(ValueError, match="not found"):
            await dispatcher.dispatch(agent_name="nonexistent", query="test")

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_dispatch_unregistered_agent_raises(self, dispatcher):
        """Agent not in AGENT_CLASSES falls through to generic dispatch and raises."""
        agent_ep = MagicMock()
        agent_ep.capabilities = ["unknown_capability"]
        dispatcher._registry.get_agent.return_value = agent_ep

        with pytest.raises(ValueError, match="no supported execution path"):
            await dispatcher.dispatch(
                agent_name="weird_agent",
                query="test",
                context={"tenant_id": "test:unit"},
            )

    @pytest.mark.asyncio
    @pytest.mark.ci_fast
    async def test_dispatch_search_capability(self, dispatcher):
        """Agent with 'search' capability routes to search handler."""
        agent_ep = MagicMock()
        agent_ep.capabilities = ["search"]
        dispatcher._registry.get_agent.return_value = agent_ep

        with patch.object(
            dispatcher,
            "_execute_search_task",
            new_callable=AsyncMock,
            return_value={"status": "success", "agent": "search_agent"},
        ) as mock_search:
            result = await dispatcher.dispatch(
                agent_name="search_agent",
                query="find cats",
                context={"tenant_id": "t1"},
            )

        mock_search.assert_called_once_with(
            "find cats", "t1", 10, conversation_history=[]
        )
        assert result["status"] == "success"


@pytest.mark.unit
class TestNoMultiAgentOrchestrator:
    """Verify that MultiAgentOrchestrator is no longer used in agent_dispatcher."""

    @pytest.mark.ci_fast
    def test_no_multi_agent_orchestrator_import(self):
        """agent_dispatcher must not reference MultiAgentOrchestrator anywhere."""
        import inspect

        source = inspect.getsource(AgentDispatcher)
        assert "MultiAgentOrchestrator" not in source, (
            "MultiAgentOrchestrator is replaced by OrchestratorAgent. "
            "Remove all references from agent_dispatcher."
        )

    @pytest.mark.ci_fast
    def test_no_get_optimizer_calls(self):
        """agent_dispatcher must not call _get_optimizer (removed from thin RoutingAgent)."""
        import inspect

        source = inspect.getsource(AgentDispatcher)
        assert "_get_optimizer" not in source
        assert "get_routing_statistics" not in source


# ── Annotation Queue HTTP endpoints ──────────────────────────────────────


def _make_annotation_request(
    span_id: str = "span-http-1",
    priority: AnnotationPriority = AnnotationPriority.MEDIUM,
) -> AnnotationRequest:
    return AnnotationRequest(
        span_id=span_id,
        timestamp=datetime.now(),
        query="http test query",
        chosen_agent="search_agent",
        routing_confidence=0.5,
        outcome=RoutingOutcome.AMBIGUOUS,
        priority=priority,
        reason="http test",
        context={},
    )


@pytest.fixture
def annotation_client():
    """
    TestClient with agents router mounted and a fresh AnnotationQueue injected.

    Overrides the module-level _annotation_queue singleton so each test
    gets an isolated queue — no cross-test state leakage.
    """
    test_app = FastAPI()
    test_app.include_router(agents_router.router, prefix="/agents")

    fresh_queue = AnnotationQueue()
    # Patch the module-level singleton directly for the duration of the test
    original = agents_router._annotation_queue
    agents_router._annotation_queue = fresh_queue
    try:
        with TestClient(test_app) as client:
            yield client, fresh_queue
    finally:
        agents_router._annotation_queue = original


@pytest.mark.unit
@pytest.mark.ci_fast
class TestAnnotationQueueEndpoints:
    """Round-trip HTTP tests for the annotation queue API."""

    def test_get_empty_queue(self, annotation_client):
        """GET /agents/annotations/queue on empty queue returns zero statistics."""
        client, _ = annotation_client
        resp = client.get("/agents/annotations/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert data["statistics"]["total"] == 0
        assert data["pending"] == []
        assert data["assigned"] == []
        assert data["expired"] == []

    def test_get_queue_shows_pending_items(self, annotation_client):
        """GET /agents/annotations/queue reflects items enqueued directly in queue."""
        client, queue = annotation_client
        queue.enqueue(_make_annotation_request("span-a"))
        queue.enqueue(_make_annotation_request("span-b", AnnotationPriority.HIGH))

        resp = client.get("/agents/annotations/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert data["statistics"]["total"] == 2
        assert data["statistics"]["by_status"]["pending"] == 2
        # HIGH priority item must appear first in the sorted pending list
        assert data["pending"][0]["span_id"] == "span-b"
        assert data["pending"][1]["span_id"] == "span-a"

    def test_assign_endpoint_round_trip(self, annotation_client):
        """POST /agents/annotations/queue/{span_id}/assign transitions PENDING→ASSIGNED."""
        client, queue = annotation_client
        queue.enqueue(_make_annotation_request("span-assign"))

        resp = client.post(
            "/agents/annotations/queue/span-assign/assign",
            json={"reviewer": "alice", "sla_hours": 8},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "assigned"
        ann = data["annotation"]
        assert ann["span_id"] == "span-assign"
        assert ann["status"] == "assigned"
        assert ann["assigned_to"] == "alice"
        assert ann["assigned_at"] is not None
        assert ann["sla_deadline"] is not None

        # Verify queue state is actually updated
        assert queue.get("span-assign").status == AnnotationStatus.ASSIGNED

    def test_assign_missing_span_returns_404(self, annotation_client):
        """POST assign on unknown span_id returns 404."""
        client, _ = annotation_client
        resp = client.post(
            "/agents/annotations/queue/nonexistent/assign",
            json={"reviewer": "bob"},
        )
        assert resp.status_code == 404

    def test_assign_already_assigned_returns_400(self, annotation_client):
        """POST assign on already-ASSIGNED span returns 400."""
        client, queue = annotation_client
        queue.enqueue(_make_annotation_request("span-dup"))
        queue.assign("span-dup", reviewer="alice")

        resp = client.post(
            "/agents/annotations/queue/span-dup/assign",
            json={"reviewer": "bob"},
        )
        assert resp.status_code == 400

    def test_complete_endpoint_round_trip(self, annotation_client):
        """POST complete transitions ASSIGNED→COMPLETED and persists label."""
        client, queue = annotation_client
        queue.enqueue(_make_annotation_request("span-complete"))
        queue.assign("span-complete", reviewer="alice")

        resp = client.post(
            "/agents/annotations/queue/span-complete/complete",
            json={"label": "correct_routing"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        ann = data["annotation"]
        assert ann["status"] == "completed"
        assert ann["completed_at"] is not None

        # Verify queue state is actually updated
        assert queue.get("span-complete").status == AnnotationStatus.COMPLETED

    def test_complete_missing_span_returns_404(self, annotation_client):
        """POST complete on unknown span_id returns 404."""
        client, _ = annotation_client
        resp = client.post(
            "/agents/annotations/queue/ghost/complete",
            json={},
        )
        assert resp.status_code == 404

    def test_complete_already_completed_returns_400(self, annotation_client):
        """POST complete on already-COMPLETED span returns 400."""
        client, queue = annotation_client
        queue.enqueue(_make_annotation_request("span-done"))
        queue.complete("span-done")

        resp = client.post(
            "/agents/annotations/queue/span-done/complete",
            json={},
        )
        assert resp.status_code == 400

    def test_full_lifecycle_via_http(self, annotation_client):
        """Full PENDING→ASSIGNED→COMPLETED lifecycle exercised through HTTP."""
        client, queue = annotation_client
        queue.enqueue(_make_annotation_request("span-lifecycle", AnnotationPriority.HIGH))

        # Step 1: Verify appears in queue as PENDING
        resp = client.get("/agents/annotations/queue")
        assert resp.status_code == 200
        assert len(resp.json()["pending"]) == 1

        # Step 2: Assign via HTTP
        resp = client.post(
            "/agents/annotations/queue/span-lifecycle/assign",
            json={"reviewer": "reviewer1"},
        )
        assert resp.status_code == 200
        assert resp.json()["annotation"]["status"] == "assigned"

        # Step 3: Verify moved to assigned in GET response
        resp = client.get("/agents/annotations/queue")
        data = resp.json()
        assert len(data["pending"]) == 0
        assert len(data["assigned"]) == 1

        # Step 4: Complete via HTTP
        resp = client.post(
            "/agents/annotations/queue/span-lifecycle/complete",
            json={"label": "correct"},
        )
        assert resp.status_code == 200
        assert resp.json()["annotation"]["status"] == "completed"

        # Step 5: Verify no longer in pending/assigned
        resp = client.get("/agents/annotations/queue")
        data = resp.json()
        assert len(data["pending"]) == 0
        assert len(data["assigned"]) == 0
        assert data["statistics"]["total"] == 1
        assert data["statistics"]["by_status"]["completed"] == 1
