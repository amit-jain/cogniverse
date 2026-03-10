"""
Integration tests for MultiAgentOrchestrator HTTP pipeline.

These tests start a real FastAPI stub server and exercise actual TCP/HTTP
round-trips through httpx — no httpx mocking. This catches URL construction,
request serialization, HTTP error handling, and timeout bugs that mock-based
chain tests cannot detect.
"""

import asyncio
import hashlib
import threading
import time
from typing import Any, Dict, List, Set
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from cogniverse_agents.multi_agent_orchestrator import MultiAgentOrchestrator
from cogniverse_agents.routing_agent import RoutingOutput

stub_app = FastAPI()

# Shared mutable state — reset before each test via fixture
_request_log: List[Dict[str, Any]] = []
_response_map: Dict[str, Dict[str, Any]] = {}  # agent_name → response dict
_error_agents: Set[str] = set()  # agents that should return 500
_slow_agents: Dict[str, float] = {}  # agent_name → delay seconds


@stub_app.post("/agents/{agent_name}/process")
async def stub_process(agent_name: str, request: Request):
    body = await request.json()
    _request_log.append({
        "agent_name": agent_name,
        "url": str(request.url),
        "body": body,
        "headers": dict(request.headers),
    })

    if agent_name in _error_agents:
        raise HTTPException(status_code=500, detail="Simulated failure")

    if agent_name in _slow_agents:
        await asyncio.sleep(_slow_agents[agent_name])

    response = _response_map.get(
        agent_name, {"result": f"default from {agent_name}", "confidence": 0.8}
    )
    return response


def _generate_stub_port(module_name: str) -> int:
    """Deterministic port in 9100-9199 range based on module name hash."""
    port_hash = int(hashlib.md5(module_name.encode()).hexdigest()[:4], 16)
    return 9100 + (port_hash % 100)


def _wait_for_server(url: str, timeout: float = 10.0):
    """Poll until the stub server accepts connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx.get(f"{url}/docs", timeout=1.0)
            return
        except (httpx.ConnectError, httpx.TimeoutException):
            time.sleep(0.1)
    raise RuntimeError(f"Stub server at {url} did not start within {timeout}s")


@pytest.fixture(scope="module")
def stub_server():
    """Start a real uvicorn server serving the stub FastAPI app."""
    port = _generate_stub_port(__name__)
    config = uvicorn.Config(stub_app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for_server(f"http://127.0.0.1:{port}")
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(autouse=True)
def _reset_stub_state():
    """Clear stub server state before each test."""
    _request_log.clear()
    _response_map.clear()
    _error_agents.clear()
    _slow_agents.clear()


def _make_planner_result(tasks_data, strategy="sequential"):
    """Create a Mock mimicking DSPy workflow_planner.forward() output."""
    result = Mock()
    result.workflow_tasks = tasks_data
    result.execution_strategy = strategy
    result.expected_outcome = "results"
    result.reasoning = "test"
    return result


@pytest.fixture
def orchestrator(stub_server, telemetry_manager_without_phoenix):
    """Real MultiAgentOrchestrator pointing at the stub server.

    httpx is NOT patched — real TCP connections to the stub server.
    """
    agents_config = {
        "search_agent": {
            "capabilities": ["video_content_search", "multimodal_retrieval"],
            "endpoint": stub_server,
            "timeout_seconds": 10,
            "parallel_capacity": 2,
        },
        "summarizer_agent": {
            "capabilities": ["content_summarization", "report_generation"],
            "endpoint": stub_server,
            "timeout_seconds": 10,
            "parallel_capacity": 2,
        },
    }

    with (
        patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent") as mock_routing_cls,
        patch("cogniverse_agents.multi_agent_orchestrator.A2AClient"),
        patch("cogniverse_agents.multi_agent_orchestrator.create_workflow_intelligence"),
    ):
        mock_routing_instance = Mock()
        mock_routing_instance.route_query = AsyncMock()
        mock_routing_cls.return_value = mock_routing_instance

        orch = MultiAgentOrchestrator(
            tenant_id="test_tenant",
            telemetry_manager=telemetry_manager_without_phoenix,
            available_agents=agents_config,
            enable_workflow_intelligence=False,
        )

        # Replace DSPy modules with controllable mocks (no LLM)
        orch.workflow_planner = Mock()
        orch.result_aggregator = Mock()

        yield orch


@pytest.mark.integration
class TestOrchestratorHttpPipeline:
    """Integration tests exercising real HTTP round-trips to a stub server."""

    @pytest.mark.asyncio
    async def test_http_abort_sets_end_time(self, orchestrator):
        """Stub returns 500 for all agents → orchestrator handles errors without crashing."""
        _error_agents.add("search_agent")
        _error_agents.add("summarizer_agent")

        orchestrator.workflow_planner.forward = Mock(
            return_value=_make_planner_result([
                {"task_id": "t1", "agent": "search_agent", "query": "q1", "dependencies": []},
                {"task_id": "t2", "agent": "search_agent", "query": "q2", "dependencies": []},
            ])
        )

        # Zero retries to avoid backoff delays
        original_plan = orchestrator._plan_workflow

        async def plan_then_zero_retries(*args, **kwargs):
            plan = await original_plan(*args, **kwargs)
            for task in plan.tasks:
                task.max_retries = 0
            return plan

        orchestrator._plan_workflow = plan_then_zero_retries

        result = await orchestrator.process_complex_query("test")

        assert result["status"] == "failed"
        # Stub actually received the requests over TCP
        assert len(_request_log) >= 1

    @pytest.mark.asyncio
    async def test_http_fallback_passes_tenant_id(self, orchestrator):
        """When planning fails, fallback calls route_query with tenant_id."""
        orchestrator.workflow_planner.forward = Mock(
            side_effect=RuntimeError("planning failed")
        )

        mock_routing_output = RoutingOutput(
            query="test",
            recommended_agent="search_agent",
            confidence=0.8,
            reasoning="fallback",
            enhanced_query="test",
        )
        orchestrator.routing_agent.route_query = AsyncMock(
            return_value=mock_routing_output
        )

        result = await orchestrator.process_complex_query("test")

        assert result["status"] == "failed"
        assert "fallback_result" in result
        orchestrator.routing_agent.route_query.assert_called_once()
        call_kwargs = orchestrator.routing_agent.route_query.call_args
        assert call_kwargs.kwargs.get("tenant_id") == "test_tenant" or (
            len(call_kwargs.args) >= 3 and call_kwargs.args[2] == "test_tenant"
        )

    @pytest.mark.asyncio
    async def test_http_request_format(self, orchestrator):
        """Verify URL, headers, and JSON body structure received by stub server."""
        orchestrator.workflow_planner.forward = Mock(
            return_value=_make_planner_result([
                {"task_id": "t1", "agent": "search_agent", "query": "find videos", "dependencies": []},
            ])
        )

        result = await orchestrator.process_complex_query("find videos")

        assert result["status"] == "completed"
        assert len(_request_log) == 1

        req = _request_log[0]
        # URL contains agent name in path
        assert "/agents/search_agent/process" in req["url"]
        # JSON body has required keys
        assert req["body"]["agent_name"] == "search_agent"
        assert "query" in req["body"]
        assert "context" in req["body"]
        # Content-Type header present
        assert "application/json" in req["headers"].get("content-type", "")

    @pytest.mark.asyncio
    async def test_http_hallucinated_agents_resolved(self, orchestrator):
        """Planner returns unknown agent names → resolved before HTTP call."""
        orchestrator.workflow_planner.forward = Mock(
            return_value=_make_planner_result([
                {"task_id": "t1", "agent": "VideoSearchAgent", "query": "search", "dependencies": []},
                {"task_id": "t2", "agent": "ContentSummarizer", "query": "summarize", "dependencies": ["t1"]},
            ])
        )

        result = await orchestrator.process_complex_query("test")

        assert result["status"] == "completed"
        # Verify the stub received resolved names, not hallucinated ones
        received_agents = [r["agent_name"] for r in _request_log]
        for name in received_agents:
            assert name in ("search_agent", "summarizer_agent"), (
                f"Hallucinated agent name '{name}' was not resolved"
            )

    @pytest.mark.asyncio
    async def test_http_post_optimization_validates_agents(
        self, stub_server, telemetry_manager_without_phoenix
    ):
        """After workflow intelligence corrupts agent names, they are re-resolved."""
        agents_config = {
            "search_agent": {
                "capabilities": ["video_content_search"],
                "endpoint": stub_server,
                "timeout_seconds": 10,
                "parallel_capacity": 1,
            },
        }

        with (
            patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent") as mock_routing_cls,
            patch("cogniverse_agents.multi_agent_orchestrator.A2AClient"),
            patch("cogniverse_agents.multi_agent_orchestrator.create_workflow_intelligence") as mock_wi_factory,
        ):
            mock_routing_cls.return_value = Mock()

            mock_wi = AsyncMock()

            async def corrupt_plan(query, plan, ctx):
                for task in plan.tasks:
                    task.agent_name = "InvalidAgent"
                return plan

            mock_wi.optimize_workflow_plan = AsyncMock(side_effect=corrupt_plan)
            mock_wi_factory.return_value = mock_wi

            orch = MultiAgentOrchestrator(
                tenant_id="test_tenant",
                telemetry_manager=telemetry_manager_without_phoenix,
                available_agents=agents_config,
                enable_workflow_intelligence=True,
            )

            orch.workflow_planner = Mock()
            orch.workflow_planner.forward = Mock(
                return_value=_make_planner_result([
                    {"task_id": "t1", "agent": "search_agent", "query": "test", "dependencies": []},
                ])
            )

            result = await orch.process_complex_query("test")

        assert result["status"] == "completed"
        # InvalidAgent should NOT appear in any request
        for req in _request_log:
            assert "InvalidAgent" not in req["url"]
            assert req["agent_name"] != "InvalidAgent"

    @pytest.mark.asyncio
    async def test_http_uses_configured_endpoint(self, orchestrator, stub_server):
        """Agent endpoint set to stub URL → stub actually receives the request."""
        orchestrator.workflow_planner.forward = Mock(
            return_value=_make_planner_result([
                {"task_id": "t1", "agent": "search_agent", "query": "test", "dependencies": []},
            ])
        )

        result = await orchestrator.process_complex_query("test")

        assert result["status"] == "completed"
        assert len(_request_log) == 1
        # URL starts with the stub server's base URL
        assert _request_log[0]["url"].startswith(stub_server.replace("http://", ""))  or \
            stub_server.split(":")[-1] in _request_log[0]["url"]

    @pytest.mark.asyncio
    async def test_http_dependency_context_serialization(self, orchestrator):
        """Second task's context.dependency_context contains first task's result as string."""
        _response_map["search_agent"] = {"result": "search data found", "confidence": 0.9}
        _response_map["summarizer_agent"] = {"result": "summary", "confidence": 0.85}

        orchestrator.workflow_planner.forward = Mock(
            return_value=_make_planner_result([
                {"task_id": "t1", "agent": "search_agent", "query": "search data", "dependencies": []},
                {"task_id": "t2", "agent": "summarizer_agent", "query": "summarize", "dependencies": ["t1"]},
            ])
        )

        result = await orchestrator.process_complex_query("test")

        assert result["status"] == "completed"
        assert len(_request_log) == 2

        # Second request should contain dependency context from first result
        second_body = _request_log[1]["body"]
        dep_context = second_body["context"].get("dependency_context")
        assert dep_context is not None
        assert isinstance(dep_context, str), f"Expected str, got {type(dep_context)}"
        assert "search data found" in dep_context

    @pytest.mark.asyncio
    async def test_http_full_pipeline_sequential(self, orchestrator):
        """2 dependent tasks → stub receives requests in order, correct summary."""
        _response_map["search_agent"] = {"result": "video results", "confidence": 0.9}
        _response_map["summarizer_agent"] = {"result": "summary of videos", "confidence": 0.85}

        orchestrator.workflow_planner.forward = Mock(
            return_value=_make_planner_result([
                {"task_id": "t1", "agent": "search_agent", "query": "find videos", "dependencies": []},
                {"task_id": "t2", "agent": "summarizer_agent", "query": "summarize findings", "dependencies": ["t1"]},
            ])
        )

        result = await orchestrator.process_complex_query("find videos and summarize")

        assert result["status"] == "completed"
        summary = result["execution_summary"]
        assert summary["total_tasks"] == 2
        assert summary["completed_tasks"] == 2
        assert summary["execution_time"] > 0
        assert "search_agent" in summary["agents_used"]
        assert "summarizer_agent" in summary["agents_used"]

        # Verify request ordering
        assert len(_request_log) == 2
        assert _request_log[0]["agent_name"] == "search_agent"
        assert _request_log[1]["agent_name"] == "summarizer_agent"

    @pytest.mark.asyncio
    async def test_http_timeout_handling(self, stub_server, telemetry_manager_without_phoenix):
        """Stub delays response beyond task timeout → orchestrator handles gracefully."""
        _slow_agents["search_agent"] = 5.0  # 5 second delay

        agents_config = {
            "search_agent": {
                "capabilities": ["video_content_search"],
                "endpoint": stub_server,
                "timeout_seconds": 1,  # 1 second timeout — will expire before response
                "parallel_capacity": 1,
            },
        }

        with (
            patch("cogniverse_agents.multi_agent_orchestrator.RoutingAgent") as mock_routing_cls,
            patch("cogniverse_agents.multi_agent_orchestrator.A2AClient"),
            patch("cogniverse_agents.multi_agent_orchestrator.create_workflow_intelligence"),
        ):
            mock_routing_cls.return_value = Mock()

            orch = MultiAgentOrchestrator(
                tenant_id="test_tenant",
                telemetry_manager=telemetry_manager_without_phoenix,
                available_agents=agents_config,
                enable_workflow_intelligence=False,
            )

            orch.workflow_planner = Mock()
            orch.workflow_planner.forward = Mock(
                return_value=_make_planner_result([
                    {"task_id": "t1", "agent": "search_agent", "query": "test", "dependencies": []},
                ])
            )

            # Zero retries to avoid waiting
            original_plan = orch._plan_workflow

            async def plan_then_zero_retries(*args, **kwargs):
                plan = await original_plan(*args, **kwargs)
                for task in plan.tasks:
                    task.max_retries = 0
                return plan

            orch._plan_workflow = plan_then_zero_retries

            result = await orch.process_complex_query("test")

        # Should fail gracefully, not crash
        assert result["status"] == "failed"
        # The stub did receive the request (TCP connection was made)
        assert len(_request_log) >= 1
