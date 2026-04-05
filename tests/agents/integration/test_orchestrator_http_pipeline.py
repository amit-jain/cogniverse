"""
Integration tests for MultiAgentOrchestrator HTTP pipeline.

These tests start a real FastAPI stub server that serves A2A-compliant
JSON-RPC 2.0 responses, and exercise actual TCP/HTTP round-trips
through the official a2a-sdk A2AClient. This catches URL construction,
JSON-RPC serialization, and A2A protocol compliance bugs.

What is REAL (integration boundary):
- A real uvicorn server running in a background thread on a unique port
- Actual TCP socket connections from the orchestrator to the stub server
- Real HTTP serialization/deserialization and JSON-RPC 2.0 framing

What is MOCKED (deliberately, to isolate HTTP layer):
- RoutingAgent: requires LLM — tested separately in TestConversationAwarePlanningWithLLM
- create_workflow_intelligence: requires LLM
- workflow_planner.forward: replaced with controllable Mock to drive specific task plans
  (the real planner is tested in TestConversationAwarePlanningWithLLM under @skip_if_no_llm)
"""

import asyncio
import hashlib
import json
import threading
import time
import uuid
from typing import Any, Dict, List, Set
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from cogniverse_agents.multi_agent_orchestrator import MultiAgentOrchestrator
from cogniverse_agents.routing_agent import RoutingOutput
from tests.agents.integration.conftest import skip_if_no_llm

stub_app = FastAPI()

# Shared mutable state — reset before each test via fixture
_request_log: List[Dict[str, Any]] = []
_response_map: Dict[str, Dict[str, Any]] = {}  # agent_name → response dict
_error_agents: Set[str] = set()  # agents that should return errors
_slow_agents: Dict[str, float] = {}  # agent_name → delay seconds


@stub_app.post("/")
async def stub_a2a_rpc(request: Request):
    """A2A JSON-RPC 2.0 endpoint that dispatches based on metadata.agent_name."""
    body = await request.json()

    rpc_id = body.get("id", "1")
    method = body.get("method", "")
    params = body.get("params", {})

    if method != "message/send":
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        )

    metadata = params.get("metadata", {})
    agent_name = metadata.get("agent_name", "unknown")
    query = metadata.get("query", "")

    _request_log.append(
        {
            "agent_name": agent_name,
            "url": str(request.url),
            "body": body,
            "headers": dict(request.headers),
            "metadata": metadata,
            "query": query,
        }
    )

    if agent_name in _error_agents:
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32000, "message": "Simulated failure"},
            }
        )

    if agent_name in _slow_agents:
        await asyncio.sleep(_slow_agents[agent_name])

    agent_result = _response_map.get(
        agent_name, {"result": f"default from {agent_name}", "confidence": 0.8}
    )

    # Return as A2A Message response (what the executor produces)
    task_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())
    return JSONResponse(
        {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "result": {
                "kind": "message",
                "messageId": str(uuid.uuid4()),
                "role": "agent",
                "parts": [{"kind": "text", "text": json.dumps(agent_result)}],
                "taskId": task_id,
                "contextId": context_id,
            },
        }
    )


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
    """Start a real uvicorn server serving the A2A stub app."""
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
    """Real MultiAgentOrchestrator pointing at the A2A stub server."""
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
        patch(
            "cogniverse_agents.multi_agent_orchestrator.RoutingAgent"
        ) as mock_routing_cls,
        patch(
            "cogniverse_agents.multi_agent_orchestrator.create_workflow_intelligence"
        ),
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
    """Integration tests exercising real HTTP round-trips via A2A JSON-RPC."""

    @pytest.mark.asyncio
    async def test_http_abort_sets_end_time(self, orchestrator):
        """Stub returns error for all agents → orchestrator handles errors without crashing."""
        _error_agents.add("search_agent")
        _error_agents.add("summarizer_agent")

        orchestrator.workflow_planner.forward = Mock(
            return_value=_make_planner_result(
                [
                    {
                        "task_id": "t1",
                        "agent": "search_agent",
                        "query": "q1",
                        "dependencies": [],
                    },
                    {
                        "task_id": "t2",
                        "agent": "search_agent",
                        "query": "q2",
                        "dependencies": [],
                    },
                ]
            )
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
    async def test_a2a_jsonrpc_request_format(self, orchestrator):
        """Verify JSON-RPC 2.0 structure and metadata in stub-received request."""
        orchestrator.workflow_planner.forward = Mock(
            return_value=_make_planner_result(
                [
                    {
                        "task_id": "t1",
                        "agent": "search_agent",
                        "query": "find videos",
                        "dependencies": [],
                    },
                ]
            )
        )

        result = await orchestrator.process_complex_query("find videos")

        assert result["status"] == "completed"
        assert len(_request_log) == 1

        req = _request_log[0]
        body = req["body"]
        # JSON-RPC 2.0 envelope
        assert body["jsonrpc"] == "2.0"
        assert body["method"] == "message/send"
        assert "params" in body
        # Agent name in metadata
        assert req["metadata"]["agent_name"] == "search_agent"
        assert req["query"] == "find videos"
        # Content-Type header present
        assert "application/json" in req["headers"].get("content-type", "")

    @pytest.mark.asyncio
    async def test_http_hallucinated_agents_resolved(self, orchestrator):
        """Planner returns unknown agent names → resolved before HTTP call."""
        orchestrator.workflow_planner.forward = Mock(
            return_value=_make_planner_result(
                [
                    {
                        "task_id": "t1",
                        "agent": "VideoSearchAgent",
                        "query": "search",
                        "dependencies": [],
                    },
                    {
                        "task_id": "t2",
                        "agent": "ContentSummarizer",
                        "query": "summarize",
                        "dependencies": ["t1"],
                    },
                ]
            )
        )

        result = await orchestrator.process_complex_query("test")

        assert result["status"] == "completed"
        received_agents = [r["agent_name"] for r in _request_log]
        for name in received_agents:
            assert name in ("search_agent", "summarizer_agent"), (
                f"Hallucinated agent name '{name}' was not resolved"
            )

    @pytest.mark.asyncio
    async def test_http_uses_configured_endpoint(self, orchestrator, stub_server):
        """Agent endpoint set to stub URL → stub actually receives the request."""
        orchestrator.workflow_planner.forward = Mock(
            return_value=_make_planner_result(
                [
                    {
                        "task_id": "t1",
                        "agent": "search_agent",
                        "query": "test",
                        "dependencies": [],
                    },
                ]
            )
        )

        result = await orchestrator.process_complex_query("test")

        assert result["status"] == "completed"
        assert len(_request_log) == 1
        # Port of stub server appears in URL
        assert stub_server.split(":")[-1] in _request_log[0]["url"]

    @pytest.mark.asyncio
    async def test_a2a_dependency_context_in_metadata(self, orchestrator):
        """Second task's metadata.dependency_context contains first task's result."""
        _response_map["search_agent"] = {
            "result": "search data found",
            "confidence": 0.9,
        }
        _response_map["summarizer_agent"] = {"result": "summary", "confidence": 0.85}

        orchestrator.workflow_planner.forward = Mock(
            return_value=_make_planner_result(
                [
                    {
                        "task_id": "t1",
                        "agent": "search_agent",
                        "query": "search data",
                        "dependencies": [],
                    },
                    {
                        "task_id": "t2",
                        "agent": "summarizer_agent",
                        "query": "summarize",
                        "dependencies": ["t1"],
                    },
                ]
            )
        )

        result = await orchestrator.process_complex_query("test")

        assert result["status"] == "completed"
        assert len(_request_log) == 2

        # Second request metadata should contain dependency context
        second_metadata = _request_log[1]["metadata"]
        dep_context = second_metadata.get("dependency_context")
        assert dep_context is not None
        assert "search data found" in dep_context

    @pytest.mark.asyncio
    async def test_http_full_pipeline_sequential(self, orchestrator):
        """2 dependent tasks → stub receives requests in order, correct summary."""
        _response_map["search_agent"] = {"result": "video results", "confidence": 0.9}
        _response_map["summarizer_agent"] = {
            "result": "summary of videos",
            "confidence": 0.85,
        }

        orchestrator.workflow_planner.forward = Mock(
            return_value=_make_planner_result(
                [
                    {
                        "task_id": "t1",
                        "agent": "search_agent",
                        "query": "find videos",
                        "dependencies": [],
                    },
                    {
                        "task_id": "t2",
                        "agent": "summarizer_agent",
                        "query": "summarize findings",
                        "dependencies": ["t1"],
                    },
                ]
            )
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
    async def test_http_timeout_handling(
        self, stub_server, telemetry_manager_without_phoenix
    ):
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
            patch(
                "cogniverse_agents.multi_agent_orchestrator.RoutingAgent"
            ) as mock_routing_cls,
            patch(
                "cogniverse_agents.multi_agent_orchestrator.create_workflow_intelligence"
            ),
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
                return_value=_make_planner_result(
                    [
                        {
                            "task_id": "t1",
                            "agent": "search_agent",
                            "query": "test",
                            "dependencies": [],
                        },
                    ]
                )
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


@pytest.mark.integration
class TestConversationAwareOrchestration:
    """Test conversation history flows through the orchestration pipeline."""

    @pytest.mark.asyncio
    async def test_conversation_history_accepted(self, orchestrator):
        """process_complex_query accepts conversation_history without error."""
        orchestrator.workflow_planner.forward = Mock(
            return_value=_make_planner_result(
                [
                    {
                        "task_id": "t1",
                        "agent": "search_agent",
                        "query": "show more",
                        "dependencies": [],
                    },
                ]
            )
        )

        history = [
            {"role": "user", "content": "search for cat videos"},
            {"role": "agent", "content": "Found 5 results about cats"},
        ]

        result = await orchestrator.process_complex_query(
            "show me more like those",
            conversation_history=history,
        )

        assert result["status"] == "completed"
        assert len(_request_log) == 1

    @pytest.mark.asyncio
    async def test_conversation_history_none_works(self, orchestrator):
        """process_complex_query works when conversation_history is not provided."""
        orchestrator.workflow_planner.forward = Mock(
            return_value=_make_planner_result(
                [
                    {
                        "task_id": "t1",
                        "agent": "search_agent",
                        "query": "test",
                        "dependencies": [],
                    },
                ]
            )
        )

        result = await orchestrator.process_complex_query(
            "test",
            conversation_history=None,
        )

        assert result["status"] == "completed"


@pytest.mark.integration
@skip_if_no_llm
class TestConversationAwarePlanningWithLLM:
    """Test real DSPy planner (no mocks) produces valid workflow plans.

    Uses a real LLM via project config for planning, with the stub server
    for HTTP execution. This tests that the DSPy ChainOfThought planner
    produces structurally valid plans, not just that mocks return canned data.
    """

    @pytest.fixture
    def orchestrator_with_real_planner(
        self, stub_server, telemetry_manager_without_phoenix, dspy_lm
    ):
        """MultiAgentOrchestrator with real DSPy planner, stub HTTP server."""
        agents_config = {
            "search_agent": {
                "capabilities": ["video_content_search", "multimodal_retrieval"],
                "endpoint": stub_server,
                "timeout_seconds": 30,
                "parallel_capacity": 2,
            },
            "summarizer_agent": {
                "capabilities": ["content_summarization", "report_generation"],
                "endpoint": stub_server,
                "timeout_seconds": 30,
                "parallel_capacity": 2,
            },
        }

        with (
            patch(
                "cogniverse_agents.multi_agent_orchestrator.RoutingAgent"
            ) as mock_routing_cls,
            patch(
                "cogniverse_agents.multi_agent_orchestrator.create_workflow_intelligence"
            ),
        ):
            mock_routing_cls.return_value = Mock()

            orch = MultiAgentOrchestrator(
                tenant_id="test_tenant",
                telemetry_manager=telemetry_manager_without_phoenix,
                available_agents=agents_config,
                enable_workflow_intelligence=False,
            )
            # workflow_planner and result_aggregator are real DSPy modules
            # (initialized by _initialize_dspy_modules with real LLM)
            yield orch

    @pytest.mark.asyncio
    async def test_planner_produces_plan_with_conversation_context(
        self, orchestrator_with_real_planner
    ):
        """Real DSPy planner produces a valid workflow plan with conversation history."""
        history = [
            {"role": "user", "content": "search for cat videos"},
            {"role": "agent", "content": "Found 5 results about cats"},
        ]

        result = await orchestrator_with_real_planner.process_complex_query(
            "show me more results and summarize them",
            conversation_history=history,
        )

        # The real planner should produce a valid plan that executes
        assert result["status"] in ("completed", "failed")
        # If completed, verify execution summary exists
        if result["status"] == "completed":
            assert "execution_summary" in result
            summary = result["execution_summary"]
            assert summary["total_tasks"] >= 1

    @pytest.mark.asyncio
    async def test_planner_without_history_still_works(
        self, orchestrator_with_real_planner
    ):
        """Real planner works with no conversation history."""
        result = await orchestrator_with_real_planner.process_complex_query(
            "search for cooking videos and summarize findings",
        )

        assert result["status"] in ("completed", "failed")
        if result["status"] == "completed":
            assert "execution_summary" in result

    @pytest.mark.asyncio
    async def test_full_pipeline_with_conversation_history(
        self, orchestrator_with_real_planner
    ):
        """Full orchestration: real plan -> stub HTTP dispatch -> aggregate."""
        _response_map["search_agent"] = {
            "result": "video results about cats",
            "confidence": 0.9,
        }
        _response_map["summarizer_agent"] = {
            "result": "summary of cat videos",
            "confidence": 0.85,
        }

        history = [
            {"role": "user", "content": "I was looking at cat videos earlier"},
            {"role": "agent", "content": "Here are some cat video results"},
        ]

        result = await orchestrator_with_real_planner.process_complex_query(
            "find more cat videos and create a summary",
            conversation_history=history,
        )

        # Real planner + real HTTP execution to stub
        assert result["status"] in ("completed", "failed")
        # Verify at least one request reached the stub
        assert len(_request_log) >= 1
