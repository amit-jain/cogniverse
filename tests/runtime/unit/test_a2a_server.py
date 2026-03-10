"""
Unit tests for the A2A protocol server mounted at /a2a.

Tests agent card discovery and message/send JSON-RPC round-trip
using a minimal in-process test setup (no real agent backends).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from starlette.testclient import TestClient

from cogniverse_runtime.a2a_executor import CogniverseAgentExecutor
from cogniverse_runtime.agent_dispatcher import AgentDispatcher


@pytest.fixture
def mock_dispatcher():
    """AgentDispatcher with mocked internals."""
    registry = MagicMock()
    config_manager = MagicMock()
    schema_loader = MagicMock()
    dispatcher = AgentDispatcher(
        agent_registry=registry,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )
    return dispatcher


@pytest.fixture
def a2a_app(mock_dispatcher):
    """Build a standalone A2A Starlette app for testing."""
    executor = CogniverseAgentExecutor(dispatcher=mock_dispatcher)

    card = AgentCard(
        name="Test Cogniverse",
        description="Test agent",
        url="http://localhost:9999/a2a",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[
            AgentSkill(
                id="search_agent",
                name="search_agent",
                description="Search for videos",
                tags=["search", "video_search"],
            ),
        ],
    )

    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=card,
        http_handler=handler,
    )
    return server.build()


@pytest.fixture
def client(a2a_app):
    return TestClient(a2a_app)


@pytest.mark.unit
class TestA2AAgentCard:
    """Test agent card discovery endpoint."""

    @pytest.mark.ci_fast
    def test_agent_card_returns_valid_json(self, client):
        """GET /.well-known/agent-card.json returns the agent card."""
        response = client.get("/.well-known/agent-card.json")
        assert response.status_code == 200

        card = response.json()
        assert card["name"] == "Test Cogniverse"
        assert card["version"] == "1.0.0"
        assert len(card["skills"]) == 1
        assert card["skills"][0]["id"] == "search_agent"

    @pytest.mark.ci_fast
    def test_agent_card_has_required_fields(self, client):
        """Agent card contains all A2A-required fields."""
        response = client.get("/.well-known/agent-card.json")
        card = response.json()

        required_fields = [
            "name", "description", "url", "version",
            "defaultInputModes", "defaultOutputModes",
            "capabilities", "skills",
        ]
        for field in required_fields:
            assert field in card, f"Missing required field: {field}"


@pytest.mark.unit
class TestA2AMessageSend:
    """Test JSON-RPC 2.0 message/send endpoint."""

    @pytest.mark.ci_fast
    def test_message_send_dispatches_to_agent(self, client, mock_dispatcher):
        """POST / with message/send dispatches to the correct agent."""
        # Mock dispatcher.dispatch to return a search result
        mock_dispatcher.dispatch = AsyncMock(
            return_value={
                "status": "success",
                "agent": "search_agent",
                "message": "Found 3 results",
                "results_count": 3,
                "results": [],
            }
        )

        # Configure registry to find the agent
        agent_ep = MagicMock()
        agent_ep.capabilities = ["search"]
        mock_dispatcher._registry.get_agent.return_value = agent_ep

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": "test-msg-1",
                    "parts": [
                        {"kind": "text", "text": "search for videos about cats"}
                    ],
                },
                "metadata": {
                    "agent_name": "search_agent",
                    "tenant_id": "test_tenant",
                },
            },
        }

        response = client.post("/", json=payload)
        assert response.status_code == 200

        body = response.json()
        assert body["jsonrpc"] == "2.0"
        assert body["id"] == 1
        # Result should contain the task/message with our dispatch result
        assert "result" in body
        result = body["result"]

        # The result is either a Task or a Message — check it has content
        assert result is not None

        # Verify dispatcher was called with correct args
        mock_dispatcher.dispatch.assert_called_once()
        call_kwargs = mock_dispatcher.dispatch.call_args
        assert call_kwargs.kwargs["agent_name"] == "search_agent"
        assert call_kwargs.kwargs["context"]["tenant_id"] == "test_tenant"

    @pytest.mark.ci_fast
    def test_message_send_without_agent_name_defaults_to_routing(
        self, client, mock_dispatcher
    ):
        """When no agent_name in metadata, defaults to routing_agent."""
        mock_dispatcher.dispatch = AsyncMock(
            return_value={
                "status": "success",
                "agent": "routing_agent",
                "recommended_agent": "search_agent",
            }
        )

        agent_ep = MagicMock()
        agent_ep.capabilities = ["routing"]
        mock_dispatcher._registry.get_agent.return_value = agent_ep

        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": "test-msg-2",
                    "parts": [
                        {"kind": "text", "text": "find videos about dogs"}
                    ],
                },
            },
        }

        response = client.post("/", json=payload)
        assert response.status_code == 200

        call_kwargs = mock_dispatcher.dispatch.call_args
        assert call_kwargs.kwargs["agent_name"] == "routing_agent"

    @pytest.mark.ci_fast
    def test_message_send_error_returns_error_text(self, client, mock_dispatcher):
        """When dispatch raises, executor returns error as text message."""
        mock_dispatcher.dispatch = AsyncMock(
            side_effect=ValueError("Agent 'bad_agent' not found in registry")
        )

        payload = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": "test-msg-3",
                    "parts": [
                        {"kind": "text", "text": "test"}
                    ],
                },
                "metadata": {
                    "agent_name": "bad_agent",
                },
            },
        }

        response = client.post("/", json=payload)
        assert response.status_code == 200

        body = response.json()
        assert "result" in body
        # The error should be in the response text
        result = body["result"]
        assert result is not None

    @pytest.mark.ci_fast
    def test_invalid_jsonrpc_method_returns_error(self, client):
        """Unknown JSON-RPC method returns error response."""
        payload = {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "nonexistent/method",
            "params": {},
        }

        response = client.post("/", json=payload)
        body = response.json()
        # Should return a JSON-RPC error for unknown method
        assert "error" in body
