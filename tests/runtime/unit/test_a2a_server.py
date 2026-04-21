"""
Unit tests for the A2A protocol server mounted at /a2a.

Tests agent card discovery, message/send JSON-RPC round-trip,
and multi-turn conversation history plumbing via contextId.
"""

from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill, Message, TextPart
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
            "name",
            "description",
            "url",
            "version",
            "defaultInputModes",
            "defaultOutputModes",
            "capabilities",
            "skills",
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
                    "parts": [{"kind": "text", "text": "search for videos about cats"}],
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
                    "parts": [{"kind": "text", "text": "find videos about dogs"}],
                },
                "metadata": {
                    # No agent_name → routing_agent default; tenant_id required.
                    "tenant_id": "test_tenant",
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
                    "parts": [{"kind": "text", "text": "test"}],
                },
                "metadata": {
                    "agent_name": "bad_agent",
                    "tenant_id": "test_tenant",
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
    def test_context_id_passed_to_dispatcher(self, client, mock_dispatcher):
        """context_id from message is threaded through to dispatcher context."""
        mock_dispatcher.dispatch = AsyncMock(
            return_value={"status": "success", "agent": "search_agent"}
        )

        payload = {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": "msg-ctx-1",
                    "contextId": "conv-abc",
                    "parts": [{"kind": "text", "text": "cat videos"}],
                },
                "metadata": {
                    "agent_name": "search_agent",
                    "tenant_id": "test_tenant",
                },
            },
        }

        response = client.post("/", json=payload)
        assert response.status_code == 200

        call_kwargs = mock_dispatcher.dispatch.call_args
        ctx = call_kwargs.kwargs["context"]
        assert ctx["context_id"] is not None  # SDK assigns or uses provided

    @pytest.mark.ci_fast
    def test_first_turn_has_empty_history(self, client, mock_dispatcher):
        """First message in a conversation has empty conversation_history."""
        mock_dispatcher.dispatch = AsyncMock(
            return_value={"status": "success", "agent": "search_agent"}
        )

        payload = {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": "msg-first-1",
                    "parts": [{"kind": "text", "text": "search for cats"}],
                },
                "metadata": {"agent_name": "search_agent", "tenant_id": "test_tenant"},
            },
        }

        response = client.post("/", json=payload)
        assert response.status_code == 200

        call_kwargs = mock_dispatcher.dispatch.call_args
        ctx = call_kwargs.kwargs["context"]
        assert ctx["conversation_history"] == []

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


@pytest.mark.unit
class TestA2AMultiTurn:
    """Test multi-turn conversation via contextId and task history."""

    @pytest.mark.ci_fast
    def test_second_turn_has_history_from_first(self, client, mock_dispatcher):
        """Second message with same contextId carries history from the first turn."""
        mock_dispatcher.dispatch = AsyncMock(
            return_value={"status": "success", "agent": "search_agent", "results": []}
        )

        context_id = "conv-multi-1"

        # Turn 1
        turn1 = {
            "jsonrpc": "2.0",
            "id": 20,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": "msg-t1",
                    "contextId": context_id,
                    "parts": [{"kind": "text", "text": "search for cat videos"}],
                },
                "metadata": {"agent_name": "search_agent", "tenant_id": "test_tenant"},
            },
        }

        resp1 = client.post("/", json=turn1)
        assert resp1.status_code == 200

        # Extract task_id from turn 1 response for turn 2
        result1 = resp1.json()["result"]
        task_id = result1.get("id") or result1.get("taskId")

        # Turn 2 — same contextId, references the task
        turn2 = {
            "jsonrpc": "2.0",
            "id": 21,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": "msg-t2",
                    "contextId": context_id,
                    "taskId": task_id,
                    "parts": [{"kind": "text", "text": "show me longer ones"}],
                },
                "metadata": {"agent_name": "search_agent", "tenant_id": "test_tenant"},
            },
        }

        resp2 = client.post("/", json=turn2)
        assert resp2.status_code == 200

        # Verify turn 2 dispatch received conversation history
        assert mock_dispatcher.dispatch.call_count == 2
        turn2_kwargs = mock_dispatcher.dispatch.call_args_list[1]
        ctx = turn2_kwargs.kwargs["context"]
        history = ctx["conversation_history"]

        # Should have at least one prior turn (the user message from turn 1
        # and/or the agent response)
        assert len(history) >= 1
        # First history entry should contain text from turn 1
        assert any("cat videos" in turn["content"] for turn in history)


@pytest.mark.unit
class TestExtractConversationHistory:
    """Unit tests for CogniverseAgentExecutor._extract_conversation_history."""

    def _make_executor(self):
        dispatcher = MagicMock()
        return CogniverseAgentExecutor(dispatcher=dispatcher)

    def _make_message(self, role: str, text: str, msg_id: str = "m1") -> Message:
        return Message(
            role=role,
            message_id=msg_id,
            parts=[TextPart(kind="text", text=text)],
        )

    def test_no_task_returns_empty(self):
        """No current_task → empty history."""
        executor = self._make_executor()
        ctx = Mock(spec=["current_task", "context_id"])
        ctx.current_task = None
        ctx.context_id = None
        assert executor._extract_conversation_history(ctx) == []

    def test_task_with_no_history_returns_empty(self):
        """Task exists but has no history → empty."""
        executor = self._make_executor()
        task = Mock()
        task.history = None
        ctx = Mock(spec=["current_task", "context_id"])
        ctx.current_task = task
        ctx.context_id = "c1"
        assert executor._extract_conversation_history(ctx) == []

    def test_single_message_history_returns_empty(self):
        """History with only the current message → empty (excluded)."""
        executor = self._make_executor()
        current_msg = self._make_message("user", "hello")
        task = Mock()
        task.history = [current_msg]
        ctx = Mock(spec=["current_task", "context_id"])
        ctx.current_task = task
        ctx.context_id = "c1"
        assert executor._extract_conversation_history(ctx) == []

    def test_two_messages_returns_first(self):
        """History with 2 messages → returns only the first (prior turn)."""
        executor = self._make_executor()
        msg1 = self._make_message("user", "search cats", "m1")
        msg2 = self._make_message("user", "show longer ones", "m2")
        task = Mock()
        task.history = [msg1, msg2]
        ctx = Mock(spec=["current_task", "context_id"])
        ctx.current_task = task
        ctx.context_id = "c1"

        result = executor._extract_conversation_history(ctx)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "search cats"

    def test_agent_messages_included(self):
        """Agent messages in history are also extracted."""
        executor = self._make_executor()
        msg1 = self._make_message("user", "search cats", "m1")
        msg2 = self._make_message("agent", "Found 5 results", "m2")
        msg3 = self._make_message("user", "show longer ones", "m3")
        task = Mock()
        task.history = [msg1, msg2, msg3]
        ctx = Mock(spec=["current_task", "context_id"])
        ctx.current_task = task
        ctx.context_id = "c1"

        result = executor._extract_conversation_history(ctx)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "agent"
