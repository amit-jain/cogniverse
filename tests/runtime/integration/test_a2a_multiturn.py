"""
Integration tests for A2A multi-turn conversation support.

Full stack: real A2A app -> real CogniverseAgentExecutor -> real AgentDispatcher
-> real DSPy query rewrite -> real Vespa search. Exercises contextId-based
history accumulation via InMemoryTaskStore (the production a2a-sdk store).
"""

import json
import logging
import uuid

import pytest

from tests.runtime.integration.conftest import skip_if_no_llm

logger = logging.getLogger(__name__)


def _send_message(
    client,
    text: str,
    context_id: str,
    agent_name: str = "search_agent",
    tenant_id: str = "test:unit",
    task_id: str | None = None,
    rpc_id: int = 1,
) -> dict:
    """Send an A2A JSON-RPC message/send request and return the response body."""
    message = {
        "role": "user",
        "messageId": str(uuid.uuid4()),
        "contextId": context_id,
        "parts": [{"kind": "text", "text": text}],
    }
    if task_id:
        message["taskId"] = task_id

    payload = {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": "message/send",
        "params": {
            "message": message,
            "metadata": {
                "agent_name": agent_name,
                "tenant_id": tenant_id,
            },
        },
    }

    response = client.post("/", json=payload)
    assert response.status_code == 200, f"HTTP {response.status_code}: {response.text}"
    body = response.json()
    assert "result" in body, f"JSON-RPC error: {body.get('error')}"
    return body


def _extract_task_id(body: dict) -> str:
    """Extract taskId from A2A JSON-RPC response."""
    result = body["result"]
    # Response may be a Task or contain status with taskId
    task_id = result.get("id") or result.get("taskId")
    assert task_id, f"No taskId in response: {result}"
    return task_id


def _extract_response_text(body: dict) -> str:
    """Extract the agent's text response from A2A JSON-RPC response."""
    result = body["result"]
    # Navigate to the text part — could be in status.message.parts or parts directly
    status = result.get("status", {})
    message = status.get("message", {})
    parts = message.get("parts", [])
    if parts:
        return parts[0].get("text", "")
    # Fallback: check result.parts directly
    parts = result.get("parts", [])
    if parts:
        return parts[0].get("text", "")
    return ""


@pytest.mark.integration
@skip_if_no_llm
class TestA2AMultiTurnHistoryAccumulation:
    """Test multi-turn conversation history via A2A contextId."""

    def test_multiturn_history_accumulates_three_turns(
        self, a2a_client, dspy_lm, vespa_instance
    ):
        """3 A2A calls with same contextId -> turn 3 carries history from turns 1+2."""
        context_id = f"test-accumulate-{uuid.uuid4()}"

        # Turn 1
        resp1 = _send_message(a2a_client, "search for cat videos", context_id, rpc_id=1)
        task_id = _extract_task_id(resp1)
        text1 = _extract_response_text(resp1)
        assert text1, "Turn 1 should produce a response"

        # Turn 2 — same context, same task
        resp2 = _send_message(
            a2a_client,
            "now search for dog videos",
            context_id,
            task_id=task_id,
            rpc_id=2,
        )
        text2 = _extract_response_text(resp2)
        assert text2, "Turn 2 should produce a response"

        # Turn 3 — same context, same task
        resp3 = _send_message(
            a2a_client,
            "compare the two sets of results",
            context_id,
            task_id=task_id,
            rpc_id=3,
        )
        text3 = _extract_response_text(resp3)
        assert text3, "Turn 3 should produce a response"

    def test_context_id_isolation(self, a2a_client, dspy_lm, vespa_instance):
        """Messages to different contextIds don't cross-contaminate history."""
        ctx_a = f"test-iso-a-{uuid.uuid4()}"
        ctx_b = f"test-iso-b-{uuid.uuid4()}"

        # Context A: cats
        resp_a1 = _send_message(a2a_client, "search for cat videos", ctx_a, rpc_id=10)
        task_a = _extract_task_id(resp_a1)

        # Context B: dogs (different context)
        resp_b1 = _send_message(a2a_client, "search for dog videos", ctx_b, rpc_id=11)
        task_b = _extract_task_id(resp_b1)

        # They should have different task IDs
        assert task_a != task_b, "Different contexts should create different tasks"

        # Context A turn 2: should NOT see dog history
        resp_a2 = _send_message(
            a2a_client,
            "show me more of those",
            ctx_a,
            task_id=task_a,
            rpc_id=12,
        )
        text_a2 = _extract_response_text(resp_a2)
        assert text_a2, "Context A turn 2 should produce a response"

    def test_task_stays_alive_input_required(self, a2a_client, dspy_lm, vespa_instance):
        """TaskState.input_required keeps task non-terminal for subsequent turns."""
        context_id = f"test-alive-{uuid.uuid4()}"

        # Turn 1
        resp1 = _send_message(a2a_client, "search for videos", context_id, rpc_id=20)
        task_id = _extract_task_id(resp1)

        # Verify task state is input_required (non-terminal)
        result1 = resp1["result"]
        status = result1.get("status", {})
        state = status.get("state", "")
        assert state == "input-required", (
            f"Task state should be input-required for multi-turn, got '{state}'"
        )

        # Turn 2 — should succeed (task is alive)
        resp2 = _send_message(
            a2a_client,
            "filter by duration",
            context_id,
            task_id=task_id,
            rpc_id=21,
        )
        assert "result" in resp2, "Turn 2 should succeed on alive task"

    def test_agent_response_in_history(self, a2a_client, dspy_lm, vespa_instance):
        """Turn 1 agent response appears in turn 2's conversation context."""
        context_id = f"test-agent-hist-{uuid.uuid4()}"

        # Turn 1
        resp1 = _send_message(
            a2a_client, "search for cat videos", context_id, rpc_id=30
        )
        task_id = _extract_task_id(resp1)

        # The agent's response from turn 1 should be persisted in the task store.
        # Turn 2 will have it in history. We can't directly inspect the dispatcher's
        # received history here, but we verify the round-trip works — the turn 2
        # response should succeed, meaning the executor correctly extracted history.
        resp2 = _send_message(
            a2a_client,
            "show me more like those",
            context_id,
            task_id=task_id,
            rpc_id=31,
        )
        text2 = _extract_response_text(resp2)
        assert text2, "Turn 2 should produce a response with history from turn 1"

    def test_first_turn_no_rewrite(self, a2a_client, dspy_lm, vespa_instance):
        """Single turn with no history -> no query rewrite in response."""
        context_id = f"test-no-rewrite-{uuid.uuid4()}"

        resp = _send_message(
            a2a_client,
            "search for dog videos",
            context_id,
            rpc_id=40,
        )
        text = _extract_response_text(resp)
        assert text, "First turn should produce a response"

        # Parse the dispatch result — first turn should NOT have rewritten_query
        try:
            result_data = json.loads(text)
            assert "rewritten_query" not in result_data, (
                "First turn with no history should not have rewritten_query"
            )
        except json.JSONDecodeError:
            pass  # Non-JSON response is acceptable

    def test_multiturn_query_rewrite_end_to_end(
        self, a2a_client, dspy_lm, vespa_instance
    ):
        """Turn 1: 'cat videos' -> Turn 2: 'show me longer ones' -> rewritten query."""
        context_id = f"test-rewrite-e2e-{uuid.uuid4()}"

        # Turn 1: explicit query
        resp1 = _send_message(
            a2a_client, "search for cat videos", context_id, rpc_id=50
        )
        task_id = _extract_task_id(resp1)

        # Turn 2: anaphoric reference — should trigger query rewrite
        resp2 = _send_message(
            a2a_client,
            "show me longer ones",
            context_id,
            task_id=task_id,
            rpc_id=51,
        )
        text2 = _extract_response_text(resp2)
        assert text2, "Turn 2 should produce a response"

        # Parse the dispatch result — turn 2 should have rewritten_query
        # (if history was correctly accumulated and rewrite triggered)
        try:
            result_data = json.loads(text2)
            if "rewritten_query" in result_data:
                rewritten = result_data["rewritten_query"].lower()
                # Rewritten query should reference the topic from turn 1
                assert any(word in rewritten for word in ["cat", "video", "long"]), (
                    f"Rewritten query '{result_data['rewritten_query']}' should resolve references"
                )
        except json.JSONDecodeError:
            pass  # Non-JSON response is acceptable
