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

from tests.runtime.integration.conftest import skip_if_no_lm

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


@pytest.fixture
def dispatch_history_spy(dispatcher, monkeypatch):
    """Record the ``conversation_history`` each ``dispatch()`` call receives.

    Wraps the real ``AgentDispatcher.dispatch`` on the same module-scoped
    instance the ``a2a_client`` drives, so a test can assert what history the
    dispatcher actually saw on each turn — the contract the docstrings claim —
    rather than only that a response came back. Returns the list of captured
    histories (one entry per dispatch, in call order). monkeypatch restores
    the original method after the test so the shared instance is left intact.
    """
    captured: list[dict] = []
    original = dispatcher.dispatch

    async def _recording_dispatch(*args, **kwargs):
        context = kwargs.get("context")
        if context is None and len(args) >= 3:
            context = args[2]
        captured.append(
            {
                "query": kwargs.get("query") or (args[1] if len(args) >= 2 else None),
                "history": list((context or {}).get("conversation_history", [])),
            }
        )
        return await original(*args, **kwargs)

    monkeypatch.setattr(dispatcher, "dispatch", _recording_dispatch)
    return captured


@pytest.mark.integration
@skip_if_no_lm
class TestA2AMultiTurnHistoryAccumulation:
    """Test multi-turn conversation history via A2A contextId."""

    def test_multiturn_history_accumulates_three_turns(
        self, a2a_client, dspy_lm, vespa_instance, dispatch_history_spy
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

        # Contract: the dispatcher must SEE accumulating history, not just
        # return a response. One dispatch per turn, in order.
        assert len(dispatch_history_spy) == 3
        turn1_hist = dispatch_history_spy[0]["history"]
        turn2_hist = dispatch_history_spy[1]["history"]
        turn3_hist = dispatch_history_spy[2]["history"]

        # Turn 1 has no prior conversation.
        assert turn1_hist == []

        # Turn 2 carries turn 1: the user's "cat videos" query AND the agent's
        # turn-1 response (role=agent), proving both directions are persisted.
        turn2_contents = [t["content"] for t in turn2_hist]
        assert any("cat videos" in c for c in turn2_contents), turn2_contents
        assert any(t["role"] == "agent" for t in turn2_hist), turn2_hist
        assert not any("dog videos" in c for c in turn2_contents), turn2_contents

        # Turn 3 carries BOTH prior user turns (1 and 2), proving accumulation
        # across more than the immediately-preceding turn.
        turn3_contents = [t["content"] for t in turn3_hist]
        assert any("cat videos" in c for c in turn3_contents), turn3_contents
        assert any("dog videos" in c for c in turn3_contents), turn3_contents

    def test_context_id_isolation(
        self, a2a_client, dspy_lm, vespa_instance, dispatch_history_spy
    ):
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

        # Contract: context A turn 2's dispatch must carry context A's history
        # ("cat videos") and must NOT contain anything from context B ("dog
        # videos"). Identify it by its query rather than call order.
        a2_dispatch = next(
            d for d in dispatch_history_spy if d["query"] == "show me more of those"
        )
        a2_contents = [t["content"] for t in a2_dispatch["history"]]
        assert any("cat videos" in c for c in a2_contents), a2_contents
        assert not any("dog videos" in c for c in a2_contents), a2_contents

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

    def test_agent_response_in_history(
        self, a2a_client, dspy_lm, vespa_instance, dispatch_history_spy
    ):
        """Turn 1 agent response appears in turn 2's conversation context."""
        context_id = f"test-agent-hist-{uuid.uuid4()}"

        # Turn 1
        resp1 = _send_message(
            a2a_client, "search for cat videos", context_id, rpc_id=30
        )
        task_id = _extract_task_id(resp1)

        resp2 = _send_message(
            a2a_client,
            "show me more like those",
            context_id,
            task_id=task_id,
            rpc_id=31,
        )
        text2 = _extract_response_text(resp2)
        assert text2, "Turn 2 should produce a response with history from turn 1"

        # Contract: turn 2's dispatch history must include turn 1's USER query
        # AND the AGENT response (role=agent) — i.e. both halves of turn 1 are
        # extracted from Task.history and threaded into turn 2's context.
        t2_dispatch = next(
            d for d in dispatch_history_spy if d["query"] == "show me more like those"
        )
        roles = [t["role"] for t in t2_dispatch["history"]]
        contents = [t["content"] for t in t2_dispatch["history"]]
        assert "user" in roles, roles
        assert "agent" in roles, roles
        assert any("cat videos" in c for c in contents), contents

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

        result_data = json.loads(text)
        assert "rewritten_query" not in result_data, (
            "First turn with no history should not have rewritten_query"
        )

    def test_multiturn_query_rewrite_end_to_end(
        self,
        a2a_client,
        dspy_lm,
        vespa_instance,
        dispatch_history_spy,
        tomoro_search_url,
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

        # Turn 2's dispatch must have received non-empty history — that's what
        # makes the dispatcher run the rewrite path.
        t2_dispatch = next(
            d for d in dispatch_history_spy if d["query"] == "show me longer ones"
        )
        assert t2_dispatch["history"], "Turn 2 should dispatch with prior history"

        # Contract (agent_dispatcher.dispatch): when history is present the
        # response carries BOTH original_query and rewritten_query — no
        # conditional, no swallowed JSON error. original is the raw turn-2
        # query; rewritten is a non-empty resolved string.
        result_data = json.loads(text2)
        assert result_data["original_query"] == "show me longer ones"
        assert "rewritten_query" in result_data, result_data
        assert (
            isinstance(result_data["rewritten_query"], str)
            and result_data["rewritten_query"].strip()
        ), result_data["rewritten_query"]
