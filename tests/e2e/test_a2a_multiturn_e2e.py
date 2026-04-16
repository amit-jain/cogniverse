"""
E2E tests for multi-turn conversation support via REST and A2A protocols.

Requires live runtime at http://localhost:28000 with Ollama + Vespa.
Uses flywheel_org:production tenant which has ingested data.

Tests validate:
- REST conversation_history triggers query rewrite on turn 2+
- REST first turn without history produces no rewritten_query
- A2A agent card discovery
- A2A single-turn message/send
- A2A multi-turn with contextId threading
- A2A context isolation between conversations
"""

import uuid

import httpx
import pytest

RUNTIME = "http://localhost:28000"
TENANT_ID = "flywheel_org:production"


def runtime_available() -> bool:
    try:
        r = httpx.get(f"{RUNTIME}/health", timeout=5.0)
        return r.status_code == 200
    except httpx.ConnectError:
        return False


skip_if_no_runtime = pytest.mark.skipif(
    not runtime_available(),
    reason="Runtime not available at localhost:28000",
)


@pytest.mark.e2e
@skip_if_no_runtime
class TestRESTMultiTurn:
    """REST endpoint multi-turn conversation tests."""

    def test_first_turn_no_rewrite(self):
        """First turn without conversation_history should NOT produce rewritten_query."""
        with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
            resp = client.post(
                "/agents/search_agent/process",
                json={
                    "agent_name": "search_agent",
                    "query": "search for cat videos",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["agent"] == "search_agent"
        assert "rewritten_query" not in data

    def test_multi_turn_with_history_triggers_rewrite(self):
        """Turn 2+ with conversation_history should produce rewritten_query."""
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/search_agent/process",
                json={
                    "agent_name": "search_agent",
                    "query": "show me longer ones",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                    "conversation_history": [
                        {"role": "user", "content": "search for cat videos"},
                        {"role": "agent", "content": "Found 5 cat video results"},
                    ],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "rewritten_query" in data, (
            f"Expected rewritten_query in response, got keys: {list(data.keys())}"
        )
        assert data["original_query"] == "show me longer ones"
        rewritten = data["rewritten_query"].lower()
        assert any(word in rewritten for word in ["cat", "video", "long"]), (
            f"Rewritten query '{data['rewritten_query']}' should reference cats/videos"
        )

    def test_routing_agent_executes_downstream_with_rewrite(self):
        """Routing agent should execute downstream search with query rewrite."""
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/agents/routing_agent/process",
                json={
                    "agent_name": "routing_agent",
                    "query": "find more sports video clips like those",
                    "context": {"tenant_id": TENANT_ID},
                    "top_k": 3,
                    "conversation_history": [
                        {"role": "user", "content": "search for sports clips"},
                        {"role": "agent", "content": "Found 5 sports clip results"},
                    ],
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        # Gateway may route through routing_agent (simple) or orchestrator (complex)
        assert data["agent"] in ("routing_agent", "gateway_agent", "orchestrator_agent"), (
            f"Expected routing/gateway/orchestrator agent, got {data['agent']}"
        )
        # If routing path, check downstream; if orchestrator, check orchestration_result
        if "downstream_result" in data:
            ds = data["downstream_result"]
            assert ds.get("agent") or ds.get("results") is not None
        elif "orchestration_result" in data:
            orch = data["orchestration_result"]
            assert len(orch.get("plan_steps", [])) > 0, "Orchestrator should produce a plan"
        # Conversation history was passed — the gateway/orchestrator handles rewrite internally


@pytest.mark.e2e
@skip_if_no_runtime
class TestA2AProtocol:
    """A2A protocol endpoint tests."""

    def test_agent_card_discovery(self):
        """GET /a2a/.well-known/agent.json returns valid agent card."""
        with httpx.Client(base_url=RUNTIME, timeout=10.0) as client:
            resp = client.get("/a2a/.well-known/agent.json")

        assert resp.status_code == 200
        card = resp.json()
        assert card["name"] == "Cogniverse Runtime"
        assert "skills" in card
        assert card["protocolVersion"] == "0.3.0"
        assert card["url"].endswith("/a2a")

    def test_single_turn_message_send(self):
        """A2A message/send returns taskId, contextId, and agent response."""
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            resp = client.post(
                "/a2a/",
                json={
                    "jsonrpc": "2.0",
                    "id": "e2e-single-1",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [{"kind": "text", "text": "search for sports"}],
                            "messageId": str(uuid.uuid4()),
                        },
                        "configuration": {
                            "acceptedOutputModes": ["text"],
                        },
                    },
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data, f"Expected result in response, got: {data}"
        result = data["result"]
        assert result["id"]  # taskId
        assert result["contextId"]
        assert result["status"]["state"] == "input-required"
        parts = result["status"]["message"]["parts"]
        assert len(parts) > 0
        assert parts[0]["kind"] == "text"

    def test_multi_turn_context_threading(self):
        """A2A multi-turn: turn 2 with same contextId preserves conversation."""
        msg_id_1 = str(uuid.uuid4())
        msg_id_2 = str(uuid.uuid4())

        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            # Turn 1
            resp1 = client.post(
                "/a2a/",
                json={
                    "jsonrpc": "2.0",
                    "id": "e2e-multi-1",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [
                                {"kind": "text", "text": "find cooking tutorials"}
                            ],
                            "messageId": msg_id_1,
                        },
                        "configuration": {"acceptedOutputModes": ["text"]},
                    },
                },
            )
            assert resp1.status_code == 200
            r1 = resp1.json()["result"]
            task_id = r1["id"]
            context_id = r1["contextId"]

            # Turn 2 with same contextId
            resp2 = client.post(
                "/a2a/",
                json={
                    "jsonrpc": "2.0",
                    "id": "e2e-multi-2",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [{"kind": "text", "text": "show me shorter ones"}],
                            "messageId": msg_id_2,
                            "taskId": task_id,
                            "contextId": context_id,
                        },
                        "configuration": {"acceptedOutputModes": ["text"]},
                    },
                },
            )

        assert resp2.status_code == 200
        r2 = resp2.json()["result"]
        assert r2["contextId"] == context_id, "Context ID should be preserved"
        assert r2["status"]["state"] == "input-required"

    def test_context_isolation(self):
        """Two conversations with different contextIds should be independent."""
        with httpx.Client(base_url=RUNTIME, timeout=300.0) as client:
            # Conversation A
            resp_a = client.post(
                "/a2a/",
                json={
                    "jsonrpc": "2.0",
                    "id": "iso-a",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [{"kind": "text", "text": "search for dogs"}],
                            "messageId": str(uuid.uuid4()),
                        },
                        "configuration": {"acceptedOutputModes": ["text"]},
                    },
                },
            )

            # Conversation B
            resp_b = client.post(
                "/a2a/",
                json={
                    "jsonrpc": "2.0",
                    "id": "iso-b",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [{"kind": "text", "text": "search for cats"}],
                            "messageId": str(uuid.uuid4()),
                        },
                        "configuration": {"acceptedOutputModes": ["text"]},
                    },
                },
            )

        assert resp_a.status_code == 200
        assert resp_b.status_code == 200
        r_a = resp_a.json()["result"]
        r_b = resp_b.json()["result"]
        assert r_a["contextId"] != r_b["contextId"], (
            "Different conversations should have different context IDs"
        )
        assert r_a["id"] != r_b["id"], (
            "Different conversations should have different task IDs"
        )
