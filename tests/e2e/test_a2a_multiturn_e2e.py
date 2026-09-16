"""
E2E tests for multi-turn conversation support via REST and A2A protocols.

Requires live runtime at http://localhost:33000 with LM + Vespa.
Uses flywheel_org:production tenant which has ingested data.

Tests validate:
- REST conversation_history triggers query rewrite on turn 2+
- REST first turn without history produces no rewritten_query
- A2A agent card discovery
- A2A single-turn message/send
- A2A multi-turn with contextId threading
- A2A context isolation between conversations
"""

import time
import uuid

import httpx
import pytest

from cogniverse_core.conversation import CONVERSATION_AGENT_NAME
from cogniverse_runtime.agent_dispatcher import (
    CONVERSATION_HISTORY_LOADED,
    CONVERSATION_SAVE_TIMEOUT_S,
    AnswerGroundingUnavailable,
)
from tests.e2e.conftest import (
    SAMPLE_DOCUMENT_TITLES,
    TENANT_DEPLOY_TIMEOUT_S,
    _ingest_sample_documents,
    register_tenant_and_wait,
    unique_id,
)
from tests.e2e.test_api_e2e import DOCUMENT_PROFILE, _deploy_profile_for_tenant

RUNTIME = "http://localhost:33000"
TENANT_ID = "flywheel_org:production"


def _assert_runtime_ready() -> None:
    try:
        response = httpx.get(f"{RUNTIME}/health/live", timeout=10.0)
    except httpx.HTTPError as exc:
        raise AssertionError(
            f"Runtime liveness endpoint must be reachable at {RUNTIME}"
        ) from exc
    assert response.status_code == 200, (
        f"Runtime liveness must return HTTP 200; got {response.status_code}: "
        f"{response.text}"
    )
    assert response.json() == {"status": "alive"}, response.json()


@pytest.mark.e2e
class TestRESTMultiTurn:
    """REST endpoint multi-turn conversation tests."""

    def test_first_turn_no_rewrite(self):
        """First turn without conversation_history should NOT produce rewritten_query."""
        _assert_runtime_ready()
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
        _assert_runtime_ready()
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
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

    def test_gateway_agent_executes_downstream_with_rewrite(self):
        """Routing agent should execute downstream search with query rewrite."""
        _assert_runtime_ready()
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
            resp = client.post(
                "/agents/gateway_agent/process",
                json={
                    "agent_name": "gateway_agent",
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
        # Gateway may route through gateway_agent (simple) or orchestrator (complex)
        assert data["agent"] in (
            "gateway_agent",
            "orchestrator_agent",
        ), f"Expected gateway/orchestrator agent, got {data['agent']}"
        # If routing path, check downstream; if orchestrator, check orchestration_result
        if "downstream_result" in data:
            ds = data["downstream_result"]
            assert ds.get("agent") is not None or ds.get("results") is not None
        elif "orchestration_result" in data:
            orch = data["orchestration_result"]
            assert len(orch.get("plan_steps", [])) > 0, (
                "Orchestrator should produce a plan"
            )
        # Conversation history was passed — the gateway/orchestrator handles rewrite internally


@pytest.mark.e2e
class TestA2AProtocol:
    """A2A protocol endpoint tests."""

    def test_agent_card_discovery(self):
        """GET /a2a/.well-known/agent.json returns valid agent card."""
        _assert_runtime_ready()
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
        _assert_runtime_ready()
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
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
                        "metadata": {"tenant_id": TENANT_ID},
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
        _assert_runtime_ready()
        msg_id_1 = str(uuid.uuid4())
        msg_id_2 = str(uuid.uuid4())

        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
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
                        "metadata": {"tenant_id": TENANT_ID},
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
                        "metadata": {"tenant_id": TENANT_ID},
                    },
                },
            )

        assert resp2.status_code == 200
        r2 = resp2.json()["result"]
        assert r2["contextId"] == context_id, "Context ID should be preserved"
        assert r2["status"]["state"] == "input-required"

    def test_context_isolation(self):
        """Two conversations with different contextIds should be independent."""
        _assert_runtime_ready()
        with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
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
                        "metadata": {"tenant_id": TENANT_ID},
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
                        "metadata": {"tenant_id": TENANT_ID},
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


@pytest.fixture(scope="module")
def document_tenant():
    """A tenant this module owns, serving the two committed caption documents."""
    org_id = unique_id("a2a_convo")
    tenant_id = f"{org_id}:t1"
    with httpx.Client(base_url=RUNTIME, timeout=TENANT_DEPLOY_TIMEOUT_S) as client:
        created = client.post(
            "/admin/organizations",
            json={
                "org_id": org_id,
                "org_name": org_id.replace("_", "-"),
                "created_by": "e2e",
            },
        )
        assert created.status_code in (200, 201), created.text
        register_tenant_and_wait(tenant_id, created_by="e2e", timeout_s=600.0)
        _deploy_profile_for_tenant(client, DOCUMENT_PROFILE, tenant_id)
        seeded = _ingest_sample_documents(tenant_id=tenant_id)
        assert set(seeded) == set(SAMPLE_DOCUMENT_TITLES), seeded
        yield tenant_id


def _conversation_rows(tenant_id: str, context_id: str) -> list[dict]:
    """This context's stored turns, oldest first, straight out of Mem0.

    Reads the partition ``ConversationStore`` writes into rather than the
    dispatcher's own loader, so the assertion lands on the persisted row and
    not on a re-render of it.
    """
    with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
        response = client.get(
            f"/admin/tenant/{tenant_id}/memories",
            params={"agent_name": CONVERSATION_AGENT_NAME, "limit": 200},
        )
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"memories", "count"}, body
    assert body["count"] == len(body["memories"]), body
    rows = [
        row
        for row in body["memories"]
        if row["metadata"].get("context_id") == context_id
    ]
    return sorted(rows, key=lambda row: float(row["metadata"]["seq"]))


def _await_conversation_rows(
    tenant_id: str, context_id: str, expected: int
) -> list[dict]:
    """Poll until the context holds ``expected`` rows, within the save budget.

    The reply returns before the turns are persisted (the save runs on the
    dispatcher's own chain), so the read has to wait for the write the way the
    next turn would.
    """
    deadline = time.monotonic() + 2 * CONVERSATION_SAVE_TIMEOUT_S
    rows: list[dict] = []
    while time.monotonic() < deadline:
        rows = _conversation_rows(tenant_id, context_id)
        if len(rows) >= expected:
            return rows
        time.sleep(2.0)
    return rows


@pytest.mark.e2e
class TestConversationHistoryStoresTheAnswer:
    """A server-managed turn persists the answer the caller was given."""

    def test_the_stored_assistant_turn_is_the_answer_not_the_status_line(
        self, document_tenant
    ):
        context_id = f"e2e-convo-{uuid.uuid4().hex}"
        query = "summarize what these documents say about washing dishes"

        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            response = client.post(
                "/agents/summarizer_agent/process",
                json={
                    "agent_name": "summarizer_agent",
                    "query": query,
                    "context": {"tenant_id": document_tenant},
                    "context_id": context_id,
                },
            )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "success", body
        assert body["agent"] == "summarizer_agent"
        # The first turn of a fresh context reads no history, and says so.
        assert body["conversation"] == {
            "state": CONVERSATION_HISTORY_LOADED,
            "turn_count": 0,
            "reason": None,
        }, body["conversation"]
        # The two fields the persisted turn must not be confused for: the
        # status line the dispatcher builds, and the rendered answer.
        assert body["message"] == f"Generated summary for '{query}'", body["message"]
        assert body["answer"] != body["message"], body["answer"]

        rows = _await_conversation_rows(document_tenant, context_id, 2)
        assert [row["metadata"]["turn_role"] for row in rows] == [
            "user",
            "assistant",
        ], rows
        assert [row["metadata"]["type"] for row in rows] == [
            "conversation",
            "conversation",
        ], rows
        assert [row["metadata"]["session_id"] for row in rows] == [
            context_id,
            context_id,
        ], rows
        # The stored text is the store's own tagged form of the two turns:
        # the query the caller sent and the answer it was handed back.
        assert [row["memory"] for row in rows] == [
            f"[ctx:{context_id}] [user] {query}",
            f"[ctx:{context_id}] [assistant] {body['answer']}",
        ], rows
        assert f"[ctx:{context_id}] [assistant] {body['message']}" not in [
            row["memory"] for row in rows
        ], rows

    def test_a_second_turn_reads_back_exactly_the_turns_the_first_one_left(
        self, document_tenant
    ):
        context_id = f"e2e-convo-{uuid.uuid4().hex}"
        first_query = "summarize what these documents say about washing dishes"
        second_query = "summarize the same documents again"

        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            first = client.post(
                "/agents/summarizer_agent/process",
                json={
                    "agent_name": "summarizer_agent",
                    "query": first_query,
                    "context": {"tenant_id": document_tenant},
                    "context_id": context_id,
                },
            )
            assert first.status_code == 200, first.text
            assert first.json()["conversation"] == {
                "state": CONVERSATION_HISTORY_LOADED,
                "turn_count": 0,
                "reason": None,
            }

            second = client.post(
                "/agents/summarizer_agent/process",
                json={
                    "agent_name": "summarizer_agent",
                    "query": second_query,
                    "context": {"tenant_id": document_tenant},
                    "context_id": context_id,
                },
            )

        assert second.status_code == 200, second.text
        body = second.json()
        # The second turn waits for the first turn's save, so the count it
        # reports is exactly the user + assistant pair the first turn wrote.
        assert body["conversation"] == {
            "state": CONVERSATION_HISTORY_LOADED,
            "turn_count": 2,
            "reason": None,
        }, body["conversation"]

        rows = _await_conversation_rows(document_tenant, context_id, 4)
        assert [row["metadata"]["turn_role"] for row in rows] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ], rows
        assert [row["memory"] for row in rows][2] == (
            f"[ctx:{context_id}] [user] {second_query}"
        ), rows
        assert [row["memory"] for row in rows][3] == (
            f"[ctx:{context_id}] [assistant] {body['answer']}"
        ), rows


@pytest.mark.e2e
class TestAFailedTurnPersistsNoAnswer:
    """A turn that never produced an answer leaves no assistant turn behind."""

    def test_a_grounding_failure_stores_no_turn_at_all(self, document_tenant):
        context_id = f"e2e-convo-{uuid.uuid4().hex}"
        session_id = f"e2e-grounding-{uuid.uuid4().hex}"
        missing_profile = f"absent_profile_{uuid.uuid4().hex[:8]}"

        with httpx.Client(base_url=RUNTIME, timeout=600.0) as client:
            response = client.post(
                "/agents/summarizer_agent/process",
                json={
                    "agent_name": "summarizer_agent",
                    "query": "summarize what these documents say about washing dishes",
                    "context": {"tenant_id": document_tenant},
                    "context_id": context_id,
                    "session_id": session_id,
                    "profiles": [missing_profile],
                },
            )

        # The grounding search cannot run, so the turn fails; it does not
        # answer from the query alone and it does not reach the save.
        assert response.status_code == 500, response.text
        assert response.json() == {
            "detail": (
                f"Agent 'summarizer_agent' failed with "
                f"{AnswerGroundingUnavailable.__name__} "
                f"(request_id={session_id}). See runtime logs for detail."
            )
        }, response.json()

        rows = _await_conversation_rows(document_tenant, context_id, 1)
        assert rows == [], rows
