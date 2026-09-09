"""What a conversation turn write owes: retry schedule, classifier, marker.

A turn whose assistant append fails permanently keeps its user turn and gains
a durable marker row in the assistant's place. The marker has to be findable
after a restart and must never reach an agent as an assistant message, and the
classifier must retry only a write that never reached a verdict -- never one
the backend refused. Both are pure functions of a stored row and of an
exception, so a recording stub manager drives them; the same store code runs
against real Mem0 in tests/memory/integration.
"""

from __future__ import annotations

import time

import httpx
import pytest
import requests

from cogniverse_core.conversation import (
    ASSISTANT_MISSING_PREFIX,
    ASSISTANT_MISSING_ROLE,
    CONVERSATION_AGENT_NAME,
    KNOWN_TURN_ROLES,
    RENDERED_TURN_ROLES,
    ConversationStore,
    is_transient_turn_write_error,
)
from cogniverse_runtime.agent_dispatcher import (
    CONVERSATION_SAVE_ATTEMPTS,
    CONVERSATION_SAVE_RETRY_BACKOFF_S,
    CONVERSATION_SAVE_STEP_RESERVE_S,
    CONVERSATION_SAVE_TIMEOUT_S,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

CTX = "c1"
TENANT = "acme:acme"


class _RecordingManager:
    """Records writes and replays a fixed row list, like the real manager."""

    def __init__(self, rows=()):
        self._rows = list(rows)
        self.adds: list = []

    def get_all_memories(self, tenant_id, agent_name, filters=None, limit=100):
        return list(self._rows)

    def add_memory(self, content, tenant_id, agent_name, metadata=None, infer=True):
        self.adds.append(
            {
                "content": content,
                "tenant_id": tenant_id,
                "agent_name": agent_name,
                "metadata": metadata,
                "infer": infer,
            }
        )
        return "mem-1"


def _row(memory, metadata):
    return {"memory": memory, "metadata": metadata}


def _turn(role, context_id, seq):
    return {
        "type": "conversation",
        "context_id": context_id,
        "session_id": context_id,
        "turn_role": role,
        "seq": seq,
    }


def test_roles_split_into_rendered_and_marker():
    assert RENDERED_TURN_ROLES == ("user", "assistant")
    assert ASSISTANT_MISSING_ROLE == "assistant_missing"
    assert KNOWN_TURN_ROLES == ("user", "assistant", "assistant_missing")


def test_marker_write_names_the_failure_type_and_carries_no_user_text():
    manager = _RecordingManager()
    store = ConversationStore(manager, TENANT)

    before = time.time()
    store.store_missing_assistant_marker(
        CTX, ConnectionError("POST /document/v1 for 'what is my bank pin' failed")
    )
    after = time.time()

    assert len(manager.adds) == 1
    written = manager.adds[0]
    seq = written["metadata"].pop("seq")
    assert type(seq) is float
    assert before <= seq <= after
    assert written == {
        "content": "[ctx:c1] [assistant_missing] assistant turn not persisted: "
        "ConnectionError",
        "tenant_id": TENANT,
        "agent_name": CONVERSATION_AGENT_NAME,
        "metadata": {
            "type": "conversation",
            "context_id": CTX,
            "session_id": CTX,
            "turn_role": "assistant_missing",
        },
        "infer": False,
    }
    # The exception's own message quoted the user's query; the row does not.
    assert "bank pin" not in written["content"]


def test_history_never_renders_a_marker_but_the_marker_stays_readable():
    rows = [
        _row("[ctx:c1] [user] what is colpali", _turn("user", CTX, 1.0)),
        _row(
            f"[ctx:c1] [assistant_missing] {ASSISTANT_MISSING_PREFIX}ConnectError",
            _turn(ASSISTANT_MISSING_ROLE, CTX, 2.0),
        ),
        _row("[ctx:c1] [user] still there?", _turn("user", CTX, 3.0)),
    ]
    store = ConversationStore(_RecordingManager(rows), TENANT)

    # The agent sees an unanswered user message, then the follow-up.
    assert store.get_history(CTX) == [
        {"role": "user", "content": "what is colpali"},
        {"role": "user", "content": "still there?"},
    ]
    assert store.get_missing_assistant_markers(CTX) == [
        {
            "role": "assistant_missing",
            "content": "assistant turn not persisted: ConnectError",
        }
    ]


def test_markers_are_scoped_to_their_context_and_ordered_by_seq():
    rows = [
        _row(
            "[ctx:c1] [assistant_missing] assistant turn not persisted: ReadTimeout",
            _turn(ASSISTANT_MISSING_ROLE, CTX, 5.0),
        ),
        _row(
            "[ctx:c1] [assistant_missing] assistant turn not persisted: ConnectError",
            _turn(ASSISTANT_MISSING_ROLE, CTX, 2.0),
        ),
        _row(
            "[ctx:other] [assistant_missing] assistant turn not persisted: RuntimeError",
            _turn(ASSISTANT_MISSING_ROLE, "other", 3.0),
        ),
    ]
    store = ConversationStore(_RecordingManager(rows), TENANT)

    assert store.get_missing_assistant_markers(CTX) == [
        {
            "role": "assistant_missing",
            "content": "assistant turn not persisted: ConnectError",
        },
        {
            "role": "assistant_missing",
            "content": "assistant turn not persisted: ReadTimeout",
        },
    ]
    assert store.get_missing_assistant_markers("other") == [
        {
            "role": "assistant_missing",
            "content": "assistant turn not persisted: RuntimeError",
        }
    ]


def test_store_turn_refuses_a_role_no_reader_renders():
    manager = _RecordingManager()
    store = ConversationStore(manager, TENANT)

    with pytest.raises(ValueError) as excinfo:
        store.store_turn(CTX, "system", "you are a helpful assistant")

    assert str(excinfo.value) == (
        "unknown conversation turn role 'system'; expected one of "
        "('user', 'assistant', 'assistant_missing')"
    )
    assert manager.adds == []


def _status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://vespa:8080/document/v1/")
    return httpx.HTTPStatusError(
        f"status {status_code}",
        request=request,
        response=httpx.Response(status_code, request=request),
    )


def _requests_http_error(status_code: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    return requests.HTTPError(f"status {status_code}", response=response)


def test_only_writes_that_never_reached_a_verdict_are_retryable():
    # The whole classification, written out: each case is an error the mem0
    # write path really raises (the ingestion client's builtin ConnectionError,
    # the DenseOn embedder's requests transport errors, pyvespa's httpx feed,
    # the vector store's rejection RuntimeError).
    cases = {
        "ingestion client unreachable": ConnectionError("Vespa is unavailable"),
        "socket timeout": TimeoutError(),
        "embedder connection": requests.ConnectionError("denseon refused"),
        "embedder read timeout": requests.ReadTimeout("denseon slow"),
        "httpx connect": httpx.ConnectError("connection refused"),
        "httpx read timeout": httpx.ReadTimeout("read timed out"),
        "httpx 503": _status_error(503),
        "httpx 429": _status_error(429),
        "httpx 400": _status_error(400),
        "httpx 404": _status_error(404),
        "requests 503": _requests_http_error(503),
        "requests 400": _requests_http_error(400),
        "rejected document": RuntimeError(
            "Mem0 insert into agent_memories persisted only 0/1 memories; "
            "failed_documents=[{'id': 'memory-1', 'state': 'rejected'}]"
        ),
        "unwired manager": RuntimeError("Mem0MemoryManager not initialized"),
        "bad payload": ValueError("expected 768 values, got 128"),
        "programming error": TypeError("store_turn() missing 1 argument"),
    }

    assert {
        name: is_transient_turn_write_error(exc) for name, exc in cases.items()
    } == {
        "ingestion client unreachable": True,
        "socket timeout": True,
        "embedder connection": True,
        "embedder read timeout": True,
        "httpx connect": True,
        "httpx read timeout": True,
        "httpx 503": True,
        "httpx 429": True,
        "httpx 400": False,
        "httpx 404": False,
        "requests 503": True,
        "requests 400": False,
        "rejected document": False,
        "unwired manager": False,
        "bad payload": False,
        "programming error": False,
    }


def test_retry_schedule_is_the_arithmetic_the_save_budget_allows():
    """The shipped schedule, and the measured budget it was derived from.

    Measured against real Mem0: a fresh process's first save costs ~7.2s
    (store build plus the embedder-warming first write) and a steady-state
    write 0.03-0.11s. The budget carries a 2x margin over a save, so the
    retries and the marker write a permanent failure adds must fit in what is
    left of half the budget.
    """
    measured_cold_save_s = 7.2
    measured_worst_write_s = 0.11

    assert CONVERSATION_SAVE_ATTEMPTS == 4
    assert CONVERSATION_SAVE_RETRY_BACKOFF_S == 0.25
    assert CONVERSATION_SAVE_STEP_RESERVE_S == 0.5

    backoffs = [
        CONVERSATION_SAVE_RETRY_BACKOFF_S * 2**retry
        for retry in range(CONVERSATION_SAVE_ATTEMPTS - 1)
    ]
    assert backoffs == [0.25, 0.5, 1.0]

    retry_cost_s = sum(backoffs) + CONVERSATION_SAVE_ATTEMPTS * measured_worst_write_s
    assert retry_cost_s == 2.19
    assert (
        measured_cold_save_s + retry_cost_s + measured_worst_write_s
        <= CONVERSATION_SAVE_TIMEOUT_S / 2
    )
    # The reserve covers the attempt about to be made and the marker write, so
    # a retry is abandoned rather than cancelled mid-flight at the deadline.
    assert CONVERSATION_SAVE_STEP_RESERVE_S >= 2 * measured_worst_write_s
