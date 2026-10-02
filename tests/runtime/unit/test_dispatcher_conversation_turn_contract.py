"""What a server-managed turn persists, and what it reports about the read.

``dispatch`` manages history itself when a caller sends a ``context_id`` with
no ``conversation_history``. Two contracts hold there:

* the assistant turn it appends is the answer the caller was given, never the
  envelope's operational ``message`` ("Generated summary for '...'"), which no
  assistant ever said and which the next turn would read back as context;
* a history read that does not answer is reported. An answer written without
  the prior turns it should have had is not the same answer as one written
  with them, and the caller can only tell from the envelope.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

from cogniverse_runtime.agent_dispatcher import (
    CONVERSATION_LOAD_TIMEOUT_S,
    AgentDispatcher,
)


def _registry_mock() -> MagicMock:
    """A registry stand-in whose shared-store refresh does nothing."""
    registry = MagicMock()
    registry.refresh = AsyncMock()
    return registry


pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TENANT = "acme:acme"


class _RecordingStore:
    """A ConversationStore-shaped double recording exactly what was stored,
    ordered by the seq each write carries, as the real store reads it."""

    def __init__(self, turns=None, read_error=None, read_block=None):
        self._rows = [(index, turn) for index, turn in enumerate(turns or [])]
        self.read_error = read_error
        self.read_block = read_block
        self.reads = 0

    @property
    def turns(self):
        return [turn for _seq, turn in sorted(self._rows, key=lambda row: row[0])]

    def get_history(self, context_id, max_turns=20):
        self.reads += 1
        if self.read_block is not None:
            self.read_block.wait()
        if self.read_error is not None:
            raise self.read_error
        return self.turns

    def store_turn(self, context_id, role, content, seq):
        self._rows.append((seq, {"role": role, "content": content}))

    def get_missing_assistant_markers(self, context_id):
        return []

    def store_missing_assistant_marker(self, context_id, reason, seq):
        self._rows.append((seq, {"role": "assistant_missing", "content": reason}))


def _dispatcher(store, result, ledger):
    config_manager = MagicMock()
    d = AgentDispatcher(
        agent_registry=_registry_mock(),
        config_manager=config_manager,
        schema_loader=MagicMock(),
        conversation_ledger=ledger,
    )
    agent = MagicMock()
    agent.capabilities = {"search"}
    d._registry.get_agent.return_value = agent
    d._conversation_store_factory = lambda tenant_id: store

    async def _skip_wiki(*args, **kwargs):
        return None

    d._maybe_auto_file_wiki = _skip_wiki
    seen_history = []

    async def _execute(query, tenant_id, top_k, conversation_history=None, **kwargs):
        seen_history.append(list(conversation_history or []))
        return _Envelope(**result).as_dict()

    d._execute_search_task = _execute
    return d, seen_history


@dataclasses.dataclass
class _Envelope:
    status: str = "success"
    agent: str = "summarizer_agent"
    message: str = ""
    result: dict = dataclasses.field(default_factory=dict)

    def as_dict(self):
        return {
            "status": self.status,
            "agent": self.agent,
            "message": self.message,
            "result": dict(self.result),
        }


async def _dispatch(dispatcher, query, context_id):
    result = await dispatcher.dispatch(
        agent_name="summarizer_agent",
        query=query,
        context={"tenant_id": TENANT, "context_id": context_id},
    )
    await dispatcher.drain_conversation_saves()
    return result


@pytest.mark.asyncio
async def test_the_persisted_assistant_turn_is_the_delivered_answer(
    conversation_ledger,
):
    """The summary envelope's ``message`` is a status line, not the answer.

    Persisting it makes the next turn read "Generated summary for '...'" as
    what the assistant said, and the summary itself is lost.
    """
    store = _RecordingStore()
    dispatcher, seen = _dispatcher(
        store,
        {
            "message": "Generated summary for 'what did the speaker say'",
            "result": {"summary": "The speaker said tides follow the moon."},
        },
        conversation_ledger,
    )

    result = await _dispatch(dispatcher, "what did the speaker say", "ctx-1")

    assert result["answer"] == "The speaker said tides follow the moon."
    assert store.turns == [
        {"role": "user", "content": "what did the speaker say"},
        {"role": "assistant", "content": "The speaker said tides follow the moon."},
    ]
    assert seen == [[]]

    await _dispatch(dispatcher, "and the tides?", "ctx-1")

    assert seen[1] == [
        {"role": "user", "content": "what did the speaker say"},
        {"role": "assistant", "content": "The speaker said tides follow the moon."},
    ]


@pytest.mark.asyncio
async def test_an_envelope_with_no_answer_persists_the_user_turn_alone(
    conversation_ledger,
):
    """An error envelope has no answer; nothing may take its place."""
    store = _RecordingStore()
    dispatcher, _ = _dispatcher(
        store,
        {
            "status": "error",
            "message": "search_agent failed: Vespa unreachable",
            "result": {},
        },
        conversation_ledger,
    )

    result = await _dispatch(dispatcher, "what did the speaker say", "ctx-2")

    assert "answer" not in result
    assert store.turns == [{"role": "user", "content": "what did the speaker say"}]


@pytest.mark.asyncio
async def test_a_loaded_history_is_reported_with_its_turn_count(conversation_ledger):
    from cogniverse_runtime.agent_dispatcher import (
        CONVERSATION_HISTORY_LOADED,
    )

    store = _RecordingStore(
        turns=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
    )
    dispatcher, seen = _dispatcher(
        store, {"message": "m", "result": {"summary": "s"}}, conversation_ledger
    )

    result = await _dispatch(dispatcher, "third", "ctx-3")

    assert result["conversation"] == {
        "state": CONVERSATION_HISTORY_LOADED,
        "turn_count": 2,
        "reason": None,
    }
    assert seen == [
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
    ]


@pytest.mark.asyncio
async def test_a_memory_outage_is_reported_not_answered_as_a_fresh_context(
    conversation_ledger,
):
    """A failed read is not an empty context.

    Without this the caller cannot tell "you have said nothing before" from
    "I could not read what you said before", and the answer that ignored the
    prior turns is presented as if it had them.
    """
    from cogniverse_runtime.agent_dispatcher import (
        CONVERSATION_HISTORY_UNAVAILABLE,
    )

    outage = ConnectionError("mem0 unreachable")
    store = _RecordingStore(read_error=outage)
    dispatcher, seen = _dispatcher(
        store, {"message": "m", "result": {"summary": "s"}}, conversation_ledger
    )

    result = await _dispatch(dispatcher, "third", "ctx-4")

    assert result["conversation"] == {
        "state": CONVERSATION_HISTORY_UNAVAILABLE,
        "turn_count": 0,
        "reason": "ConnectionError('mem0 unreachable')",
    }
    assert seen == [[]]
    assert store.turns == [
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "s"},
    ]


@pytest.mark.asyncio
async def test_a_read_that_exceeds_its_budget_is_reported_as_unavailable(
    conversation_ledger,
):
    from cogniverse_runtime.agent_dispatcher import (
        CONVERSATION_HISTORY_UNAVAILABLE,
        ConversationHistory,
    )

    release = threading.Event()
    store = _RecordingStore(read_block=release)
    dispatcher, _ = _dispatcher(
        store, {"message": "m", "result": {"summary": "s"}}, conversation_ledger
    )

    try:
        history = await asyncio.wait_for(
            dispatcher._load_conversation_history(TENANT, "ctx-5"),
            timeout=CONVERSATION_LOAD_TIMEOUT_S + 5,
        )
    finally:
        release.set()

    assert history == ConversationHistory(
        turns=[],
        state=CONVERSATION_HISTORY_UNAVAILABLE,
        reason="TimeoutError()",
    )


@pytest.mark.asyncio
async def test_concurrent_tenants_report_their_own_read_outcome(conversation_ledger):
    """One tenant's memory outage must not be attributed to another's turn.

    Both dispatches are in flight together, so the degrade has to ride the
    request rather than any shared dispatcher state.
    """
    from cogniverse_runtime.agent_dispatcher import (
        CONVERSATION_HISTORY_LOADED,
        CONVERSATION_HISTORY_UNAVAILABLE,
    )

    healthy = _RecordingStore(turns=[{"role": "user", "content": "prior"}])
    broken = _RecordingStore(read_error=ConnectionError("mem0 unreachable"))
    stores = {"acme:acme": healthy, "peer:peer": broken}

    dispatcher = AgentDispatcher(
        agent_registry=_registry_mock(),
        config_manager=MagicMock(),
        schema_loader=MagicMock(),
        conversation_ledger=conversation_ledger,
    )
    agent = MagicMock()
    agent.capabilities = {"search"}
    dispatcher._registry.get_agent.return_value = agent

    async def _skip_wiki(*args, **kwargs):
        return None

    dispatcher._maybe_auto_file_wiki = _skip_wiki
    dispatcher._conversation_store_factory = lambda tenant_id: stores[tenant_id]

    started = asyncio.Event()

    async def _execute(query, tenant_id, top_k, conversation_history=None, **kwargs):
        started.set()
        await asyncio.sleep(0.02)
        return {
            "status": "success",
            "agent": "summarizer_agent",
            "message": "status",
            "result": {"summary": f"answer for {tenant_id}"},
        }

    dispatcher._execute_search_task = _execute

    async def _one(tenant_id):
        return await dispatcher.dispatch(
            agent_name="summarizer_agent",
            query=tenant_id,
            context={"tenant_id": tenant_id, "context_id": "shared-ctx"},
        )

    ok, down = await asyncio.gather(_one("acme:acme"), _one("peer:peer"))
    await dispatcher.drain_conversation_saves()

    assert started.is_set() is True
    assert ok["conversation"] == {
        "state": CONVERSATION_HISTORY_LOADED,
        "turn_count": 1,
        "reason": None,
    }
    assert down["conversation"] == {
        "state": CONVERSATION_HISTORY_UNAVAILABLE,
        "turn_count": 0,
        "reason": "ConnectionError('mem0 unreachable')",
    }
    assert healthy.turns == [
        {"role": "user", "content": "prior"},
        {"role": "user", "content": "acme:acme"},
        {"role": "assistant", "content": "answer for acme:acme"},
    ]
    assert broken.turns == [
        {"role": "user", "content": "peer:peer"},
        {"role": "assistant", "content": "answer for peer:peer"},
    ]
