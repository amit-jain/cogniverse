"""A turn's assistant append is retried, and a half-turn is marked, in real Mem0.

The dispatcher persists a turn as two sequential appends. When the second one
fails, the first is already durable, so the contract is: retry the failed
append inside the save budget, never repeat an append that succeeded, and when
the reply still cannot be stored keep the user turn and write a durable marker
row in the assistant's place -- never fabricated assistant prose. Every
assertion here is on rows read back from real Mem0 (real Vespa + real DenseOn);
only the assistant append is faulted, through a subclass of the production
ConversationStore, so the user append, the marker write and the history read
all run the real code against the real backend.
"""

from __future__ import annotations

import threading
import time
import uuid

import httpx
import pytest
import requests

from cogniverse_core.conversation import (
    ASSISTANT_MISSING_ROLE,
    ConversationStore,
)
from cogniverse_runtime.agent_dispatcher import (
    CONVERSATION_SAVE_ATTEMPTS,
    CONVERSATION_SAVE_RETRY_BACKOFF_S,
    CONVERSATION_SAVE_TIMEOUT_S,
    ConversationPersistFailed,
)
from tests.memory.integration.test_dispatcher_conversation_history_real_mem0 import (
    TENANT,
    _build_manager,
    _dispatch,
    _dispatcher_with_real_store,
    _reply_with,
)

pytestmark = [pytest.mark.integration]

QUERY = "what is colpali"
REPLY = "a late-interaction model"
FOLLOW_UP = "how many dims"
FOLLOW_UP_REPLY = "128"


def _retry_backoff_total() -> float:
    """Wall time the shipped schedule sleeps when every retry is taken."""
    return sum(
        CONVERSATION_SAVE_RETRY_BACKOFF_S * 2**attempt
        for attempt in range(CONVERSATION_SAVE_ATTEMPTS - 1)
    )


class _FaultyAssistantStore(ConversationStore):
    """The production store with the assistant append faulted N times.

    Subclasses ConversationStore so the user append, the marker write and the
    history read run the real code against real Mem0; only the assistant write
    raises, and only for its first ``failures`` attempts. Every attempt is
    recorded as ``(role, "raised" | "stored")`` so the retry count is read off
    the store rather than inferred.
    """

    def __init__(self, memory_manager, tenant_id, *, failures, error_factory):
        super().__init__(memory_manager, tenant_id)
        # ``failures=None`` fails every attempt, whatever the schedule allows.
        self._remaining = float("inf") if failures is None else failures
        self._error_factory = error_factory
        self.attempts: list[tuple[str, str]] = []
        self.write_durations: list[float] = []

    def heal(self):
        """Stop faulting: the backend the injection stood in for recovered."""
        self._remaining = 0

    def store_turn(self, context_id, role, content):
        if role == "assistant" and self._remaining > 0:
            self._remaining -= 1
            self.attempts.append((role, "raised"))
            raise self._error_factory()
        self.attempts.append((role, "stored"))
        started = time.monotonic()
        super().store_turn(context_id, role, content)
        self.write_durations.append(time.monotonic() - started)


def _dispatcher_with_faulty_store(mm, *, failures, error_factory):
    store = _FaultyAssistantStore(
        mm, TENANT, failures=failures, error_factory=error_factory
    )
    dispatcher = _dispatcher_with_real_store(mm)
    dispatcher._conversation_store_factory = lambda _tenant_id: store
    return dispatcher, store


@pytest.mark.asyncio
async def test_transient_assistant_failure_retries_into_one_clean_pair(
    shared_memory_vespa, shared_denseon
):
    """A blip on the assistant append is retried until it lands, and the user
    turn that already succeeded is never written twice."""
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    # Two consecutive blips on the reply -- a fixed adversarial count, not one
    # derived from the shipped schedule, so a schedule that stopped retrying
    # fails here instead of quietly matching a smaller expectation.
    d, store = _dispatcher_with_faulty_store(
        mm,
        failures=2,
        error_factory=lambda: requests.ConnectionError("denseon connection reset"),
    )
    ctx = f"chat{uuid.uuid4().hex[:10]}"
    seen: list = []
    _reply_with(d, {QUERY: REPLY}, seen)

    started = time.monotonic()
    result = await _dispatch(d, QUERY, ctx)
    assert result["message"] == REPLY
    assert await d.drain_conversation_saves() is True
    elapsed = time.monotonic() - started
    print(f"RETRIED_SAVE_ELAPSED_S={elapsed:.3f}")

    # One user append, then the assistant append tried until it landed.
    assert store.attempts == [
        ("user", "stored"),
        ("assistant", "raised"),
        ("assistant", "raised"),
        ("assistant", "stored"),
    ]

    reader = ConversationStore(mm, TENANT)
    assert reader.get_history(ctx) == [
        {"role": "user", "content": QUERY},
        {"role": "assistant", "content": REPLY},
    ]
    assert reader.get_missing_assistant_markers(ctx) == []
    assert d.conversation_persist_status() == {"pending": 0, "failed": []}
    assert d.conversation_persist_failure(TENANT, ctx) is None
    # The retried save slept the whole shipped backoff schedule and still fit
    # the budget the save is bounded by.
    assert elapsed > _retry_backoff_total()
    assert elapsed < CONVERSATION_SAVE_TIMEOUT_S


@pytest.mark.asyncio
async def test_exhausted_retries_keep_the_user_turn_and_mark_the_missing_reply(
    shared_memory_vespa, shared_denseon
):
    """When the reply cannot be stored the user turn stays, a durable marker
    names the failure, and the next turn reads an unanswered user message --
    never fabricated assistant prose."""
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    d, store = _dispatcher_with_faulty_store(
        mm,
        failures=None,
        error_factory=lambda: httpx.ConnectError("vespa refused the feed"),
    )
    ctx = f"chat{uuid.uuid4().hex[:10]}"
    seen: list = []
    _reply_with(d, {QUERY: REPLY, FOLLOW_UP: FOLLOW_UP_REPLY}, seen)

    result = await _dispatch(d, QUERY, ctx)
    assert result["message"] == REPLY
    assert await d.drain_conversation_saves() is True

    # Every attempt the schedule allows was spent on the reply, then the
    # marker took its place.
    assert store.attempts == (
        [("user", "stored")]
        + [("assistant", "raised")] * CONVERSATION_SAVE_ATTEMPTS
        + [(ASSISTANT_MISSING_ROLE, "stored")]
    )

    reader = ConversationStore(mm, TENANT)
    assert reader.get_history(ctx) == [{"role": "user", "content": QUERY}]
    assert reader.get_missing_assistant_markers(ctx) == [
        {
            "role": "assistant_missing",
            "content": "assistant turn not persisted: ConnectError",
        }
    ]
    assert d.conversation_persist_status() == {"pending": 0, "failed": [(TENANT, ctx)]}
    failure = d.conversation_persist_failure(TENANT, ctx)
    assert type(failure) is ConversationPersistFailed
    assert type(failure.__cause__) is httpx.ConnectError
    assert failure.context_id == ctx
    assert failure.tenant_id == TENANT

    # The next dispatch on this context sees the unanswered user message and
    # nothing standing in for the reply that was lost. Its own reply lands:
    # the outage the injection stood in for has cleared.
    store.heal()
    follow_up = await _dispatch(d, FOLLOW_UP, ctx)
    assert follow_up["message"] == FOLLOW_UP_REPLY
    assert seen[1] == [{"role": "user", "content": QUERY}]
    assert await d.drain_conversation_saves() is True

    assert store.attempts == (
        [("user", "stored")]
        + [("assistant", "raised")] * CONVERSATION_SAVE_ATTEMPTS
        + [
            (ASSISTANT_MISSING_ROLE, "stored"),
            ("user", "stored"),
            ("assistant", "stored"),
        ]
    )
    assert reader.get_history(ctx) == [
        {"role": "user", "content": QUERY},
        {"role": "user", "content": FOLLOW_UP},
        {"role": "assistant", "content": FOLLOW_UP_REPLY},
    ]
    assert reader.get_missing_assistant_markers(ctx) == [
        {
            "role": "assistant_missing",
            "content": "assistant turn not persisted: ConnectError",
        }
    ]
    # A later save that lands clears the context's failure record.
    assert d.conversation_persist_status() == {"pending": 0, "failed": []}


@pytest.mark.asyncio
async def test_a_rejected_write_is_marked_without_a_second_attempt(
    shared_memory_vespa, shared_denseon
):
    """A write the backend refused is a verdict, not a blip: it is never
    retried, and the half-turn is marked immediately."""
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    rejection = (
        "Mem0 insert into agent_memories persisted only 0/1 memories; "
        "failed_documents=[{'id': 'memory-1', 'state': 'rejected', "
        "'status_code': 400}]"
    )
    d, store = _dispatcher_with_faulty_store(
        mm,
        failures=None,
        error_factory=lambda: RuntimeError(rejection),
    )
    ctx = f"chat{uuid.uuid4().hex[:10]}"
    seen: list = []
    _reply_with(d, {QUERY: REPLY}, seen)

    await _dispatch(d, QUERY, ctx)
    assert await d.drain_conversation_saves() is True

    # One attempt at the reply, then the marker: no backoff was ever served.
    assert store.attempts == [
        ("user", "stored"),
        ("assistant", "raised"),
        (ASSISTANT_MISSING_ROLE, "stored"),
    ]

    reader = ConversationStore(mm, TENANT)
    assert reader.get_history(ctx) == [{"role": "user", "content": QUERY}]
    assert reader.get_missing_assistant_markers(ctx) == [
        {
            "role": "assistant_missing",
            "content": "assistant turn not persisted: RuntimeError",
        }
    ]
    failure = d.conversation_persist_failure(TENANT, ctx)
    assert type(failure) is ConversationPersistFailed
    assert type(failure.__cause__) is RuntimeError
    assert str(failure.__cause__) == rejection


@pytest.mark.asyncio
async def test_retry_schedule_fits_the_measured_save_budget(
    shared_memory_vespa, shared_denseon
):
    """The shipped retry schedule is sized from what a save really costs.

    Two saves are timed against real Mem0 here -- one that serves every
    backoff, one healthy -- and the schedule must leave the budget the same 2x
    margin over a save that it carries today.
    """
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    # Three blips: the most the shipped schedule absorbs, so every backoff in
    # it is served and timed.
    d, store = _dispatcher_with_faulty_store(
        mm,
        failures=3,
        error_factory=lambda: httpx.ReadTimeout("vespa read timed out"),
    )
    retried_ctx = f"chat{uuid.uuid4().hex[:10]}"
    healthy_ctx = f"chat{uuid.uuid4().hex[:10]}"
    _reply_with(d, {QUERY: REPLY}, [])

    await _dispatch(d, QUERY, retried_ctx)
    retried_started = time.monotonic()
    assert await d.drain_conversation_saves() is True
    retried_save_s = time.monotonic() - retried_started

    await _dispatch(d, QUERY, healthy_ctx)
    healthy_started = time.monotonic()
    assert await d.drain_conversation_saves() is True
    healthy_save_s = time.monotonic() - healthy_started

    worst_write_s = max(store.write_durations)
    print(
        f"RETRIED_SAVE_S={retried_save_s:.3f} HEALTHY_SAVE_S={healthy_save_s:.3f} "
        f"WORST_WRITE_S={worst_write_s:.3f} "
        f"WRITE_DURATIONS_S={[round(x, 3) for x in store.write_durations]} "
        f"BACKOFF_SCHEDULE_S={_retry_backoff_total():.3f} "
        f"ATTEMPTS={CONVERSATION_SAVE_ATTEMPTS} BUDGET_S={CONVERSATION_SAVE_TIMEOUT_S}"
    )

    assert store.attempts == [
        ("user", "stored"),
        ("assistant", "raised"),
        ("assistant", "raised"),
        ("assistant", "raised"),
        ("assistant", "stored"),
        ("user", "stored"),
        ("assistant", "stored"),
    ]
    reader = ConversationStore(mm, TENANT)
    assert reader.get_history(retried_ctx) == [
        {"role": "user", "content": QUERY},
        {"role": "assistant", "content": REPLY},
    ]
    assert reader.get_history(healthy_ctx) == [
        {"role": "user", "content": QUERY},
        {"role": "assistant", "content": REPLY},
    ]
    assert d.conversation_persist_status() == {"pending": 0, "failed": []}

    # The retried save served every backoff and still fit the budget.
    assert retried_save_s > _retry_backoff_total()
    assert retried_save_s < CONVERSATION_SAVE_TIMEOUT_S
    # A healthy save plus the whole schedule plus the marker write a permanent
    # failure adds stays inside half the budget -- the margin the budget was
    # sized with. A longer schedule (more attempts, or a bigger base backoff)
    # spends that margin and fails here.
    assert (
        healthy_save_s + _retry_backoff_total() + worst_write_s
        <= CONVERSATION_SAVE_TIMEOUT_S / 2
    )


@pytest.mark.asyncio
async def test_a_marked_half_turn_leaves_a_concurrent_context_untouched(
    shared_memory_vespa, shared_denseon
):
    """Two contexts save at the same time -- proven by a barrier both user
    appends must reach -- and the one whose reply is lost takes the marker
    alone: the other context's turns land in dispatch order, unmarked."""
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    ctx_a = f"chat{uuid.uuid4().hex[:10]}"
    ctx_b = f"chat{uuid.uuid4().hex[:10]}"
    barrier = threading.Barrier(2, timeout=60.0)
    lock = threading.Lock()
    state = {"in_flight": 0, "peak": 0}
    synchronised: set = set()

    class _ConcurrentStore(ConversationStore):
        def store_turn(self, context_id, role, content):
            with lock:
                first_write = context_id not in synchronised
                if first_write:
                    synchronised.add(context_id)
            if first_write:
                # Both contexts' first appends are in flight together or this
                # raises, so the assertions below cannot pass on a serial run.
                barrier.wait()
            with lock:
                state["in_flight"] += 1
                state["peak"] = max(state["peak"], state["in_flight"])
            try:
                if context_id == ctx_a and role == "assistant":
                    raise ConnectionError("vespa reset context A's reply")
                super().store_turn(context_id, role, content)
            finally:
                with lock:
                    state["in_flight"] -= 1

    d = _dispatcher_with_real_store(mm)
    d._conversation_store_factory = lambda tenant_id: _ConcurrentStore(mm, tenant_id)
    seen: list = []
    _reply_with(
        d,
        {"in A": "answer A", "in B": "answer B", "more B": "answer more B"},
        seen,
    )

    await _dispatch(d, "in A", ctx_a)
    await _dispatch(d, "in B", ctx_b)
    await _dispatch(d, "more B", ctx_b)
    assert await d.drain_conversation_saves() is True

    assert barrier.broken is False
    assert state["peak"] == 2

    reader = ConversationStore(mm, TENANT)
    # B is untouched by A's failure, in dispatch order, with no marker.
    assert reader.get_history(ctx_b) == [
        {"role": "user", "content": "in B"},
        {"role": "assistant", "content": "answer B"},
        {"role": "user", "content": "more B"},
        {"role": "assistant", "content": "answer more B"},
    ]
    assert reader.get_missing_assistant_markers(ctx_b) == []
    # A keeps its user turn and takes the marker alone.
    assert reader.get_history(ctx_a) == [{"role": "user", "content": "in A"}]
    assert reader.get_missing_assistant_markers(ctx_a) == [
        {
            "role": "assistant_missing",
            "content": "assistant turn not persisted: ConnectionError",
        }
    ]
    assert d.conversation_persist_status() == {
        "pending": 0,
        "failed": [(TENANT, ctx_a)],
    }
