"""Behavioural contract for the inbound-messaging primitive.

Locks the observable behaviour of ``InboundMessage``, ``InboundQueue``,
and ``InboundQueueRegistry`` — the foundation Phase 1 wires the HTTP
route + orchestrator integration onto. Subsequent phases test the
INTEGRATION; this file tests only the primitive's promises to its
in-process consumers.

Every assertion below pins a behaviour the consumer (HTTP route or
running agent) relies on:

* The HTTP route reads ``get_queue(session_id)``: must return None
  for unknown sessions, must return the SAME instance the agent
  registered (not a copy).
* The agent's drain loop calls ``drain()``: must return the messages
  in submission order AND clear the buffer atomically, so the next
  iteration's drain sees only fresh messages.
* The agent's ``finally`` block calls ``close_queue(s)``: subsequent
  enqueues must raise so the HTTP route can surface a clean 404
  instead of orphaning messages.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import pytest

from cogniverse_runtime.messaging import (
    InboundMessage,
    InboundQueue,
    InboundQueueRegistry,
    QueueClosedError,
    get_inbound_queue_registry,
    reset_inbound_queue_registry_for_testing,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def _msg(content: str, *, deadline_ms: int | None = None) -> InboundMessage:
    return InboundMessage(
        session_id="sess-1",
        role="user",
        content=content,
        tags=("constraint",),
        created_at=datetime.now(timezone.utc).isoformat(),
        deadline_ms=deadline_ms,
    )


# --------------------------------------------------------------------- #
# Queue drain returns work-in-progress in submission order              #
# --------------------------------------------------------------------- #


async def test_drain_returns_messages_in_submission_order_then_clears():
    """The agent's iteration loop drains the queue between iterations.
    Drain MUST return everything pending in submission order so the
    agent processes constraints in the order the caller sent them,
    AND clear the buffer atomically so the next iteration doesn't
    re-process the same constraint.
    """
    q = InboundQueue("sess-1", "tenant-x")
    await q.enqueue(_msg("first"))
    await q.enqueue(_msg("second"))
    await q.enqueue(_msg("third"))

    first_drain = await q.drain()
    # Exact list — content + order — not "at least 3", not "contains 'first'".
    assert [m.content for m in first_drain] == ["first", "second", "third"]

    # Second drain immediately after sees an empty buffer; the agent's
    # next iteration MUST NOT re-process the same constraints.
    second_drain = await q.drain()
    assert second_drain == []

    # New enqueue after the drain shows up in a third drain.
    await q.enqueue(_msg("fourth"))
    third_drain = await q.drain()
    assert [m.content for m in third_drain] == ["fourth"]


# --------------------------------------------------------------------- #
# Past-deadline messages drop at drain (not at enqueue)                  #
# --------------------------------------------------------------------- #


async def test_past_deadline_messages_dropped_only_at_drain():
    """A sender that races a deadline must always see the same enqueue
    success path. The deadline check fires at drain time so an old
    message that expired in the buffer never reaches the agent.
    """
    q = InboundQueue("sess-1", "tenant-x")
    past = _msg("expired", deadline_ms=int(time.time() * 1000) - 1000)
    fresh = _msg("fresh", deadline_ms=int(time.time() * 1000) + 60_000)
    none = _msg("no-deadline", deadline_ms=None)

    # Enqueue all three — none raises, the past-deadline one is buffered
    # but invisible to the consumer.
    await q.enqueue(past)
    await q.enqueue(fresh)
    await q.enqueue(none)

    drained = await q.drain()
    assert [m.content for m in drained] == ["fresh", "no-deadline"]


# --------------------------------------------------------------------- #
# close() prevents further enqueue                                       #
# --------------------------------------------------------------------- #


async def test_close_prevents_further_enqueue():
    """After the agent's ``process()`` ``finally`` block closes the
    queue, late-arriving messages from the HTTP route must surface a
    clear error rather than silently buffering messages no one will
    ever drain.
    """
    q = InboundQueue("sess-1", "tenant-x")
    await q.enqueue(_msg("before"))
    q.close()
    assert q.is_closed is True

    # Idempotent — double-close doesn't raise.
    q.close()
    assert q.is_closed is True

    # Post-close enqueue raises with explicit reason.
    with pytest.raises(QueueClosedError) as exc_info:
        await q.enqueue(_msg("after"))
    assert "sess-1" in str(exc_info.value)
    assert "closed" in str(exc_info.value)


# --------------------------------------------------------------------- #
# Registry returns the SAME INSTANCE on re-resolve                       #
# --------------------------------------------------------------------- #


async def test_registry_get_or_create_returns_same_instance():
    """The HTTP route resolves the queue independently from the agent.
    Both must see the same queue object so a message enqueued by the
    HTTP route lands in the same buffer the agent drains. Returning
    a fresh instance silently would split the buffer and the agent
    would never see HTTP-side messages.
    """
    registry = InboundQueueRegistry()
    queue_a = await registry.get_or_create_queue("sess-1", "tenant-x")
    queue_b = await registry.get_or_create_queue("sess-1", "tenant-x")
    # Same instance — `is`, not `==`. Two object equality is the
    # weaker guarantee; identity is the strong one the consumers
    # rely on.
    assert queue_a is queue_b

    # Concurrent get_or_create from a different cooperative task must
    # also return the same instance (no race produces two queues).
    queue_c = await asyncio.gather(
        registry.get_or_create_queue("sess-1", "tenant-x"),
        registry.get_or_create_queue("sess-1", "tenant-x"),
    )
    assert queue_c[0] is queue_a
    assert queue_c[1] is queue_a


# --------------------------------------------------------------------- #
# Cross-tenant collision on session_id raises                            #
# --------------------------------------------------------------------- #


async def test_cross_tenant_session_id_collision_raises():
    """Session ids are tenant-scoped. If the same session_id is
    registered under two different tenants, the second registration
    raises — silent fall-through would let one tenant's messages
    land in another tenant's queue.
    """
    registry = InboundQueueRegistry()
    await registry.get_or_create_queue("shared-id", "alice")
    with pytest.raises(ValueError) as exc_info:
        await registry.get_or_create_queue("shared-id", "bob")
    assert "alice" in str(exc_info.value)
    assert "bob" in str(exc_info.value)


# --------------------------------------------------------------------- #
# get_queue() — None for unknown, exact instance for known               #
# --------------------------------------------------------------------- #


async def test_get_queue_returns_none_for_unknown_else_exact_instance():
    """The HTTP route uses ``get_queue`` to decide 202 vs 404. None
    must mean "no such active session" — not "create one for me".
    """
    registry = InboundQueueRegistry()
    # Never-seen session → None.
    assert await registry.get_queue("never-existed") is None

    created = await registry.get_or_create_queue("sess-1", "tenant-x")
    looked_up = await registry.get_queue("sess-1")
    # Same instance the agent registered — not a fresh queue.
    assert looked_up is created


# --------------------------------------------------------------------- #
# close_queue() returns True iff existed; subsequent get_queue → None    #
# --------------------------------------------------------------------- #


async def test_close_queue_returns_true_iff_existed_and_makes_get_queue_return_none():
    """Closing the queue removes it from the registry. The HTTP route
    must then return 404 for further messages — proven by
    ``get_queue`` returning None after close.
    """
    registry = InboundQueueRegistry()
    await registry.get_or_create_queue("sess-1", "tenant-x")

    assert await registry.close_queue("sess-1") is True
    # Idempotent for unknown ids — second close is False, not an error.
    assert await registry.close_queue("sess-1") is False
    assert await registry.close_queue("never-existed") is False

    # After close, get_queue returns None — HTTP route reads this
    # and responds 404.
    assert await registry.get_queue("sess-1") is None


# --------------------------------------------------------------------- #
# close_queue() also closes the live queue (post-close enqueue raises)   #
# --------------------------------------------------------------------- #


async def test_close_queue_propagates_close_to_live_queue():
    """The agent may hold a reference to the queue object after
    ``close_queue()`` removes it from the registry. That reference
    MUST also become unusable — otherwise a stale reference would
    silently accept messages no consumer ever drains.
    """
    registry = InboundQueueRegistry()
    q = await registry.get_or_create_queue("sess-1", "tenant-x")
    assert q.is_closed is False

    await registry.close_queue("sess-1")
    assert q.is_closed is True
    with pytest.raises(QueueClosedError):
        await q.enqueue(_msg("after-close"))


# --------------------------------------------------------------------- #
# Tenant-scoped listing — cross-tenant invisibility                       #
# --------------------------------------------------------------------- #


async def test_list_active_queues_filters_by_tenant():
    """Admin endpoints will use ``list_active_queues(tenant_id=...)`` to
    show only the calling tenant's sessions. A leak here would expose
    other tenants' session ids — the strong assertion: a tenant sees
    EXACTLY their own sessions, not a superset.
    """
    registry = InboundQueueRegistry()
    await registry.get_or_create_queue("alice-1", "alice")
    await registry.get_or_create_queue("alice-2", "alice")
    await registry.get_or_create_queue("bob-1", "bob")

    alice_sessions = sorted(
        e["session_id"] for e in await registry.list_active_queues("alice")
    )
    bob_sessions = sorted(
        e["session_id"] for e in await registry.list_active_queues("bob")
    )
    no_filter = sorted(e["session_id"] for e in await registry.list_active_queues())

    assert alice_sessions == ["alice-1", "alice-2"]
    assert bob_sessions == ["bob-1"]
    # No-filter call sees all three — same as a union of the two
    # tenant-scoped lists.
    assert no_filter == ["alice-1", "alice-2", "bob-1"]

    # Each entry carries the right tenant_id (not just session_id).
    alice_entries = await registry.list_active_queues("alice")
    assert all(e["tenant_id"] == "alice" for e in alice_entries)


# --------------------------------------------------------------------- #
# Singleton getter — same instance across calls                          #
# --------------------------------------------------------------------- #


async def test_singleton_registry_is_shared_across_callers():
    """The HTTP route and the orchestrator both call
    ``get_inbound_queue_registry()``. They MUST see the same instance,
    else messages from the HTTP route never reach the agent.
    """
    reset_inbound_queue_registry_for_testing()
    a = get_inbound_queue_registry()
    b = get_inbound_queue_registry()
    assert a is b

    # Different callers see the agent's registered sessions.
    await a.get_or_create_queue("sess-1", "tenant-x")
    found = await b.get_queue("sess-1")
    assert found is not None
    assert found.session_id == "sess-1"


# --------------------------------------------------------------------- #
# Concurrent enqueue / drain — atomicity                                  #
# --------------------------------------------------------------------- #


async def test_concurrent_enqueue_and_drain_preserves_every_message():
    """Concurrent producers + a draining consumer must not lose
    messages. 100 messages enqueued by 10 producers; the consumer
    drains repeatedly until all 100 have been seen. The set of
    consumed contents must equal the set of enqueued contents.
    """
    q = InboundQueue("sess-1", "tenant-x")
    expected = {f"msg-{i:03d}" for i in range(100)}

    async def produce(start: int) -> None:
        for i in range(start, start + 10):
            await q.enqueue(_msg(f"msg-{i:03d}"))

    producers = [asyncio.create_task(produce(s)) for s in range(0, 100, 10)]
    seen: set[str] = set()
    # Drain in a loop while producers run + a final drain after.
    while len(seen) < 100:
        batch = await q.drain()
        seen.update(m.content for m in batch)
        if not all(p.done() for p in producers):
            await asyncio.sleep(0.001)
    await asyncio.gather(*producers)
    seen.update(m.content for m in await q.drain())

    assert seen == expected
