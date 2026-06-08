"""Per-session inbound messaging for running agents.

Mirrors the ``cogniverse_core.events`` outbound pattern (``EventQueue``
+ ``BaseQueueManager``) but inverted: the caller pushes messages INTO
a running agent's session, and the agent drains them at well-defined
boundaries (between iterative-retrieval iterations, between RLM REPL
turns, etc.).

Three primitives:

* :class:`InboundMessage` — frozen dataclass carrying ``session_id``,
  ``role``, ``content``, ``tags``, ``created_at``, ``deadline_ms``.
  Tags drive agent behaviour: ``("stop",)`` triggers cooperative
  cancellation; ``("constraint",)`` / ``("interrupt",)`` inject
  context into the next iteration; ``("system",)`` is reserved for
  supervisor messages.

* :class:`InboundQueue` — async per-session FIFO. ``enqueue()`` is
  non-blocking; ``drain()`` returns all currently-buffered messages
  AND atomically clears the buffer (so the next drain returns ``[]``
  until new messages arrive). Past-deadline messages are silently
  dropped at drain time, not at enqueue — so a slow agent that
  drains rarely still sees fresh messages even if older ones expired
  in the buffer.

* :class:`InboundQueueRegistry` — registry of
  ``(session_id) -> InboundQueue``. The HTTP route + agent
  integration both go through the registry so a single
  ``session_id`` resolves to the same in-process queue regardless of
  which side opened it. ``get_or_create_queue()`` is idempotent:
  second call returns the exact same queue instance, so the
  agent's existing reference stays valid even if the HTTP route
  re-resolves the session.

This module is the in-pod registry implementation. For multi-pod
routing + durability across pod restarts, callers should resolve
the registry through ``agents._resolve_inbound_registry`` (or
``orchestrator_agent._resolve_inbound_registry_for_orchestrator``)
which swaps in :class:`cogniverse_runtime.messaging_redis.RedisInboundQueueRegistry`
when ``REDIS_URL`` is set.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class QueueClosedError(RuntimeError):
    """Raised by :meth:`InboundQueue.enqueue` after the queue has been closed.

    Closure happens when the owning agent's ``process()`` returns —
    its ``finally`` block calls :meth:`InboundQueueRegistry.close_queue`
    so further inbound messages have no listener and we surface that
    fact rather than silently dropping.
    """


@dataclass(frozen=True)
class InboundMessage:
    """A single inbound message bound for a running agent session.

    Frozen so it's hashable + safe to share across the async boundary
    between the HTTP route handler and the agent's drain loop.
    """

    session_id: str
    role: str
    content: str
    tags: Tuple[str, ...]
    created_at: str
    deadline_ms: Optional[int] = None


class InboundQueue:
    """Per-session async FIFO of :class:`InboundMessage`.

    Buffered in-memory. Order-preserving via list append + pop-all
    semantics in :meth:`drain`. Past-deadline messages drop at drain
    so consumers always see a clean buffer of still-valid messages.
    """

    def __init__(self, session_id: str, tenant_id: str) -> None:
        self._session_id = session_id
        self._tenant_id = tenant_id
        self._buffer: List[InboundMessage] = []
        self._lock = asyncio.Lock()
        self._closed = False
        self._created_at = datetime.now(timezone.utc)

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    async def is_closed(self) -> bool:
        # Async to match the durable RedisInboundQueue (which needs a round
        # trip) so the two backends are interchangeable behind ``await``.
        return self._closed

    @property
    def created_at(self) -> datetime:
        return self._created_at

    async def enqueue(self, msg: InboundMessage) -> None:
        """Append ``msg`` to the buffer.

        Raises :class:`QueueClosedError` if the queue has been closed
        (agent's ``process()`` returned). Past-deadline messages are
        accepted at enqueue and dropped at drain — letting senders
        race a deadline without seeing different success codes.
        """
        if self._closed:
            raise QueueClosedError(
                f"queue '{self._session_id}' is closed; agent session "
                "has already finished"
            )
        async with self._lock:
            self._buffer.append(msg)

    async def drain(self) -> List[InboundMessage]:
        """Return all currently-buffered messages AND clear the buffer.

        Atomic under ``self._lock``: a concurrent ``enqueue()`` either
        lands fully before the drain (and is returned) or fully after
        (and is preserved for the next drain). Past-deadline messages
        are filtered out — they never reach the consumer.
        """
        now_ms = int(time.time() * 1000)
        async with self._lock:
            batch = self._buffer
            self._buffer = []
        # Drop past-deadline outside the lock — pure filter, no shared state.
        return [m for m in batch if m.deadline_ms is None or m.deadline_ms >= now_ms]

    def close(self) -> None:
        """Mark the queue closed. Idempotent — second call is a no-op."""
        self._closed = True


@dataclass
class _RegistryEntry:
    queue: InboundQueue
    tenant_id: str = field(default="")


class InboundQueueRegistry:
    """Registry of ``(session_id) -> InboundQueue``.

    Single source of truth for in-flight agent sessions in this pod.
    The HTTP route looks up the target queue by session_id; the
    agent's ``process()`` registers itself at start and closes at
    finish. Tenant id is recorded so ``list_active_queues`` can
    filter cross-tenant.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, _RegistryEntry] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_queue(
        self, session_id: str, tenant_id: str
    ) -> InboundQueue:
        """Return the existing queue for ``session_id`` OR create one.

        Idempotent: a second call with the same ``session_id`` returns
        the same instance (``a is b``), so the agent's stored
        reference stays valid even if the HTTP route resolves the
        session again concurrently. Cross-tenant collision: if
        ``session_id`` is reused under a different ``tenant_id``,
        raises ``ValueError`` — session ids are scoped per-tenant and
        an unintentional collision indicates a routing bug upstream.
        """
        async with self._lock:
            existing = self._entries.get(session_id)
            if existing is not None:
                if existing.tenant_id and existing.tenant_id != tenant_id:
                    raise ValueError(
                        f"session_id '{session_id}' already registered under "
                        f"tenant '{existing.tenant_id}', cannot rebind to "
                        f"tenant '{tenant_id}'"
                    )
                return existing.queue
            queue = InboundQueue(session_id, tenant_id)
            self._entries[session_id] = _RegistryEntry(queue=queue, tenant_id=tenant_id)
            return queue

    async def get_queue(self, session_id: str) -> Optional[InboundQueue]:
        """Return the queue if ``session_id`` is active, else ``None``.

        Read-only; never creates. The HTTP route uses this to decide
        between 202 (active) and 404 (not active).
        """
        async with self._lock:
            entry = self._entries.get(session_id)
            return entry.queue if entry else None

    async def close_queue(self, session_id: str) -> bool:
        """Close and remove the queue for ``session_id``.

        Returns ``True`` if the queue existed and was closed,
        ``False`` if no such session was registered. The agent's
        ``process()`` ``finally`` block calls this — subsequent
        :meth:`get_queue` returns ``None`` and the HTTP route 404s
        on further messages.
        """
        async with self._lock:
            entry = self._entries.pop(session_id, None)
        if entry is None:
            return False
        entry.queue.close()
        return True

    async def list_active_queues(
        self, tenant_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """List currently-active session ids.

        If ``tenant_id`` is given, filter to that tenant only.
        Returned dicts carry ``session_id``, ``tenant_id``, and
        ``created_at`` ISO-8601 — enough for an admin endpoint to
        show "what's running" without exposing the queues themselves.
        """
        async with self._lock:
            entries = list(self._entries.items())
        return [
            {
                "session_id": sess,
                "tenant_id": entry.tenant_id,
                "created_at": entry.queue.created_at.isoformat(),
            }
            for sess, entry in entries
            if tenant_id is None or entry.tenant_id == tenant_id
        ]


# Module-level singleton — mirrors the cogniverse_runtime pattern of
# in-pod registries (see EventQueueRegistry, AgentRegistry).
# When ``REDIS_URL`` is set in the env, callers should resolve the
# registry via the async factory in ``messaging_redis`` instead,
# which returns a cross-pod durable backend bound to the same Redis
# instance as the ingestion queue.
_singleton_registry: Optional[InboundQueueRegistry] = None


def get_inbound_queue_registry() -> InboundQueueRegistry:
    """Return the process-wide in-pod :class:`InboundQueueRegistry`.

    Single-pod / no-Redis path. For multi-pod or durable cross-pod
    state, use ``messaging_redis.get_redis_inbound_queue_registry``.
    """
    global _singleton_registry
    if _singleton_registry is None:
        _singleton_registry = InboundQueueRegistry()
    return _singleton_registry


def reset_inbound_queue_registry_for_testing() -> None:
    """Test-only: drop the singleton so each test starts clean.

    Production code never calls this. The orchestrator + HTTP route
    both go through :func:`get_inbound_queue_registry` so resetting
    the singleton between tests is the only way to keep them
    isolated without a shared fixture.
    """
    global _singleton_registry
    _singleton_registry = None
