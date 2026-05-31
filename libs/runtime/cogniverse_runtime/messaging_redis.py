"""Redis-backed durable cross-pod inbound messaging.

Replaces the in-pod
:class:`cogniverse_runtime.messaging.InboundQueueRegistry`
singleton with a Redis-backed implementation so messages survive
pod restarts AND route correctly across pods sharing the same
Redis instance. Activated when ``REDIS_URL`` is set in the runtime
env (see ``routers.agents._resolve_inbound_registry``).

Redis state shape:

* ``session:<session_id>:tenant`` — string with TTL. Value is the
  tenant_id. Existence of this key means the session is "active";
  its value is the only authoritative tenant binding. Cross-tenant
  collision is detected here (NX semantics on SET).
* ``inbound:<tenant_id>:<session_id>`` — list. ``enqueue`` does
  ``LPUSH``; ``drain`` runs a Lua script that returns all elements
  AND clears the list atomically.

Wiring:

* ``REDIS_URL`` env set → callers use :class:`RedisInboundQueueRegistry`
  via :func:`get_inbound_queue_registry_from_env`.
* ``REDIS_URL`` unset → in-pod singleton from ``messaging.py``.

The two paths share the same public surface (``InboundQueue`` +
``InboundQueueRegistry`` shape) so the HTTP route and orchestrator
don't branch — only the registry-factory does.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import redis.asyncio as aioredis

from cogniverse_runtime.messaging import (
    InboundMessage,
    QueueClosedError,
)

logger = logging.getLogger(__name__)


_DEFAULT_ACTIVE_TTL_SECONDS = 3600


# Lua script for atomic drain: returns all elements then clears the list.
# Without this, a concurrent enqueue between LRANGE and DEL would be
# either returned twice or lost. The script runs atomically server-side.
_DRAIN_LUA = """
local items = redis.call('LRANGE', KEYS[1], 0, -1)
redis.call('DEL', KEYS[1])
return items
"""


def _list_key(tenant_id: str, session_id: str) -> str:
    return f"inbound:{tenant_id}:{session_id}"


def _active_key(session_id: str) -> str:
    return f"session:{session_id}:tenant"


def _session_id_from_active_key(key: str) -> str:
    """Inverse of ``_active_key`` — strips the fixed prefix/suffix so a
    session_id containing a colon round-trips intact."""
    return key[len("session:") : -len(":tenant")]


def _serialize(msg: InboundMessage) -> str:
    """JSON encode an InboundMessage. ``tags`` is a tuple — JSON does
    not preserve tuple vs list, so deserialize coerces back."""
    d = asdict(msg)
    d["tags"] = list(msg.tags)
    return json.dumps(d, sort_keys=True)


def _deserialize(s: str) -> InboundMessage:
    d = json.loads(s)
    return InboundMessage(
        session_id=d["session_id"],
        role=d["role"],
        content=d["content"],
        tags=tuple(d.get("tags") or ()),
        created_at=d["created_at"],
        deadline_ms=d.get("deadline_ms"),
    )


class RedisInboundQueue:
    """Redis-backed per-session FIFO of :class:`InboundMessage`.

    All state lives in Redis. This class holds only a reference to
    the shared aioredis client + the session/tenant identity. Two
    instances bound to the same (session_id, tenant_id, redis) act
    as one queue — drains and enqueues from either land in the same
    list.
    """

    def __init__(
        self,
        session_id: str,
        tenant_id: str,
        redis: aioredis.Redis,
        created_at: Optional[datetime] = None,
        active_ttl_seconds: int = _DEFAULT_ACTIVE_TTL_SECONDS,
    ) -> None:
        self._session_id = session_id
        self._tenant_id = tenant_id
        self._redis = redis
        self._created_at = created_at or datetime.now(timezone.utc)
        self._active_ttl = active_ttl_seconds

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    async def is_closed(self) -> bool:
        # No active-marker → closed. Lookup is one round-trip.
        v = await self._redis.get(_active_key(self._session_id))
        return v is None

    async def _check_open(self) -> None:
        v = await self._redis.get(_active_key(self._session_id))
        if v is None:
            raise QueueClosedError(
                f"queue '{self._session_id}' is closed; agent session "
                "has already finished"
            )

    async def enqueue(self, msg: InboundMessage) -> None:
        """Append ``msg`` to the Redis list for this session.

        Refuses to enqueue when the active-marker is gone (session
        closed or never registered) — surfaces ``QueueClosedError``
        with the same shape as the in-pod version.
        """
        await self._check_open()
        # Bound the list lifetime to the active-marker TTL so an abandoned
        # session (never explicitly closed) self-expires instead of leaking.
        list_key = _list_key(self._tenant_id, self._session_id)
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.lpush(list_key, _serialize(msg))
            pipe.expire(list_key, self._active_ttl)
            await pipe.execute()

    async def drain(self) -> List[InboundMessage]:
        """Atomically return all currently-buffered messages AND
        clear the list. Server-side Lua script guarantees no
        concurrent ``enqueue`` is partially observed. Past-deadline
        messages are filtered client-side after the Redis read."""
        raw = await self._redis.eval(
            _DRAIN_LUA, 1, _list_key(self._tenant_id, self._session_id)
        )
        # Redis LPUSH stores in LIFO order — reverse so consumers see
        # submission order (oldest first).
        msgs = [_deserialize(s) for s in reversed(raw or [])]
        now_ms = int(time.time() * 1000)
        return [m for m in msgs if m.deadline_ms is None or m.deadline_ms >= now_ms]


class RedisInboundQueueRegistry:
    """Cross-pod registry for :class:`RedisInboundQueue` instances.

    Mirrors the public surface of the in-pod
    :class:`cogniverse_runtime.messaging.InboundQueueRegistry` so the
    HTTP route and orchestrator can use either backend
    interchangeably.

    Session state lives entirely in Redis — Python instances of
    ``RedisInboundQueue`` are stateless handles. Two pods sharing
    one Redis see the same sessions.
    """

    def __init__(
        self,
        redis: aioredis.Redis,
        active_ttl_seconds: int = _DEFAULT_ACTIVE_TTL_SECONDS,
    ) -> None:
        self._redis = redis
        self._active_ttl = active_ttl_seconds

    async def get_or_create_queue(
        self, session_id: str, tenant_id: str
    ) -> RedisInboundQueue:
        """Atomic create-or-noop via ``SET NX``.

        If the active-marker doesn't exist, set it to ``tenant_id``.
        If it exists with the SAME tenant_id, refresh the TTL and
        return a handle. If it exists with a DIFFERENT tenant_id,
        raise ``ValueError`` so a routing bug surfaces rather than
        cross-tenant bleed.
        """
        key = _active_key(session_id)
        # NX returns True on first set, None when key already exists.
        set_ok = await self._redis.set(key, tenant_id, ex=self._active_ttl, nx=True)
        if not set_ok:
            existing = await self._redis.get(key)
            if existing != tenant_id:
                raise ValueError(
                    f"session_id '{session_id}' already registered under "
                    f"tenant '{existing}', cannot rebind to "
                    f"tenant '{tenant_id}'"
                )
            # Refresh TTL so long-running sessions don't expire mid-flight.
            await self._redis.expire(key, self._active_ttl)
        return RedisInboundQueue(
            session_id=session_id,
            tenant_id=tenant_id,
            redis=self._redis,
            active_ttl_seconds=self._active_ttl,
        )

    async def get_queue(self, session_id: str) -> Optional[RedisInboundQueue]:
        """Return a queue handle if the session is active in Redis.

        Read-only: no creation. The HTTP route uses this to decide
        between 202 (active) and 404 (not active).
        """
        existing = await self._redis.get(_active_key(session_id))
        if existing is None:
            return None
        return RedisInboundQueue(
            session_id=session_id,
            tenant_id=existing,
            redis=self._redis,
            active_ttl_seconds=self._active_ttl,
        )

    async def close_queue(self, session_id: str) -> bool:
        """Close + remove the session AND drop any unconsumed messages.

        Returns ``True`` if the active-marker existed,
        ``False`` otherwise. The list under
        ``inbound:<tenant>:<session>`` is deleted too so closed
        sessions don't leave orphaned data accumulating in Redis.
        """
        existing = await self._redis.get(_active_key(session_id))
        if existing is None:
            return False
        # Delete BOTH keys atomically via pipeline.
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.delete(_active_key(session_id))
            pipe.delete(_list_key(existing, session_id))
            await pipe.execute()
        return True

    async def list_active_queues(
        self, tenant_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Scan all active sessions; filter by tenant if requested.

        Uses ``SCAN`` (non-blocking) rather than ``KEYS``. Cluster
        churn on the scan is acceptable — the result is best-effort
        for admin endpoints.
        """
        out: List[Dict[str, str]] = []
        async for key in self._redis.scan_iter(match="session:*:tenant"):
            value = await self._redis.get(key)
            if value is None:
                continue
            session_id = (
                _session_id_from_active_key(key) if isinstance(key, str) else key
            )
            if tenant_id is None or value == tenant_id:
                out.append(
                    {
                        "session_id": session_id,
                        "tenant_id": value,
                        # created_at is approximated as "now" since
                        # the active-marker doesn't store it; admin
                        # consumers that need precise create-time
                        # should read the per-session metadata key.
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
        return out


# --------------------------------------------------------------------- #
# Factory: env-driven registry selection                                  #
# --------------------------------------------------------------------- #


_redis_registry_singleton: Optional[RedisInboundQueueRegistry] = None


async def get_redis_inbound_queue_registry(
    redis_url: str,
    *,
    active_ttl_seconds: int = _DEFAULT_ACTIVE_TTL_SECONDS,
) -> RedisInboundQueueRegistry:
    """Process-wide singleton for the Redis-backed registry.

    Multiple callers in the same process share one aioredis pool.
    Different pods share one Redis instance and therefore one
    logical registry.
    """
    global _redis_registry_singleton
    if _redis_registry_singleton is None:
        from cogniverse_runtime.ingestion_worker.redis_client import get_redis

        redis = await get_redis(redis_url)
        _redis_registry_singleton = RedisInboundQueueRegistry(
            redis, active_ttl_seconds=active_ttl_seconds
        )
    return _redis_registry_singleton


def reset_redis_inbound_queue_registry_for_testing() -> None:
    """Test-only: drop the Python-side singleton.

    The REDIS STATE persists across this reset by design — the
    durability assertions rely on it. To wipe Redis between tests,
    call ``RedisInboundQueueRegistry`` methods or ``FLUSHDB``
    explicitly in the test fixture.
    """
    global _redis_registry_singleton
    _redis_registry_singleton = None
