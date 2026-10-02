"""Per-session state the runtime's processes share through Redis.

* ``ConversationLedger`` orders a server-managed context's turns and tracks
  the saves still landing, so whichever process serves the next turn waits
  for them and then reads them.
* ``ContinuationStore`` holds a suspended ``/v1`` turn's state until its tool
  results come back, readable once from any process.

Every command, connect and wait for a pooled connection is bounded by the
client's timeout. An unreachable or silent Redis raises
``SessionStateUnavailable``; nothing falls back to process memory.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from redis.asyncio import BlockingConnectionPool, Redis
from redis.exceptions import RedisError

from cogniverse_runtime.shared_state import redacted_redis_url

# Bound on one Redis command, a connect, and the wait for a pooled connection.
SESSION_REDIS_TIMEOUT_SECONDS = 5.0
SESSION_REDIS_MAX_CONNECTIONS = 64
_HEALTH_CHECK_INTERVAL_SECONDS = 30

CONVERSATION_KEY_PREFIX = "cogniverse:conversation"
CONTINUATION_KEY_PREFIX = "cogniverse:continuation"

# How long a context's ordering clock and failure record outlive its last
# turn. The clock restarts from Redis time once it expires, which still orders
# after every position it handed out.
CONVERSATION_STATE_RETENTION_S = 7 * 24 * 3600.0

# How often a load re-reads the saves it waits for.
CONVERSATION_PENDING_POLL_S = 0.05

# How long a suspended /v1 turn's state waits for its tool results.
CONTINUATION_TTL_SECONDS = 600.0


class SessionStateUnavailable(RuntimeError):
    """Redis could not complete a session-state operation."""


class ConversationPersistFailed(Exception):
    """A dispatched turn's conversation history was not persisted.

    Names the tenant, the context, the failure's type and the turn's position;
    the failure's message stays in the runtime log, since it can quote the
    turn's text.
    """

    def __init__(
        self, tenant_id: str, context_id: str, error_type: str, position: int
    ) -> None:
        super().__init__(
            f"conversation turns for context {context_id} (tenant {tenant_id}) "
            f"were not persisted: {error_type}"
        )
        self.tenant_id = tenant_id
        self.context_id = context_id
        self.error_type = error_type
        self.position = position


async def open_session_redis(
    redis_url: str,
    *,
    timeout_seconds: float = SESSION_REDIS_TIMEOUT_SECONDS,
    max_connections: int = SESSION_REDIS_MAX_CONNECTIONS,
) -> Redis:
    """Connect to Redis and confirm it answers."""
    if not redis_url.strip():
        raise ValueError("redis_url must be non-empty")
    client = Redis.from_pool(
        BlockingConnectionPool.from_url(
            redis_url,
            decode_responses=True,
            max_connections=max_connections,
            timeout=timeout_seconds,
            socket_timeout=timeout_seconds,
            socket_connect_timeout=timeout_seconds,
            socket_keepalive=True,
            health_check_interval=_HEALTH_CHECK_INTERVAL_SECONDS,
        )
    )
    try:
        await client.ping()
    except RedisError as exc:
        await client.aclose()
        raise SessionStateUnavailable(
            f"session state store unavailable: connect to "
            f"{redacted_redis_url(redis_url)}"
        ) from exc
    return client


def _digest(*parts: str) -> str:
    return hashlib.sha256(json.dumps(list(parts)).encode()).hexdigest()


# Positions come from Redis' clock in microseconds and step by two per turn:
# the user turn takes the position and the reply the next one. A context's
# clock never hands out a position at or below its last one.
_ACCEPT_SCRIPT = """
local now = redis.call('TIME')
local now_us = tonumber(now[1]) * 1000000 + tonumber(now[2])
local now_ms = math.floor(now_us / 1000)
local last = tonumber(redis.call('HGET', KEYS[1], 'clock') or '0')
local position = last + 2
if position < now_us then
  position = now_us
end
local text = string.format('%.0f', position)
redis.call('HSET', KEYS[1], 'clock', text)
redis.call('PEXPIRE', KEYS[1], ARGV[2])
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now_ms)
redis.call('ZADD', KEYS[2], now_ms + tonumber(ARGV[1]), text)
redis.call('PEXPIRE', KEYS[2], ARGV[1])
return text
"""

_PENDING_SCRIPT = """
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
return redis.call('ZRANGE', KEYS[1], '(' .. now_ms, '+inf', 'BYSCORE')
"""

# A failure is recovered once a later turn of the same context lands, in
# whichever order the two outcomes reach Redis.
_LANDED_SCRIPT = """
redis.call('ZREM', KEYS[2], ARGV[1])
local position = tonumber(ARGV[1])
local landed = tonumber(redis.call('HGET', KEYS[1], 'landed') or '0')
if position > landed then
  redis.call('HSET', KEYS[1], 'landed', ARGV[1])
  landed = position
end
local failed = redis.call('HGET', KEYS[1], 'failed_position')
if failed and tonumber(failed) < landed then
  redis.call('HDEL', KEYS[1], 'failed_position', 'error_type')
  redis.call('ZREM', KEYS[3], KEYS[1])
end
redis.call('PEXPIRE', KEYS[1], ARGV[2])
return 0
"""

_FAILED_SCRIPT = """
redis.call('ZREM', KEYS[2], ARGV[1])
local position = tonumber(ARGV[1])
local landed = tonumber(redis.call('HGET', KEYS[1], 'landed') or '0')
local failed = redis.call('HGET', KEYS[1], 'failed_position')
if landed > position or (failed and tonumber(failed) > position) then
  return 0
end
redis.call('HSET', KEYS[1], 'tenant_id', ARGV[3], 'context_id', ARGV[4],
  'failed_position', ARGV[1], 'error_type', ARGV[5])
redis.call('PEXPIRE', KEYS[1], ARGV[2])
redis.call('ZADD', KEYS[3], redis.call('INCR', KEYS[4]), KEYS[1])
local excess = redis.call('ZCARD', KEYS[3]) - tonumber(ARGV[6])
if excess > 0 then
  local evicted = redis.call('ZRANGE', KEYS[3], 0, excess - 1)
  for _, state in ipairs(evicted) do
    redis.call('HDEL', state, 'failed_position', 'error_type')
  end
  redis.call('ZREMRANGEBYRANK', KEYS[3], 0, excess - 1)
end
return 1
"""

_FAILURES_SCRIPT = """
local out = {}
for _, state in ipairs(redis.call('ZRANGE', KEYS[1], 0, -1)) do
  local row = redis.call('HMGET', state, 'tenant_id', 'context_id', 'failed_position')
  if row[3] then
    table.insert(out, row[1])
    table.insert(out, row[2])
  else
    redis.call('ZREM', KEYS[1], state)
  end
end
return out
"""


class ConversationLedger:
    """Turn order and in-flight saves of server-managed conversations.

    A turn takes its position when its reply is accepted, before the reply
    returns, and stays pending until its save lands or fails. A load waits
    for the pending turns of its context, so the next turn reads the previous
    one whichever process served it, and positions order the stored turns
    whichever save lands first. A pending turn whose process died expires
    with its lease.
    """

    def __init__(
        self,
        redis: Redis,
        *,
        save_lease_s: float,
        failure_capacity: int,
        key_prefix: str = CONVERSATION_KEY_PREFIX,
        retention_s: float = CONVERSATION_STATE_RETENTION_S,
        poll_interval_s: float = CONVERSATION_PENDING_POLL_S,
    ) -> None:
        if save_lease_s <= 0 or retention_s <= save_lease_s:
            raise ValueError(
                f"retention_s ({retention_s}) must exceed save_lease_s "
                f"({save_lease_s}) > 0"
            )
        if failure_capacity < 1:
            raise ValueError(f"failure_capacity must be >= 1, got {failure_capacity}")
        self._redis = redis
        self._prefix = key_prefix.rstrip(":")
        self._lease_ms = int(save_lease_s * 1000)
        self._retention_ms = int(retention_s * 1000)
        self._failure_capacity = failure_capacity
        self._poll_interval_s = poll_interval_s
        self._accept = redis.register_script(_ACCEPT_SCRIPT)
        self._pending = redis.register_script(_PENDING_SCRIPT)
        self._landed = redis.register_script(_LANDED_SCRIPT)
        self._failed = redis.register_script(_FAILED_SCRIPT)
        self._failures = redis.register_script(_FAILURES_SCRIPT)

    def _state_key(self, tenant_id: str, context_id: str) -> str:
        return f"{self._prefix}:context:{_digest(tenant_id, context_id)}"

    def _keys(self, tenant_id: str, context_id: str) -> Tuple[str, str, str, str]:
        state = self._state_key(tenant_id, context_id)
        failures = f"{self._prefix}:failures"
        return state, f"{state}:pending", failures, f"{failures}:order"

    async def accept(self, tenant_id: str, context_id: str) -> int:
        """Give the context's next turn its position and mark it pending."""
        state, pending, _, _ = self._keys(tenant_id, context_id)
        try:
            position = await self._accept(
                keys=[state, pending], args=[self._lease_ms, self._retention_ms]
            )
        except RedisError as exc:
            raise self._unavailable("accept a turn of", context_id) from exc
        return int(position)

    async def pending(self, tenant_id: str, context_id: str) -> List[int]:
        """Positions of the context's turns whose saves have not settled."""
        _, pending, _, _ = self._keys(tenant_id, context_id)
        try:
            positions = await self._pending(keys=[pending], args=[])
        except RedisError as exc:
            raise self._unavailable("read the pending turns of", context_id) from exc
        return sorted(int(position) for position in positions)

    async def wait_settled(
        self,
        tenant_id: str,
        context_id: str,
        positions: Sequence[int],
        timeout_s: float,
    ) -> List[int]:
        """Wait until ``positions`` settle; return those still pending at the
        deadline."""
        waiting = set(positions)
        deadline = time.monotonic() + timeout_s
        while waiting:
            waiting &= set(await self.pending(tenant_id, context_id))
            if not waiting or time.monotonic() >= deadline:
                break
            await asyncio.sleep(
                min(self._poll_interval_s, max(deadline - time.monotonic(), 0.0))
            )
        return sorted(waiting)

    async def landed(self, tenant_id: str, context_id: str, position: int) -> None:
        """Settle a turn whose save landed."""
        state, pending, failures, _ = self._keys(tenant_id, context_id)
        try:
            await self._landed(
                keys=[state, pending, failures],
                args=[str(position), self._retention_ms],
            )
        except RedisError as exc:
            raise self._unavailable("settle a turn of", context_id) from exc

    async def failed(
        self, tenant_id: str, context_id: str, position: int, error_type: str
    ) -> None:
        """Settle a turn whose save failed and record the loss."""
        try:
            await self._failed(
                keys=list(self._keys(tenant_id, context_id)),
                args=[
                    str(position),
                    self._retention_ms,
                    tenant_id,
                    context_id,
                    error_type,
                    self._failure_capacity,
                ],
            )
        except RedisError as exc:
            raise self._unavailable("record a lost turn of", context_id) from exc

    async def failure(
        self, tenant_id: str, context_id: str
    ) -> Optional[ConversationPersistFailed]:
        """The context's unrecovered persistence failure, if any."""
        state = self._state_key(tenant_id, context_id)
        try:
            position, error_type = await self._redis.hmget(
                state, "failed_position", "error_type"
            )
        except RedisError as exc:
            raise self._unavailable("read the lost turns of", context_id) from exc
        if position is None:
            return None
        return ConversationPersistFailed(
            tenant_id, context_id, error_type, int(position)
        )

    async def failures(self) -> List[Tuple[str, str]]:
        """``(tenant_id, context_id)`` of every unrecovered failure, oldest
        first, at most ``failure_capacity`` of them."""
        try:
            flat = await self._failures(keys=[f"{self._prefix}:failures"], args=[])
        except RedisError as exc:
            raise SessionStateUnavailable(
                "session state store unavailable: list lost conversation turns"
            ) from exc
        return list(zip(flat[0::2], flat[1::2]))

    @staticmethod
    def _unavailable(action: str, context_id: str) -> SessionStateUnavailable:
        return SessionStateUnavailable(
            f"session state store unavailable: {action} context {context_id}"
        )


class ContinuationStore:
    """Suspended ``/v1`` turns keyed by tenant, agent, seed and call ids.

    The tenant and the agent are part of the key: without them a second
    tenant replaying the same opening message and the same call ids would be
    handed the first tenant's plan, and because the read is one-shot the
    owner would resume with nothing. A read takes the state atomically, so
    two resumes racing on different processes never both get it.
    """

    def __init__(
        self,
        redis: Redis,
        *,
        key_prefix: str = CONTINUATION_KEY_PREFIX,
        ttl_seconds: float = CONTINUATION_TTL_SECONDS,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be > 0, got {ttl_seconds}")
        self._redis = redis
        self._prefix = key_prefix.rstrip(":")
        self._ttl_ms = int(ttl_seconds * 1000)

    def _key(
        self, tenant_id: str, agent_name: str, seed: str, call_ids: Sequence[Any]
    ) -> str:
        ids = sorted(str(call_id) for call_id in call_ids)
        return f"{self._prefix}:{_digest(tenant_id, agent_name, seed, *ids)}"

    async def put(
        self,
        tenant_id: str,
        agent_name: str,
        seed: str,
        call_ids: Sequence[Any],
        state: Dict[str, Any],
    ) -> None:
        """Keep a suspended turn's state until its tool results come back."""
        payload = json.dumps(state)
        try:
            await self._redis.set(
                self._key(tenant_id, agent_name, seed, call_ids),
                payload,
                px=self._ttl_ms,
            )
        except RedisError as exc:
            raise SessionStateUnavailable(
                f"session state store unavailable: keep the suspended turn of "
                f"{agent_name}"
            ) from exc

    async def pop(
        self,
        tenant_id: str,
        agent_name: str,
        seed: str,
        call_ids: Sequence[Any],
    ) -> Optional[Dict[str, Any]]:
        """Take a suspended turn's state; None once taken or expired."""
        try:
            payload = await self._redis.getdel(
                self._key(tenant_id, agent_name, seed, call_ids)
            )
        except RedisError as exc:
            raise SessionStateUnavailable(
                f"session state store unavailable: resume the suspended turn of "
                f"{agent_name}"
            ) from exc
        return None if payload is None else json.loads(payload)

    async def count(self) -> int:
        """Suspended turns currently held."""
        try:
            return len(
                [key async for key in self._redis.scan_iter(f"{self._prefix}:*")]
            )
        except RedisError as exc:
            raise SessionStateUnavailable(
                "session state store unavailable: count suspended turns"
            ) from exc
