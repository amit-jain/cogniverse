"""Redis persistence for A2A protocol tasks."""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from collections.abc import AsyncIterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

from a2a.server.context import ServerCallContext
from a2a.server.events import Event
from a2a.server.tasks import TaskStore
from a2a.types import (
    Message,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_text_message
from pydantic import TypeAdapter, ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError

_UNAVAILABLE = "shared A2A task store unavailable"


class A2ATaskStoreError(RuntimeError):
    """Raised when the shared A2A task store cannot complete an operation."""


class A2ATaskCapacityError(A2ATaskStoreError):
    """Raised when capacity is full and every retained task is active."""


class A2ATaskConflictError(A2ATaskStoreError):
    """Raised when another request owns a task execution lease."""


class A2ATaskOwnershipLostError(A2ATaskStoreError):
    """Raised when an execution tries to write after losing ownership."""


class A2ACancelTimeoutError(A2ATaskStoreError):
    """Raised when a task owner does not acknowledge cancellation in time."""


@dataclass(frozen=True)
class TaskLease:
    """A generation-fenced execution lease for one task."""

    task_id: str
    replica_id: str
    request_id: str
    generation: int
    expires_at_ms: int


@dataclass(frozen=True)
class CancelCommand:
    """Cancellation request delivered to the replica that owns a task."""

    request_id: str
    task_id: str
    reply_key: str


_bound_lease: ContextVar[TaskLease | None] = ContextVar(
    "cogniverse_a2a_task_lease", default=None
)
_CALL_CONTEXT_LEASE_KEY = "cogniverse.a2a.task_lease"
_EVENT_ADAPTER = TypeAdapter(
    Message | Task | TaskStatusUpdateEvent | TaskArtifactUpdateEvent
)


_SAVE_SCRIPT = """
local tasks_key = KEYS[1]
local inactive_key = KEYS[2]
local active_key = KEYS[3]
local leases_key = KEYS[4]
local task_id = ARGV[1]
local payload = ARGV[2]
local active = ARGV[3]
local max_tasks = tonumber(ARGV[4])
local enforce_lease = ARGV[5]
local request_id = ARGV[6]
local generation = tonumber(ARGV[7])
if enforce_lease == '1' then
    local raw_lease = redis.call('HGET', leases_key, task_id)
    if not raw_lease then
        return {-1, ''}
    end
    local lease = cjson.decode(raw_lease)
    local now = redis.call('TIME')
    local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
    if lease.request_id ~= request_id
        or tonumber(lease.generation) ~= generation
        or tonumber(lease.expires_at_ms) <= now_ms then
        return {-1, ''}
    end
end
local existed = redis.call('HEXISTS', tasks_key, task_id)
local evicted = ''

if existed == 0 and redis.call('HLEN', tasks_key) >= max_tasks then
    local victims = redis.call('ZRANGE', inactive_key, 0, 0)
    if #victims == 0 then
        return {0, ''}
    end
    evicted = victims[1]
    redis.call('HDEL', tasks_key, evicted)
    redis.call('ZREM', inactive_key, evicted)
    redis.call('SREM', active_key, evicted)
end

local now = redis.call('TIME')
local score = tonumber(now[1]) * 1000000 + tonumber(now[2])
redis.call('HSET', tasks_key, task_id, payload)
if active == '1' then
    redis.call('SADD', active_key, task_id)
    redis.call('ZREM', inactive_key, task_id)
else
    redis.call('SREM', active_key, task_id)
    redis.call('ZADD', inactive_key, score, task_id)
end
return {1, evicted}
"""

_GET_SCRIPT = """
local payload = redis.call('HGET', KEYS[1], ARGV[1])
if payload and redis.call('SISMEMBER', KEYS[3], ARGV[1]) == 0 then
    local now = redis.call('TIME')
    local score = tonumber(now[1]) * 1000000 + tonumber(now[2])
    redis.call('ZADD', KEYS[2], score, ARGV[1])
end
return payload
"""

_DELETE_SCRIPT = """
redis.call('HDEL', KEYS[1], ARGV[1])
redis.call('ZREM', KEYS[2], ARGV[1])
redis.call('SREM', KEYS[3], ARGV[1])
return 1
"""

_ACQUIRE_SCRIPT = """
local leases_key = KEYS[1]
local generations_key = KEYS[2]
local active_key = KEYS[3]
local task_id = ARGV[1]
local replica_id = ARGV[2]
local request_id = ARGV[3]
local ttl_ms = tonumber(ARGV[4])
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
local raw_lease = redis.call('HGET', leases_key, task_id)
if raw_lease then
    local lease = cjson.decode(raw_lease)
    if tonumber(lease.expires_at_ms) > now_ms then
        return {
            'conflict',
            lease.replica_id,
            tostring(tonumber(lease.expires_at_ms) - now_ms),
            tostring(lease.generation)
        }
    end
    if redis.call('SISMEMBER', active_key, task_id) == 1 then
        return {
            'orphaned',
            lease.replica_id,
            '0',
            tostring(lease.generation)
        }
    end
end
local generation = redis.call('HINCRBY', generations_key, task_id, 1)
local expires_at_ms = now_ms + ttl_ms
redis.call('HSET', leases_key, task_id, cjson.encode({
    replica_id = replica_id,
    request_id = request_id,
    generation = generation,
    expires_at_ms = expires_at_ms
}))
return {
    'acquired',
    replica_id,
    tostring(expires_at_ms),
    tostring(generation)
}
"""

_RENEW_SCRIPT = """
local raw_lease = redis.call('HGET', KEYS[1], ARGV[1])
if not raw_lease then
    return 0
end
local lease = cjson.decode(raw_lease)
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
if lease.request_id ~= ARGV[2]
    or tonumber(lease.generation) ~= tonumber(ARGV[3])
    or tonumber(lease.expires_at_ms) <= now_ms then
    return 0
end
lease.expires_at_ms = now_ms + tonumber(ARGV[4])
redis.call('HSET', KEYS[1], ARGV[1], cjson.encode(lease))
return tostring(lease.expires_at_ms)
"""

_RELEASE_SCRIPT = """
local raw_lease = redis.call('HGET', KEYS[1], ARGV[1])
if not raw_lease then
    return 0
end
local lease = cjson.decode(raw_lease)
if lease.request_id ~= ARGV[2]
    or tonumber(lease.generation) ~= tonumber(ARGV[3]) then
    return 0
end
redis.call('HDEL', KEYS[1], ARGV[1])
return 1
"""

_BEGIN_CANCEL_SCRIPT = """
local raw_lease = redis.call('HGET', KEYS[1], ARGV[1])
if not raw_lease then
    return {'missing', '', '0', '0'}
end
local lease = cjson.decode(raw_lease)
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
if lease.replica_id ~= ARGV[2] or tonumber(lease.expires_at_ms) <= now_ms then
    return {'lost', lease.replica_id, tostring(lease.generation), '0'}
end
local generation = redis.call('HINCRBY', KEYS[2], ARGV[1], 1)
local expires_at_ms = now_ms + tonumber(ARGV[4])
redis.call('HSET', KEYS[1], ARGV[1], cjson.encode({
    replica_id = ARGV[2],
    request_id = ARGV[3],
    generation = generation,
    expires_at_ms = expires_at_ms
}))
return {'acquired', ARGV[2], tostring(generation), tostring(expires_at_ms)}
"""

_INTERRUPT_SCRIPT = """
local raw_lease = redis.call('HGET', KEYS[4], ARGV[1])
if not raw_lease then
    return 0
end
local lease = cjson.decode(raw_lease)
local now = redis.call('TIME')
local now_ms = tonumber(now[1]) * 1000 + math.floor(tonumber(now[2]) / 1000)
if tonumber(lease.expires_at_ms) > now_ms then
    return -1
end
if redis.call('SISMEMBER', KEYS[3], ARGV[1]) == 0 then
    return 0
end
local score = tonumber(now[1]) * 1000000 + tonumber(now[2])
redis.call('HINCRBY', KEYS[5], ARGV[1], 1)
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('SREM', KEYS[3], ARGV[1])
redis.call('ZADD', KEYS[2], score, ARGV[1])
redis.call('HDEL', KEYS[4], ARGV[1])
return 1
"""

_ACTIVE_STATES = frozenset(
    {
        TaskState.submitted,
        TaskState.working,
        TaskState.auth_required,
    }
)


class RedisTaskStore(TaskStore):
    """Bounded A2A task storage shared by every runtime replica."""

    def __init__(
        self,
        redis: Redis,
        *,
        max_tasks: int = 10000,
        key_prefix: str = "cogniverse:a2a",
        owns_client: bool = False,
        enforce_leases: bool = False,
    ) -> None:
        if max_tasks < 1:
            raise ValueError(f"max_tasks must be >= 1, got {max_tasks}")
        if not key_prefix.strip():
            raise ValueError("key_prefix must be non-empty")
        self._redis = redis
        self._max_tasks = max_tasks
        self._key_prefix = key_prefix.rstrip(":")
        self._owns_client = owns_client
        self._enforce_leases = enforce_leases
        self._tasks_key = f"{self._key_prefix}:tasks"
        self._inactive_key = f"{self._key_prefix}:inactive-lru"
        self._active_key = f"{self._key_prefix}:active"
        self._leases_key = f"{self._key_prefix}:leases"
        self._generations_key = f"{self._key_prefix}:generations"

    @classmethod
    async def from_url(
        cls,
        redis_url: str,
        *,
        max_tasks: int = 10000,
        key_prefix: str = "cogniverse:a2a",
        enforce_leases: bool = False,
    ) -> RedisTaskStore:
        """Connect to Redis and validate it before serving A2A requests."""
        if not redis_url.strip():
            raise ValueError("redis_url must be non-empty")
        client = Redis.from_url(redis_url, decode_responses=True)
        try:
            await client.ping()
        except RedisError as exc:
            await client.aclose()
            raise A2ATaskStoreError(f"{_UNAVAILABLE}: connect to {redis_url}") from exc
        return cls(
            client,
            max_tasks=max_tasks,
            key_prefix=key_prefix,
            owns_client=True,
            enforce_leases=enforce_leases,
        )

    async def save(self, task: Task, context: ServerCallContext | None = None) -> None:
        """Atomically save a task and evict the inactive LRU when required."""
        try:
            bound_lease = self._context_lease(context) or _bound_lease.get()
            result: list[Any] = await self._redis.eval(
                _SAVE_SCRIPT,
                4,
                self._tasks_key,
                self._inactive_key,
                self._active_key,
                self._leases_key,
                task.id,
                task.model_dump_json(by_alias=True),
                "1" if task.status.state in _ACTIVE_STATES else "0",
                self._max_tasks,
                "1" if self._enforce_leases else "0",
                bound_lease.request_id if bound_lease else "",
                bound_lease.generation if bound_lease else 0,
            )
        except RedisError as exc:
            raise A2ATaskStoreError(f"{_UNAVAILABLE}: save task {task.id}") from exc
        result_code = int(result[0])
        if result_code == -1:
            generation = bound_lease.generation if bound_lease else 0
            raise A2ATaskOwnershipLostError(
                f"save task {task.id} with stale ownership generation {generation}"
            )
        if result_code != 1:
            raise A2ATaskCapacityError(
                f"capacity {self._max_tasks} is full of active tasks; "
                f"rejected task {task.id}"
            )

    async def get(
        self, task_id: str, context: ServerCallContext | None = None
    ) -> Task | None:
        """Load a task exactly and refresh its inactive LRU position."""
        del context
        try:
            payload = await self._redis.eval(
                _GET_SCRIPT,
                3,
                self._tasks_key,
                self._inactive_key,
                self._active_key,
                task_id,
            )
        except RedisError as exc:
            raise A2ATaskStoreError(f"{_UNAVAILABLE}: get task {task_id}") from exc
        if payload is None:
            return None
        try:
            return Task.model_validate_json(payload)
        except (ValidationError, ValueError, TypeError) as exc:
            raise A2ATaskStoreError(f"decode task {task_id}") from exc

    async def delete(
        self, task_id: str, context: ServerCallContext | None = None
    ) -> None:
        """Atomically remove a task and all retention bookkeeping."""
        del context
        try:
            await self._redis.eval(
                _DELETE_SCRIPT,
                3,
                self._tasks_key,
                self._inactive_key,
                self._active_key,
                task_id,
            )
        except RedisError as exc:
            raise A2ATaskStoreError(f"{_UNAVAILABLE}: delete task {task_id}") from exc

    async def close(self) -> None:
        """Close the Redis client when this store created it."""
        if self._owns_client:
            await self._redis.aclose()

    def attach_execution(self, context: ServerCallContext, lease: TaskLease) -> None:
        """Attach ownership to the SDK call context used by task saves."""
        context.state[_CALL_CONTEXT_LEASE_KEY] = lease

    async def acquire_execution(
        self,
        task_id: str,
        *,
        replica_id: str,
        lease_seconds: float,
    ) -> TaskLease:
        """Acquire exclusive execution ownership before reading task state."""
        ttl_ms = self._ttl_ms(lease_seconds)
        request_id = uuid.uuid4().hex
        try:
            result: list[Any] = await self._redis.eval(
                _ACQUIRE_SCRIPT,
                3,
                self._leases_key,
                self._generations_key,
                self._active_key,
                task_id,
                replica_id,
                request_id,
                ttl_ms,
            )
        except RedisError as exc:
            raise A2ATaskStoreError(f"{_UNAVAILABLE}: acquire task {task_id}") from exc
        outcome = self._text(result[0])
        owner = self._text(result[1])
        if outcome == "conflict":
            retry_ms = int(self._text(result[2]))
            raise A2ATaskConflictError(
                f"task {task_id} is owned by {owner}; retry after {retry_ms}ms"
            )
        if outcome == "orphaned":
            raise A2ATaskOwnershipLostError(
                f"task {task_id} lost owner {owner} during active execution"
            )
        return TaskLease(
            task_id=task_id,
            replica_id=replica_id,
            request_id=request_id,
            generation=int(self._text(result[3])),
            expires_at_ms=int(self._text(result[2])),
        )

    async def renew_execution(
        self, lease: TaskLease, *, lease_seconds: float
    ) -> TaskLease:
        """Renew an owned lease without changing its fencing generation."""
        try:
            expires_at = await self._redis.eval(
                _RENEW_SCRIPT,
                1,
                self._leases_key,
                lease.task_id,
                lease.request_id,
                lease.generation,
                self._ttl_ms(lease_seconds),
            )
        except RedisError as exc:
            raise A2ATaskStoreError(
                f"{_UNAVAILABLE}: renew task {lease.task_id}"
            ) from exc
        if int(expires_at) == 0:
            raise A2ATaskOwnershipLostError(
                f"renew task {lease.task_id} with stale ownership generation "
                f"{lease.generation}"
            )
        return TaskLease(
            task_id=lease.task_id,
            replica_id=lease.replica_id,
            request_id=lease.request_id,
            generation=lease.generation,
            expires_at_ms=int(expires_at),
        )

    async def release_execution(self, lease: TaskLease) -> bool:
        """Release a lease only when its request and generation still match."""
        try:
            released = await self._redis.eval(
                _RELEASE_SCRIPT,
                1,
                self._leases_key,
                lease.task_id,
                lease.request_id,
                lease.generation,
            )
        except RedisError as exc:
            raise A2ATaskStoreError(
                f"{_UNAVAILABLE}: release task {lease.task_id}"
            ) from exc
        return int(released) == 1

    async def begin_cancel(
        self,
        task_id: str,
        *,
        replica_id: str,
        lease_seconds: float,
    ) -> TaskLease:
        """Fence the executing generation before the owner performs cancellation."""
        request_id = uuid.uuid4().hex
        try:
            result: list[Any] = await self._redis.eval(
                _BEGIN_CANCEL_SCRIPT,
                2,
                self._leases_key,
                self._generations_key,
                task_id,
                replica_id,
                request_id,
                self._ttl_ms(lease_seconds),
            )
        except RedisError as exc:
            raise A2ATaskStoreError(
                f"{_UNAVAILABLE}: begin cancel task {task_id}"
            ) from exc
        outcome = self._text(result[0])
        if outcome != "acquired":
            raise A2ATaskOwnershipLostError(
                f"cancel task {task_id} is not owned by replica {replica_id}"
            )
        return TaskLease(
            task_id=task_id,
            replica_id=replica_id,
            request_id=request_id,
            generation=int(self._text(result[2])),
            expires_at_ms=int(self._text(result[3])),
        )

    async def get_execution_lease(self, task_id: str) -> TaskLease | None:
        """Return the recorded owner, including an expired owner for recovery."""
        try:
            payload = await self._redis.hget(self._leases_key, task_id)
        except RedisError as exc:
            raise A2ATaskStoreError(
                f"{_UNAVAILABLE}: get task {task_id} owner"
            ) from exc
        if payload is None:
            return None
        try:
            data = json.loads(payload)
            return TaskLease(task_id=task_id, **data)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise A2ATaskStoreError(f"decode task {task_id} owner") from exc

    async def has_live_owner(self, task_id: str) -> bool:
        """Report whether some replica still holds an unexpired lease."""
        lease = await self.get_execution_lease(task_id)
        if lease is None:
            return False
        try:
            seconds, microseconds = await self._redis.time()
        except RedisError as exc:
            raise A2ATaskStoreError(f"{_UNAVAILABLE}: read server time") from exc
        now_ms = int(seconds) * 1000 + int(microseconds) // 1000
        return lease.expires_at_ms > now_ms

    async def mark_owner_lost(self, task_id: str) -> bool:
        """Atomically persist interruption if an active task's owner expired."""
        task = await self.get(task_id)
        if task is None:
            return False
        interrupted = task.model_copy(deep=True)
        interrupted.status = TaskStatus(
            state=TaskState.failed,
            message=new_agent_text_message(
                "Execution interrupted because its owning runtime stopped before "
                "completion."
            ),
        )
        try:
            result = await self._redis.eval(
                _INTERRUPT_SCRIPT,
                5,
                self._tasks_key,
                self._inactive_key,
                self._active_key,
                self._leases_key,
                self._generations_key,
                task_id,
                interrupted.model_dump_json(by_alias=True),
            )
        except RedisError as exc:
            raise A2ATaskStoreError(
                f"{_UNAVAILABLE}: mark task {task_id} owner lost"
            ) from exc
        return int(result) == 1

    @contextmanager
    def bind_execution(self, lease: TaskLease) -> Iterator[None]:
        """Bind a lease to SDK task-store saves in the current async context."""
        token = _bound_lease.set(lease)
        try:
            yield
        finally:
            _bound_lease.reset(token)

    async def request_cancel(
        self,
        *,
        owner_replica_id: str,
        task_id: str,
        timeout_seconds: float,
    ) -> Task:
        """Route cancellation to the owning replica and await its task result."""
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")
        request_id = uuid.uuid4().hex
        reply_key = f"{self._key_prefix}:cancel-reply:{request_id}"
        control_key = f"{self._key_prefix}:control:{owner_replica_id}"
        command = json.dumps(
            {
                "request_id": request_id,
                "task_id": task_id,
                "reply_key": reply_key,
            },
            separators=(",", ":"),
        )
        try:
            pipeline = self._redis.pipeline(transaction=True)
            pipeline.lpush(control_key, command)
            pipeline.expire(control_key, max(30, math.ceil(timeout_seconds * 2)))
            await pipeline.execute()
            async with asyncio.timeout(timeout_seconds + 1):
                response = await self._redis.brpop(
                    reply_key, timeout=max(1, math.ceil(timeout_seconds))
                )
            if response is None:
                raise A2ACancelTimeoutError(
                    f"owner {owner_replica_id} did not acknowledge cancel for "
                    f"task {task_id} within {timeout_seconds:g}s"
                )
            payload = json.loads(response[1])
            if payload.get("error"):
                raise A2ATaskStoreError(
                    f"owner {owner_replica_id} rejected cancel for task {task_id}: "
                    f"{payload['error']}"
                )
            return Task.model_validate(payload["task"])
        except TimeoutError as exc:
            raise A2ACancelTimeoutError(
                f"owner {owner_replica_id} did not acknowledge cancel for "
                f"task {task_id} within {timeout_seconds:g}s"
            ) from exc
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            raise A2ATaskStoreError(
                f"decode cancel acknowledgement for task {task_id}"
            ) from exc
        except RedisError as exc:
            raise A2ATaskStoreError(
                f"{_UNAVAILABLE}: route cancel for task {task_id}"
            ) from exc
        finally:
            try:
                await self._redis.delete(reply_key)
            except RedisError:
                pass

    async def next_cancel(
        self, *, replica_id: str, timeout_seconds: float = 1
    ) -> CancelCommand | None:
        """Wait briefly for the next cancellation addressed to this replica."""
        control_key = f"{self._key_prefix}:control:{replica_id}"
        try:
            response = await self._redis.brpop(
                control_key, timeout=max(1, math.ceil(timeout_seconds))
            )
        except RedisError as exc:
            raise A2ATaskStoreError(
                f"{_UNAVAILABLE}: listen for cancels on replica {replica_id}"
            ) from exc
        if response is None:
            return None
        try:
            payload = json.loads(response[1])
            return CancelCommand(**payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise A2ATaskStoreError(
                f"decode cancel command for replica {replica_id}"
            ) from exc

    async def acknowledge_cancel(
        self,
        command: CancelCommand,
        *,
        task: Task | None = None,
        error: str | None = None,
    ) -> None:
        """Publish one bounded cancellation acknowledgement."""
        if (task is None) == (error is None):
            raise ValueError("exactly one of task or error is required")
        payload = json.dumps(
            {
                "task": task.model_dump(mode="json", by_alias=True) if task else None,
                "error": error,
            },
            separators=(",", ":"),
        )
        try:
            pipeline = self._redis.pipeline(transaction=True)
            pipeline.lpush(command.reply_key, payload)
            pipeline.expire(command.reply_key, 30)
            await pipeline.execute()
        except RedisError as exc:
            raise A2ATaskStoreError(
                f"{_UNAVAILABLE}: acknowledge cancel for task {command.task_id}"
            ) from exc

    async def publish_event(self, task_id: str, event: Event) -> None:
        """Append an active task event for cross-replica resubscription."""
        stream_key = f"{self._key_prefix}:events:{task_id}"
        try:
            pipeline = self._redis.pipeline(transaction=True)
            pipeline.xadd(
                stream_key,
                {"payload": _EVENT_ADAPTER.dump_json(event).decode()},
                maxlen=1000,
                approximate=True,
            )
            # A later turn reuses the stream key the previous turn's close
            # left an expiry on; drop it so this relay cannot vanish mid-turn.
            pipeline.persist(stream_key)
            await pipeline.execute()
        except RedisError as exc:
            raise A2ATaskStoreError(
                f"{_UNAVAILABLE}: publish event for task {task_id}"
            ) from exc

    async def close_event_stream(self, task_id: str) -> None:
        """Close an active event relay while retaining a short drain window."""
        stream_key = f"{self._key_prefix}:events:{task_id}"
        try:
            pipeline = self._redis.pipeline(transaction=True)
            pipeline.xadd(stream_key, {"closed": "1"})
            pipeline.expire(stream_key, 60)
            await pipeline.execute()
        except RedisError as exc:
            raise A2ATaskStoreError(
                f"{_UNAVAILABLE}: close event stream for task {task_id}"
            ) from exc

    async def subscribe_events(self, task_id: str) -> AsyncIterator[Event]:
        """Yield future owner events from the shared relay until it closes.

        A relay whose owner dies is never closed, so a quiet read also checks
        that the owning lease is still held; losing it ends the subscription
        instead of holding the caller's stream open forever.
        """
        stream_key = f"{self._key_prefix}:events:{task_id}"
        try:
            try:
                stream_info = await self._redis.xinfo_stream(stream_key)
                cursor = self._text(stream_info["last-generated-id"])
            except ResponseError as exc:
                if "no such key" not in str(exc).lower():
                    raise
                cursor = "0-0"
            while True:
                batches = await self._redis.xread(
                    {stream_key: cursor}, count=100, block=1000
                )
                if not batches and not await self.has_live_owner(task_id):
                    return
                for _, records in batches:
                    for event_id, fields in records:
                        cursor = self._text(event_id)
                        if fields.get("closed") == "1":
                            return
                        payload = fields.get("payload")
                        if payload is None:
                            raise A2ATaskStoreError(
                                f"event stream for task {task_id} has no payload"
                            )
                        yield _EVENT_ADAPTER.validate_json(payload)
        except (ValidationError, ValueError, TypeError) as exc:
            raise A2ATaskStoreError(f"decode event for task {task_id}") from exc
        except RedisError as exc:
            raise A2ATaskStoreError(
                f"{_UNAVAILABLE}: subscribe to task {task_id}"
            ) from exc

    @staticmethod
    def _ttl_ms(lease_seconds: float) -> int:
        if lease_seconds <= 0:
            raise ValueError(f"lease_seconds must be > 0, got {lease_seconds}")
        return max(1, int(lease_seconds * 1000))

    @staticmethod
    def _text(value: Any) -> str:
        return value.decode() if isinstance(value, bytes) else str(value)

    @staticmethod
    def _context_lease(context: ServerCallContext | None) -> TaskLease | None:
        if context is None:
            return None
        lease = context.state.get(_CALL_CONTEXT_LEASE_KEY)
        return lease if isinstance(lease, TaskLease) else None
