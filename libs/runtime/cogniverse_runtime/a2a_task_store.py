"""Redis persistence for A2A protocol tasks."""

from __future__ import annotations

from typing import Any

from a2a.server.context import ServerCallContext
from a2a.server.tasks import TaskStore
from a2a.types import Task, TaskState
from pydantic import ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError


class A2ATaskStoreError(RuntimeError):
    """Raised when the shared A2A task store cannot complete an operation."""


class A2ATaskCapacityError(A2ATaskStoreError):
    """Raised when capacity is full and every retained task is active."""


_SAVE_SCRIPT = """
local tasks_key = KEYS[1]
local inactive_key = KEYS[2]
local active_key = KEYS[3]
local task_id = ARGV[1]
local payload = ARGV[2]
local active = ARGV[3]
local max_tasks = tonumber(ARGV[4])
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
    ) -> None:
        if max_tasks < 1:
            raise ValueError(f"max_tasks must be >= 1, got {max_tasks}")
        if not key_prefix.strip():
            raise ValueError("key_prefix must be non-empty")
        self._redis = redis
        self._max_tasks = max_tasks
        self._key_prefix = key_prefix.rstrip(":")
        self._owns_client = owns_client
        self._tasks_key = f"{self._key_prefix}:tasks"
        self._inactive_key = f"{self._key_prefix}:inactive-lru"
        self._active_key = f"{self._key_prefix}:active"

    @classmethod
    async def from_url(
        cls,
        redis_url: str,
        *,
        max_tasks: int = 10000,
        key_prefix: str = "cogniverse:a2a",
    ) -> RedisTaskStore:
        """Connect to Redis and validate it before serving A2A requests."""
        if not redis_url.strip():
            raise ValueError("redis_url must be non-empty")
        client = Redis.from_url(redis_url, decode_responses=True)
        try:
            await client.ping()
        except RedisError as exc:
            await client.aclose()
            raise A2ATaskStoreError(
                f"connect to shared A2A task store at {redis_url}"
            ) from exc
        return cls(
            client,
            max_tasks=max_tasks,
            key_prefix=key_prefix,
            owns_client=True,
        )

    async def save(self, task: Task, context: ServerCallContext | None = None) -> None:
        """Atomically save a task and evict the inactive LRU when required."""
        del context
        try:
            result: list[Any] = await self._redis.eval(
                _SAVE_SCRIPT,
                3,
                self._tasks_key,
                self._inactive_key,
                self._active_key,
                task.id,
                task.model_dump_json(by_alias=True),
                "1" if task.status.state in _ACTIVE_STATES else "0",
                self._max_tasks,
            )
        except RedisError as exc:
            raise A2ATaskStoreError(f"save task {task.id}") from exc
        if int(result[0]) != 1:
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
            raise A2ATaskStoreError(f"get task {task_id}") from exc
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
            raise A2ATaskStoreError(f"delete task {task_id}") from exc

    async def close(self) -> None:
        """Close the Redis client when this store created it."""
        if self._owns_client:
            await self._redis.aclose()
