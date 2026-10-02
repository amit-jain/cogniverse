"""Redis store for the agent registrations every runtime process serves.

One hash maps each agent name to its registration, or to a removal marker
that hides a configured agent. A second hash carries the version every
process compares before re-reading the registrations: an epoch created with
the first change and a counter bumped by each change, both written in the
same script as the change itself.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict

from redis.asyncio import Redis
from redis.exceptions import RedisError

from cogniverse_core.registries.agent_registry import (
    AgentRegistryUnavailableError,
    RegistrySnapshot,
    RegistryVersion,
)

_UNAVAILABLE = "shared agent registry unavailable"
_REMOVED = json.dumps({"removed": True})

# KEYS: entries, meta. ARGV: name, record, epoch for a store without one.
_REGISTER_SCRIPT = """
redis.call('HSETNX', KEYS[2], 'epoch', ARGV[3])
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
redis.call('HINCRBY', KEYS[2], 'counter', 1)
return 1
"""

# KEYS: entries, meta. ARGV: name, '1' when configured, removal marker, epoch.
# Whether the agent is served is decided in the same script that removes it,
# so of two concurrent removals exactly one reports it.
_UNREGISTER_SCRIPT = """
local current = redis.call('HGET', KEYS[1], ARGV[1])
local served
if current then
  served = current ~= ARGV[3]
else
  served = ARGV[2] == '1'
end
if not served then
  return 0
end
redis.call('HSETNX', KEYS[2], 'epoch', ARGV[4])
if ARGV[2] == '1' then
  redis.call('HSET', KEYS[1], ARGV[1], ARGV[3])
else
  redis.call('HDEL', KEYS[1], ARGV[1])
end
redis.call('HINCRBY', KEYS[2], 'counter', 1)
return 1
"""


def _version(epoch: Any, counter: Any) -> RegistryVersion:
    return RegistryVersion(epoch=epoch or "", counter=int(counter or 0))


class RedisAgentRegistryStore:
    """:class:`AgentRegistryStore` kept in Redis."""

    def __init__(
        self, redis: Redis, *, key_prefix: str = "cogniverse:agent-registry"
    ) -> None:
        prefix = key_prefix.rstrip(":")
        if not prefix:
            raise ValueError("key_prefix must be non-empty")
        self._redis = redis
        self._entries_key = f"{prefix}:entries"
        self._meta_key = f"{prefix}:meta"
        self._register = redis.register_script(_REGISTER_SCRIPT)
        self._unregister = redis.register_script(_UNREGISTER_SCRIPT)

    async def version(self) -> RegistryVersion:
        try:
            epoch, counter = await self._redis.hmget(
                self._meta_key, ["epoch", "counter"]
            )
        except RedisError as exc:
            raise AgentRegistryUnavailableError(
                f"{_UNAVAILABLE}: read version"
            ) from exc
        return _version(epoch, counter)

    async def snapshot(self) -> RegistrySnapshot:
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.hmget(self._meta_key, ["epoch", "counter"])
                pipe.hgetall(self._entries_key)
                (epoch, counter), entries = await pipe.execute()
        except RedisError as exc:
            raise AgentRegistryUnavailableError(
                f"{_UNAVAILABLE}: read registrations"
            ) from exc
        registered: Dict[str, Dict[str, Any]] = {}
        removed = set()
        for name, record in entries.items():
            if record == _REMOVED:
                removed.add(name)
            else:
                registered[name] = json.loads(record)
        return RegistrySnapshot(
            version=_version(epoch, counter),
            registered=registered,
            removed=frozenset(removed),
        )

    async def register(self, name: str, data: Dict[str, Any]) -> None:
        try:
            await self._register(
                keys=[self._entries_key, self._meta_key],
                args=[name, json.dumps(data, sort_keys=True), uuid.uuid4().hex],
            )
        except RedisError as exc:
            raise AgentRegistryUnavailableError(
                f"{_UNAVAILABLE}: register agent {name}"
            ) from exc

    async def unregister(self, name: str, *, configured: bool) -> bool:
        try:
            removed = await self._unregister(
                keys=[self._entries_key, self._meta_key],
                args=[name, "1" if configured else "0", _REMOVED, uuid.uuid4().hex],
            )
        except RedisError as exc:
            raise AgentRegistryUnavailableError(
                f"{_UNAVAILABLE}: unregister agent {name}"
            ) from exc
        return bool(removed)
