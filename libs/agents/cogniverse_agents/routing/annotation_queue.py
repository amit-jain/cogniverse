"""
Annotation queue with reviewer assignment and SLA tracking, kept in Redis.

Every runtime process serves the same queue. Each request is a Redis hash;
one sorted set per status indexes the requests, and every transition runs in
one script that checks the current status before it moves the request:

  PENDING → ASSIGNED → COMPLETED
                    ↘ EXPIRED (past its SLA deadline)

A completion that persists the reviewer's label elsewhere first claims the
request (:meth:`AnnotationQueue.begin_completion`), so of several processes
completing the same request at once exactly one writes the label.
Completed and expired requests are removed ``retention_seconds`` after they
finished; open requests stay until they finish, at most ``max_open`` of them.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence

from redis.asyncio import Redis
from redis.exceptions import RedisError

from cogniverse_agents.routing.annotation_agent import (
    AnnotationPriority,
    AnnotationRequest,
    AnnotationStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_SLA_HOURS = {
    AnnotationPriority.HIGH: 4,
    AnnotationPriority.MEDIUM: 24,
    AnnotationPriority.LOW: 72,
}

# How long a completed or expired request stays readable.
ANNOTATION_RETENTION_SECONDS = 7 * 24 * 3600
# Pending plus assigned requests the queue holds; a batch that would pass it
# is refused whole.
ANNOTATION_QUEUE_MAX_OPEN = 10_000
# How long a completion claim holds a request while its label is persisted;
# a claim whose process died frees the request after this.
ANNOTATION_COMPLETION_CLAIM_SECONDS = 300
# Finished requests one bulk read removes at most.
_SWEEP_BATCH = 500

_PRIORITY_RANK = {
    AnnotationPriority.HIGH: 0,
    AnnotationPriority.MEDIUM: 1,
    AnnotationPriority.LOW: 2,
}
_PRIORITY_SPAN = 10**13  # above any millisecond timestamp
_OPTIONAL_FIELDS = (
    "assigned_to",
    "assigned_at",
    "sla_deadline",
    "completed_at",
    "label",
    "tenant_id",
)
_UNAVAILABLE = "annotation queue unavailable"


class AnnotationQueueUnavailableError(RuntimeError):
    """Raised when the queue's Redis cannot complete an operation."""


class AnnotationQueueFullError(RuntimeError):
    """Raised when a batch would take the open requests past the cap."""


class AnnotationCompletionInProgressError(RuntimeError):
    """Raised when another completion holds or took the request's claim."""


@dataclass(frozen=True)
class CompletionClaim:
    """A held right to complete one request."""

    span_id: str
    token: str
    request: AnnotationRequest


@dataclass(frozen=True)
class EnqueueOutcome:
    enqueued: int
    total: int


@dataclass(frozen=True)
class QueueSnapshot:
    """Statistics and the first requests of each open list, read at once."""

    statistics: Dict
    pending: List[AnnotationRequest]
    assigned: List[AnnotationRequest]
    expired: List[AnnotationRequest]


# Shared by the scripts below. ARGV[1] of every script is the key prefix.
_LUA_HELPERS = (
    """
local p = ARGV[1]
local function item(span) return p .. ':item:' .. span end
local function total()
  return redis.call('ZCARD', p .. ':pending') + redis.call('ZCARD', p .. ':assigned')
    + redis.call('ZCARD', p .. ':completed') + redis.call('ZCARD', p .. ':expired')
end
local function forget_priority(priority)
  if priority and redis.call('HINCRBY', p .. ':priority', priority, -1) <= 0 then
    redis.call('HDEL', p .. ':priority', priority)
  end
end
local function sweep(cutoff)
  for _, status in ipairs({'completed', 'expired'}) do
    local index = p .. ':' .. status
    for _, span in ipairs(redis.call('ZRANGEBYSCORE', index, '-inf', '(' .. cutoff,
                                     'LIMIT', 0, %d)) do
      forget_priority(redis.call('HGET', item(span), 'priority'))
      redis.call('DEL', item(span))
      redis.call('ZREM', index, span)
    end
  end
end
"""
    % _SWEEP_BATCH
)

# ARGV: prefix, max_open, sweep cutoff ms, count, then per request: span id,
# pending score, priority, field count, fields and values.
_ENQUEUE_SCRIPT = (
    _LUA_HELPERS
    + """
sweep(ARGV[3])
local open = redis.call('ZCARD', p .. ':pending') + redis.call('ZCARD', p .. ':assigned')
local fresh, seen, pos = {}, {}, 5
for _ = 1, tonumber(ARGV[4]) do
  local span, count = ARGV[pos], tonumber(ARGV[pos + 3])
  if not seen[span] and redis.call('EXISTS', item(span)) == 0 then
    fresh[#fresh + 1] = pos
  end
  seen[span] = true
  pos = pos + 4 + 2 * count
end
if open + #fresh > tonumber(ARGV[2]) then
  return {-1, open}
end
for _, at in ipairs(fresh) do
  local span = ARGV[at]
  local fields = {}
  for j = 1, 2 * tonumber(ARGV[at + 3]) do
    fields[j] = ARGV[at + 3 + j]
  end
  redis.call('HSET', item(span), unpack(fields))
  redis.call('ZADD', p .. ':pending', ARGV[at + 1], span)
  redis.call('HINCRBY', p .. ':priority', ARGV[at + 2], 1)
end
return {#fresh, total()}
"""
)

# ARGV: prefix, span id, reviewer, assigned at, deadline, deadline ms.
_ASSIGN_SCRIPT = (
    _LUA_HELPERS
    + """
local key = item(ARGV[2])
local status = redis.call('HGET', key, 'status')
if not status then return {'missing'} end
if status ~= 'pending' then return {'status', status} end
redis.call('HSET', key, 'status', 'assigned', 'assigned_to', ARGV[3],
           'assigned_at', ARGV[4], 'sla_deadline', ARGV[5])
redis.call('ZREM', p .. ':pending', ARGV[2])
redis.call('ZADD', p .. ':assigned', ARGV[6], ARGV[2])
return {'ok', redis.call('HGETALL', key)}
"""
)

# ARGV: prefix, span id, claim token, now ms, claim ms.
_BEGIN_COMPLETION_SCRIPT = (
    _LUA_HELPERS
    + """
local key = item(ARGV[2])
local status = redis.call('HGET', key, 'status')
if not status then return {'missing'} end
if status ~= 'pending' and status ~= 'assigned' then return {'status', status} end
if tonumber(redis.call('HGET', key, 'claim_until') or '0') > tonumber(ARGV[4]) then
  return {'busy'}
end
redis.call('HSET', key, 'claim', ARGV[3],
           'claim_until', tostring(tonumber(ARGV[4]) + tonumber(ARGV[5])))
return {'ok', redis.call('HGETALL', key)}
"""
)

# ARGV: prefix, span id, claim token, completed at, completed ms, '1' when a
# label is given, label.
_FINISH_COMPLETION_SCRIPT = (
    _LUA_HELPERS
    + """
local key = item(ARGV[2])
if redis.call('HGET', key, 'claim') ~= ARGV[3] then return {'lost'} end
redis.call('HSET', key, 'status', 'completed', 'completed_at', ARGV[4])
if ARGV[6] == '1' then
  redis.call('HSET', key, 'label', ARGV[7])
else
  redis.call('HDEL', key, 'label')
end
redis.call('HDEL', key, 'claim', 'claim_until')
for _, status in ipairs({'pending', 'assigned', 'expired'}) do
  redis.call('ZREM', p .. ':' .. status, ARGV[2])
end
redis.call('ZADD', p .. ':completed', ARGV[5], ARGV[2])
return {'ok', redis.call('HGETALL', key)}
"""
)

# ARGV: prefix, span id, claim token.
_ABANDON_COMPLETION_SCRIPT = (
    _LUA_HELPERS
    + """
local key = item(ARGV[2])
if redis.call('HGET', key, 'claim') ~= ARGV[3] then return 0 end
redis.call('HDEL', key, 'claim', 'claim_until')
return 1
"""
)

# ARGV: prefix, now ms, sweep cutoff ms, list limit. Expires assigned
# requests past their deadline (unless a completion holds them), removes
# finished requests past retention, then reads the counts and lists.
_SNAPSHOT_SCRIPT = (
    _LUA_HELPERS
    + """
local now = tonumber(ARGV[2])
for _, span in ipairs(redis.call('ZRANGEBYSCORE', p .. ':assigned', '-inf', now,
                                 'LIMIT', 0, %d)) do
  if tonumber(redis.call('HGET', item(span), 'claim_until') or '0') <= now then
    redis.call('HSET', item(span), 'status', 'expired')
    redis.call('ZREM', p .. ':assigned', span)
    redis.call('ZADD', p .. ':expired', now, span)
  end
end
sweep(ARGV[3])
local limit = tonumber(ARGV[4])
local function records(ids)
  local out = {}
  for i, span in ipairs(ids) do out[i] = redis.call('HGETALL', item(span)) end
  return out
end
local pending, assigned, expired = {}, {}, {}
if limit > 0 then
  pending = records(redis.call('ZRANGE', p .. ':pending', 0, limit - 1))
  assigned = records(redis.call('ZRANGE', p .. ':assigned', 0, limit - 1))
  expired = records(redis.call('ZREVRANGE', p .. ':expired', 0, limit - 1))
end
return {
  {redis.call('ZCARD', p .. ':pending'), redis.call('ZCARD', p .. ':assigned'),
   redis.call('ZCARD', p .. ':completed'), redis.call('ZCARD', p .. ':expired')},
  redis.call('HGETALL', p .. ':priority'), pending, assigned, expired,
}
"""
    % _SWEEP_BATCH
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1000)


def _pairs(flat: Sequence[str]) -> Dict[str, str]:
    return dict(zip(flat[0::2], flat[1::2]))


def _encode(request: AnnotationRequest) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for name, value in request.to_dict().items():
        if value is None:
            continue
        if name == "context":
            fields[name] = json.dumps(value)
        elif name == "routing_confidence":
            fields[name] = repr(float(value))
        else:
            fields[name] = str(value)
    return fields


def _decode(fields: Dict[str, str]) -> AnnotationRequest:
    data = {name: fields.get(name) for name in _OPTIONAL_FIELDS}
    data.update(
        span_id=fields["span_id"],
        timestamp=fields["timestamp"],
        query=fields["query"],
        chosen_agent=fields["chosen_agent"],
        routing_confidence=float(fields["routing_confidence"]),
        outcome=fields["outcome"],
        priority=fields["priority"],
        reason=fields["reason"],
        context=json.loads(fields["context"]),
        agent_type=fields["agent_type"],
    )
    request = AnnotationRequest.from_dict(data)
    request.status = AnnotationStatus(fields["status"])
    request.assigned_to = fields.get("assigned_to")
    for name in ("assigned_at", "sla_deadline", "completed_at"):
        if fields.get(name) is not None:
            setattr(request, name, datetime.fromisoformat(fields[name]))
    return request


class AnnotationQueue:
    """
    Annotation queue every runtime process shares, with assignment, SLA
    tracking and status transitions.

    Every method raises :class:`AnnotationQueueUnavailableError` when Redis
    cannot complete it.
    """

    def __init__(
        self,
        redis: Redis,
        *,
        key_prefix: str = "cogniverse:annotation-queue",
        sla_hours: Dict[AnnotationPriority, int] | None = None,
        retention_seconds: int = ANNOTATION_RETENTION_SECONDS,
        max_open: int = ANNOTATION_QUEUE_MAX_OPEN,
        claim_seconds: int = ANNOTATION_COMPLETION_CLAIM_SECONDS,
        clock: Callable[[], datetime] = _utc_now,
    ):
        prefix = key_prefix.rstrip(":")
        if not prefix:
            raise ValueError("key_prefix must be non-empty")
        self._redis = redis
        self._prefix = prefix
        self._sla_hours = sla_hours or dict(DEFAULT_SLA_HOURS)
        self._retention_ms = retention_seconds * 1000
        self._max_open = max_open
        self._claim_ms = claim_seconds * 1000
        self._clock = clock
        self._enqueue_script = redis.register_script(_ENQUEUE_SCRIPT)
        self._assign_script = redis.register_script(_ASSIGN_SCRIPT)
        self._begin_script = redis.register_script(_BEGIN_COMPLETION_SCRIPT)
        self._finish_script = redis.register_script(_FINISH_COMPLETION_SCRIPT)
        self._abandon_script = redis.register_script(_ABANDON_COMPLETION_SCRIPT)
        self._snapshot_script = redis.register_script(_SNAPSHOT_SCRIPT)

    async def _run(self, script, operation: str, *args):
        try:
            return await script(args=[self._prefix, *args])
        except RedisError as exc:
            raise AnnotationQueueUnavailableError(
                f"{_UNAVAILABLE}: {operation}"
            ) from exc

    async def enqueue(self, request: AnnotationRequest) -> bool:
        """Add a request; False when its span is already queued."""
        outcome = await self.enqueue_batch([request])
        return outcome.enqueued == 1

    async def enqueue_batch(self, requests: List[AnnotationRequest]) -> EnqueueOutcome:
        """Add the requests whose spans are not queued, all or none.

        Raises:
            AnnotationQueueFullError: If the new requests would take the open
                requests past ``max_open``; none of them is added.
        """
        now = self._clock()
        args: List[str] = [
            str(self._max_open),
            str(_ms(now) - self._retention_ms),
            str(len(requests)),
        ]
        for request in requests:
            # Rebuilt from its payload: a request enters PENDING, unassigned.
            pending = AnnotationRequest.from_dict(request.to_dict())
            fields = _encode(pending)
            score = _PRIORITY_RANK[pending.priority] * _PRIORITY_SPAN + _ms(
                pending.timestamp
            )
            args += [
                pending.span_id,
                str(score),
                pending.priority.value,
                str(len(fields)),
            ]
            for name, value in fields.items():
                args += [name, value]
        enqueued, count = await self._run(
            self._enqueue_script, "enqueue requests", *args
        )
        if enqueued < 0:
            raise AnnotationQueueFullError(
                f"annotation queue holds {count} open requests; adding this "
                f"batch would pass the limit of {self._max_open}"
            )
        return EnqueueOutcome(enqueued=enqueued, total=count)

    async def get(self, span_id: str) -> Optional[AnnotationRequest]:
        """Get a request by span_id, or None."""
        try:
            fields = await self._redis.hgetall(f"{self._prefix}:item:{span_id}")
        except RedisError as exc:
            raise AnnotationQueueUnavailableError(
                f"{_UNAVAILABLE}: get span {span_id}"
            ) from exc
        return _decode(fields) if fields else None

    async def assign(
        self,
        span_id: str,
        reviewer: str,
        sla_hours: int | None = None,
    ) -> AnnotationRequest:
        """
        Assign a pending request to a reviewer.

        Raises:
            KeyError: If span_id not in queue
            ValueError: If request is not in PENDING status
        """
        request = await self.get(span_id)
        if request is None:
            raise KeyError(f"Span {span_id} not found in annotation queue")
        hours = (
            self._sla_hours.get(request.priority, 24)
            if sla_hours is None
            else sla_hours
        )
        now = self._clock()
        deadline = now + timedelta(hours=hours)
        reply = await self._run(
            self._assign_script,
            f"assign span {span_id}",
            span_id,
            reviewer,
            now.isoformat(),
            deadline.isoformat(),
            str(_ms(deadline)),
        )
        return self._transitioned(reply, span_id, "assign")

    async def begin_completion(self, span_id: str) -> CompletionClaim:
        """Claim a pending or assigned request for completion.

        Raises:
            KeyError: If span_id not in queue
            ValueError: If the request is neither PENDING nor ASSIGNED
            AnnotationCompletionInProgressError: If another completion holds it
        """
        token = uuid.uuid4().hex
        reply = await self._run(
            self._begin_script,
            f"claim span {span_id}",
            span_id,
            token,
            str(_ms(self._clock())),
            str(self._claim_ms),
        )
        if reply[0] == "busy":
            raise AnnotationCompletionInProgressError(
                f"Span {span_id} is being completed by another request"
            )
        return CompletionClaim(
            span_id=span_id,
            token=token,
            request=self._transitioned(reply, span_id, "complete"),
        )

    async def finish_completion(
        self, claim: CompletionClaim, label: str | None = None
    ) -> AnnotationRequest:
        """Mark the claimed request COMPLETED with ``label``.

        Raises:
            AnnotationCompletionInProgressError: If the claim lapsed and another
                completion took the request.
        """
        now = self._clock()
        reply = await self._run(
            self._finish_script,
            f"complete span {claim.span_id}",
            claim.span_id,
            claim.token,
            now.isoformat(),
            str(_ms(now)),
            "1" if label is not None else "0",
            label or "",
        )
        if reply[0] == "lost":
            raise AnnotationCompletionInProgressError(
                f"Span {claim.span_id} completion claim lapsed and was taken over"
            )
        return _decode(_pairs(reply[1]))

    async def abandon_completion(self, claim: CompletionClaim) -> None:
        """Release a claim whose completion could not be finished."""
        await self._run(
            self._abandon_script,
            f"release span {claim.span_id}",
            claim.span_id,
            claim.token,
        )

    async def complete(
        self, span_id: str, label: str | None = None
    ) -> AnnotationRequest:
        """
        Mark a pending or assigned request as completed.

        Raises:
            KeyError: If span_id not in queue
            ValueError: If request is not ASSIGNED or PENDING
            AnnotationCompletionInProgressError: If another completion holds it
        """
        claim = await self.begin_completion(span_id)
        return await self.finish_completion(claim, label)

    async def snapshot(self, limit: int = 50) -> QueueSnapshot:
        """Statistics plus the first ``limit`` requests of each open list.

        Pending requests come by priority then timestamp, assigned ones by
        SLA deadline, expired ones most recently expired first. Assigned
        requests past their deadline turn EXPIRED first, and finished
        requests past retention are removed.
        """
        now = _ms(self._clock())
        counts, priorities, pending, assigned, expired = await self._run(
            self._snapshot_script,
            "read queue",
            str(now),
            str(now - self._retention_ms),
            str(limit),
        )
        by_status = {
            status: count
            for status, count in zip(
                ("pending", "assigned", "completed", "expired"), counts
            )
            if count
        }
        statistics = {
            "total": sum(counts),
            "by_status": by_status,
            "by_priority": {
                priority: int(count)
                for priority, count in _pairs(priorities).items()
                if int(count)
            },
        }
        return QueueSnapshot(
            statistics=statistics,
            pending=[_decode(_pairs(record)) for record in pending],
            assigned=[_decode(_pairs(record)) for record in assigned],
            expired=[_decode(_pairs(record)) for record in expired],
        )

    async def statistics(self) -> Dict:
        """Queue statistics by status and priority."""
        return (await self.snapshot(limit=0)).statistics

    @staticmethod
    def _transitioned(reply, span_id: str, action: str) -> AnnotationRequest:
        if reply[0] == "missing":
            raise KeyError(f"Span {span_id} not found in annotation queue")
        if reply[0] == "status":
            raise ValueError(f"Cannot {action} span {span_id}: status is {reply[1]}")
        return _decode(_pairs(reply[1]))
