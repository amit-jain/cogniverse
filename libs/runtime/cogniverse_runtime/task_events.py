"""Task event queues every runtime process shares through Redis.

A workflow (an orchestration or a deep-research run) and an ingestion job each
report their progress as task events. Any process streams a task's events,
cancels it and lists a tenant's active tasks, whichever process runs it:

* a task is a hash holding its kind, tenant, state and cancellation, listed in
  its tenant's active set until it ends;
* its events are a Redis stream, capped and expiring once the task goes quiet:
  ``<prefix>:task:<id>:events`` for a workflow, and for an ingestion job the
  status stream queue-driven ingestion already writes
  (``ingest:status:<id>``), so both routes read one source;
* the process running a task holds a lease on it. One poller per process
  renews the leases of the tasks it runs and carries a cancellation recorded
  by any process to the task's queue, whose producer stops at its next phase
  boundary. A task whose lease lapses before it ends stopped reporting.

An event's offset is its position among every event the task ever appended,
so a reader resumes where it left off; a reader behind the retained window
resumes at the oldest retained event. A workflow ends when its producer
appends the terminal event; an ingestion job also ends when its stream's
newest entry is a terminal status (``complete``, ``failed``, ``cancelled``).

Every change is one Redis script, which reads Redis' own clock. A Redis error
raises ``TaskEventsUnavailable``; nothing falls back to process memory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from pydantic import TypeAdapter
from redis.asyncio import Redis
from redis.exceptions import RedisError

from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_core.events import (
    BaseEventQueue,
    CompleteEvent,
    ErrorEvent,
    StatusEvent,
    TaskEvent,
    TaskState,
)
from cogniverse_runtime.ingestion_worker.queue import (
    STATUS_STREAM_KEY_PREFIX,
    STATUS_STREAM_MAXLEN,
    STATUS_STREAM_TTL_SECONDS,
    TERMINAL_STATUS_STATES,
)

logger = logging.getLogger(__name__)

TASK_EVENTS_KEY_PREFIX = "cogniverse:task-events"

WORKFLOW = "workflow"
INGESTION = "ingestion"
TASK_KINDS = (WORKFLOW, INGESTION)

# A workflow keeps this many newest events, readable for this long after its
# last one.
WORKFLOW_EVENTS_MAXLEN = 1000
WORKFLOW_EVENT_RETENTION_S = 30 * 60

# A running task's lease; its process renews it at half its length.
PRODUCER_LEASE_S = 30.0
# A queue-driven ingestion job is leased from submit for as long as its
# submission stays in flight, so a queued job reads as active.
QUEUED_INGESTION_LEASE_S = 6 * 60 * 60
# How often each process renews its leases and picks up cancellations.
POLL_INTERVAL_S = 0.5
# A stream reader counts as a subscriber this long after its last read.
SUBSCRIBER_LEASE_S = 30.0
# How often a caught-up stream reader reads again.
READ_INTERVAL_S = 0.25
READ_BATCH = 100

_UNAVAILABLE = "task event store unavailable"

_LUA_COMMON = """
local function now_ms()
  local t = redis.call('TIME')
  return tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
end
local function task_key(prefix, id)
  return prefix .. ':task:' .. id
end
local function events_key(prefix, ingest, kind, id)
  if kind == 'ingestion' then
    return ingest .. id
  end
  return prefix .. ':task:' .. id .. ':events'
end
local function stream_state(events)
  if redis.call('EXISTS', events) == 0 then
    return 0, 0, false
  end
  local info = redis.call('XINFO', 'STREAM', events)
  local length, added, first = 0, 0, false
  for i = 1, #info, 2 do
    local field = info[i]
    if field == 'length' then
      length = tonumber(info[i + 1])
    elseif field == 'entries-added' then
      added = tonumber(info[i + 1])
    elseif field == 'first-entry' and info[i + 1] then
      first = info[i + 1][1]
    end
  end
  return length, added, first
end
local function terminal_set(csv)
  local set = {}
  for state in string.gmatch(csv, '[^,]+') do
    set[state] = true
  end
  return set
end
local function ended(kind, events, terminal)
  if kind ~= 'ingestion' then
    return false
  end
  local last = redis.call('XREVRANGE', events, '+', '-', 'COUNT', 1)
  if #last == 0 then
    return false
  end
  local fields = last[1][2]
  for i = 1, #fields, 2 do
    if fields[i] == 'data' then
      local ok, decoded = pcall(cjson.decode, fields[i + 1])
      if ok and type(decoded) == 'table' and terminal[decoded['state']] then
        return true
      end
    end
  end
  return false
end
local function id_at_most(a, b)
  local am, as = string.match(a, '^(%d+)%-(%d+)$')
  local bm, bs = string.match(b, '^(%d+)%-(%d+)$')
  am, as, bm, bs = tonumber(am), tonumber(as), tonumber(bm), tonumber(bs)
  if am ~= bm then
    return am < bm
  end
  return as <= bs
end
"""

# Create a task (mode "create"), or take over the lease of a task a submit
# registered (mode "attach", creating it when absent). ARGV: prefix, ingestion
# stream prefix, task id, kind, tenant, mode, lease ms, retention ms, active
# set retention ms, terminal states.
_OPEN_SCRIPT = (
    _LUA_COMMON
    + """
local prefix, ingest, id, kind, tenant, mode = ARGV[1], ARGV[2], ARGV[3], ARGV[4], ARGV[5], ARGV[6]
local lease, retention, active_retention = tonumber(ARGV[7]), ARGV[8], ARGV[9]
local now = now_ms()
local task = task_key(prefix, id)
local active = prefix .. ':active:' .. tenant
local h = redis.call('HMGET', task, 'kind', 'tenant_id', 'closed', 'cancelled', 'cancel_reason', 'created_ms')
if h[1] then
  if mode == 'create' then
    return {'exists', h[1], h[2]}
  end
  if h[1] ~= kind or h[2] ~= tenant then
    return {'exists', h[1], h[2]}
  end
  if h[3] == '1' or ended(kind, events_key(prefix, ingest, kind, id), terminal_set(ARGV[10])) then
    return {'closed', h[1], h[2]}
  end
  redis.call('HSET', task, 'lease_until', now + lease)
  redis.call('PEXPIRE', task, retention)
  redis.call('ZADD', active, h[6], id)
  redis.call('PEXPIRE', active, active_retention)
  return {'attached', h[4] or '0', h[5] or ''}
end
redis.call('HSET', task, 'kind', kind, 'tenant_id', tenant, 'created_ms', now,
  'closed', '0', 'cancelled', '0', 'lease_until', now + lease)
redis.call('PEXPIRE', task, retention)
redis.call('ZADD', active, now, id)
redis.call('PEXPIRE', active, active_retention)
return {'created', '0', ''}
"""
)

# Append one event (empty data appends none) and, when closing, end the task.
# ARGV: prefix, ingestion stream prefix, task id, data, maxlen, retention ms,
# close flag, outcome.
_APPEND_SCRIPT = (
    _LUA_COMMON
    + """
local prefix, ingest, id, data = ARGV[1], ARGV[2], ARGV[3], ARGV[4]
local maxlen, retention, close, outcome = ARGV[5], ARGV[6], ARGV[7], ARGV[8]
local task = task_key(prefix, id)
local h = redis.call('HMGET', task, 'kind', 'tenant_id', 'closed', 'cancelled', 'cancel_reason')
if not h[1] then
  return {'missing'}
end
if h[3] == '1' then
  return {'closed'}
end
local offset = -1
if data ~= '' then
  local events = events_key(prefix, ingest, h[1], id)
  redis.call('XADD', events, 'MAXLEN', maxlen, '*', 'data', data)
  redis.call('PEXPIRE', events, retention)
  local _, added = stream_state(events)
  offset = added - 1
end
redis.call('PEXPIRE', task, retention)
if redis.call('EXISTS', task .. ':subscribers') == 1 then
  redis.call('PEXPIRE', task .. ':subscribers', retention)
end
if close == '1' then
  redis.call('HSET', task, 'closed', '1', 'closed_ms', now_ms(), 'outcome', outcome)
  redis.call('ZREM', prefix .. ':active:' .. h[2], id)
end
return {'ok', h[4] or '0', h[5] or '', offset}
"""
)

# Record a cancellation for the process running the task to pick up.
# ARGV: prefix, ingestion stream prefix, task id, kind, reason, terminal states.
_CANCEL_SCRIPT = (
    _LUA_COMMON
    + """
local prefix, ingest, id, kind, reason = ARGV[1], ARGV[2], ARGV[3], ARGV[4], ARGV[5]
local task = task_key(prefix, id)
local now = now_ms()
local h = redis.call('HMGET', task, 'kind', 'closed', 'cancelled', 'lease_until')
if not h[1] or h[1] ~= kind then
  return 'missing'
end
if h[2] == '1' or ended(kind, events_key(prefix, ingest, kind, id), terminal_set(ARGV[6])) then
  return 'finished'
end
if tonumber(h[4] or '0') < now then
  return 'stopped'
end
if h[3] ~= '1' then
  redis.call('HSET', task, 'cancelled', '1', 'cancel_reason', reason, 'cancelled_ms', now)
end
return 'cancelled'
"""
)

# Renew this process's leases at half their length and read their
# cancellations. ARGV: prefix, lease ms, active set retention ms, task ids...
_POLL_SCRIPT = (
    _LUA_COMMON
    + """
local prefix, lease, active_retention = ARGV[1], tonumber(ARGV[2]), ARGV[3]
local now = now_ms()
local out = {}
for i = 4, #ARGV do
  local id = ARGV[i]
  local task = task_key(prefix, id)
  local h = redis.call('HMGET', task, 'tenant_id', 'closed', 'cancelled', 'cancel_reason', 'lease_until', 'created_ms')
  if not h[1] then
    table.insert(out, {'missing', ''})
  else
    if h[2] ~= '1' and tonumber(h[5] or '0') - now < lease / 2 then
      redis.call('HSET', task, 'lease_until', now + lease)
      local active = prefix .. ':active:' .. h[1]
      redis.call('ZADD', active, h[6], id)
      redis.call('PEXPIRE', active, active_retention)
    end
    table.insert(out, {h[3] or '0', h[4] or ''})
  end
end
return out
"""
)

# A task's state and the events after an offset; registers the reader as a
# subscriber when it names one. ARGV: prefix, ingestion stream prefix, task id,
# kind ('' for any), subscriber id ('' for none), subscriber lease ms,
# retention ms, after offset, last delivered stream id ('' for none), count,
# terminal states.
_READ_SCRIPT = (
    _LUA_COMMON
    + """
local prefix, ingest, id, kind, sub = ARGV[1], ARGV[2], ARGV[3], ARGV[4], ARGV[5]
local sub_lease, retention = tonumber(ARGV[6]), ARGV[7]
local after, last_id, count = tonumber(ARGV[8]), ARGV[9], tonumber(ARGV[10])
local task = task_key(prefix, id)
local h = redis.call('HMGET', task, 'kind', 'tenant_id', 'created_ms', 'closed', 'cancelled', 'cancel_reason', 'lease_until')
if not h[1] or (kind ~= '' and h[1] ~= kind) then
  return {'missing'}
end
local now = now_ms()
local subscribers = task .. ':subscribers'
if sub ~= '' then
  redis.call('ZADD', subscribers, now + sub_lease, sub)
  redis.call('PEXPIRE', subscribers, retention)
end
local events = events_key(prefix, ingest, h[1], id)
local length, added, first = stream_state(events)
local base = added - length
local start = -1
local flat = {}
if count > 0 and length > 0 and after + 1 < added then
  local entries
  if last_id ~= '' and first and id_at_most(first, last_id) then
    entries = redis.call('XRANGE', events, '(' .. last_id, '+', 'COUNT', count)
    start = after + 1
  else
    local skip = math.max(0, after + 1 - base)
    local scanned = redis.call('XRANGE', events, '-', '+', 'COUNT', skip + count)
    entries = {}
    for i = skip + 1, #scanned do
      table.insert(entries, scanned[i])
    end
    start = base + skip
  end
  for _, entry in ipairs(entries) do
    local data = ''
    for i = 1, #entry[2], 2 do
      if entry[2][i] == 'data' then
        data = entry[2][i + 1]
      end
    end
    table.insert(flat, entry[1])
    table.insert(flat, data)
  end
end
local closed = h[4] == '1' or ended(h[1], events, terminal_set(ARGV[11]))
local lapsed = (not closed) and tonumber(h[7] or '0') < now
local reading = redis.call('ZCOUNT', subscribers, now, '+inf')
local out = {'ok', h[1], h[2], h[3], closed and 1 or 0, h[5] or '0', h[6] or '',
  lapsed and 1 or 0, added, length, reading, start}
for _, value in ipairs(flat) do
  table.insert(out, value)
end
return out
"""
)

# The tenant's active tasks; prunes those that ended or stopped reporting.
# ARGV: prefix, ingestion stream prefix, tenant, terminal states.
_LIST_SCRIPT = (
    _LUA_COMMON
    + """
local prefix, ingest, tenant = ARGV[1], ARGV[2], ARGV[3]
local terminal = terminal_set(ARGV[4])
local active = prefix .. ':active:' .. tenant
local now = now_ms()
local out = {}
for _, id in ipairs(redis.call('ZRANGE', active, 0, -1)) do
  local task = task_key(prefix, id)
  local h = redis.call('HMGET', task, 'kind', 'created_ms', 'closed', 'cancelled', 'lease_until')
  local keep = false
  if h[1] and h[3] ~= '1' and tonumber(h[5] or '0') >= now then
    local events = events_key(prefix, ingest, h[1], id)
    if not ended(h[1], events, terminal) then
      keep = true
      local length = stream_state(events)
      table.insert(out, id)
      table.insert(out, h[1])
      table.insert(out, h[2])
      table.insert(out, h[4] or '0')
      table.insert(out, length)
      table.insert(out, redis.call('ZCOUNT', task .. ':subscribers', now, '+inf'))
    end
  end
  if not keep then
    redis.call('ZREM', active, id)
  end
end
return out
"""
)


class TaskEventsUnavailable(RuntimeError):
    """Redis could not complete a task event operation."""


class TaskAlreadyExists(ValueError):
    """A task with this id already exists; ids are not reused."""

    def __init__(self, task_id: str, kind: str, tenant_id: str) -> None:
        super().__init__(f"task {task_id} already exists")
        self.task_id = task_id
        self.kind = kind
        self.tenant_id = tenant_id


class TaskClosedError(RuntimeError):
    """An event was sent to a task that already ended."""


@dataclass(frozen=True)
class TaskRead:
    """One read of a task: its state and the events after the read's offset."""

    kind: str
    tenant_id: str
    created_ms: int
    closed: bool
    cancelled: bool
    cancel_reason: Optional[str]
    stopped_reporting: bool
    appended: int
    retained: int
    subscribers: int
    events: Tuple[Tuple[int, str, str], ...]
    """``(offset, stream entry id, data)`` in order."""

    def info(self, task_id: str) -> Dict[str, Any]:
        return {
            "task_id": task_id,
            "kind": self.kind,
            "tenant_id": self.tenant_id,
            "event_count": self.retained,
            "subscriber_count": self.subscribers,
            "is_closed": self.closed,
            "is_cancelled": self.cancelled,
            "created_at": _iso(self.created_ms),
        }


def _iso(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat()


_TASK_EVENT = TypeAdapter(TaskEvent)

_INGESTION_TERMINAL_EVENTS = {
    TaskState.COMPLETED.value,
    TaskState.FAILED.value,
    TaskState.CANCELLED.value,
}


def _entry_timestamp(entry_id: str) -> datetime:
    return datetime.fromtimestamp(int(entry_id.split("-")[0]) / 1000, tz=timezone.utc)


def ingestion_status_event(
    task_id: str, tenant_id: str, entry_id: str, status: Dict[str, Any]
) -> Dict[str, Any]:
    """The task event an ingestion status entry reports.

    An entry carrying the pipeline's own event reports that event; otherwise
    its ``state`` maps to one, stamped with the entry's time and an id derived
    from the entry, so a replay reports the same event.
    """
    embedded = status.get("event")
    if isinstance(embedded, dict):
        return embedded
    state = status.get("state")
    common = {
        "event_id": f"evt_{entry_id}",
        "task_id": task_id,
        "tenant_id": tenant_id,
        "timestamp": _entry_timestamp(entry_id),
    }
    if state == "complete":
        event: TaskEvent = CompleteEvent(result=status.get("result") or {}, **common)
    elif state == "failed":
        event = ErrorEvent(
            error_type=str(status.get("error_type") or "IngestionFailed"),
            error_message=str(status.get("error") or "ingestion failed"),
            recoverable=False,
            **common,
        )
    elif state == "cancelled":
        event = StatusEvent(
            state=TaskState.CANCELLED,
            phase="cancelled",
            message=status.get("reason"),
            **common,
        )
    elif state == "queued":
        event = StatusEvent(state=TaskState.PENDING, phase="queued", **common)
    else:
        event = StatusEvent(
            state=TaskState.WORKING,
            phase=str(state),
            message=status.get("error"),
            **common,
        )
    return event.model_dump(mode="json")


def stream_event(kind: str, task_id: str, tenant_id: str, entry_id: str, data: str):
    """The task event a stream entry of a task of ``kind`` reports, as a dict."""
    decoded = json.loads(data)
    if kind == INGESTION:
        return ingestion_status_event(task_id, tenant_id, entry_id, decoded)
    return decoded


class TaskEventStore:
    """Task events, cancellations and the active-task index in Redis.

    One store per process, on the process's shared-state client. ``start``
    runs the poller that renews this process's leases and delivers
    cancellations to its queues; ``close`` stops it.
    """

    def __init__(
        self,
        redis: Redis,
        *,
        key_prefix: str = TASK_EVENTS_KEY_PREFIX,
        ingestion_stream_prefix: str = STATUS_STREAM_KEY_PREFIX,
        producer_lease_s: float = PRODUCER_LEASE_S,
        queued_ingestion_lease_s: float = QUEUED_INGESTION_LEASE_S,
        poll_interval_s: float = POLL_INTERVAL_S,
        subscriber_lease_s: float = SUBSCRIBER_LEASE_S,
        read_interval_s: float = READ_INTERVAL_S,
        workflow_retention_s: float = WORKFLOW_EVENT_RETENTION_S,
        ingestion_retention_s: float = STATUS_STREAM_TTL_SECONDS,
        workflow_maxlen: int = WORKFLOW_EVENTS_MAXLEN,
        ingestion_maxlen: int = STATUS_STREAM_MAXLEN,
    ) -> None:
        prefix = key_prefix.rstrip(":")
        if not prefix:
            raise ValueError("key_prefix must be non-empty")
        if not 0 < poll_interval_s < producer_lease_s / 2:
            raise ValueError(
                "poll_interval_s must be > 0 and below half of producer_lease_s, "
                f"got {poll_interval_s} and {producer_lease_s}"
            )
        self._redis = redis
        self._prefix = prefix
        self._ingest = ingestion_stream_prefix
        self._lease_ms = int(producer_lease_s * 1000)
        self._queued_lease_ms = int(queued_ingestion_lease_s * 1000)
        self._poll_interval_s = poll_interval_s
        self._sub_lease_ms = int(subscriber_lease_s * 1000)
        self.read_interval_s = read_interval_s
        self._retention_ms = {
            WORKFLOW: int(workflow_retention_s * 1000),
            INGESTION: int(ingestion_retention_s * 1000),
        }
        self._active_retention_ms = max(self._retention_ms.values())
        self._maxlen = {WORKFLOW: workflow_maxlen, INGESTION: ingestion_maxlen}
        self._terminal = ",".join(sorted(TERMINAL_STATUS_STATES))
        self._open = redis.register_script(_OPEN_SCRIPT)
        self._append = redis.register_script(_APPEND_SCRIPT)
        self._cancel = redis.register_script(_CANCEL_SCRIPT)
        self._poll = redis.register_script(_POLL_SCRIPT)
        self._read = redis.register_script(_READ_SCRIPT)
        self._list = redis.register_script(_LIST_SCRIPT)
        self._producers: Dict[str, RedisTaskEventQueue] = {}
        self._poller: Optional[asyncio.Task] = None
        self._poll_failing = False

    async def _run(self, script, operation: str, *args) -> Any:
        try:
            return await script(args=[self._prefix, *args])
        except RedisError as exc:
            raise TaskEventsUnavailable(f"{_UNAVAILABLE}: {operation}") from exc

    @staticmethod
    def _check_kind(kind: str) -> None:
        if kind not in TASK_KINDS:
            raise ValueError(f"kind must be one of {TASK_KINDS}, got {kind!r}")

    async def open_task(
        self, kind: str, task_id: str, tenant_id: str
    ) -> RedisTaskEventQueue:
        """Create a task this process runs and return its queue.

        Raises:
            TaskAlreadyExists: A task with ``task_id`` exists.
        """
        self._check_kind(kind)
        outcome = await self._run(
            self._open,
            f"open task {task_id}",
            self._ingest,
            task_id,
            kind,
            tenant_id,
            "create",
            self._lease_ms,
            self._retention_ms[kind],
            self._active_retention_ms,
            self._terminal,
        )
        if outcome[0] == "exists":
            raise TaskAlreadyExists(task_id, outcome[1], outcome[2])
        return self._producer(kind, task_id, tenant_id)

    async def register_queued(self, task_id: str, tenant_id: str) -> None:
        """Record a submitted ingestion job before a worker runs it, leased
        for as long as its submission stays in flight."""
        outcome = await self._run(
            self._open,
            f"register task {task_id}",
            self._ingest,
            task_id,
            INGESTION,
            tenant_id,
            "create",
            self._queued_lease_ms,
            self._retention_ms[INGESTION],
            self._active_retention_ms,
            self._terminal,
        )
        if outcome[0] == "exists":
            raise TaskAlreadyExists(task_id, outcome[1], outcome[2])

    async def attach(
        self, kind: str, task_id: str, tenant_id: str
    ) -> Optional[RedisTaskEventQueue]:
        """Take over the lease of a registered task this process now runs,
        recording the task first if it is not; None once the task ended.

        Raises:
            TaskAlreadyExists: The id belongs to a task of another kind or
                tenant.
        """
        self._check_kind(kind)
        outcome = await self._run(
            self._open,
            f"attach to task {task_id}",
            self._ingest,
            task_id,
            kind,
            tenant_id,
            "attach",
            self._lease_ms,
            self._retention_ms[kind],
            self._active_retention_ms,
            self._terminal,
        )
        if outcome[0] == "exists":
            raise TaskAlreadyExists(task_id, outcome[1], outcome[2])
        if outcome[0] == "closed":
            return None
        queue = self._producer(kind, task_id, tenant_id)
        if outcome[1] == "1":
            queue.cancellation_token.cancel(outcome[2] or None)
        return queue

    def _producer(self, kind: str, task_id: str, tenant_id: str) -> RedisTaskEventQueue:
        queue = RedisTaskEventQueue(self, kind, task_id, tenant_id)
        self._producers[task_id] = queue
        return queue

    def _release(self, task_id: str) -> None:
        self._producers.pop(task_id, None)

    async def append(
        self,
        kind: str,
        task_id: str,
        data: str,
        *,
        close: bool = False,
        outcome: str = "",
    ) -> Tuple[int, bool, Optional[str]]:
        """Append ``data`` (none when empty) to the task's events, ending the
        task when ``close``; returns the event's offset (-1 for none) and the
        task's cancellation.

        Raises:
            KeyError: There is no such task.
            TaskClosedError: The task already ended.
        """
        result = await self._run(
            self._append,
            f"append to task {task_id}",
            self._ingest,
            task_id,
            data,
            self._maxlen[kind],
            self._retention_ms[kind],
            "1" if close else "0",
            outcome,
        )
        if result[0] == "missing":
            raise KeyError(f"task {task_id} does not exist")
        if result[0] == "closed":
            raise TaskClosedError(f"task {task_id} already ended")
        return int(result[3]), result[1] == "1", result[2] or None

    async def cancel(self, kind: str, task_id: str, reason: Optional[str]) -> str:
        """Record a cancellation. Returns ``cancelled``, ``missing`` (no task
        of ``kind``), ``finished`` or ``stopped`` (its lease lapsed)."""
        self._check_kind(kind)
        return await self._run(
            self._cancel,
            f"cancel task {task_id}",
            self._ingest,
            task_id,
            kind,
            reason or "",
            self._terminal,
        )

    async def read(
        self,
        task_id: str,
        *,
        kind: Optional[str] = None,
        after_offset: int = -1,
        last_entry_id: Optional[str] = None,
        subscriber: Optional[str] = None,
        count: int = READ_BATCH,
    ) -> Optional[TaskRead]:
        """The task's state and up to ``count`` events after ``after_offset``;
        None when there is no such task (of ``kind``, when given)."""
        retention = self._retention_ms[kind] if kind else self._active_retention_ms
        result = await self._run(
            self._read,
            f"read task {task_id}",
            self._ingest,
            task_id,
            kind or "",
            subscriber or "",
            self._sub_lease_ms,
            retention,
            after_offset,
            last_entry_id or "",
            count,
            self._terminal,
        )
        if result[0] == "missing":
            return None
        start = int(result[11])
        flat = result[12:]
        events = tuple(
            (start + index, flat[2 * index], flat[2 * index + 1])
            for index in range(len(flat) // 2)
        )
        return TaskRead(
            kind=result[1],
            tenant_id=result[2],
            created_ms=int(result[3]),
            closed=result[4] == 1,
            cancelled=result[5] == "1",
            cancel_reason=result[6] or None,
            stopped_reporting=result[7] == 1,
            appended=int(result[8]),
            retained=int(result[9]),
            subscribers=int(result[10]),
            events=events,
        )

    async def leave(self, task_id: str, subscriber: str) -> None:
        """Stop counting ``subscriber`` as reading the task."""
        try:
            await self._redis.zrem(
                f"{self._prefix}:task:{task_id}:subscribers", subscriber
            )
        except RedisError as exc:
            raise TaskEventsUnavailable(
                f"{_UNAVAILABLE}: leave task {task_id}"
            ) from exc

    async def list_active(self, tenant_id: str) -> List[Dict[str, Any]]:
        """The tenant's tasks that are running or queued, oldest first."""
        flat = await self._run(
            self._list,
            f"list the active tasks of tenant {tenant_id}",
            self._ingest,
            tenant_id,
            self._terminal,
        )
        return [
            {
                "task_id": flat[i],
                "kind": flat[i + 1],
                "tenant_id": tenant_id,
                "event_count": int(flat[i + 4]),
                "subscriber_count": int(flat[i + 5]),
                "is_closed": False,
                "is_cancelled": flat[i + 3] == "1",
                "created_at": _iso(int(flat[i + 2])),
            }
            for i in range(0, len(flat), 6)
        ]

    async def poll_once(self) -> None:
        """Renew this process's leases and deliver recorded cancellations."""
        producers = dict(self._producers)
        if not producers:
            return
        rows = await self._run(
            self._poll,
            "renew task leases",
            self._lease_ms,
            self._active_retention_ms,
            *producers,
        )
        for queue, (cancelled, reason) in zip(producers.values(), rows):
            if cancelled == "1" and not queue.cancellation_token.is_cancelled:
                queue.cancellation_token.cancel(reason or None)
                logger.info(
                    "Task %s was cancelled: %s", queue.task_id, reason or "no reason"
                )

    async def _poll_forever(self) -> None:
        while True:
            await asyncio.sleep(self._poll_interval_s)
            try:
                await self.poll_once()
            except Exception as exc:
                if not self._poll_failing:
                    logger.warning(
                        "Task leases and cancellations not read: %s (cause: %r)",
                        exc,
                        exc.__cause__,
                    )
                self._poll_failing = True
            else:
                if self._poll_failing:
                    logger.info("Task leases and cancellations read again")
                self._poll_failing = False

    def start(self) -> None:
        """Start this process's poller."""
        if self._poller is None:
            self._poller = asyncio.create_task(
                self._poll_forever(), name="task-event-poller"
            )

    async def close(self) -> None:
        """Stop this process's poller."""
        if self._poller is not None:
            self._poller.cancel()
            await asyncio.gather(self._poller, return_exceptions=True)
            self._poller = None


class RedisTaskEventQueue(BaseEventQueue):
    """The queue a process reports one task's events to.

    A workflow's events are stored as they are. An ingestion job's are stored
    as status entries of its status stream, ``{"state": "running", "event":
    ...}``; the pipeline's own end-of-job event is held and stored with the
    job's terminal status by ``finish_ingestion``, after the job's outcome is
    recorded.
    """

    def __init__(
        self, store: TaskEventStore, kind: str, task_id: str, tenant_id: str
    ) -> None:
        super().__init__(task_id, tenant_id)
        self._store = store
        self._kind = kind
        self._held: Optional[TaskEvent] = None

    @property
    def kind(self) -> str:
        return self._kind

    def _check(self, event: TaskEvent) -> TaskEvent:
        """The event as the task stores it: its tenant in the canonical form
        the task is indexed under."""
        if event.task_id != self._task_id:
            raise ValueError(
                f"event task_id '{event.task_id}' does not match "
                f"queue task_id '{self._task_id}'"
            )
        if canonical_tenant_id(event.tenant_id) != self._tenant_id:
            raise ValueError(
                f"event tenant_id '{event.tenant_id}' does not match "
                f"queue tenant_id '{self._tenant_id}'"
            )
        return event.model_copy(update={"tenant_id": self._tenant_id})

    async def _write(self, data: str, *, close: bool, outcome: str = "") -> int:
        try:
            offset, cancelled, reason = await self._store.append(
                self._kind, self._task_id, data, close=close, outcome=outcome
            )
        except (KeyError, TaskClosedError) as exc:
            raise TaskClosedError(f"Queue {self._task_id} is closed") from exc
        if cancelled and not self._cancellation_token.is_cancelled:
            self._cancellation_token.cancel(reason)
        return offset

    async def enqueue(self, event: TaskEvent) -> None:
        event = self._check(event)
        if self._closed:
            raise TaskClosedError(f"Queue {self._task_id} is closed")
        if self._kind == WORKFLOW:
            await self._write(event.model_dump_json(), close=False)
            return
        if isinstance(event, CompleteEvent) or (
            isinstance(event, StatusEvent) and event.state in _INGESTION_TERMINAL_EVENTS
        ):
            self._held = event
            return
        await self._write(
            json.dumps(
                {
                    "state": "running",
                    "ingest_id": self._task_id,
                    "event": event.model_dump(mode="json"),
                }
            ),
            close=False,
        )

    async def finish(self, event: TaskEvent) -> None:
        """End a workflow with its terminal event."""
        event = self._check(event)
        try:
            await self._write(
                event.model_dump_json(), close=True, outcome=_outcome(event)
            )
        finally:
            self._closed = True
            self._store._release(self._task_id)

    async def finish_ingestion(self, state: str, **fields: Any) -> None:
        """End an ingestion job with its terminal status entry, carrying the
        pipeline's held end-of-job event when it reported one."""
        status: Dict[str, Any] = {"state": state, "ingest_id": self._task_id, **fields}
        if self._held is not None:
            status["event"] = self._held.model_dump(mode="json")
        try:
            await self._write(json.dumps(status), close=True, outcome=state)
        finally:
            self._closed = True
            self._store._release(self._task_id)

    def release(self) -> None:
        """Stop renewing this task's lease without ending it: the process no
        longer runs it."""
        self._closed = True
        self._store._release(self._task_id)

    async def subscribe(self, from_offset: int = 0) -> AsyncIterator[TaskEvent]:
        subscriber = uuid.uuid4().hex
        after, last_id = from_offset - 1, None
        try:
            while True:
                read = await self._store.read(
                    self._task_id,
                    kind=self._kind,
                    after_offset=after,
                    last_entry_id=last_id,
                    subscriber=subscriber,
                )
                if read is None:
                    return
                for offset, entry_id, data in read.events:
                    yield _TASK_EVENT.validate_python(
                        stream_event(
                            self._kind, self._task_id, read.tenant_id, entry_id, data
                        )
                    )
                    after, last_id = offset, entry_id
                if read.closed and after + 1 >= read.appended:
                    return
                if read.stopped_reporting:
                    return
                if not read.events:
                    await asyncio.sleep(self._store.read_interval_s)
        finally:
            await self._store.leave(self._task_id, subscriber)

    async def get_latest_offset(self) -> int:
        read = await self._store.read(self._task_id, kind=self._kind, count=0)
        return 0 if read is None else read.appended

    async def close(self) -> None:
        if self._closed:
            return
        try:
            await self._write("", close=True, outcome="closed")
        finally:
            self._closed = True
            self._store._release(self._task_id)


def _outcome(event: TaskEvent) -> str:
    """The state a task ends in with ``event`` as its terminal event."""
    if isinstance(event, StatusEvent):
        return str(event.state)
    if isinstance(event, ErrorEvent):
        return TaskState.FAILED.value
    return TaskState.COMPLETED.value
