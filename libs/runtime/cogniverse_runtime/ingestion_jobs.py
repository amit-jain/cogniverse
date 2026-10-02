"""Status of ``/ingestion/start`` jobs, kept in Redis.

The process that accepts a job runs it and writes its record; every process
answers ``/ingestion/status`` from the same record. While the job runs, its
process renews a lease on it. A job still running whose lease lapsed — its
process stopped — reads as failed with :data:`ABANDONED_ERROR`. A finished
job is final: no later write changes it. Every record expires
``retention_seconds`` after its last write.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

JOB_RETENTION_SECONDS = 24 * 3600
JOB_LEASE_SECONDS = 30
JOB_HEARTBEAT_SECONDS = 10
RUNNING_STATES = ("started", "processing")
ABANDONED_ERROR = "the runtime process running this job stopped before it finished"
_UNAVAILABLE = "ingestion job store unavailable"

# KEYS: record, lease. ARGV: retention ms, '1' to end the lease, then field
# and value pairs. Writes only while the job runs.
_UPDATE_SCRIPT = """
local status = redis.call('HGET', KEYS[1], 'status')
if not status then return 0 end
if status ~= 'started' and status ~= 'processing' then return -1 end
local fields = {}
for i = 3, #ARGV do fields[#fields + 1] = ARGV[i] end
redis.call('HSET', KEYS[1], unpack(fields))
redis.call('PEXPIRE', KEYS[1], ARGV[1])
if ARGV[2] == '1' then redis.call('DEL', KEYS[2]) end
return 1
"""

# KEYS: record, lease. ARGV: lease ms, owner, retention ms. Holds the lease
# while the job runs, and the record with it however long the job takes.
_RENEW_SCRIPT = """
local status = redis.call('HGET', KEYS[1], 'status')
if status ~= 'started' and status ~= 'processing' then return 0 end
redis.call('SET', KEYS[2], ARGV[2], 'PX', ARGV[1])
redis.call('PEXPIRE', KEYS[1], ARGV[3])
return 1
"""

# KEYS: record, lease. ARGV: retention ms, abandoned error as a JSON string.
# A running job without a lease fails here, once: the first read that finds
# it marks it, every later one reads the mark.
_READ_SCRIPT = """
local status = redis.call('HGET', KEYS[1], 'status')
if not status then return {} end
if (status == 'started' or status == 'processing')
    and redis.call('EXISTS', KEYS[2]) == 0 then
  local errors = redis.call('HGET', KEYS[1], 'errors')
  if errors == '[]' then
    errors = '[' .. ARGV[2] .. ']'
  else
    errors = string.sub(errors, 1, -2) .. ',' .. ARGV[2] .. ']'
  end
  redis.call('HSET', KEYS[1], 'status', 'failed', 'errors', errors)
  redis.call('PEXPIRE', KEYS[1], ARGV[1])
end
return redis.call('HGETALL', KEYS[1])
"""


class IngestionJobStoreUnavailableError(RuntimeError):
    """Raised when the job store's Redis cannot complete an operation."""


def _record(flat: List[str]) -> Dict[str, Any]:
    fields = dict(zip(flat[0::2], flat[1::2]))
    return {
        "job_id": fields["job_id"],
        "status": fields["status"],
        "videos_processed": int(fields["videos_processed"]),
        "videos_total": int(fields["videos_total"]),
        "errors": json.loads(fields["errors"]),
    }


class IngestionJobStore:
    """Job records every runtime process reads; written by the job's owner."""

    def __init__(
        self,
        redis: Redis,
        *,
        owner: str,
        key_prefix: str = "cogniverse:ingestion-job",
        retention_seconds: int = JOB_RETENTION_SECONDS,
        lease_seconds: float = JOB_LEASE_SECONDS,
        heartbeat_seconds: float = JOB_HEARTBEAT_SECONDS,
    ) -> None:
        prefix = key_prefix.rstrip(":")
        if not prefix:
            raise ValueError("key_prefix must be non-empty")
        if not 0 < heartbeat_seconds < lease_seconds:
            raise ValueError(
                "heartbeat_seconds must be > 0 and below lease_seconds, got "
                f"{heartbeat_seconds} and {lease_seconds}"
            )
        self._redis = redis
        self._owner = owner
        self._prefix = prefix
        self._retention_ms = int(retention_seconds * 1000)
        self._lease_ms = int(lease_seconds * 1000)
        self._heartbeat_seconds = heartbeat_seconds
        self._update = redis.register_script(_UPDATE_SCRIPT)
        self._renew = redis.register_script(_RENEW_SCRIPT)
        self._read = redis.register_script(_READ_SCRIPT)

    def _keys(self, job_id: str) -> List[str]:
        return [f"{self._prefix}:{job_id}", f"{self._prefix}:{job_id}:lease"]

    @staticmethod
    def _unavailable(operation: str, job_id: str) -> IngestionJobStoreUnavailableError:
        return IngestionJobStoreUnavailableError(
            f"{_UNAVAILABLE}: {operation} job {job_id}"
        )

    async def create(self, job_id: str) -> Dict[str, Any]:
        """Record a started job owned by this process."""
        record, lease = self._keys(job_id)
        fields = {
            "job_id": job_id,
            "status": "started",
            "videos_processed": "0",
            "videos_total": "0",
            "errors": "[]",
        }
        try:
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.hset(record, mapping=fields)
                pipe.pexpire(record, self._retention_ms)
                pipe.set(lease, self._owner, px=self._lease_ms)
                await pipe.execute()
        except RedisError as exc:
            raise self._unavailable("create", job_id) from exc
        return {
            "job_id": job_id,
            "status": "started",
            "videos_processed": 0,
            "videos_total": 0,
            "errors": [],
        }

    async def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        """The job's record, or None when there is none."""
        try:
            flat = await self._read(
                keys=self._keys(job_id),
                args=[self._retention_ms, json.dumps(ABANDONED_ERROR)],
            )
        except RedisError as exc:
            raise self._unavailable("read", job_id) from exc
        return _record(flat) if flat else None

    async def update(self, job_id: str, **fields: Any) -> bool:
        """Set fields of a running job; False once the job has finished."""
        return await self._write(job_id, fields, final=False)

    async def finish(
        self,
        job_id: str,
        *,
        status: str,
        errors: List[str],
        videos_processed: Optional[int] = None,
    ) -> bool:
        """Record the job's outcome; False when it had already finished."""
        if status in RUNNING_STATES:
            raise ValueError(f"{status!r} is not a finished job status")
        fields: Dict[str, Any] = {"status": status, "errors": errors}
        if videos_processed is not None:
            fields["videos_processed"] = videos_processed
        return await self._write(job_id, fields, final=True)

    async def _write(self, job_id: str, fields: Dict[str, Any], final: bool) -> bool:
        args: List[Any] = [self._retention_ms, "1" if final else "0"]
        for name, value in fields.items():
            args += [name, json.dumps(value) if name == "errors" else str(value)]
        try:
            written = await self._update(keys=self._keys(job_id), args=args)
        except RedisError as exc:
            raise self._unavailable("update", job_id) from exc
        if written != 1:
            logger.warning(
                "Ingestion job %s is %s; its update %s was not written",
                job_id,
                "finished" if written == -1 else "gone",
                sorted(fields),
            )
        return written == 1

    async def renew(self, job_id: str) -> bool:
        """Extend this process's lease on a running job."""
        try:
            renewed = await self._renew(
                keys=self._keys(job_id),
                args=[self._lease_ms, self._owner, self._retention_ms],
            )
        except RedisError as exc:
            raise self._unavailable("renew", job_id) from exc
        return renewed == 1

    @contextlib.asynccontextmanager
    async def lease(self, job_id: str) -> AsyncIterator[None]:
        """Renew the job's lease every heartbeat while the block runs."""

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(self._heartbeat_seconds)
                try:
                    if not await self.renew(job_id):
                        return
                except IngestionJobStoreUnavailableError as exc:
                    logger.warning(
                        "Ingestion job %s lease not renewed: %s", job_id, exc
                    )

        task = asyncio.create_task(heartbeat(), name=f"ingestion-job-lease-{job_id}")
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
