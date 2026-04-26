"""Two-axis backpressure for ``POST /ingest``.

Submit is rejected when EITHER axis is exceeded:

  - cluster: ``XLEN ingest:queue >= queue_depth_limit`` — protects
    workers from a stampede that would let the queue grow unbounded.
  - tenant:  ``ingest:active:<tenant> >= per_tenant_concurrency`` —
    one tenant can't monopolise all workers.

Both limits come from chart values
(``ingestor.queueDepthLimit`` / ``ingestor.perTenantConcurrency``)
and flow through to env (``INGEST_QUEUE_DEPTH_LIMIT`` /
``INGEST_PER_TENANT_CONCURRENCY``) read at the runtime startup boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import redis.asyncio as aioredis

from cogniverse_runtime.ingestion_v2 import queue


@dataclass(frozen=True)
class BackpressureRejection:
    """Why a submission was rejected. Produced by ``check`` and shown
    to the caller in the 429 response so they can decide which axis
    to back off on."""

    axis: str  # "cluster" | "tenant"
    current: int
    limit: int

    @property
    def message(self) -> str:
        if self.axis == "cluster":
            return (
                f"Cluster ingest queue depth {self.current} >= limit {self.limit}. "
                "Retry after current jobs drain."
            )
        return (
            f"Tenant in-flight ingestions {self.current} >= limit {self.limit}. "
            "Wait for an active job to complete."
        )


async def check(
    redis: aioredis.Redis,
    tenant_id: str,
    *,
    queue_depth_limit: int,
    per_tenant_concurrency: int,
) -> BackpressureRejection | None:
    """Return ``None`` if a new submission would be accepted, else a
    BackpressureRejection describing why it should be rejected.

    The check is best-effort — an in-flight increment from another
    submitter between our read and the caller's INCR can race past
    the limit by a small constant. That's acceptable; the limits are
    safety thresholds, not strict gates.
    """
    depth = await queue.queue_depth(redis)
    if depth >= queue_depth_limit:
        return BackpressureRejection("cluster", depth, queue_depth_limit)

    active = await queue.get_active(redis, tenant_id)
    if active >= per_tenant_concurrency:
        return BackpressureRejection("tenant", active, per_tenant_concurrency)

    return None
