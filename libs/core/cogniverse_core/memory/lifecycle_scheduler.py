"""Scheduled memory lifecycle maintenance.

A.9 — runs cleanup_expired_memories on a periodic tick. Per-tenant cleanup
is initiated for every tenant whose Mem0 instance is currently warm in the
process LRU. Tenants that are evicted from the cache will be cleaned up the
next time they are accessed (which warms the instance back).

This is a thin scheduler; the actual deletion logic lives in
``Mem0MemoryManager.cleanup_expired_memories``. A.7 will replace the bulk
``max_age_seconds`` knob with per-schema policies and add soft-delete
semantics; A.9 just makes sure the existing cleanup runs on a schedule.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_SECONDS = 3600.0  # 1 hour
_DEFAULT_MAX_AGE_SECONDS = 30 * 24 * 3600  # 30 days


class LifecycleScheduler:
    """Periodic cleanup runner across warm tenant Mem0 instances.

    Args:
        get_warm_managers: Callable returning the currently-warm
            ``Mem0MemoryManager`` instances. The scheduler does not own the
            tenant cache — it asks the cache for the current set on each
            tick. This keeps the contract narrow and avoids holding stale
            references across LRU evictions.
        interval_seconds: Tick cadence. Default 1 hour.
        max_age_seconds: Memories older than this are cleaned up. Default
            30 days. A.7 will replace this single knob with per-schema
            policies; for now it's a global cutoff.
    """

    def __init__(
        self,
        get_warm_managers: Callable[[], Iterable],
        interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
        max_age_seconds: int = _DEFAULT_MAX_AGE_SECONDS,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if max_age_seconds <= 0:
            raise ValueError("max_age_seconds must be positive")
        self._get_warm = get_warm_managers
        self._interval = interval_seconds
        self._max_age = max_age_seconds
        self._task: Optional[asyncio.Task] = None
        self._stop_evt: Optional[asyncio.Event] = None
        self._last_run_summary: Optional[dict] = None

    @property
    def last_run_summary(self) -> Optional[dict]:
        """Last tick's per-tenant deletion counts plus aggregate total."""
        return self._last_run_summary

    async def tick_once(self) -> dict:
        """Run cleanup across all currently-warm tenants. Returns a summary.

        Each warm Mem0MemoryManager is processed sequentially. Errors on a
        single tenant do not abort the run — the offending tenant is
        recorded in the summary so operators can investigate.
        """
        per_tenant: dict[str, int | str] = {}
        total = 0

        for manager in list(self._get_warm()):
            tenant_id = getattr(manager, "tenant_id", None) or "unknown"
            try:
                deleted = await asyncio.to_thread(
                    manager.cleanup_expired_memories, self._max_age
                )
                per_tenant[tenant_id] = deleted
                total += deleted
            except Exception as exc:
                logger.warning(
                    "Lifecycle cleanup failed for tenant %s: %s", tenant_id, exc
                )
                per_tenant[tenant_id] = f"error: {type(exc).__name__}"

        summary = {
            "tenants": per_tenant,
            "total_deleted": total,
            "max_age_seconds": self._max_age,
        }
        self._last_run_summary = summary
        logger.info(
            "Lifecycle tick complete: %d memories deleted across %d tenants",
            total,
            len(per_tenant),
        )
        return summary

    def start(self) -> None:
        """Schedule the periodic tick on the running event loop."""
        if self._task is not None and not self._task.done():
            logger.debug("LifecycleScheduler already running; start() is a no-op")
            return
        loop = asyncio.get_running_loop()
        self._stop_evt = asyncio.Event()
        self._task = loop.create_task(
            self._run_loop(), name="memory_lifecycle_scheduler"
        )
        logger.info(
            "Memory lifecycle scheduler started (interval=%.0fs, max_age=%ds)",
            self._interval,
            self._max_age,
        )

    async def stop(self) -> None:
        """Stop the scheduler and await clean shutdown."""
        if self._stop_evt is not None:
            self._stop_evt.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=self._interval + 1)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None
        logger.info("Memory lifecycle scheduler stopped")

    async def _run_loop(self) -> None:
        assert self._stop_evt is not None
        while not self._stop_evt.is_set():
            try:
                await self.tick_once()
            except Exception:
                logger.exception("Unhandled error during lifecycle tick")
            try:
                await asyncio.wait_for(self._stop_evt.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass
