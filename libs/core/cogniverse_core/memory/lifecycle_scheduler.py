"""Scheduled memory lifecycle maintenance.

Per-tenant cleanup is initiated for every tenant whose Mem0 instance is
currently warm in the process LRU. Tenants that are evicted from the
cache will be cleaned up the next time they are accessed (which warms
the instance back).

Cleanup is **schema-driven only**: each tick consults the
``KnowledgeRegistry`` for the per-kind retention policy. There is no
bulk-age fallback — every memory must declare its kind, and every kind's
schema decides whether it expires. Pinned memories are skipped via the
``pin_lookup`` callable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Iterable, Optional

from cogniverse_core.memory.schema import KnowledgeRegistry

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_SECONDS = 3600.0  # 1 hour


class LifecycleScheduler:
    """Periodic schema-driven cleanup runner across warm tenant Mem0 instances.

    Args:
        get_warm_managers: Callable returning the currently-warm
            ``Mem0MemoryManager`` instances. The scheduler does not own the
            tenant cache — it asks the cache for the current set on each
            tick. This keeps the contract narrow and avoids holding stale
            references across LRU evictions.
        registry: Knowledge schema registry. Required — every tick uses
            per-kind retention from this registry. Soft-delete at TTL,
            hard-delete at 2× TTL (see Mem0MemoryManager.cleanup_with_schema).
        interval_seconds: Tick cadence. Default 1 hour.
        pin_lookup: Optional callable that returns the set of pinned
            memory ids for a given Mem0 manager. Called once per warm
            manager per tick. When omitted, no memories are treated as
            pinned (lifecycle proceeds without pin protection).
    """

    def __init__(
        self,
        get_warm_managers: Callable[[], Iterable],
        registry: KnowledgeRegistry,
        interval_seconds: float = _DEFAULT_INTERVAL_SECONDS,
        pin_lookup: Optional[Callable[[object], set]] = None,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        if registry is None:
            raise ValueError(
                "LifecycleScheduler requires a KnowledgeRegistry — schema-driven "
                "retention is the only supported mode"
            )
        self._get_warm = get_warm_managers
        self._interval = interval_seconds
        self._registry = registry
        self._pin_lookup = pin_lookup
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

        Per-tenant entries are ``{kind: deleted_count}`` dicts (with
        ``{kind}:archived`` keys for soft-delete events).
        """
        per_tenant: dict[str, object] = {}
        total = 0

        for manager in list(self._get_warm()):
            tenant_id = getattr(manager, "tenant_id", None) or "unknown"
            try:
                pinned_ids = self._pin_lookup(manager) if self._pin_lookup else set()
                deleted_by_kind = await asyncio.to_thread(
                    manager.cleanup_with_schema,
                    self._registry,
                    pinned_ids,
                )
                per_tenant[tenant_id] = deleted_by_kind
                total += sum(v for v in deleted_by_kind.values() if isinstance(v, int))
            except Exception as exc:
                logger.warning(
                    "Lifecycle cleanup failed for tenant %s: %s", tenant_id, exc
                )
                per_tenant[tenant_id] = f"error: {type(exc).__name__}"

        summary = {
            "tenants": per_tenant,
            "total_deleted": total,
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
            "Memory lifecycle scheduler started (interval=%.0fs)",
            self._interval,
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
