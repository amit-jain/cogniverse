"""Agent memory writes that run after the response, off the event loop.

``remember_success`` / ``remember_failure`` reach Mem0's ``add(infer=True)``,
two LLM round trips on the shared model. A request hands them to the one
process-wide :class:`BackgroundMemoryWriter` instead of waiting for them.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable, Dict, List, Optional, Tuple

from cogniverse_foundation.common.tenant_utils import canonical_tenant_id

logger = logging.getLogger(__name__)

# Writes running at once per process; each holds one call on the shared small
# model, so two keep background extraction to a small share of its slots.
MEMORY_WRITE_CONCURRENCY = 2

# Writes accepted but not finished (running + queued), at most 4 of them per
# tenant so one tenant's burst cannot take the others' slots. A write past
# either bound is dropped, newest first: blocking would put the wait back on
# the response, and an unbounded queue would outgrow the shutdown drain.
MEMORY_WRITE_MAX_PENDING = 16
MEMORY_WRITE_MAX_PENDING_PER_TENANT = 4

# Shutdown budget for pending writes, after the A2A drain. Writes still pending
# when the process is killed are lost.
MEMORY_WRITE_DRAIN_TIMEOUT_S = 30.0

# Dropped writes are counted; the warning repeats at most this often per tenant.
MEMORY_WRITE_DROP_LOG_INTERVAL_S = 60.0

_Label = Tuple[Optional[str], Optional[str]]


def _tenant_key(tenant_id: Optional[str]) -> Optional[str]:
    try:
        return canonical_tenant_id(tenant_id) if tenant_id else tenant_id
    except ValueError:
        return tenant_id


class BackgroundMemoryWriter:
    """Bounded, per-tenant-fair queue for fire-and-forget memory writes.

    ``submit`` never blocks and never raises into the caller: it copies the
    caller's context (tenant, session and trace ContextVars), so the write
    runs as ``asyncio.to_thread`` would have run it, and it is callable from
    the event loop or from a worker thread. Workers are daemon threads, so a
    write still running after the drain does not hold process exit.
    """

    def __init__(
        self,
        *,
        concurrency: int,
        max_pending: int,
        max_pending_per_tenant: Optional[int] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.concurrency = concurrency
        self.max_pending = max_pending
        self.max_pending_per_tenant = max_pending_per_tenant or max_pending
        self._clock = clock
        self._queue: "queue.SimpleQueue[tuple]" = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._pending: Dict[Future, _Label] = {}
        self._pending_per_tenant: Dict[Optional[str], int] = {}
        self._dropped: Dict[_Label, int] = {}
        self._drop_reports: Dict[Optional[str], Tuple[float, int]] = {}
        self._workers: List[threading.Thread] = []

    def submit(
        self,
        write: Callable[[], Any],
        *,
        tenant_id: Optional[str],
        agent_name: Optional[str],
    ) -> bool:
        """Queue ``write``; False when a bound is full and it was dropped."""
        key = _tenant_key(tenant_id)
        with self._lock:
            if len(self._pending) >= self.max_pending:
                self._record_drop(key, tenant_id, agent_name, "writer queue full")
                return False
            if self._pending_per_tenant.get(key, 0) >= self.max_pending_per_tenant:
                self._record_drop(key, tenant_id, agent_name, "tenant's share full")
                return False
            future: Future = Future()
            self._pending[future] = (tenant_id, agent_name)
            self._pending_per_tenant[key] = self._pending_per_tenant.get(key, 0) + 1
            while len(self._workers) < self.concurrency:
                worker = threading.Thread(
                    target=self._work,
                    name=f"memory-write-{len(self._workers)}",
                    daemon=True,
                )
                worker.start()
                self._workers.append(worker)
        future.add_done_callback(self._settle)
        self._queue.put(
            (future, contextvars.copy_context(), write, tenant_id, agent_name)
        )
        return True

    def _record_drop(
        self,
        key: Optional[str],
        tenant_id: Optional[str],
        agent_name: Optional[str],
        reason: str,
    ) -> None:
        """Count a drop; warn at most once per interval per tenant. Holds _lock."""
        label = (tenant_id, agent_name)
        self._dropped[label] = self._dropped.get(label, 0) + 1
        now = self._clock()
        reported_at, unreported = self._drop_reports.get(key, (None, 0))
        unreported += 1
        if (
            reported_at is not None
            and now - reported_at < MEMORY_WRITE_DROP_LOG_INTERVAL_S
        ):
            self._drop_reports[key] = (reported_at, unreported)
            return
        self._drop_reports[key] = (now, 0)
        logger.warning(
            "%d memory write(s) dropped for tenant %s (latest agent %s) since the "
            "last report: %s",
            unreported,
            tenant_id,
            agent_name,
            reason,
        )

    def dropped_counts(self) -> Dict[_Label, int]:
        """Writes dropped since start, by ``(tenant_id, agent_name)``."""
        with self._lock:
            return dict(self._dropped)

    def _work(self) -> None:
        while True:
            future, context, write, tenant_id, agent_name = self._queue.get()
            if not future.set_running_or_notify_cancel():
                continue
            try:
                context.run(self._run, write, tenant_id, agent_name)
            finally:
                future.set_result(None)

    @staticmethod
    def _run(
        write: Callable[[], Any], tenant_id: Optional[str], agent_name: Optional[str]
    ) -> None:
        try:
            write()
        except Exception as exc:
            logger.error(
                "Background memory write failed for tenant %s agent %s: %r",
                tenant_id,
                agent_name,
                exc,
                exc_info=True,
            )

    def _settle(self, future: Future) -> None:
        with self._lock:
            label = self._pending.pop(future, None)
            if label is None:
                return
            key = _tenant_key(label[0])
            remaining = self._pending_per_tenant.get(key, 1) - 1
            if remaining > 0:
                self._pending_per_tenant[key] = remaining
            else:
                self._pending_per_tenant.pop(key, None)

    def cancel_tenant(self, tenant_id: str) -> int:
        """Cancel ``tenant_id``'s queued writes; returns how many were cancelled.

        A write already running finishes.
        """
        key = _tenant_key(tenant_id)
        with self._lock:
            targets = [
                (future, label)
                for future, label in self._pending.items()
                if _tenant_key(label[0]) == key
            ]
        cancelled = [label[1] for future, label in targets if future.cancel()]
        if cancelled:
            logger.warning(
                "Cancelled %d queued memory write(s) for tenant %s (agents %s)",
                len(cancelled),
                tenant_id,
                cancelled,
            )
        return len(cancelled)

    async def drain(self, timeout_s: float) -> bool:
        """Wait up to ``timeout_s`` for every pending write to finish.

        At the budget, queued writes that never started are cancelled and
        every write still pending is logged by tenant and agent. The writer
        keeps accepting writes afterwards.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                pending = list(self._pending)
            if not pending:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.wait(
                [asyncio.wrap_future(future) for future in pending],
                timeout=remaining,
            )
        with self._lock:
            pending_labels = dict(self._pending)
        running, cancelled = [], []
        for future, (tenant_id, agent_name) in pending_labels.items():
            label = f"{tenant_id}/{agent_name}"
            (cancelled if future.cancel() else running).append(label)
        logger.warning(
            "Memory-write drain left %d write(s) unfinished after %.1fs; still "
            "running: %s; cancelled before starting: %s",
            len(pending_labels),
            timeout_s,
            running,
            cancelled,
        )
        return False


_writer: Optional[BackgroundMemoryWriter] = None
_writer_lock = threading.Lock()


def get_background_memory_writer() -> BackgroundMemoryWriter:
    """The process-wide writer every agent's background writes share."""
    global _writer
    with _writer_lock:
        if _writer is None:
            _writer = BackgroundMemoryWriter(
                concurrency=MEMORY_WRITE_CONCURRENCY,
                max_pending=MEMORY_WRITE_MAX_PENDING,
                max_pending_per_tenant=MEMORY_WRITE_MAX_PENDING_PER_TENANT,
            )
        return _writer


async def drain_background_memory_writes(
    timeout_s: float = MEMORY_WRITE_DRAIN_TIMEOUT_S,
) -> bool:
    """Land pending background memory writes within ``timeout_s``."""
    return await get_background_memory_writer().drain(timeout_s)
