"""Agent memory writes that run after the response, off the event loop.

``remember_success`` / ``remember_failure`` reach Mem0's ``add(infer=True)``,
two LLM round trips on the shared model. A request hands them to the one
process-wide :class:`BackgroundMemoryWriter` instead of waiting for them.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Writes running at once per process; each holds one call on the shared model.
MEMORY_WRITE_CONCURRENCY = 2

# Writes accepted but not finished (running + queued). A write beyond this is
# dropped and logged rather than queued without bound.
MEMORY_WRITE_MAX_PENDING = 16

# Shutdown budget for pending writes, after the A2A drain.
MEMORY_WRITE_DRAIN_TIMEOUT_S = 30.0


class BackgroundMemoryWriter:
    """Bounded executor for fire-and-forget memory writes.

    ``submit`` never blocks and never raises into the caller: it copies the
    caller's context (tenant, session and trace ContextVars), so the write
    runs as ``asyncio.to_thread`` would have run it, and it is callable from
    the event loop or from a worker thread.
    """

    def __init__(self, *, concurrency: int, max_pending: int):
        self.concurrency = concurrency
        self.max_pending = max_pending
        self._executor = ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="memory-write"
        )
        self._slots = threading.BoundedSemaphore(max_pending)
        self._lock = threading.Lock()
        self._pending: Dict[Future, Tuple[Optional[str], Optional[str]]] = {}

    def submit(
        self,
        write: Callable[[], Any],
        *,
        tenant_id: Optional[str],
        agent_name: Optional[str],
    ) -> bool:
        """Queue ``write``; False when the queue is full and it was dropped."""
        if not self._slots.acquire(blocking=False):
            logger.warning(
                "Memory write for tenant %s agent %s dropped: %d writes already "
                "pending",
                tenant_id,
                agent_name,
                self.max_pending,
            )
            return False
        context = contextvars.copy_context()
        try:
            future = self._executor.submit(
                context.run, self._run, write, tenant_id, agent_name
            )
        except RuntimeError as exc:
            self._slots.release()
            logger.error(
                "Memory write for tenant %s agent %s dropped: %s",
                tenant_id,
                agent_name,
                exc,
            )
            return False
        with self._lock:
            self._pending[future] = (tenant_id, agent_name)
        future.add_done_callback(self._settle)
        return True

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
            self._pending.pop(future, None)
        self._slots.release()

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
            pending = dict(self._pending)
        running, cancelled = [], []
        for future, (tenant_id, agent_name) in pending.items():
            label = f"{tenant_id}/{agent_name}"
            (cancelled if future.cancel() else running).append(label)
        logger.warning(
            "Memory-write drain left %d write(s) unfinished after %.1fs; still "
            "running: %s; cancelled before starting: %s",
            len(pending),
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
            )
        return _writer


async def drain_background_memory_writes(
    timeout_s: float = MEMORY_WRITE_DRAIN_TIMEOUT_S,
) -> bool:
    """Land pending background memory writes within ``timeout_s``."""
    return await get_background_memory_writer().drain(timeout_s)
