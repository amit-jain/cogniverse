"""A cluster-events channel with the test process as its only worker."""

from __future__ import annotations

import asyncio


class InProcessClusterEvents:
    """The cluster-events channel with this process as its only worker: each
    event runs its handler here and answers as one acknowledgement."""

    worker_id = "unit-worker"

    def __init__(self, handlers):
        self._handlers = handlers
        self.published: list = []

    async def publish(self, kind, payload, *, timeout_s):
        self.published.append((kind, payload))
        return {self.worker_id: await asyncio.to_thread(self._handlers[kind], payload)}
