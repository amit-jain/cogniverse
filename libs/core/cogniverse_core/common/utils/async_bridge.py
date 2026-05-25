"""Bridge async coroutines into synchronous call sites."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def run_coro_blocking(coro: Any) -> Any:
    """Run a coroutine to completion from synchronous code.

    When an event loop is already running in this thread (e.g. a sync method
    invoked from within an async request path), the coroutine is driven on a
    worker thread via ``asyncio.run`` in a ``ThreadPoolExecutor`` to avoid a
    "loop already running" error; otherwise it runs directly via
    ``asyncio.run``. Used by the agent artifact-load paths that call async
    ArtifactManager / telemetry APIs from sync code.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    return asyncio.run(coro)
