"""Async helpers for the Streamlit dashboard."""

import asyncio
from typing import Any


def run_async_in_streamlit(coro: Any, *, timeout_s: float | None = None) -> Any:
    """Run an async coroutine from Streamlit's sync context.

    If an event loop is already running, the coroutine is driven on a worker
    thread (``asyncio.run`` in a ``ThreadPoolExecutor``); otherwise it runs
    directly. Used by the dashboard tabs to call async backend/provider APIs.
    """

    async def _bounded() -> Any:
        if timeout_s is None:
            return await coro
        return await asyncio.wait_for(coro, timeout=timeout_s)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _bounded())
                # The executor's own timeout, so a hung coroutine cannot hold the
                # render thread even if wait_for is somehow not reached.
                return future.result(
                    timeout=None if timeout_s is None else timeout_s + 1
                )
        return asyncio.run(_bounded())
    except RuntimeError:
        # No event loop in this thread.
        return asyncio.run(_bounded())
