"""Async helpers for the Streamlit dashboard."""

import asyncio
from typing import Any


def run_async_in_streamlit(coro: Any) -> Any:
    """Run an async coroutine from Streamlit's sync context.

    If an event loop is already running, the coroutine is driven on a worker
    thread (``asyncio.run`` in a ``ThreadPoolExecutor``); otherwise it runs
    directly. Used by the dashboard tabs to call async backend/provider APIs.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        return asyncio.run(coro)
    except RuntimeError:
        # No event loop in this thread.
        return asyncio.run(coro)
