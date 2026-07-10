"""Unit tests for the sync->async bridge shared by agent artifact loaders."""

import asyncio
import threading

import pytest

from cogniverse_core.common.utils.async_bridge import run_coro_blocking


async def _returns(value):
    return value


async def _raises():
    raise ValueError("boom from coroutine")


async def _current_thread_id():
    return threading.get_ident()


@pytest.mark.unit
@pytest.mark.ci_fast
def test_runs_coroutine_with_no_running_loop():
    """From plain sync code (no loop) the coroutine runs and returns its value."""
    assert run_coro_blocking(_returns(42)) == 42
    assert run_coro_blocking(_returns("artifact-blob")) == "artifact-blob"


@pytest.mark.unit
@pytest.mark.ci_fast
def test_runs_coroutine_from_within_running_loop():
    """Called from inside a running loop, it must NOT raise 'loop already
    running' — it bridges onto a worker thread and still returns the value."""

    async def main():
        return run_coro_blocking(_returns(7))

    assert asyncio.run(main()) == 7


def test_bridges_to_a_separate_thread_when_loop_running():
    """The loop-running path drives the coroutine on a different OS thread
    than the caller; the no-loop path runs inline on the caller thread."""
    caller_no_loop = threading.get_ident()
    assert run_coro_blocking(_current_thread_id()) == caller_no_loop

    async def main():
        caller_thread = threading.get_ident()
        worker_thread = run_coro_blocking(_current_thread_id())
        return caller_thread, worker_thread

    caller_thread, worker_thread = asyncio.run(main())
    assert worker_thread != caller_thread


@pytest.mark.unit
@pytest.mark.ci_fast
def test_propagates_coroutine_exception_no_loop():
    with pytest.raises(ValueError, match="boom from coroutine"):
        run_coro_blocking(_raises())


def test_propagates_coroutine_exception_within_loop():
    async def main():
        return run_coro_blocking(_raises())

    with pytest.raises(ValueError, match="boom from coroutine"):
        asyncio.run(main())
