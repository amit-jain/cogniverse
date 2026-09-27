"""The shared background writer that takes memory writes off the response path.

Every test drives a real ``BackgroundMemoryWriter`` with real threads; the
writes are plain blocking callables the test holds and releases.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import subprocess
import sys
import textwrap
import threading
import time

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from cogniverse_agents.background_memory_writes import (
    MEMORY_WRITE_CONCURRENCY,
    MEMORY_WRITE_MAX_PENDING,
    BackgroundMemoryWriter,
    get_background_memory_writer,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

_REQUEST_TAG: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "bg_memory_test_request_tag", default=None
)


class _HeldWrites:
    """Blocking writes that record when they run and wait for a release."""

    def __init__(self):
        self.release = threading.Event()
        self.lock = threading.Lock()
        self.running = 0
        self.max_running = 0
        self.finished: list[str] = []

    def write(self, name: str) -> None:
        with self.lock:
            self.running += 1
            self.max_running = max(self.max_running, self.running)
        try:
            assert self.release.wait(10) is True
        finally:
            with self.lock:
                self.running -= 1
                self.finished.append(name)

    def wait_running(self, count: int, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.running >= count:
                    return
            time.sleep(0.01)
        raise AssertionError(f"{count} writes never ran together ({self.running})")


@pytest.mark.asyncio
async def test_submit_returns_while_the_write_is_still_running():
    writer = BackgroundMemoryWriter(concurrency=2, max_pending=4)
    held = _HeldWrites()

    started = time.monotonic()
    queued = writer.submit(
        functools.partial(held.write, "w1"), tenant_id="acme:acme", agent_name="a"
    )
    returned_after = time.monotonic() - started
    held.wait_running(1)

    assert queued is True
    assert returned_after < 0.5
    assert held.finished == []
    held.release.set()
    assert await writer.drain(5.0) is True
    assert held.finished == ["w1"]


@pytest.mark.asyncio
async def test_the_write_runs_off_the_event_loop_with_the_submitter_context():
    writer = BackgroundMemoryWriter(concurrency=1, max_pending=2)
    tracer = TracerProvider().get_tracer(__name__)
    seen: dict = {}

    def write():
        seen["thread"] = threading.get_ident()
        seen["tag"] = _REQUEST_TAG.get()
        seen["trace_id"] = trace.get_current_span().get_span_context().trace_id

    with tracer.start_as_current_span("request") as span:
        _REQUEST_TAG.set("request-7")
        assert writer.submit(write, tenant_id="acme:acme", agent_name="a") is True
        request_trace_id = span.get_span_context().trace_id
    _REQUEST_TAG.set(None)

    assert await writer.drain(5.0) is True
    assert seen["thread"] != threading.get_ident()
    assert seen["tag"] == "request-7"
    assert seen["trace_id"] == request_trace_id


@pytest.mark.asyncio
async def test_a_write_submitted_from_a_worker_thread_keeps_that_thread_context():
    """Sync search paths run inside ``asyncio.to_thread`` and submit from
    there; the write sees the request context the worker was given."""
    writer = BackgroundMemoryWriter(concurrency=1, max_pending=2)
    seen: dict = {}

    def write():
        seen["tag"] = _REQUEST_TAG.get()

    def sync_request_path():
        return writer.submit(write, tenant_id="acme:acme", agent_name="a")

    _REQUEST_TAG.set("threaded-request")
    queued = await asyncio.to_thread(sync_request_path)
    _REQUEST_TAG.set(None)

    assert queued is True
    assert await writer.drain(5.0) is True
    assert seen == {"tag": "threaded-request"}


@pytest.mark.asyncio
async def test_a_failing_write_is_logged_with_tenant_and_agent_and_never_raises(
    caplog,
):
    writer = BackgroundMemoryWriter(concurrency=1, max_pending=2)

    def write():
        raise RuntimeError("mem0 vector store unreachable")

    with caplog.at_level(logging.ERROR):
        queued = writer.submit(
            write, tenant_id="acme:acme", agent_name="document_agent"
        )
        assert await writer.drain(5.0) is True

    assert queued is True
    failures = [
        r
        for r in caplog.records
        if r.name == "cogniverse_agents.background_memory_writes"
        and r.levelno == logging.ERROR
    ]
    assert len(failures) == 1
    message = failures[0].getMessage()
    assert "acme:acme" in message
    assert "document_agent" in message
    assert "mem0 vector store unreachable" in message


@pytest.mark.asyncio
async def test_concurrency_is_bounded_and_a_full_queue_drops_with_a_log(caplog):
    writer = BackgroundMemoryWriter(concurrency=2, max_pending=4)
    held = _HeldWrites()

    queued = [
        writer.submit(
            functools.partial(held.write, f"w{i}"),
            tenant_id="acme:acme",
            agent_name="a",
        )
        for i in range(4)
    ]
    held.wait_running(2)
    with caplog.at_level(logging.WARNING):
        overflow = writer.submit(
            functools.partial(held.write, "w4"),
            tenant_id="acme:acme",
            agent_name="search_agent",
        )
    time.sleep(0.2)
    running_while_held = held.running

    assert queued == [True, True, True, True]
    assert overflow is False
    assert running_while_held == 2
    dropped = [
        r.getMessage()
        for r in caplog.records
        if r.name == "cogniverse_agents.background_memory_writes"
    ]
    assert len(dropped) == 1
    assert "acme:acme" in dropped[0]
    assert "search_agent" in dropped[0]
    held.release.set()
    assert await writer.drain(5.0) is True
    assert sorted(held.finished) == ["w0", "w1", "w2", "w3"]
    assert held.max_running == 2


@pytest.mark.asyncio
async def test_a_slot_frees_once_a_write_lands():
    writer = BackgroundMemoryWriter(concurrency=1, max_pending=1)

    assert writer.submit(lambda: None, tenant_id="t", agent_name="a") is True
    assert await writer.drain(5.0) is True
    assert writer.submit(lambda: None, tenant_id="t", agent_name="a") is True
    assert await writer.drain(5.0) is True


@pytest.mark.asyncio
async def test_drain_logs_what_is_still_pending_at_the_budget(caplog):
    writer = BackgroundMemoryWriter(concurrency=1, max_pending=4)
    held = _HeldWrites()
    writer.submit(
        functools.partial(held.write, "running"),
        tenant_id="acme:acme",
        agent_name="orchestrator_agent",
    )
    writer.submit(
        functools.partial(held.write, "queued"),
        tenant_id="beta:beta",
        agent_name="search_agent",
    )
    held.wait_running(1)

    with caplog.at_level(logging.WARNING):
        started = time.monotonic()
        drained = await writer.drain(0.3)
        elapsed = time.monotonic() - started

    assert drained is False
    assert elapsed < 2.0
    report = [
        r.getMessage()
        for r in caplog.records
        if r.name == "cogniverse_agents.background_memory_writes"
    ]
    assert len(report) == 1
    assert "acme:acme/orchestrator_agent" in report[0]
    assert "beta:beta/search_agent" in report[0]
    held.release.set()
    assert await writer.drain(5.0) is True
    # The queued write never started; the drain cancelled it at the budget.
    assert held.finished == ["running"]


@pytest.mark.asyncio
async def test_drain_with_nothing_pending_returns_at_once():
    writer = BackgroundMemoryWriter(concurrency=1, max_pending=1)

    started = time.monotonic()
    drained = await writer.drain(5.0)

    assert drained is True
    assert time.monotonic() - started < 0.5


@pytest.mark.asyncio
async def test_the_writer_keeps_accepting_writes_after_a_drain():
    """Several runtime lifespans share one process; a drain is not a close."""
    writer = BackgroundMemoryWriter(concurrency=1, max_pending=1)
    assert await writer.drain(1.0) is True

    ran = threading.Event()
    assert writer.submit(ran.set, tenant_id="t", agent_name="a") is True
    assert await writer.drain(5.0) is True
    assert ran.is_set()


def test_the_shared_writer_uses_the_shipped_bounds():
    writer = get_background_memory_writer()

    assert writer is get_background_memory_writer()
    assert writer.concurrency == MEMORY_WRITE_CONCURRENCY
    assert writer.max_pending == MEMORY_WRITE_MAX_PENDING


def test_the_shared_writer_caps_each_tenant_inside_the_global_bound():
    from cogniverse_agents.background_memory_writes import (
        MEMORY_WRITE_MAX_PENDING_PER_TENANT,
    )

    writer = get_background_memory_writer()

    assert MEMORY_WRITE_MAX_PENDING_PER_TENANT == 4
    assert writer.max_pending_per_tenant == MEMORY_WRITE_MAX_PENDING_PER_TENANT
    assert writer.max_pending_per_tenant < writer.max_pending


def _writer_logs(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == "cogniverse_agents.background_memory_writes"
    ]


@pytest.mark.asyncio
async def test_one_tenant_cannot_take_the_other_tenants_queue_slots(caplog):
    writer = BackgroundMemoryWriter(
        concurrency=1, max_pending=6, max_pending_per_tenant=2
    )
    held = _HeldWrites()

    def queue(tenant_id, name):
        return writer.submit(
            functools.partial(held.write, name), tenant_id=tenant_id, agent_name="a"
        )

    with caplog.at_level(logging.WARNING):
        busy = [queue("acme:busy", f"busy{i}") for i in range(3)]
        quiet = [queue("acme:quiet", f"quiet{i}") for i in range(2)]
    held.release.set()
    assert await writer.drain(5.0) is True

    assert busy == [True, True, False]
    assert quiet == [True, True]
    assert sorted(held.finished) == ["busy0", "busy1", "quiet0", "quiet1"]
    logged = _writer_logs(caplog)
    assert len(logged) == 1
    assert "acme:busy" in logged[0]
    assert "tenant's share full" in logged[0]


@pytest.mark.asyncio
async def test_drops_are_counted_and_their_warning_is_rate_limited(caplog):
    now = [1000.0]
    writer = BackgroundMemoryWriter(concurrency=1, max_pending=1, clock=lambda: now[0])
    held = _HeldWrites()
    assert writer.submit(
        functools.partial(held.write, "kept"), tenant_id="acme:a", agent_name="a"
    )

    def drop():
        return writer.submit(lambda: None, tenant_id="acme:a", agent_name="search")

    with caplog.at_level(logging.WARNING):
        first_minute = [drop() for _ in range(5)]
        now[0] += 61.0
        next_minute = drop()
    held.release.set()
    assert await writer.drain(5.0) is True

    assert first_minute == [False] * 5
    assert next_minute is False
    assert writer.dropped_counts() == {("acme:a", "search"): 6}
    warnings = _writer_logs(caplog)
    assert len(warnings) == 2
    assert "acme:a" in warnings[0]
    assert "1 memory write(s) dropped" in warnings[0]
    assert "5 memory write(s) dropped" in warnings[1]


@pytest.mark.asyncio
async def test_cancel_tenant_cancels_only_that_tenants_queued_writes(caplog):
    writer = BackgroundMemoryWriter(concurrency=1, max_pending=8)
    held = _HeldWrites()
    writer.submit(
        functools.partial(held.write, "running"), tenant_id="acme:a", agent_name="a"
    )
    held.wait_running(1)
    writer.submit(
        functools.partial(held.write, "queued-a"), tenant_id="acme", agent_name="a"
    )
    writer.submit(
        functools.partial(held.write, "queued-b"), tenant_id="acme:b", agent_name="b"
    )

    with caplog.at_level(logging.WARNING):
        cancelled = writer.cancel_tenant("acme:acme")
    held.release.set()
    assert await writer.drain(5.0) is True

    assert cancelled == 1
    assert sorted(held.finished) == ["queued-b", "running"]
    logged = _writer_logs(caplog)
    assert len(logged) == 1
    assert "acme" in logged[0]


def test_a_write_still_running_after_the_drain_does_not_hold_process_exit():
    script = textwrap.dedent(
        """
        import asyncio
        import threading

        from cogniverse_agents.background_memory_writes import (
            get_background_memory_writer,
        )

        writer = get_background_memory_writer()
        writer.submit(threading.Event().wait, tenant_id="t", agent_name="a")
        print("drained", asyncio.run(writer.drain(0.2)), flush=True)
        """
    )

    started = time.monotonic()
    finished = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )
    elapsed = time.monotonic() - started

    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.strip() == "drained False"
    assert elapsed < 20


def test_a_write_counts_as_pending_before_any_worker_can_take_it():
    """A drain between accepting a write and queueing it must still see it."""
    writer = BackgroundMemoryWriter(concurrency=1, max_pending=2)
    drained_mid_submit = []

    class _ObservedQueue:
        def __init__(self, inner):
            self._inner = inner

        def put(self, item):
            drained_mid_submit.append(asyncio.run(writer.drain(0.0)))
            self._inner.put(item)

        def get(self):
            return self._inner.get()

    writer._queue = _ObservedQueue(writer._queue)

    queued = writer.submit(lambda: None, tenant_id="t", agent_name="a")

    assert queued is True
    assert drained_mid_submit == [False]
