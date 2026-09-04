"""Write-behind queue for admin config-blob persistence.

A pin-quota PUT used to run load->delete->create inline against Phoenix —
0.06s idle, 35s measured while span ingestion held the store — so the admin
request stalled behind telemetry load. The queue accepts the write
immediately and applies it in the background, with three contracts:
accepted is reportable as distinct from applied, readers see their own
pending write, and a write the queue ultimately cannot persist surfaces as
a typed error instead of silently reverting to the stale durable value.
"""

import asyncio
import logging

import pytest

from cogniverse_runtime.blob_write_queue import BlobWriteFailed, BlobWriteQueue

TENANT = "org:tenant"


class GatedApplier:
    """Records apply calls; optionally blocks until released or fails."""

    def __init__(self, fail_times: int = 0):
        self.calls: list[tuple[str, str, str, str]] = []
        self.gate = asyncio.Event()
        self.gate.set()
        self._fail_times = fail_times

    async def __call__(self, tenant_id: str, kind: str, key: str, content: str):
        await self.gate.wait()
        self.calls.append((tenant_id, kind, key, content))
        if self._fail_times > 0:
            self._fail_times -= 1
            raise ConnectionError("phoenix unreachable")


@pytest.mark.asyncio
async def test_enqueue_is_accepted_before_apply_runs():
    applier = GatedApplier()
    applier.gate.clear()
    queue = BlobWriteQueue(applier)

    queue.enqueue(TENANT, "config", "pin_quotas", '{"user": 3}')

    assert applier.calls == []
    assert queue.status() == {"pending": 1, "failed": []}

    applier.gate.set()
    await queue.flush()
    assert applier.calls == [(TENANT, "config", "pin_quotas", '{"user": 3}')]
    assert queue.status() == {"pending": 0, "failed": []}


@pytest.mark.asyncio
async def test_pending_content_serves_read_your_write():
    applier = GatedApplier()
    applier.gate.clear()
    queue = BlobWriteQueue(applier)

    queue.enqueue(TENANT, "config", "pin_quotas", '{"user": 9}')
    assert queue.pending_content(TENANT, "config", "pin_quotas") == '{"user": 9}'

    applier.gate.set()
    await queue.flush()
    assert queue.pending_content(TENANT, "config", "pin_quotas") is None


@pytest.mark.asyncio
async def test_writes_to_one_key_coalesce_to_the_last_content():
    applier = GatedApplier()
    applier.gate.clear()
    queue = BlobWriteQueue(applier)

    for n in range(1, 6):
        queue.enqueue(TENANT, "config", "pin_quotas", f'{{"user": {n}}}')

    applier.gate.set()
    await queue.flush()
    assert applier.calls == [(TENANT, "config", "pin_quotas", '{"user": 5}')]


@pytest.mark.asyncio
async def test_distinct_keys_apply_in_enqueue_order():
    applier = GatedApplier()
    applier.gate.clear()
    queue = BlobWriteQueue(applier)

    queue.enqueue(TENANT, "config", "pin_quotas", "a")
    queue.enqueue(TENANT, "config", "signature_variants", "b")
    queue.enqueue("org:other", "config", "pin_quotas", "c")

    applier.gate.set()
    await queue.flush()
    assert applier.calls == [
        (TENANT, "config", "pin_quotas", "a"),
        (TENANT, "config", "signature_variants", "b"),
        ("org:other", "config", "pin_quotas", "c"),
    ]


@pytest.mark.asyncio
async def test_terminal_failure_is_a_typed_error_that_survives():
    applier = GatedApplier(fail_times=99)
    queue = BlobWriteQueue(applier, max_attempts=2, backoff_s=0)

    queue.enqueue(TENANT, "config", "pin_quotas", '{"user": 3}')
    await queue.flush()

    assert queue.status() == {
        "pending": 0,
        "failed": [(TENANT, "config", "pin_quotas")],
    }
    for _ in range(2):
        with pytest.raises(BlobWriteFailed) as exc_info:
            queue.raise_if_failed(TENANT, "config", "pin_quotas")
        message = str(exc_info.value)
        assert TENANT in message
        assert "pin_quotas" in message
        assert "phoenix unreachable" in message
    assert len(applier.calls) == 2  # max_attempts, then terminal


@pytest.mark.asyncio
async def test_transient_failures_are_retried_to_success():
    applier = GatedApplier(fail_times=2)
    queue = BlobWriteQueue(applier, max_attempts=3, backoff_s=0)

    queue.enqueue(TENANT, "config", "pin_quotas", '{"user": 4}')
    await queue.flush()

    assert queue.status() == {"pending": 0, "failed": []}
    assert len(applier.calls) == 3
    queue.raise_if_failed(TENANT, "config", "pin_quotas")


@pytest.mark.asyncio
async def test_new_enqueue_supersedes_a_failed_write():
    applier = GatedApplier(fail_times=2)
    queue = BlobWriteQueue(applier, max_attempts=2, backoff_s=0)

    queue.enqueue(TENANT, "config", "pin_quotas", '{"user": 3}')
    await queue.flush()
    assert queue.status()["failed"] == [(TENANT, "config", "pin_quotas")]

    queue.enqueue(TENANT, "config", "pin_quotas", '{"user": 8}')
    queue.raise_if_failed(TENANT, "config", "pin_quotas")  # cleared by re-enqueue
    await queue.flush()

    assert queue.status() == {"pending": 0, "failed": []}
    assert applier.calls[-1] == (TENANT, "config", "pin_quotas", '{"user": 8}')


@pytest.mark.asyncio
async def test_concurrent_enqueues_across_keys_each_apply_exactly_once():
    applier = GatedApplier()
    queue = BlobWriteQueue(applier)
    barrier = asyncio.Barrier(8)

    async def put(n: int):
        await barrier.wait()
        queue.enqueue(f"org:t{n}", "config", "pin_quotas", f"v{n}")

    async with asyncio.TaskGroup() as tg:
        for n in range(8):
            tg.create_task(put(n))
    await queue.flush()

    assert sorted(applier.calls) == [
        (f"org:t{n}", "config", "pin_quotas", f"v{n}") for n in range(8)
    ]


@pytest.mark.asyncio
async def test_write_enqueued_during_inflight_apply_is_not_lost():
    applier = GatedApplier()
    queue = BlobWriteQueue(applier)

    applier.gate.clear()
    queue.enqueue(TENANT, "config", "pin_quotas", "old")
    # Give the worker a chance to take the snapshot and block inside apply.
    await asyncio.sleep(0.05)
    queue.enqueue(TENANT, "config", "pin_quotas", "new")
    applier.gate.set()
    await queue.flush()

    assert applier.calls[-1] == (TENANT, "config", "pin_quotas", "new")
    assert queue.pending_content(TENANT, "config", "pin_quotas") is None


@pytest.mark.asyncio
async def test_flush_on_empty_queue_returns():
    queue = BlobWriteQueue(GatedApplier())
    assert await queue.flush() is None
    assert queue.status() == {"pending": 0, "failed": []}


@pytest.mark.asyncio
async def test_failed_error_retains_the_accepted_content():
    """The PUT recovery path merges from the last ACCEPTED state, so the
    terminal error must carry the content that never persisted."""
    applier = GatedApplier(fail_times=99)
    queue = BlobWriteQueue(applier, max_attempts=1, backoff_s=0)

    queue.enqueue(TENANT, "config", "pin_quotas", '{"user": 5}')
    await queue.flush()

    error = queue.failed_error(TENANT, "config", "pin_quotas")
    assert isinstance(error, BlobWriteFailed)
    assert error.content == '{"user": 5}'
    assert queue.failed_error(TENANT, "config", "other") is None


QUEUE_LOGGER = "cogniverse_runtime.blob_write_queue"


def _records(caplog, level):
    return [
        r.getMessage()
        for r in caplog.records
        if r.name == QUEUE_LOGGER and r.levelno == level
    ]


@pytest.mark.asyncio
async def test_applied_write_is_logged_with_its_identity(caplog):
    """An operator reads pod logs, not the queue object. Terminal failures
    log; an applied write logged nothing, so a write that landed was
    indistinguishable from one the worker never drained."""
    queue = BlobWriteQueue(GatedApplier())

    with caplog.at_level(logging.INFO, logger=QUEUE_LOGGER):
        queue.enqueue(TENANT, "config", "pin_quotas", '{"user": 3}')
        await queue.flush()

    assert _records(caplog, logging.INFO) == [
        "Blob write config/pin_quotas for tenant org:tenant applied"
    ]


@pytest.mark.asyncio
async def test_retried_write_logs_every_failed_attempt(caplog):
    """A store degrading under load retried silently: a write that succeeded
    on attempt 3 looked identical to one that succeeded on attempt 1."""
    queue = BlobWriteQueue(GatedApplier(fail_times=2), backoff_s=0)

    with caplog.at_level(logging.INFO, logger=QUEUE_LOGGER):
        queue.enqueue(TENANT, "config", "pin_quotas", '{"user": 3}')
        await queue.flush()

    assert _records(caplog, logging.WARNING) == [
        "Blob write config/pin_quotas for tenant org:tenant failed on "
        "attempt 1/3, retrying: phoenix unreachable",
        "Blob write config/pin_quotas for tenant org:tenant failed on "
        "attempt 2/3, retrying: phoenix unreachable",
    ]
    assert _records(caplog, logging.INFO) == [
        "Blob write config/pin_quotas for tenant org:tenant applied"
    ]
