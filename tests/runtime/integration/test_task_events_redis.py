"""Task events shared through a real Redis: workflow and ingestion progress,
cancellations and the active-task index every runtime process serves.

Interleavings are executed, not reasoned about: concurrent producers and
cancellers are separate OS processes released together by a barrier. Outages
are real: a port nothing listens on, and a Redis container the test owns and
pauses.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import socket
import time
import uuid

import pytest
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from cogniverse_core.agents.base import AgentBase, AgentDeps, AgentInput, AgentOutput
from cogniverse_core.events import (
    TaskCancelled,
    TaskState,
    bind_event_queue,
    create_complete_event,
    create_progress_event,
    create_status_event,
    publish_phase,
)
from cogniverse_runtime.ingestion_worker import queue as ingest_queue
from cogniverse_runtime.shared_state import connect_shared_state_redis
from cogniverse_runtime.task_events import (
    INGESTION,
    WORKFLOW,
    TaskAlreadyExists,
    TaskClosedError,
    TaskEventStore,
    TaskEventsUnavailable,
    stream_event,
)

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]

TENANT = "acme:acme"
PROCESSES = 4
EVENTS_PER_PROCESS = 25


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _prefix() -> str:
    return f"test:task-events:{uuid.uuid4().hex}"


def _store(redis, prefix=None, **kwargs) -> TaskEventStore:
    kwargs.setdefault("ingestion_stream_prefix", f"{prefix or _prefix()}:ingest:")
    return TaskEventStore(redis, key_prefix=prefix or _prefix(), **kwargs)


def _phases(read):
    return [(offset, json.loads(data).get("phase")) for offset, _, data in read.events]


def events_json(read, index: int) -> str:
    """The pipeline event a status entry of ``read`` carries, as stored."""
    return json.dumps(json.loads(read.events[index][2])["event"])


def _working(task_id: str, phase: str, tenant: str = TENANT):
    return create_status_event(task_id, tenant, TaskState.WORKING, phase=phase)


class TestWorkflowEvents:
    async def test_events_read_back_in_order_with_their_offsets(
        self, shared_state_redis
    ):
        store = _store(shared_state_redis)
        queue = await store.open_task(WORKFLOW, "wf", TENANT)
        for phase in ("planning", "execution", "aggregating"):
            await queue.enqueue(_working("wf", phase))

        everything = await store.read("wf", kind=WORKFLOW)
        replayed = await store.read("wf", kind=WORKFLOW, after_offset=0)
        resumed = await store.read(
            "wf",
            kind=WORKFLOW,
            after_offset=1,
            last_entry_id=everything.events[1][1],
        )

        assert _phases(everything) == [
            (0, "planning"),
            (1, "execution"),
            (2, "aggregating"),
        ]
        assert _phases(replayed) == [(1, "execution"), (2, "aggregating")]
        assert _phases(resumed) == [(2, "aggregating")]
        assert (everything.kind, everything.tenant_id) == (WORKFLOW, TENANT)
        assert (everything.appended, everything.retained) == (3, 3)
        assert (everything.closed, everything.cancelled) == (False, False)
        assert everything.stopped_reporting is False

    async def test_a_reader_behind_the_retained_window_resumes_at_the_oldest(
        self, shared_state_redis
    ):
        store = _store(shared_state_redis, workflow_maxlen=3)
        queue = await store.open_task(WORKFLOW, "wf", TENANT)
        for index in range(7):
            await queue.enqueue(_working("wf", f"p{index}"))

        behind = await store.read("wf", kind=WORKFLOW, after_offset=1)
        latest = await store.read("wf", kind=WORKFLOW, after_offset=5)

        assert _phases(behind) == [(4, "p4"), (5, "p5"), (6, "p6")]
        assert _phases(latest) == [(6, "p6")]
        assert (behind.appended, behind.retained) == (7, 3)

    async def test_the_terminal_event_ends_the_task_and_refuses_later_events(
        self, shared_state_redis
    ):
        store = _store(shared_state_redis)
        queue = await store.open_task(WORKFLOW, "wf", TENANT)
        await queue.enqueue(_working("wf", "planning"))
        await queue.finish(create_complete_event("wf", TENANT, result={"ok": True}))

        read = await store.read("wf", kind=WORKFLOW)
        with pytest.raises(TaskClosedError) as refused:
            await queue.enqueue(_working("wf", "late"))
        cancel = await store.cancel(WORKFLOW, "wf", "too late")

        assert [json.loads(data)["event_type"] for _, _, data in read.events] == [
            "status",
            "complete",
        ]
        assert json.loads(read.events[1][2])["result"] == {"ok": True}
        assert read.closed is True
        assert str(refused.value) == "Queue wf is closed"
        assert cancel == "finished"
        assert await store.list_active(TENANT) == []

    async def test_a_task_id_is_never_reused(self, shared_state_redis):
        store = _store(shared_state_redis)
        await store.open_task(WORKFLOW, "wf", TENANT)

        with pytest.raises(TaskAlreadyExists) as taken:
            await store.open_task(WORKFLOW, "wf", "other:other")

        assert (taken.value.task_id, taken.value.kind, taken.value.tenant_id) == (
            "wf",
            WORKFLOW,
            TENANT,
        )

    async def test_an_event_for_another_task_or_tenant_is_refused(
        self, shared_state_redis
    ):
        store = _store(shared_state_redis)
        queue = await store.open_task(WORKFLOW, "wf", TENANT)

        with pytest.raises(ValueError) as wrong_task:
            await queue.enqueue(_working("other", "planning"))
        with pytest.raises(ValueError) as wrong_tenant:
            await queue.enqueue(_working("wf", "planning", tenant="evil:evil"))
        await queue.enqueue(_working("wf", "planning", tenant="acme"))
        read = await store.read("wf", kind=WORKFLOW)

        assert str(wrong_task.value) == (
            "event task_id 'other' does not match queue task_id 'wf'"
        )
        assert str(wrong_tenant.value) == (
            "event tenant_id 'evil:evil' does not match queue tenant_id 'acme:acme'"
        )
        # A tenant named in its simple form is stored in the canonical form
        # the task is indexed under.
        assert [json.loads(data)["tenant_id"] for _, _, data in read.events] == [TENANT]

    async def test_a_phase_report_publishes_and_stops_at_a_cancellation(
        self, shared_state_redis
    ):
        store = _store(shared_state_redis)
        queue = await store.open_task(WORKFLOW, "wf", TENANT)

        with bind_event_queue(queue):
            await publish_phase("planning", "Creating execution plan...")
            assert await store.cancel(WORKFLOW, "wf", "operator stop") == "cancelled"
            await publish_phase("complete", "done", check_cancelled=False)
            with pytest.raises(TaskCancelled) as stopped:
                await publish_phase("execution", "Executing...")
        read = await store.read("wf", kind=WORKFLOW)

        # The append that follows a recorded cancellation carries it back.
        assert _phases(read) == [(0, "planning"), (1, "complete"), (2, "execution")]
        assert (stopped.value.task_id, stopped.value.reason) == ("wf", "operator stop")
        assert str(stopped.value) == "task wf was cancelled: operator stop"
        assert read.cancelled is True
        assert read.cancel_reason == "operator stop"


class TestCancellation:
    async def test_cancel_outcomes_name_what_the_task_is(self, shared_state_redis):
        store = _store(shared_state_redis, producer_lease_s=1.0, poll_interval_s=0.1)
        await store.open_task(WORKFLOW, "running", TENANT)
        finished = await store.open_task(WORKFLOW, "finished", TENANT)
        await finished.finish(create_complete_event("finished", TENANT, result={}))
        silent = await store.open_task(WORKFLOW, "silent", TENANT)
        silent.release()
        await asyncio.sleep(1.2)

        assert await store.cancel(WORKFLOW, "running", "stop") == "stopped"
        running = await store.open_task(WORKFLOW, "running-2", TENANT)
        assert await store.cancel(WORKFLOW, "running-2", "stop") == "cancelled"
        assert await store.cancel(WORKFLOW, "running-2", "again") == "cancelled"
        assert await store.cancel(INGESTION, "running-2", "stop") == "missing"
        assert await store.cancel(WORKFLOW, "absent", "stop") == "missing"
        assert await store.cancel(WORKFLOW, "finished", "stop") == "finished"
        assert await store.cancel(WORKFLOW, "silent", "stop") == "stopped"
        # A second cancel keeps the first reason.
        read = await store.read("running-2", kind=WORKFLOW, count=0)
        assert (read.cancelled, read.cancel_reason) == (True, "stop")
        assert running.cancellation_token.is_cancelled is False

    async def test_the_poller_delivers_a_cancellation_from_another_process(
        self, shared_state_redis, shared_state_redis_url
    ):
        prefix = _prefix()
        store = _store(shared_state_redis, prefix, poll_interval_s=0.05)
        queue = await store.open_task(WORKFLOW, "wf", TENANT)
        store.start()
        try:
            await asyncio.to_thread(
                _run_process,
                _cancel_in_process,
                (shared_state_redis_url, prefix, "wf", "operator"),
            )
            deadline = time.monotonic() + 5
            while not queue.cancellation_token.is_cancelled:
                assert time.monotonic() < deadline, "cancellation never delivered"
                await asyncio.sleep(0.02)
        finally:
            await store.close()

        assert queue.cancellation_token.reason == "operator"

    async def test_concurrent_cancels_from_processes_record_one_cancellation(
        self, shared_state_redis, shared_state_redis_url
    ):
        prefix = _prefix()
        store = _store(shared_state_redis, prefix)
        await store.open_task(WORKFLOW, "wf", TENANT)

        outcomes = _run_processes(
            _cancel_at_once, (shared_state_redis_url, prefix, "wf")
        )
        read = await store.read("wf", kind=WORKFLOW, count=0)

        assert sorted(outcome for outcome, _ in outcomes) == ["cancelled"] * PROCESSES
        reasons = {reason for _, reason in outcomes}
        assert len(reasons) == PROCESSES
        assert read.cancel_reason in reasons
        assert read.cancelled is True

    async def test_cancelling_a_tenant_cancels_its_running_and_queued_tasks(
        self, shared_state_redis
    ):
        store = _store(shared_state_redis, producer_lease_s=1.0, poll_interval_s=0.1)
        silent = await store.open_task(WORKFLOW, "silent", TENANT)
        silent.release()
        await asyncio.sleep(1.2)
        running = await store.open_task(WORKFLOW, "running", TENANT)
        await store.register_queued("queued", TENANT)
        await store.open_task(WORKFLOW, "first", TENANT)
        await store.cancel(WORKFLOW, "first", "operator")
        ended = await store.open_task(WORKFLOW, "ended", TENANT)
        await ended.finish(create_complete_event("ended", TENANT, result={}))
        # A queue worker ends a job by its status stream's terminal entry.
        await store.register_queued("job-done", TENANT)
        await shared_state_redis.xadd(
            f"{store._ingest}job-done", {"data": json.dumps({"state": "complete"})}
        )
        other = await store.open_task(WORKFLOW, "other", "other:other")

        # "acme" names "acme:acme".
        cancelled = await store.cancel_tenant("acme", "tenant deleted")
        await store.poll_once()
        reads = {
            task: await store.read(task, count=0)
            for task in ("silent", "running", "queued", "first", "ended", "job-done")
        }
        active = await shared_state_redis.zrange(
            f"{store._prefix}:active:{TENANT}", 0, -1
        )

        assert sorted(cancelled) == ["first", "queued", "running"]
        assert {
            task: (read.cancelled, read.cancel_reason) for task, read in reads.items()
        } == {
            "silent": (False, None),
            "running": (True, "tenant deleted"),
            "queued": (True, "tenant deleted"),
            "first": (True, "operator"),
            "ended": (False, None),
            "job-done": (False, None),
        }
        assert running.cancellation_token.reason == "tenant deleted"
        assert other.cancellation_token.is_cancelled is False
        assert (await store.read("other", count=0)).cancelled is False
        # Tasks that ended or stopped reporting leave the tenant's index.
        assert sorted(active) == ["first", "queued", "running"]
        assert await store.cancel_tenant(TENANT, "again") == cancelled
        # The listing names the tenant the way the cancel does.
        assert [row["task_id"] for row in await store.list_active("acme")] == [
            row["task_id"] for row in await store.list_active(TENANT)
        ]
        assert {row["tenant_id"] for row in await store.list_active("acme")} == {TENANT}

    async def test_a_tenant_cancel_racing_opens_cancels_exactly_what_it_reports(
        self, shared_state_redis, shared_state_redis_url
    ):
        """Processes open tasks of the tenant and of another tenant while one
        cancels the tenant: the tasks it reports are exactly the tenant's
        tasks that read cancelled, and the other tenant's are untouched."""
        prefix = _prefix()
        outcomes = _run_processes(
            _open_or_cancel_tenant_at_once, (shared_state_redis_url, prefix)
        )
        store = _store(shared_state_redis, prefix)
        opened = [
            task for kind, tasks in outcomes if kind == "opened" for task in tasks
        ]
        (reported,) = [tasks for kind, tasks in outcomes if kind == "cancelled"]
        reads = {task: await store.read(task, count=0) for task in opened}

        assert len(opened) == (PROCESSES - 1) * EVENTS_PER_PROCESS
        assert len(reported) == len(set(reported))
        assert set(reported) == {
            task
            for task, read in reads.items()
            if read.cancelled and read.tenant_id == TENANT
        }
        assert {task for task, read in reads.items() if read.tenant_id != TENANT} == {
            task for task in opened if task.startswith("other-")
        }
        assert all(
            not read.cancelled for read in reads.values() if read.tenant_id != TENANT
        )
        assert all(
            read.cancel_reason == "tenant deleted"
            for read in reads.values()
            if read.cancelled
        )


class TestConcurrentProducers:
    async def test_concurrent_appends_from_processes_get_every_offset_once(
        self, shared_state_redis, shared_state_redis_url
    ):
        prefix = _prefix()
        store = _store(shared_state_redis, prefix)
        await store.open_task(WORKFLOW, "wf", TENANT)

        outcomes = _run_processes(
            _append_at_once, (shared_state_redis_url, prefix, "wf")
        )
        read = await store.read("wf", kind=WORKFLOW, count=PROCESSES * 100)

        total = PROCESSES * EVENTS_PER_PROCESS
        offsets = sorted(offset for offsets in outcomes for offset in offsets)
        assert offsets == list(range(total))
        assert [offset for offset, _, _ in read.events] == list(range(total))
        # Each process's own events stay in the order it appended them.
        stored = [json.loads(data)["phase"] for _, _, data in read.events]
        for writer in range(PROCESSES):
            mine = [phase for phase in stored if phase.startswith(f"w{writer}-")]
            assert mine == [f"w{writer}-{i}" for i in range(EVENTS_PER_PROCESS)]
        assert read.appended == total

    async def test_concurrent_opens_of_one_id_admit_exactly_one(
        self, shared_state_redis, shared_state_redis_url
    ):
        prefix = _prefix()
        outcomes = _run_processes(_open_at_once, (shared_state_redis_url, prefix))

        assert sorted(outcomes) == ["created"] + ["exists"] * (PROCESSES - 1)
        listed = await _store(shared_state_redis, prefix).list_active(TENANT)
        assert [row["task_id"] for row in listed] == ["contended"]


class TestActiveTasks:
    async def test_the_listing_holds_running_and_queued_tasks_of_the_tenant(
        self, shared_state_redis
    ):
        store = _store(shared_state_redis)
        workflow = await store.open_task(WORKFLOW, "wf", TENANT)
        await workflow.enqueue(_working("wf", "planning"))
        await store.register_queued("job-queued", TENANT)
        ended = await store.open_task(WORKFLOW, "wf-ended", TENANT)
        await ended.finish(create_complete_event("wf-ended", TENANT, result={}))
        await store.open_task(WORKFLOW, "wf-other", "other:other")
        await store.cancel(WORKFLOW, "wf", "stop")
        await store.read("wf", kind=WORKFLOW, subscriber="reader-1")
        await store.read("wf", kind=WORKFLOW, subscriber="reader-2")
        await store.leave("wf", "reader-2")

        listed = await store.list_active(TENANT)

        assert sorted(
            ({key: row[key] for key in row if key != "created_at"} for row in listed),
            key=lambda row: row["task_id"],
        ) == [
            {
                "task_id": "job-queued",
                "kind": INGESTION,
                "tenant_id": TENANT,
                "event_count": 0,
                "subscriber_count": 0,
                "is_closed": False,
                "is_cancelled": False,
            },
            {
                "task_id": "wf",
                "kind": WORKFLOW,
                "tenant_id": TENANT,
                "event_count": 1,
                "subscriber_count": 1,
                "is_closed": False,
                "is_cancelled": True,
            },
        ]

    async def test_a_task_whose_lease_lapses_stopped_reporting(
        self, shared_state_redis
    ):
        store = _store(shared_state_redis, producer_lease_s=1.0, poll_interval_s=0.1)
        queue = await store.open_task(WORKFLOW, "wf", TENANT)
        queue.release()
        await asyncio.sleep(1.2)

        read = await store.read("wf", kind=WORKFLOW)

        assert read.stopped_reporting is True
        assert read.closed is False
        assert await store.list_active(TENANT) == []

    async def test_the_poller_keeps_a_running_task_leased(self, shared_state_redis):
        store = _store(shared_state_redis, producer_lease_s=1.0, poll_interval_s=0.1)
        await store.open_task(WORKFLOW, "wf", TENANT)
        store.start()
        try:
            await asyncio.sleep(2.5)
            read = await store.read("wf", kind=WORKFLOW)
            listed = [row["task_id"] for row in await store.list_active(TENANT)]
        finally:
            await store.close()

        assert read.stopped_reporting is False
        assert listed == ["wf"]


class TestIngestionEvents:
    async def test_pipeline_events_are_status_entries_and_end_with_the_outcome(
        self, shared_state_redis
    ):
        prefix = _prefix()
        ingest = f"{prefix}:ingest:"
        store = _store(shared_state_redis, prefix, ingestion_stream_prefix=ingest)
        queue = await store.open_task(INGESTION, "job", TENANT)
        await queue.enqueue(_working("job", "starting"))
        await queue.enqueue(
            create_progress_event("job", TENANT, current=0, total=2, step="video_1")
        )
        pipeline_done = create_complete_event(
            "job", TENANT, result={"successful": 2, "failed": 0, "total": 2}
        )
        await queue.enqueue(pipeline_done)
        before_outcome = await store.read("job", kind=INGESTION)
        await queue.finish_ingestion(
            "complete", result={"status": "completed", "videos_processed": 2}
        )

        entries = [
            json.loads(fields["data"])
            for _, fields in await shared_state_redis.xrange(f"{ingest}job")
        ]
        read = await store.read("job", kind=INGESTION)
        events = [
            stream_event(INGESTION, "job", TENANT, entry, data)
            for _, entry, data in read.events
        ]

        # The pipeline's own end-of-job event waits for the job's outcome.
        assert before_outcome.closed is False
        assert [offset for offset, _, _ in before_outcome.events] == [0, 1]
        assert [entry["state"] for entry in entries] == [
            "running",
            "running",
            "complete",
        ]
        assert entries[0] == {
            "state": "running",
            "ingest_id": "job",
            "event": json.loads(events_json(read, 0)),
        }
        assert entries[2] == {
            "state": "complete",
            "ingest_id": "job",
            "result": {"status": "completed", "videos_processed": 2},
            "event": pipeline_done.model_dump(mode="json"),
        }
        assert [event["event_type"] for event in events] == [
            "status",
            "progress",
            "complete",
        ]
        assert events[2]["result"] == {"successful": 2, "failed": 0, "total": 2}
        assert read.closed is True

    async def test_queue_worker_statuses_report_as_task_events(
        self, shared_state_redis
    ):
        """The statuses queue-driven ingestion publishes, read through the
        task's events; the terminal status ends the task."""
        job = f"job-{uuid.uuid4().hex}"
        store = _store(
            shared_state_redis,
            ingestion_stream_prefix=ingest_queue.STATUS_STREAM_KEY_PREFIX,
        )
        await store.register_queued(job, TENANT)
        for status in (
            {"state": "queued", "ingest_id": job},
            {"state": "running", "ingest_id": job, "consumer_id": "w1"},
            {"state": "retrying", "ingest_id": job, "error": "graph stage"},
            {
                "state": "failed",
                "ingest_id": job,
                "error": "pipeline failed",
                "error_type": "IngestPipelineError",
            },
        ):
            await ingest_queue.publish_status(shared_state_redis, job, status)

        try:
            read = await store.read(job, kind=INGESTION)
            late_cancel = await store.cancel(INGESTION, job, "late")
            listed = await store.list_active(TENANT)
        finally:
            await shared_state_redis.delete(ingest_queue._status_stream_key(job))
        events = [
            stream_event(INGESTION, job, TENANT, entry, data)
            for _, entry, data in read.events
        ]

        assert [
            (event["event_type"], event.get("state"), event.get("phase"))
            for event in events
        ] == [
            ("status", "pending", "queued"),
            ("status", "working", "running"),
            ("status", "working", "retrying"),
            ("error", None, None),
        ]
        assert events[2]["message"] == "graph stage"
        assert (events[3]["error_type"], events[3]["error_message"]) == (
            "IngestPipelineError",
            "pipeline failed",
        )
        assert events[3]["recoverable"] is False
        assert {event["task_id"] for event in events} == {job}
        # Each event is stamped with its entry: a replay reports the same one.
        assert [event["event_id"] for event in events] == [
            f"evt_{entry}" for _, entry, _ in read.events
        ]
        assert read.closed is True
        assert late_cancel == "finished"
        assert listed == []

    async def test_a_cancelled_queued_job_is_seen_by_the_worker_that_claims_it(
        self, shared_state_redis
    ):
        store = _store(shared_state_redis)
        await store.register_queued("job", TENANT)
        assert await store.cancel(INGESTION, "job", "not needed") == "cancelled"

        claimed = await store.attach(INGESTION, "job", TENANT)

        assert claimed.cancellation_token.is_cancelled is True
        assert claimed.cancellation_token.reason == "not needed"


def _dispatcher(store):
    from unittest.mock import MagicMock

    from cogniverse_runtime.agent_dispatcher import AgentDispatcher

    return AgentDispatcher(
        agent_registry=MagicMock(),
        config_manager=MagicMock(),
        schema_loader=MagicMock(),
        task_events=store,
    )


def _stored(read):
    return [json.loads(data) for _, _, data in read.events]


class TestWorkflowRuns:
    """The dispatcher reports each orchestration or deep-research run as a
    workflow task, its queue bound to the request."""

    async def test_a_run_reports_its_phases_and_completes(self, shared_state_redis):
        store = _store(shared_state_redis)
        dispatcher = _dispatcher(store)

        async with dispatcher.workflow_run(
            "orchestrator_agent", {"workflow_id": "wf-run"}, "acme"
        ) as run:
            await publish_phase("planning", "Creating execution plan...")
            run.result = {"status": "success"}
            run.summary = "Orchestrated"
        read = await store.read("wf-run", kind=WORKFLOW)
        stored = _stored(read)

        assert run.workflow_id == "wf-run"
        assert [
            (event["event_type"], event.get("phase"), event.get("message"))
            for event in stored
        ] == [
            ("status", "started", "orchestrator_agent started"),
            ("status", "planning", "Creating execution plan..."),
            ("complete", None, None),
        ]
        assert (stored[2]["result"], stored[2]["summary"]) == (
            {"status": "success"},
            "Orchestrated",
        )
        assert {event["tenant_id"] for event in stored} == {TENANT}
        assert read.closed is True
        assert store._producers == {}

    async def test_a_cancelled_run_ends_cancelled_and_stops(self, shared_state_redis):
        store = _store(shared_state_redis)
        dispatcher = _dispatcher(store)

        with pytest.raises(TaskCancelled) as stopped:
            async with dispatcher.workflow_run(
                "deep_research_agent", {"workflow_id": "wf-stop"}, TENANT
            ):
                await store.cancel(WORKFLOW, "wf-stop", "operator stop")
                await publish_phase("decompose", "Decomposing...")
                await publish_phase("search", "never reported")
        stored = _stored(await store.read("wf-stop", kind=WORKFLOW))

        assert str(stopped.value) == "task wf-stop was cancelled: operator stop"
        assert [
            (event["event_type"], event.get("state"), event.get("phase"))
            for event in stored
        ] == [
            ("status", "working", "started"),
            ("status", "working", "decompose"),
            ("status", "cancelled", "cancelled"),
        ]
        assert stored[2]["message"] == "operator stop"

    async def test_a_failed_run_ends_with_its_failure(self, shared_state_redis):
        store = _store(shared_state_redis)
        dispatcher = _dispatcher(store)

        with pytest.raises(LookupError):
            async with dispatcher.workflow_run(
                "orchestrator_agent", {"workflow_id": "wf-fail"}, TENANT
            ):
                raise LookupError("secret backend url http://user:pw@host")
        stored = _stored(await store.read("wf-fail", kind=WORKFLOW))

        # The failure's text stays in the log: it can carry credentials.
        assert {
            key: stored[1][key]
            for key in stored[1]
            if key in ("event_type", "error_type", "error_message", "recoverable")
        } == {
            "event_type": "error",
            "error_type": "LookupError",
            "error_message": "orchestrator_agent failed with LookupError",
            "recoverable": False,
        }
        assert len(stored) == 2

    async def test_a_run_whose_request_ends_first_ends_cancelled(
        self, shared_state_redis
    ):
        store = _store(shared_state_redis)
        dispatcher = _dispatcher(store)
        entered = asyncio.Event()

        async def request():
            async with dispatcher.workflow_run(
                "orchestrator_agent", {"workflow_id": "wf-gone"}, TENANT
            ):
                entered.set()
                await asyncio.sleep(60)

        task = asyncio.create_task(request())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        stored = _stored(await store.read("wf-gone", kind=WORKFLOW))

        assert [(event.get("state"), event.get("message")) for event in stored] == [
            ("working", "orchestrator_agent started"),
            (
                "cancelled",
                "the request running the workflow ended before it finished",
            ),
        ]

    async def test_a_named_workflow_id_is_used_once(self, shared_state_redis):
        store = _store(shared_state_redis)
        dispatcher = _dispatcher(store)
        async with dispatcher.workflow_run(
            "orchestrator_agent", {"workflow_id": "wf-once"}, TENANT
        ):
            pass

        with pytest.raises(TaskAlreadyExists) as taken:
            async with dispatcher.workflow_run(
                "orchestrator_agent", {"workflow_id": "wf-once"}, TENANT
            ):
                pass
        async with dispatcher.workflow_run("orchestrator_agent", {}, TENANT) as run:
            unnamed = run.workflow_id

        assert taken.value.task_id == "wf-once"
        assert unnamed.startswith("workflow_")
        assert len(unnamed) == len("workflow_") + 32

    async def test_without_a_store_a_run_is_unreported_unless_named(self):
        dispatcher = _dispatcher(None)

        async with dispatcher.workflow_run("orchestrator_agent", {}, TENANT) as run:
            unreported = run
        with pytest.raises(TaskEventsUnavailable) as refused:
            async with dispatcher.workflow_run(
                "orchestrator_agent", {"workflow_id": "wf-x"}, TENANT
            ):
                pass

        assert unreported is None
        assert str(refused.value) == (
            "task event store unavailable: workflow events need the shared task "
            "event store, and none is configured"
        )

    async def test_concurrent_runs_on_one_dispatcher_each_report_their_own(
        self, shared_state_redis
    ):
        """One dispatcher and one agent serve many requests at once: each run's
        phases land on its own task, never on another's."""
        store = _store(shared_state_redis)
        dispatcher = _dispatcher(store)
        runs = 8
        barrier = asyncio.Barrier(runs)

        async def request(index):
            async with dispatcher.workflow_run(
                "orchestrator_agent", {"workflow_id": f"wf-{index}"}, TENANT
            ):
                await barrier.wait()
                for step in range(3):
                    await publish_phase(f"r{index}-s{step}", "step")
                    await asyncio.sleep(0)

        await asyncio.gather(*(request(index) for index in range(runs)))
        phases = {
            index: [
                event.get("phase")
                for event in _stored(await store.read(f"wf-{index}", kind=WORKFLOW))
            ]
            for index in range(runs)
        }

        assert phases == {
            index: [
                "started",
                f"r{index}-s0",
                f"r{index}-s1",
                f"r{index}-s2",
                None,
            ]
            for index in range(runs)
        }


class _PhasedInput(AgentInput):
    query: str


class _PhasedOutput(AgentOutput):
    answer: str


class _PhasedAgent(AgentBase[_PhasedInput, _PhasedOutput, AgentDeps]):
    """A workflow agent reporting two phases; ``hold`` keeps it between them."""

    def __init__(self, hold: asyncio.Event | None = None):
        super().__init__(deps=AgentDeps())
        self.hold = hold
        self.reached: list[str] = []

    async def _process_impl(self, input: _PhasedInput) -> _PhasedOutput:
        await self.report_phase("planning", "Creating execution plan...")
        self.reached.append("planning")
        if self.hold is not None:
            await self.hold.wait()
        await self.report_phase("execution", "Executing...")
        self.reached.append("execution")
        return _PhasedOutput(answer=f"answered {input.query}")


def _streaming_dispatcher(store, agent):
    from unittest.mock import AsyncMock, MagicMock

    dispatcher = _dispatcher(store)
    endpoint = MagicMock()
    endpoint.capabilities = ["orchestration", "planning"]
    dispatcher._registry.get_agent.return_value = endpoint

    async def streaming_agent(agent_name, query, tenant_id, context=None):
        return agent, _PhasedInput(query=query)

    dispatcher.create_streaming_agent = streaming_agent
    dispatcher.resolve_artefact_for_request = AsyncMock(return_value=None)
    dispatcher._init_agent_memory = lambda *_args: None
    dispatcher._bind_graph_manager = lambda *_args: None
    return dispatcher


class TestStreamedWorkflows:
    """A streamed orchestrator run (A2A ``message/stream``, ``/v1`` with
    ``stream``) is reported as a workflow task like a dispatched one."""

    async def test_a_streamed_run_reports_its_phases_and_completes(
        self, shared_state_redis
    ):
        from cogniverse_runtime.a2a_executor import stream_agent_events

        store = _store(shared_state_redis)
        dispatcher = _streaming_dispatcher(store, _PhasedAgent())

        streamed = [
            event
            async for event in stream_agent_events(
                dispatcher,
                "orchestrator_agent",
                "find cats",
                TENANT,
                {"workflow_id": "wf-stream"},
            )
        ]
        stored = _stored(await store.read("wf-stream", kind=WORKFLOW))

        assert [(event["type"], event.get("phase")) for event in streamed] == [
            ("status", "planning"),
            ("status", "execution"),
            ("final", None),
        ]
        assert streamed[2]["data"] == {"answer": "answered find cats"}
        assert [(event["event_type"], event.get("phase")) for event in stored] == [
            ("status", "started"),
            ("status", "planning"),
            ("status", "execution"),
            ("complete", None),
        ]

    async def test_a_cancelled_streamed_run_ends_with_the_cancellation(
        self, shared_state_redis, caplog
    ):
        import logging

        from cogniverse_runtime.a2a_executor import stream_agent_events

        caplog.set_level(logging.INFO, logger="cogniverse_core.agents.base")

        store = _store(shared_state_redis)
        hold = asyncio.Event()
        agent = _PhasedAgent(hold)
        dispatcher = _streaming_dispatcher(store, agent)
        streamed = []

        async def consume():
            async for event in stream_agent_events(
                dispatcher,
                "orchestrator_agent",
                "find cats",
                TENANT,
                {"workflow_id": "wf-stream-stop"},
            ):
                streamed.append(event)
                if event.get("phase") == "planning":
                    assert (
                        await store.cancel(WORKFLOW, "wf-stream-stop", "operator")
                        == "cancelled"
                    )
                    await store.poll_once()
                    hold.set()

        await consume()
        stored = _stored(await store.read("wf-stream-stop", kind=WORKFLOW))

        # The run reports the boundary it stopped at, then the cancellation.
        assert agent.reached == ["planning"]
        assert [(event["type"], event.get("phase")) for event in streamed] == [
            ("status", "planning"),
            ("status", "execution"),
            ("final", None),
        ]
        assert streamed[2]["data"] == {
            "status": "cancelled",
            "agent": "orchestrator_agent",
            "workflow_id": "wf-stream-stop",
            "message": "Workflow wf-stream-stop was cancelled: operator",
        }
        assert [(event.get("state"), event.get("phase")) for event in stored] == [
            ("working", "started"),
            ("working", "planning"),
            ("working", "execution"),
            ("cancelled", "cancelled"),
        ]
        # A cancellation is where the run was told to stop, not a failure.
        assert [
            (record.levelname, record.getMessage())
            for record in caplog.records
            if record.name == "cogniverse_core.agents.base"
        ] == [
            (
                "INFO",
                "_PhasedAgent stopped at a cancellation: task wf-stream-stop was "
                "cancelled: operator",
            )
        ]

    async def test_a_stream_its_caller_leaves_ends_the_run_cancelled(
        self, shared_state_redis
    ):
        import contextlib

        from cogniverse_runtime.a2a_executor import stream_agent_events

        store = _store(shared_state_redis)
        agent = _PhasedAgent(asyncio.Event())
        dispatcher = _streaming_dispatcher(store, agent)

        async with contextlib.aclosing(
            stream_agent_events(
                dispatcher,
                "orchestrator_agent",
                "find cats",
                TENANT,
                {"workflow_id": "wf-stream-left"},
            )
        ) as events:
            first = await anext(events)
            while agent.reached != ["planning"]:
                await asyncio.sleep(0.01)
        stored = _stored(await store.read("wf-stream-left", kind=WORKFLOW))

        assert first["phase"] == "planning"
        assert agent.reached == ["planning"]
        assert [(event.get("state"), event.get("message")) for event in stored] == [
            ("working", "orchestrator_agent started"),
            ("working", "Creating execution plan..."),
            (
                "cancelled",
                "the request running the workflow ended before it finished",
            ),
        ]


class TestTheProcessRoute:
    """``POST /agents/{name}/process`` answers a workflow whose task cannot be
    reported, rather than running it unreported."""

    @staticmethod
    def _app(store, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock

        import httpx
        from fastapi import FastAPI

        from cogniverse_runtime.routers import agents as agents_router

        dispatcher = _dispatcher(store)
        endpoint = MagicMock()
        endpoint.capabilities = {"deep_research"}
        dispatcher._registry.refresh = AsyncMock()
        dispatcher._registry.get_agent.return_value = endpoint

        async def research(query, tenant_id, context=None):
            async with dispatcher.workflow_run(
                "deep_research_agent", context, tenant_id
            ):
                return {"status": "success", "message": f"Researched {query}"}

        async def resolve(query, history):
            return query

        dispatcher._execute_deep_research_task = research
        dispatcher._resolve_history_query = resolve
        dispatcher._maybe_auto_file_wiki = AsyncMock(return_value=None)
        monkeypatch.setattr(agents_router, "_dispatcher", dispatcher, raising=False)
        app = FastAPI()
        app.include_router(agents_router.router, prefix="/agents")
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://runtime"
        )

    @staticmethod
    def _task(workflow_id):
        return {
            "agent_name": "deep_research_agent",
            "query": "q",
            "context": {"tenant_id": TENANT, "workflow_id": workflow_id},
        }

    async def test_a_taken_workflow_id_is_a_conflict(
        self, shared_state_redis, monkeypatch
    ):
        store = _store(shared_state_redis)
        await store.open_task(WORKFLOW, "wf-taken", TENANT)

        async with self._app(store, monkeypatch) as client:
            taken = await client.post(
                "/agents/deep_research_agent/process", json=self._task("wf-taken")
            )
            fresh = await client.post(
                "/agents/deep_research_agent/process", json=self._task("wf-fresh")
            )
        stored = _stored(await store.read("wf-fresh", kind=WORKFLOW))

        assert (taken.status_code, taken.json()) == (
            409,
            {"detail": "Workflow wf-taken already exists; name a new workflow_id"},
        )
        assert fresh.status_code == 200
        assert fresh.json()["message"] == "Researched q"
        assert [event["event_type"] for event in stored] == ["status", "complete"]

    async def test_a_store_that_does_not_answer_is_a_503(self, monkeypatch, caplog):
        import logging

        caplog.set_level(logging.ERROR, logger="cogniverse_runtime.http_errors")
        dead = Redis.from_url(
            f"redis://127.0.0.1:{_free_port()}/0",
            decode_responses=True,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )

        async with self._app(_store(dead), monkeypatch) as client:
            response = await client.post(
                "/agents/deep_research_agent/process", json=self._task("wf-down")
            )
        await dead.aclose()

        assert (response.status_code, response.json()) == (
            503,
            {
                "detail": {
                    "error": "task_events_unavailable",
                    "message": "Agent 'deep_research_agent' could not complete: "
                    "the task event store did not answer; retry.",
                    "failure": "TaskEventsUnavailable",
                    "agent": "deep_research_agent",
                    "request_id": response.json()["detail"]["request_id"],
                }
            },
        )
        assert [
            record.getMessage()
            for record in caplog.records
            if record.name == "cogniverse_runtime.http_errors"
        ] == [
            "task_events_unavailable: TaskEventsUnavailable: task event store "
            "unavailable: open task wf-down"
        ]


class TestV1:
    async def test_a_store_that_does_not_answer_is_a_503_naming_it(self):
        """A deep-research turn over ``/v1`` cannot open its workflow task:
        the plain turn answers 503 and the streamed turn ends with a
        ``service_unavailable`` frame, both naming the task event store."""
        import httpx
        from fastapi import FastAPI

        from cogniverse_core.common.agent_models import AgentEndpoint
        from cogniverse_core.registries.agent_registry import AgentRegistry
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_runtime.agent_dispatcher import AgentDispatcher
        from cogniverse_runtime.routers import openai_compat
        from tests.utils.memory_store import InMemoryConfigStore

        config_store = InMemoryConfigStore()
        config_store.initialize()
        config_manager = ConfigManager(store=config_store)
        registry = AgentRegistry(tenant_id=TENANT, config_manager=config_manager)
        registry.register_agent(
            AgentEndpoint(
                name="deep_research_agent",
                url="http://localhost:8000",
                capabilities=["deep_research"],
            )
        )
        dead = Redis.from_url(
            f"redis://127.0.0.1:{_free_port()}/0",
            decode_responses=True,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        dispatcher = AgentDispatcher(
            agent_registry=registry,
            config_manager=config_manager,
            schema_loader=None,
            task_events=_store(dead),
        )
        openai_compat.set_dispatcher_provider(lambda: dispatcher)
        openai_compat.set_api_keys({"key-a": "acme"})
        openai_compat.set_model_map({"cogniverse/deep-research": "deep_research_agent"})
        openai_compat.set_key_resolver(None)
        app = FastAPI()
        app.include_router(openai_compat.router, prefix="/v1")
        body = {
            "model": "cogniverse/deep-research",
            "messages": [{"role": "user", "content": "what changed?"}],
        }
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://runtime"
            ) as client:
                plain = await client.post(
                    "/v1/chat/completions",
                    json={**body, "stream": False},
                    headers={"Authorization": "Bearer key-a"},
                )
                streamed = await client.post(
                    "/v1/chat/completions",
                    json={**body, "stream": True},
                    headers={"Authorization": "Bearer key-a"},
                )
        finally:
            openai_compat.set_dispatcher_provider(None)
            openai_compat.set_api_keys({})
            openai_compat.set_model_map({})
            await dead.aclose()

        assert (plain.status_code, plain.json()) == (
            503,
            {
                "error": {
                    "message": (
                        "The task event store is unavailable "
                        "(TaskEventsUnavailable). See server logs for detail."
                    ),
                    "type": "server_error",
                    "code": "service_unavailable",
                    "error_type": "TaskEventsUnavailable",
                }
            },
        )
        frames = [
            line[len("data: ") :]
            for line in streamed.text.splitlines()
            if line.startswith("data: ")
        ]
        assert frames[-1] == "[DONE]"
        assert json.loads(frames[-2]) == {
            "error": {
                "message": (
                    "deep_research_agent failed with TaskEventsUnavailable. "
                    "See server logs for detail."
                ),
                "agent": "deep_research_agent",
                "error_type": "TaskEventsUnavailable",
                "type": "server_error",
                "code": "service_unavailable",
            }
        }


class TestThreadedProducers:
    async def test_an_offloaded_rlm_reports_on_the_queue_loop(self, shared_state_redis):
        """The RLM runs on a worker thread; its events reach the workflow's
        task through the loop the queue was created on."""
        from cogniverse_agents.inference.instrumented_rlm import InstrumentedRLM

        store = _store(shared_state_redis)
        queue = await store.open_task(WORKFLOW, "wf-rlm", TENANT)
        rlm = InstrumentedRLM(
            "context, query -> answer",
            event_queue=queue,
            task_id="wf-rlm",
            tenant_id=TENANT,
            max_iterations=3,
        )

        await asyncio.to_thread(
            rlm._emit_sync,
            lambda: create_status_event(
                "wf-rlm", TENANT, TaskState.WORKING, phase="rlm_start"
            ),
        )
        stored = _stored(await store.read("wf-rlm", kind=WORKFLOW))

        assert [(event["phase"], event["task_id"]) for event in stored] == [
            ("rlm_start", "wf-rlm")
        ]

    async def test_an_offloaded_rlm_fails_when_its_event_is_refused(
        self, shared_state_redis
    ):
        from cogniverse_agents.inference.instrumented_rlm import InstrumentedRLM

        store = _store(shared_state_redis)
        queue = await store.open_task(WORKFLOW, "wf-rlm-closed", TENANT)
        await queue.finish(create_complete_event("wf-rlm-closed", TENANT, result={}))
        rlm = InstrumentedRLM(
            "context, query -> answer",
            event_queue=queue,
            task_id="wf-rlm-closed",
            tenant_id=TENANT,
        )

        with pytest.raises(TaskClosedError) as refused:
            await asyncio.to_thread(
                rlm._emit_sync,
                lambda: _working("wf-rlm-closed", "rlm_iteration"),
            )

        assert str(refused.value) == "Queue wf-rlm-closed is closed"


class TestOutages:
    async def test_every_operation_raises_on_a_dead_redis(self):
        redis = Redis.from_url(
            f"redis://127.0.0.1:{_free_port()}/0",
            decode_responses=True,
            socket_connect_timeout=1.0,
            socket_timeout=1.0,
        )
        store = _store(redis)
        calls = {
            "open": lambda: store.open_task(WORKFLOW, "wf", TENANT),
            "register": lambda: store.register_queued("job", TENANT),
            "attach": lambda: store.attach(INGESTION, "job", TENANT),
            "append": lambda: store.append(WORKFLOW, "wf", "{}"),
            "cancel": lambda: store.cancel(WORKFLOW, "wf", "stop"),
            "read": lambda: store.read("wf", kind=WORKFLOW),
            "leave": lambda: store.leave("wf", "reader"),
            "list": lambda: store.list_active(TENANT),
            "cancel_tenant": lambda: store.cancel_tenant(TENANT, "deleted"),
        }
        raised = {}
        for name, call in calls.items():
            with pytest.raises(TaskEventsUnavailable) as failure:
                await call()
            raised[name] = (str(failure.value), type(failure.value.__cause__))
        await redis.aclose()

        unavailable = "task event store unavailable"
        assert raised == {
            "open": (f"{unavailable}: open task wf", RedisConnectionError),
            "register": (f"{unavailable}: register task job", RedisConnectionError),
            "attach": (f"{unavailable}: attach to task job", RedisConnectionError),
            "append": (f"{unavailable}: append to task wf", RedisConnectionError),
            "cancel": (f"{unavailable}: cancel task wf", RedisConnectionError),
            "read": (f"{unavailable}: read task wf", RedisConnectionError),
            "leave": (f"{unavailable}: leave task wf", RedisConnectionError),
            "list": (
                f"{unavailable}: list the active tasks of tenant {TENANT}",
                RedisConnectionError,
            ),
            "cancel_tenant": (
                f"{unavailable}: cancel the tasks of tenant {TENANT}",
                RedisConnectionError,
            ),
        }

    async def test_a_paused_redis_fails_a_producer_within_its_bound(self, own_redis):
        url, pause, resume = own_redis
        redis = await connect_shared_state_redis(url, timeout_seconds=1.0)
        store = _store(redis, poll_interval_s=0.05)
        queue = await store.open_task(WORKFLOW, "wf", TENANT)
        store.start()
        pause()
        try:
            started = time.monotonic()
            with pytest.raises(TaskEventsUnavailable) as failure:
                await queue.enqueue(_working("wf", "planning"))
            elapsed = time.monotonic() - started
        finally:
            resume()
        try:
            # The poller recovers with Redis and still delivers a
            # cancellation recorded after the outage.
            assert await store.cancel(WORKFLOW, "wf", "after") == "cancelled"
            deadline = time.monotonic() + 5
            while not queue.cancellation_token.is_cancelled:
                assert time.monotonic() < deadline, "cancellation never delivered"
                await asyncio.sleep(0.02)
        finally:
            await store.close()
            await redis.aclose()

        assert str(failure.value) == "task event store unavailable: append to task wf"
        assert type(failure.value.__cause__) is RedisTimeoutError
        assert elapsed < 3.0, elapsed
        assert queue.cancellation_token.reason == "after"


def _run_processes(target, args):
    """Run ``target`` in PROCESSES spawned processes released by one barrier;
    return each one's result."""
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(PROCESSES)
    results = context.Queue()
    processes = [
        context.Process(target=target, args=(*args, index, barrier, results))
        for index in range(PROCESSES)
    ]
    for process in processes:
        process.start()
    try:
        outcomes = [results.get(timeout=120) for _ in processes]
    finally:
        for process in processes:
            process.join(timeout=30)
    assert [process.exitcode for process in processes] == [0] * PROCESSES
    return outcomes


def _in_process(redis_url, run):
    async def main():
        redis = await connect_shared_state_redis(redis_url)
        try:
            return await run(redis)
        finally:
            await redis.aclose()

    return asyncio.run(main())


def _append_at_once(redis_url, prefix, task_id, index, barrier, results):
    async def run(redis):
        store = _store(redis, prefix)
        barrier.wait(timeout=60)
        offsets = []
        for event in range(EVENTS_PER_PROCESS):
            offset, _, _ = await store.append(
                WORKFLOW,
                task_id,
                _working(task_id, f"w{index}-{event}").model_dump_json(),
            )
            offsets.append(offset)
        return offsets

    results.put(_in_process(redis_url, run))


def _open_at_once(redis_url, prefix, index, barrier, results):
    async def run(redis):
        store = _store(redis, prefix)
        barrier.wait(timeout=60)
        try:
            await store.open_task(WORKFLOW, "contended", TENANT)
        except TaskAlreadyExists:
            return "exists"
        return "created"

    results.put(_in_process(redis_url, run))


def _cancel_at_once(redis_url, prefix, task_id, index, barrier, results):
    async def run(redis):
        store = _store(redis, prefix)
        barrier.wait(timeout=60)
        reason = f"canceller-{index}"
        return await store.cancel(WORKFLOW, task_id, reason), reason

    results.put(_in_process(redis_url, run))


def _open_or_cancel_tenant_at_once(redis_url, prefix, index, barrier, results):
    """The last process cancels the tenant; every other one opens tasks,
    alternating between the tenant and another tenant."""

    async def run(redis):
        store = _store(redis, prefix)
        barrier.wait(timeout=60)
        if index == PROCESSES - 1:
            await asyncio.sleep(0.005)
            return "cancelled", await store.cancel_tenant(TENANT, "tenant deleted")
        opened = []
        for task in range(EVENTS_PER_PROCESS):
            if task % 2:
                task_id, tenant = f"other-{index}-{task}", "other:other"
            else:
                task_id, tenant = f"own-{index}-{task}", TENANT
            await store.open_task(WORKFLOW, task_id, tenant)
            opened.append(task_id)
        return "opened", opened

    results.put(_in_process(redis_url, run))


def _run_process(target, args):
    """Run ``target`` in one spawned process and return its result."""
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    process = context.Process(target=target, args=(*args, results))
    process.start()
    try:
        outcome = results.get(timeout=120)
    finally:
        process.join(timeout=30)
    assert process.exitcode == 0
    return outcome


def _cancel_in_process(redis_url, prefix, task_id, reason, results):
    async def run(redis):
        return await _store(redis, prefix).cancel(WORKFLOW, task_id, reason)

    results.put(_in_process(redis_url, run))
