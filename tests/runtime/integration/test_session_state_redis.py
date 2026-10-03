"""Per-session state shared through Redis: the conversation ledger and the
``/v1`` continuation store, against a real Redis.

Interleavings are executed, not reasoned about: concurrent writers are separate
OS processes released together by a barrier, and their Redis-clock windows are
asserted to overlap. Outages are real: a port nothing listens on, and a Redis
container this module owns and pauses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import socket
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from cogniverse_agents.routing.annotation_agent import (
    AnnotationPriority,
    AnnotationRequest,
)
from cogniverse_agents.routing.annotation_queue import (
    AnnotationQueue,
    AnnotationQueueUnavailableError,
)
from cogniverse_core.registries.agent_registry import (
    AgentRegistryUnavailableError,
    RegistryVersion,
)
from cogniverse_evaluation.evaluators.routing_evaluator import RoutingOutcome
from cogniverse_runtime.agent_dispatcher import (
    CONVERSATION_HISTORY_INCOMPLETE,
    AgentDispatcher,
)
from cogniverse_runtime.agent_registry_store import RedisAgentRegistryStore
from cogniverse_runtime.ingestion_jobs import (
    IngestionJobStore,
    IngestionJobStoreUnavailableError,
)
from cogniverse_runtime.routers import agents as agents_router
from cogniverse_runtime.routers import openai_compat
from cogniverse_runtime.session_state import (
    ContinuationStore,
    ConversationLedger,
    ConversationPersistFailed,
    SessionStateUnavailable,
)
from cogniverse_runtime.shared_state import (
    SharedStateUnavailableError,
    connect_shared_state_redis,
)

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]

TENANT = "acme:acme"
PEER_TENANT = "peer:peer"
PROCESSES = 4
ACCEPTS_PER_PROCESS = 10
PAUSED_TIMEOUT_S = 1.0


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _prefix(kind: str) -> str:
    return f"test:{kind}:{uuid.uuid4().hex}"


def _ledger(redis, prefix, *, lease_s=60.0, capacity=8) -> ConversationLedger:
    return ConversationLedger(
        redis, save_lease_s=lease_s, failure_capacity=capacity, key_prefix=prefix
    )


async def _redis_now_us(redis) -> int:
    seconds, micros = await redis.time()
    return seconds * 1_000_000 + micros


@pytest.fixture(scope="module")
def pausable_redis():
    """A Redis this module owns, so pausing it stalls no other test."""
    port = _free_port()
    name = f"cogniverse-session-state-{os.getpid()}-{port}"
    started = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            "-p",
            f"{port}:6379",
            "redis:7.4-alpine",
        ],
        capture_output=True,
        text=True,
    )
    assert started.returncode == 0, started.stderr
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        ping = subprocess.run(
            ["docker", "exec", name, "redis-cli", "ping"],
            capture_output=True,
            text=True,
        )
        if ping.stdout.strip() == "PONG":
            break
        time.sleep(0.25)
    else:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        pytest.fail("the pausable Redis did not answer PING within 30 seconds")
    try:
        yield {"url": f"redis://127.0.0.1:{port}/0", "container": name}
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.fixture
def dead_redis():
    """A client on a port nothing listens on."""
    port = _free_port()
    return Redis.from_url(
        f"redis://127.0.0.1:{port}/0",
        decode_responses=True,
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
    )


class TestConversationLedger:
    async def test_positions_come_from_redis_time_and_step_by_two(self, session_redis):
        ledger = _ledger(session_redis, _prefix("conversation"))

        before = await _redis_now_us(session_redis)
        first = await ledger.accept(TENANT, "ctx")
        second = await ledger.accept(TENANT, "ctx")
        after = await _redis_now_us(session_redis)

        assert before <= first <= after
        assert first + 2 <= second <= after + 2
        assert await ledger.pending(TENANT, "ctx") == [first, second]
        # Another tenant's context of the same name has its own order.
        assert await ledger.pending(PEER_TENANT, "ctx") == []

    async def test_a_clock_ahead_of_redis_time_is_never_stepped_back(
        self, session_redis
    ):
        prefix = _prefix("conversation")
        ledger = _ledger(session_redis, prefix)
        first = await ledger.accept(TENANT, "ctx")
        ahead = first + 3_600_000_000
        await session_redis.hset(ledger._state_key(TENANT, "ctx"), "clock", ahead)

        assert await ledger.accept(TENANT, "ctx") == ahead + 2

    async def test_an_expired_clock_restarts_after_every_position_it_gave(
        self, session_redis
    ):
        ledger = _ledger(session_redis, _prefix("conversation"))
        first = await ledger.accept(TENANT, "ctx")
        await session_redis.delete(ledger._state_key(TENANT, "ctx"))

        restarted = await ledger.accept(TENANT, "ctx")

        assert restarted > first
        assert await ledger.pending(TENANT, "ctx") == [first, restarted]

    async def test_a_settled_turn_stops_holding_the_context(self, session_redis):
        prefix = _prefix("conversation")
        ledger = _ledger(session_redis, prefix)
        peer = _ledger(session_redis, prefix)
        first = await ledger.accept(TENANT, "ctx")
        second = await ledger.accept(TENANT, "ctx")

        await peer.landed(TENANT, "ctx", first)

        assert await ledger.pending(TENANT, "ctx") == [second]
        started = time.monotonic()
        assert await ledger.wait_settled(TENANT, "ctx", [first], 5.0) == []
        assert time.monotonic() - started < 0.5

    async def test_a_wait_returns_what_is_still_pending_at_its_deadline(
        self, session_redis
    ):
        ledger = _ledger(session_redis, _prefix("conversation"))
        held = await ledger.accept(TENANT, "ctx")

        started = time.monotonic()
        remaining = await ledger.wait_settled(TENANT, "ctx", [held], 0.4)
        elapsed = time.monotonic() - started

        assert remaining == [held]
        assert 0.4 <= elapsed < 0.4 + 0.3

    async def test_a_wait_ignores_turns_accepted_after_it_began(self, session_redis):
        ledger = _ledger(session_redis, _prefix("conversation"))
        earlier = await ledger.accept(TENANT, "ctx")
        snapshot = await ledger.pending(TENANT, "ctx")
        later = await ledger.accept(TENANT, "ctx")

        async def settle_earlier():
            await asyncio.sleep(0.1)
            await ledger.landed(TENANT, "ctx", earlier)

        settling = asyncio.create_task(settle_earlier())
        remaining = await ledger.wait_settled(TENANT, "ctx", snapshot, 5.0)
        await settling

        assert snapshot == [earlier]
        assert remaining == []
        assert await ledger.pending(TENANT, "ctx") == [later]

    async def test_a_turn_whose_process_died_stops_holding_at_its_lease(
        self, session_redis
    ):
        ledger = _ledger(session_redis, _prefix("conversation"), lease_s=0.5)
        orphan = await ledger.accept(TENANT, "ctx")

        assert await ledger.pending(TENANT, "ctx") == [orphan]
        await asyncio.sleep(0.6)
        assert await ledger.pending(TENANT, "ctx") == []

    async def test_a_failure_is_readable_until_a_later_turn_lands(self, session_redis):
        prefix = _prefix("conversation")
        ledger = _ledger(session_redis, prefix)
        reader = _ledger(session_redis, prefix)
        lost = await ledger.accept(TENANT, "ctx")
        later = await ledger.accept(TENANT, "ctx")

        await ledger.failed(TENANT, "ctx", lost, "TimeoutError")

        failure = await reader.failure(TENANT, "ctx")
        assert type(failure) is ConversationPersistFailed
        assert (
            failure.tenant_id,
            failure.context_id,
            failure.error_type,
            failure.position,
        ) == (TENANT, "ctx", "TimeoutError", lost)
        assert str(failure) == (
            f"conversation turns for context ctx (tenant {TENANT}) were not "
            "persisted: TimeoutError"
        )
        assert await reader.failures() == [(TENANT, "ctx")]
        assert await reader.pending(TENANT, "ctx") == [later]

        await ledger.landed(TENANT, "ctx", later)

        assert await reader.failure(TENANT, "ctx") is None
        assert await reader.failures() == []

    async def test_recovery_does_not_depend_on_which_outcome_arrives_first(
        self, session_redis
    ):
        ledger = _ledger(session_redis, _prefix("conversation"))
        lost = await ledger.accept(TENANT, "ctx")
        later = await ledger.accept(TENANT, "ctx")

        await ledger.landed(TENANT, "ctx", later)
        await ledger.failed(TENANT, "ctx", lost, "ConnectionError")

        assert await ledger.failure(TENANT, "ctx") is None
        assert await ledger.failures() == []
        assert await ledger.pending(TENANT, "ctx") == []

    async def test_an_earlier_failure_never_replaces_a_later_one(self, session_redis):
        ledger = _ledger(session_redis, _prefix("conversation"))
        earlier = await ledger.accept(TENANT, "ctx")
        later = await ledger.accept(TENANT, "ctx")

        await ledger.failed(TENANT, "ctx", later, "TimeoutError")
        await ledger.failed(TENANT, "ctx", earlier, "ConnectionError")

        failure = await ledger.failure(TENANT, "ctx")
        assert (failure.error_type, failure.position) == ("TimeoutError", later)

    async def test_failures_keep_the_newest_contexts_up_to_capacity(
        self, session_redis
    ):
        ledger = _ledger(session_redis, _prefix("conversation"), capacity=3)
        contexts = [f"ctx-{index}" for index in range(5)]
        for context_id in contexts:
            position = await ledger.accept(TENANT, context_id)
            await ledger.failed(TENANT, context_id, position, "TimeoutError")

        assert await ledger.failures() == [(TENANT, ctx) for ctx in contexts[2:]]
        assert await ledger.failure(TENANT, contexts[1]) is None
        assert (await ledger.failure(TENANT, contexts[2])).context_id == contexts[2]

    async def test_concurrent_processes_get_distinct_ordered_positions(
        self, workflow_state_redis_url, session_redis
    ):
        """Accepts from four OS processes at once, ten concurrent accepts in
        each, all on one context: every position is distinct, each is at least
        two above the one before, and the ledger holds exactly those."""
        prefix = _prefix("conversation")
        outcomes = _run_processes(
            _accept_in_process,
            (workflow_state_redis_url, prefix, TENANT, "busy-ctx"),
        )

        windows = [window for window, _ in outcomes]
        positions = sorted(p for _, accepted in outcomes for p in accepted)
        assert max(start for start, _ in windows) < min(end for _, end in windows)
        assert len(positions) == PROCESSES * ACCEPTS_PER_PROCESS
        assert len(set(positions)) == len(positions)
        assert [
            later - earlier >= 2 for earlier, later in zip(positions, positions[1:])
        ] == [True] * (len(positions) - 1)
        ledger = _ledger(session_redis, prefix)
        assert await ledger.pending(TENANT, "busy-ctx") == positions


class TestContinuationStore:
    async def test_a_suspended_turn_is_taken_once(self, continuation_store):
        await continuation_store.put(
            TENANT, "tool_echo_agent", "seed", ["c2", "c1"], {"plan": "p", "step": 1}
        )

        assert await continuation_store.count() == 1
        assert await continuation_store.pop(
            TENANT, "tool_echo_agent", "seed", ["c1", "c2"]
        ) == {"plan": "p", "step": 1}
        assert (
            await continuation_store.pop(
                TENANT, "tool_echo_agent", "seed", ["c1", "c2"]
            )
            is None
        )
        assert await continuation_store.count() == 0

    async def test_every_key_part_separates_turns(self, continuation_store):
        await continuation_store.put(TENANT, "agent", "seed", ["c1"], {"plan": "own"})

        assert (
            await continuation_store.pop(PEER_TENANT, "agent", "seed", ["c1"]) is None
        )
        assert await continuation_store.pop(TENANT, "other", "seed", ["c1"]) is None
        assert await continuation_store.pop(TENANT, "agent", "other", ["c1"]) is None
        assert await continuation_store.pop(TENANT, "agent", "seed", ["c9"]) is None
        assert await continuation_store.pop(TENANT, "agent", "seed", ["c1"]) == {
            "plan": "own"
        }

    async def test_retention_is_bounded(self, session_redis):
        store = ContinuationStore(
            session_redis, key_prefix=_prefix("continuation"), ttl_seconds=0.3
        )
        await store.put(TENANT, "agent", "seed", ["c1"], {"plan": "p"})

        assert await store.count() == 1
        await asyncio.sleep(0.4)
        assert await store.count() == 0
        assert await store.pop(TENANT, "agent", "seed", ["c1"]) is None

    async def test_concurrent_resumes_from_processes_have_one_winner(
        self, workflow_state_redis_url, continuation_store, session_redis
    ):
        prefix = _prefix("continuation")
        store = ContinuationStore(session_redis, key_prefix=prefix)
        await store.put(TENANT, "agent", "seed", ["c1"], {"plan": "only once"})

        outcomes = _run_processes(
            _pop_in_process, (workflow_state_redis_url, prefix, TENANT)
        )

        windows = [window for window, _ in outcomes]
        assert max(start for start, _ in windows) < min(end for _, end in windows)
        assert sorted(json.dumps(state) for _, state in outcomes) == sorted(
            [json.dumps({"plan": "only once"})] + [json.dumps(None)] * (PROCESSES - 1)
        )
        assert await store.count() == 0


def _run_processes(target, args):
    """Run ``target`` in PROCESSES spawned processes released by one barrier;
    return each one's ``((start_us, end_us), result)``."""
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(PROCESSES)
    results = context.Queue()
    processes = [
        context.Process(target=target, args=(*args, barrier, results))
        for _ in range(PROCESSES)
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


def _accept_in_process(redis_url, prefix, tenant, context_id, barrier, results):
    async def run():
        redis = await connect_shared_state_redis(redis_url)
        try:
            ledger = _ledger(redis, prefix)
            barrier.wait(timeout=60)
            start = await _redis_now_us(redis)
            accepted = await asyncio.gather(
                *(ledger.accept(tenant, context_id) for _ in range(ACCEPTS_PER_PROCESS))
            )
            end = await _redis_now_us(redis)
            return (start, end), list(accepted)
        finally:
            await redis.aclose()

    results.put(asyncio.run(run()))


def _pop_in_process(redis_url, prefix, tenant, barrier, results):
    async def run():
        redis = await connect_shared_state_redis(redis_url)
        try:
            store = ContinuationStore(redis, key_prefix=prefix)
            barrier.wait(timeout=60)
            start = await _redis_now_us(redis)
            state = await store.pop(tenant, "agent", "seed", ["c1"])
            # Hold the window open until every peer has started, so the
            # overlap the caller asserts is the one the pops ran in.
            await asyncio.sleep(0.2)
            end = await _redis_now_us(redis)
            return (start, end), state
        finally:
            await redis.aclose()

    results.put(asyncio.run(run()))


def _annotation(span_id: str) -> AnnotationRequest:
    return AnnotationRequest(
        span_id=span_id,
        timestamp=datetime(2026, 9, 1, 11, 0, tzinfo=timezone.utc),
        query="find clips of animals",
        chosen_agent="search_agent",
        routing_confidence=0.42,
        outcome=RoutingOutcome.AMBIGUOUS,
        priority=AnnotationPriority.MEDIUM,
        reason="one client",
        context={},
    )


def _state_stores(redis) -> dict:
    """Every store the runtime hands its one shared-state client."""
    return {
        "registry": RedisAgentRegistryStore(redis, key_prefix=_prefix("registry")),
        "annotations": AnnotationQueue(redis, key_prefix=_prefix("annotations")),
        "jobs": IngestionJobStore(redis, owner="owner", key_prefix=_prefix("jobs")),
        "ledger": _ledger(redis, _prefix("conversation")),
        "continuations": ContinuationStore(redis, key_prefix=_prefix("continuation")),
    }


class TestOneClientServesEveryStateStore:
    """The runtime opens one shared-state client per process and gives it to
    the shared stores and the session stores alike."""

    OPERATIONS_PER_STORE = 8
    POOL_SIZE = 3

    async def test_concurrent_operations_of_every_store_share_one_bounded_pool(
        self, workflow_state_redis_url
    ):
        name = f"test-state-{uuid.uuid4().hex}"
        redis = await connect_shared_state_redis(
            workflow_state_redis_url, max_connections=self.POOL_SIZE, client_name=name
        )
        observer = Redis.from_url(workflow_state_redis_url, decode_responses=True)
        stores = _state_stores(redis)
        count = self.OPERATIONS_PER_STORE
        await stores["continuations"].put(
            TENANT, "agent", "seed", ["c1"], {"plan": "resume"}
        )
        operations = (
            [
                lambda i=i: stores["registry"].register(
                    f"agent-{i}", {"url": f"http://agent-{i}:9000"}
                )
                for i in range(count)
            ]
            + [
                lambda i=i: stores["annotations"].enqueue(_annotation(f"span-{i}"))
                for i in range(count)
            ]
            + [lambda i=i: stores["jobs"].create(f"job-{i}") for i in range(count)]
            + [lambda: stores["ledger"].accept(TENANT, "ctx") for _ in range(count)]
            + [
                lambda: stores["continuations"].pop(TENANT, "agent", "seed", ["c1"])
                for _ in range(count)
            ]
        )
        barrier = asyncio.Barrier(len(operations))

        async def at_once(operation):
            await barrier.wait()
            return await operation()

        try:
            results = await asyncio.gather(*(at_once(op) for op in operations))
            snapshot = await stores["registry"].snapshot()
            jobs = [await stores["jobs"].get(f"job-{i}") for i in range(count)]
            spans = [
                (await stores["annotations"].get(f"span-{i}")).span_id
                for i in range(count)
            ]
            connections = [
                row for row in await observer.client_list() if row["name"] == name
            ]
        finally:
            await redis.aclose()
            await observer.aclose()

        registered, enqueued, created, accepted, popped = (
            results[i * count : (i + 1) * count] for i in range(5)
        )
        assert registered == [None] * count
        assert snapshot.registered == {
            f"agent-{i}": {"url": f"http://agent-{i}:9000"} for i in range(count)
        }
        assert enqueued == [True] * count
        assert spans == [f"span-{i}" for i in range(count)]
        assert [job["status"] for job in created] == ["started"] * count
        assert jobs == created
        assert len(set(accepted)) == count
        ordered = sorted(accepted)
        assert [b - a >= 2 for a, b in zip(ordered, ordered[1:])] == [True] * (
            count - 1
        )
        assert sorted(popped, key=lambda state: state is not None) == [None] * (
            count - 1
        ) + [{"plan": "resume"}]
        # Forty operations at once waited on the pool rather than opening a
        # connection each.
        assert len(connections) == self.POOL_SIZE

    async def test_a_paused_redis_fails_every_store_with_its_own_error(
        self, pausable_redis
    ):
        redis = await connect_shared_state_redis(
            pausable_redis["url"], timeout_seconds=PAUSED_TIMEOUT_S
        )
        stores = _state_stores(redis)
        calls = {
            "registry": lambda: stores["registry"].version(),
            "annotations": lambda: stores["annotations"].get("span-1"),
            "jobs": lambda: stores["jobs"].get("job-1"),
            "ledger": lambda: stores["ledger"].accept(TENANT, "ctx"),
            "continuations": lambda: stores["continuations"].pop(
                TENANT, "agent", "seed", ["c1"]
            ),
        }
        subprocess.run(
            ["docker", "pause", pausable_redis["container"]],
            check=True,
            capture_output=True,
        )
        raised = {}
        try:
            for store, call in calls.items():
                with pytest.raises(Exception) as failure:
                    await call()
                raised[store] = (
                    type(failure.value),
                    str(failure.value),
                    type(failure.value.__cause__),
                )
        finally:
            subprocess.run(
                ["docker", "unpause", pausable_redis["container"]],
                check=True,
                capture_output=True,
            )
        try:
            recovered = {store: await call() for store, call in calls.items()}
            pending = await stores["ledger"].pending(TENANT, "ctx")
        finally:
            await redis.aclose()

        assert raised == {
            "registry": (
                AgentRegistryUnavailableError,
                "shared agent registry unavailable: read version",
                RedisTimeoutError,
            ),
            "annotations": (
                AnnotationQueueUnavailableError,
                "annotation queue unavailable: get span span-1",
                RedisTimeoutError,
            ),
            "jobs": (
                IngestionJobStoreUnavailableError,
                "ingestion job store unavailable: read job job-1",
                RedisTimeoutError,
            ),
            "ledger": (
                SessionStateUnavailable,
                "session state store unavailable: accept a turn of context ctx",
                RedisTimeoutError,
            ),
            "continuations": (
                SessionStateUnavailable,
                "session state store unavailable: resume the suspended turn of agent",
                RedisTimeoutError,
            ),
        }
        # The same client serves every store again once Redis answers.
        assert {
            store: value for store, value in recovered.items() if store != "ledger"
        } == {
            "registry": RegistryVersion(epoch="", counter=0),
            "annotations": None,
            "jobs": None,
            "continuations": None,
        }
        assert pending == [recovered["ledger"]]


class TestOutages:
    async def test_opening_an_unreachable_redis_names_it_without_credentials(self):
        """The stores run on the process's shared-state client; opening it on
        a Redis nothing answers names the Redis without its credentials."""
        port = _free_port()

        with pytest.raises(SharedStateUnavailableError) as refused:
            await connect_shared_state_redis(
                f"redis://cogniverse:s3cret@127.0.0.1:{port}/0", timeout_seconds=1.0
            )

        assert str(refused.value) == (
            f"shared state Redis unavailable at redis://127.0.0.1:{port}/0"
        )
        assert type(refused.value.__cause__) is RedisConnectionError

    async def test_every_ledger_operation_raises_on_a_dead_redis(self, dead_redis):
        ledger = _ledger(dead_redis, _prefix("conversation"))
        calls = {
            "accept": lambda: ledger.accept(TENANT, "ctx"),
            "pending": lambda: ledger.pending(TENANT, "ctx"),
            "wait_settled": lambda: ledger.wait_settled(TENANT, "ctx", [1], 1.0),
            "landed": lambda: ledger.landed(TENANT, "ctx", 1),
            "failed": lambda: ledger.failed(TENANT, "ctx", 1, "TimeoutError"),
            "failure": lambda: ledger.failure(TENANT, "ctx"),
            "failures": lambda: ledger.failures(),
        }
        raised = {}
        for name, call in calls.items():
            with pytest.raises(SessionStateUnavailable) as failure:
                await call()
            raised[name] = (str(failure.value), type(failure.value.__cause__))

        unavailable = "session state store unavailable"
        assert raised == {
            "accept": (
                f"{unavailable}: accept a turn of context ctx",
                RedisConnectionError,
            ),
            "pending": (
                f"{unavailable}: read the pending turns of context ctx",
                RedisConnectionError,
            ),
            "wait_settled": (
                f"{unavailable}: read the pending turns of context ctx",
                RedisConnectionError,
            ),
            "landed": (
                f"{unavailable}: settle a turn of context ctx",
                RedisConnectionError,
            ),
            "failed": (
                f"{unavailable}: record a lost turn of context ctx",
                RedisConnectionError,
            ),
            "failure": (
                f"{unavailable}: read the lost turns of context ctx",
                RedisConnectionError,
            ),
            "failures": (
                f"{unavailable}: list lost conversation turns",
                RedisConnectionError,
            ),
        }
        await dead_redis.aclose()

    async def test_every_continuation_operation_raises_on_a_dead_redis(
        self, dead_redis
    ):
        store = ContinuationStore(dead_redis, key_prefix=_prefix("continuation"))
        calls = {
            "put": lambda: store.put(TENANT, "agent", "seed", ["c1"], {"plan": "p"}),
            "pop": lambda: store.pop(TENANT, "agent", "seed", ["c1"]),
            "count": lambda: store.count(),
        }
        raised = {}
        for name, call in calls.items():
            with pytest.raises(SessionStateUnavailable) as failure:
                await call()
            raised[name] = (str(failure.value), type(failure.value.__cause__))

        unavailable = "session state store unavailable"
        assert raised == {
            "put": (
                f"{unavailable}: keep the suspended turn of agent",
                RedisConnectionError,
            ),
            "pop": (
                f"{unavailable}: resume the suspended turn of agent",
                RedisConnectionError,
            ),
            "count": (f"{unavailable}: count suspended turns", RedisConnectionError),
        }
        await dead_redis.aclose()

    async def test_a_paused_redis_fails_each_store_within_its_bound(
        self, pausable_redis
    ):
        """A Redis that stops answering raises within the command bound rather
        than hanging the turn."""
        redis = await connect_shared_state_redis(
            pausable_redis["url"], timeout_seconds=PAUSED_TIMEOUT_S
        )
        ledger = _ledger(redis, _prefix("conversation"))
        store = ContinuationStore(redis, key_prefix=_prefix("continuation"))
        subprocess.run(
            ["docker", "pause", pausable_redis["container"]],
            check=True,
            capture_output=True,
        )
        timings = {}
        causes = {}
        try:
            for name, call in {
                "accept": lambda: ledger.accept(TENANT, "ctx"),
                "pop": lambda: store.pop(TENANT, "agent", "seed", ["c1"]),
            }.items():
                started = time.monotonic()
                with pytest.raises(SessionStateUnavailable) as failure:
                    await call()
                timings[name] = time.monotonic() - started
                causes[name] = type(failure.value.__cause__)
        finally:
            subprocess.run(
                ["docker", "unpause", pausable_redis["container"]],
                check=True,
                capture_output=True,
            )
            await redis.aclose()

        print(f"PAUSED_REDIS_ELAPSED_S={timings}")
        assert causes == {"accept": RedisTimeoutError, "pop": RedisTimeoutError}
        assert [
            PAUSED_TIMEOUT_S <= elapsed < PAUSED_TIMEOUT_S + 1.0
            for elapsed in timings.values()
        ] == [True, True]


def _dispatcher(ledger, store, replies, calls):
    dispatcher = AgentDispatcher(
        agent_registry=MagicMock(),
        config_manager=MagicMock(),
        schema_loader=MagicMock(),
        conversation_ledger=ledger,
    )
    endpoint = MagicMock()
    endpoint.capabilities = {"search"}
    dispatcher._registry.refresh = AsyncMock()
    dispatcher._registry.get_agent.return_value = endpoint
    dispatcher._conversation_store_factory = lambda _tenant: store

    async def skip_wiki(*_args, **_kwargs):
        return None

    async def answer(query, tenant_id, top_k, conversation_history=None, **kwargs):
        calls.append({"query": query, "history": list(conversation_history or [])})
        reply = replies[query]
        if callable(reply):
            reply = await reply()
        return {"message": reply, "entities": []}

    dispatcher._maybe_auto_file_wiki = skip_wiki
    dispatcher._execute_search_task = answer
    return dispatcher


class _Store:
    """A ConversationStore-shaped record of what reached the store, read back
    in seq order as the real store reads it."""

    def __init__(self):
        self.rows = []

    def get_history(self, context_id, max_turns=10):
        return [turn for _seq, turn in sorted(self.rows, key=lambda row: row[0])]

    def store_turn(self, context_id, role, content, seq):
        self.rows.append((seq, {"role": role, "content": content}))


async def _until_the_agent_runs(turn: asyncio.Task, at_agent: asyncio.Event) -> None:
    """Wait for ``turn`` to reach its agent. A dispatch that ends before it
    raises its own error here instead of leaving the wait open."""
    reached = asyncio.ensure_future(at_agent.wait())
    done, _ = await asyncio.wait({turn, reached}, return_when=asyncio.FIRST_COMPLETED)
    if reached not in done:
        reached.cancel()
        await turn
        raise AssertionError("the turn ended without reaching its agent")


async def _dispatch(dispatcher, query, context_id="ctx"):
    return await dispatcher.dispatch(
        agent_name="search_agent",
        query=query,
        context={"tenant_id": TENANT, "context_id": context_id},
    )


class TestManagedTurnsAgainstTheLedger:
    async def test_a_dead_ledger_refuses_the_turn_before_the_agent_runs(
        self, dead_redis
    ):
        store, calls = _Store(), []
        dispatcher = _dispatcher(
            _ledger(dead_redis, _prefix("conversation")), store, {"q": "a"}, calls
        )

        with pytest.raises(SessionStateUnavailable) as refused:
            await _dispatch(dispatcher, "q")

        assert str(refused.value) == (
            "session state store unavailable: read the pending turns of context ctx"
        )
        assert calls == []
        assert store.rows == []
        assert await dispatcher.drain_conversation_saves() is True
        await dead_redis.aclose()

    async def test_an_unconfigured_ledger_refuses_the_turn(self):
        store, calls = _Store(), []
        dispatcher = _dispatcher(None, store, {"q": "a"}, calls)

        with pytest.raises(SessionStateUnavailable) as refused:
            await _dispatch(dispatcher, "q")

        assert str(refused.value) == (
            "server-managed conversation history needs the shared conversation "
            "ledger, and none is configured"
        )
        assert calls == []

    async def test_redis_lost_while_the_agent_ran_refuses_the_answer(
        self, pausable_redis
    ):
        """The answer exists but its turn cannot be ordered: the dispatch
        raises rather than reply with a turn no one will store."""
        redis = await connect_shared_state_redis(
            pausable_redis["url"], timeout_seconds=PAUSED_TIMEOUT_S
        )
        store, calls = _Store(), []
        at_agent, release = asyncio.Event(), asyncio.Event()

        async def held_reply():
            at_agent.set()
            await release.wait()
            return "answered"

        dispatcher = _dispatcher(
            _ledger(redis, _prefix("conversation")), store, {"q": held_reply}, calls
        )
        turn = asyncio.create_task(_dispatch(dispatcher, "q"))
        await _until_the_agent_runs(turn, at_agent)
        subprocess.run(
            ["docker", "pause", pausable_redis["container"]],
            check=True,
            capture_output=True,
        )
        try:
            release.set()
            with pytest.raises(SessionStateUnavailable) as refused:
                await turn
        finally:
            subprocess.run(
                ["docker", "unpause", pausable_redis["container"]],
                check=True,
                capture_output=True,
            )
            await redis.aclose()

        assert str(refused.value) == (
            "session state store unavailable: accept a turn of context ctx"
        )
        assert calls == [{"query": "q", "history": []}]
        assert await dispatcher.drain_conversation_saves() is True
        assert store.rows == []

    async def test_a_turn_redis_could_not_settle_holds_until_its_lease(
        self, pausable_redis, caplog
    ):
        """The rows landed but Redis missed the settle: the loss of the settle
        is logged with the turn's position, and the context's next turn is held
        only until the turn's lease lapses."""
        redis = await connect_shared_state_redis(
            pausable_redis["url"], timeout_seconds=PAUSED_TIMEOUT_S
        )
        write = threading.Event()

        class _GatedStore(_Store):
            def store_turn(self, context_id, role, content, seq):
                assert write.wait(30), "the write was never released"
                super().store_turn(context_id, role, content, seq)

        store = _GatedStore()
        ledger = _ledger(redis, _prefix("conversation"), lease_s=3.0)
        dispatcher = _dispatcher(ledger, store, {"q": "a"}, [])
        try:
            await _dispatch(dispatcher, "q")
            [position] = await ledger.pending(TENANT, "ctx")
            subprocess.run(
                ["docker", "pause", pausable_redis["container"]],
                check=True,
                capture_output=True,
            )
            try:
                with caplog.at_level(
                    logging.ERROR, logger="cogniverse_runtime.agent_dispatcher"
                ):
                    write.set()
                    assert await dispatcher.drain_conversation_saves() is True
            finally:
                subprocess.run(
                    ["docker", "unpause", pausable_redis["container"]],
                    check=True,
                    capture_output=True,
                )

            assert store.get_history("ctx") == [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
            assert [
                record.getMessage()
                for record in caplog.records
                if record.name == "cogniverse_runtime.agent_dispatcher"
            ] == [
                f"Conversation turn at position {position} of context ctx could "
                "not be settled in the shared ledger; the context's next turn "
                "waits out its lease: SessionStateUnavailable('session state "
                "store unavailable: settle a turn of context ctx')"
            ]
            assert await ledger.pending(TENANT, "ctx") == [position]
            await asyncio.sleep(3.1)
            assert await ledger.pending(TENANT, "ctx") == []
        finally:
            await redis.aclose()

    async def test_a_turn_still_saving_elsewhere_is_reported_incomplete(
        self, session_redis, monkeypatch
    ):
        """Another process accepted a turn and has not settled it; past the
        wait budget the next turn reads without it and says so."""
        from cogniverse_runtime import agent_dispatcher

        monkeypatch.setattr(agent_dispatcher, "CONVERSATION_SAVE_TIMEOUT_S", 1.0)
        prefix = _prefix("conversation")
        elsewhere = _ledger(session_redis, prefix)
        await elsewhere.accept(TENANT, "ctx")
        store, calls = _Store(), []
        dispatcher = _dispatcher(
            _ledger(session_redis, prefix), store, {"q": "a"}, calls
        )

        started = time.monotonic()
        result = await _dispatch(dispatcher, "q")
        elapsed = time.monotonic() - started

        assert result["conversation"] == {
            "state": CONVERSATION_HISTORY_INCOMPLETE,
            "turn_count": 0,
            "reason": "1 earlier turn(s) still saving after 1s",
        }
        assert 1.0 <= elapsed < 1.0 + 0.5
        assert await dispatcher.drain_conversation_saves() is True

    async def test_turns_read_back_in_accept_order_whichever_lands_first(
        self, session_redis
    ):
        """Two dispatchers share one ledger as two processes would, and answer
        one context concurrently. The first-accepted turn's save is held until
        the second turn's has landed; the context still reads the first turn
        first."""
        prefix = _prefix("conversation")
        loop = asyncio.get_running_loop()
        hold_first_save = asyncio.Event()
        landed: list = []
        store = _Store()

        def store_turn(context_id, role, content, seq, _base=store.store_turn):
            if content in ("one", "first answer"):
                asyncio.run_coroutine_threadsafe(hold_first_save.wait(), loop).result(
                    30
                )
            _base(context_id, role, content, seq)
            landed.append(content)

        store.store_turn = store_turn
        at_agent = {"one": asyncio.Event(), "two": asyncio.Event()}
        answer = {"one": asyncio.Event(), "two": asyncio.Event()}

        def gated(query, reply):
            async def run():
                at_agent[query].set()
                await answer[query].wait()
                return reply

            return run

        first_calls, second_calls = [], []
        first = _dispatcher(
            _ledger(session_redis, prefix),
            store,
            {"one": gated("one", "first answer")},
            first_calls,
        )
        second = _dispatcher(
            _ledger(session_redis, prefix),
            store,
            {"two": gated("two", "second answer")},
            second_calls,
        )
        turns = [
            asyncio.create_task(_dispatch(first, "one")),
            asyncio.create_task(_dispatch(second, "two")),
        ]
        await _until_the_agent_runs(turns[0], at_agent["one"])
        await _until_the_agent_runs(turns[1], at_agent["two"])
        answer["one"].set()
        await turns[0]
        answer["two"].set()
        await turns[1]
        positions = await _ledger(session_redis, prefix).pending(TENANT, "ctx")
        assert len(positions) == 2
        assert (
            await _ledger(session_redis, prefix).wait_settled(
                TENANT, "ctx", positions[1:], 10.0
            )
            == []
        )
        hold_first_save.set()
        assert await first.drain_conversation_saves() is True
        assert await second.drain_conversation_saves() is True

        assert first_calls == [{"query": "one", "history": []}]
        assert second_calls == [{"query": "two", "history": []}]
        assert landed == ["two", "second answer", "one", "first answer"]
        assert store.get_history("ctx") == [
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "two"},
            {"role": "assistant", "content": "second answer"},
        ]


class TestRoutesReportTheOutage:
    async def test_the_process_route_answers_503_naming_the_store(
        self, dead_redis, monkeypatch, caplog
    ):
        caplog.set_level(logging.ERROR, logger="cogniverse_runtime.http_errors")
        dispatcher = _dispatcher(
            _ledger(dead_redis, _prefix("conversation")), _Store(), {"q": "a"}, []
        )
        monkeypatch.setattr(agents_router, "_dispatcher", dispatcher, raising=False)
        app = FastAPI()
        app.include_router(agents_router.router, prefix="/agents")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://runtime"
        ) as client:
            response = await client.post(
                "/agents/search_agent/process",
                json={
                    "agent_name": "search_agent",
                    "query": "q",
                    "context": {"tenant_id": TENANT},
                    "context_id": "ctx",
                },
            )

        assert response.status_code == 503
        assert response.json() == {
            "detail": {
                "error": "session_state_unavailable",
                "message": "Agent 'search_agent' could not complete: the session "
                "state store did not answer; retry.",
                "failure": "SessionStateUnavailable",
                "agent": "search_agent",
                "context_id": "ctx",
                "request_id": "ctx",
            }
        }
        assert [
            record.getMessage()
            for record in caplog.records
            if record.name == "cogniverse_runtime.http_errors"
        ] == [
            "session_state_unavailable: SessionStateUnavailable: session state "
            "store unavailable: read the pending turns of context ctx"
        ]
        await dead_redis.aclose()

    async def test_v1_suspension_answers_503_when_the_store_is_down(self, dead_redis):
        from cogniverse_core.common.agent_models import AgentEndpoint
        from cogniverse_core.registries.agent_registry import AgentRegistry
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_runtime.config_loader import ConfigLoader
        from tests.runtime.integration.test_openai_compat_endpoint import (
            _AGENT_CLASSES,
            KEY_A,
            MODEL_MAP,
            TENANT_A_RAW,
            TOOL_DEFS,
            _auth,
            _body,
        )
        from tests.utils.memory_store import InMemoryConfigStore

        config_store = InMemoryConfigStore()
        config_store.initialize()
        config_manager = ConfigManager(store=config_store)
        registry = AgentRegistry(tenant_id="acme:acme", config_manager=config_manager)
        registry.register_agent(
            AgentEndpoint(
                name="tool_echo_agent",
                url="http://localhost:8000",
                capabilities=["tool_echo"],
            )
        )
        ConfigLoader.AGENT_CLASSES.update(_AGENT_CLASSES)
        dispatcher = AgentDispatcher(
            agent_registry=registry, config_manager=config_manager, schema_loader=None
        )
        openai_compat.set_dispatcher_provider(lambda: dispatcher)
        openai_compat.set_api_keys({KEY_A: TENANT_A_RAW})
        openai_compat.set_model_map(MODEL_MAP)
        openai_compat.set_key_resolver(None)
        openai_compat.set_continuation_store(
            ContinuationStore(dead_redis, key_prefix=_prefix("continuation"))
        )
        app = FastAPI()
        app.include_router(openai_compat.router, prefix="/v1")
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://runtime"
            ) as client:
                plain = await client.post(
                    "/v1/chat/completions",
                    json=_body(model="cogniverse/tools", tools=TOOL_DEFS),
                    headers=_auth(KEY_A),
                )
                streamed = await client.post(
                    "/v1/chat/completions",
                    json=_body(model="cogniverse/tools", tools=TOOL_DEFS, stream=True),
                    headers=_auth(KEY_A),
                )
        finally:
            openai_compat.set_dispatcher_provider(None)
            openai_compat.set_api_keys({})
            openai_compat.set_model_map({})
            openai_compat.set_continuation_store(None)
            for agent_name in _AGENT_CLASSES:
                ConfigLoader.AGENT_CLASSES.pop(agent_name, None)
            await dead_redis.aclose()

        assert plain.status_code == 503
        assert plain.json() == {
            "error": {
                "message": (
                    "The session state store is unavailable "
                    "(SessionStateUnavailable). See server logs for detail."
                ),
                "type": "server_error",
                "code": "service_unavailable",
                "error_type": "SessionStateUnavailable",
            }
        }
        frames = [
            line[len("data: ") :]
            for line in streamed.text.splitlines()
            if line.startswith("data: ")
        ]
        assert streamed.status_code == 200
        assert frames[-1] == "[DONE]"
        assert json.loads(frames[-2]) == {
            "error": {
                "message": (
                    "tool_echo_agent failed with SessionStateUnavailable. "
                    "See server logs for detail."
                ),
                "agent": "tool_echo_agent",
                "error_type": "SessionStateUnavailable",
                "type": "server_error",
                "code": "service_unavailable",
            }
        }
