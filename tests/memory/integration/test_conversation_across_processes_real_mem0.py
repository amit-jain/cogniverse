"""A server-managed conversation served by two processes, in real Mem0.

Each process runs the production dispatcher with the production conversation
ledger on a real Redis and the production ConversationStore on real Mem0
(Vespa + DenseOn); only the agent's answer is stubbed. Consecutive turns of
one context alternate between the processes, the way replicas and uvicorn
workers receive them, and every turn must read every turn answered before it
— including the one whose save is still landing on the other process — and the
context must read back complete and in order.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import uuid

import pytest

from cogniverse_core.conversation import ConversationStore
from cogniverse_runtime.agent_dispatcher import (
    CONVERSATION_HISTORY_LOADED,
    CONVERSATION_SAVE_TIMEOUT_S,
)
from tests.memory.integration.test_dispatcher_conversation_history_real_mem0 import (
    TENANT,
    _build_manager,
)

pytestmark = [pytest.mark.integration]

REPLIES = {
    "what is colpali": "a late-interaction model",
    "how many dims": "128",
    "and per patch": "one vector per patch",
    "which index": "an HNSW index",
}
QUERIES = list(REPLIES)
# What a turn answered by both processes at once takes before replying, so both
# have read their history before either is accepted.
TOGETHER_AGENT_S = 1.0


def _serve_turns(vespa_ports, denseon_url, redis_url, prefix, barrier, conn):
    """One runtime process: commands arrive on ``conn`` while the loop keeps
    landing saves in the background."""
    from unittest.mock import AsyncMock, MagicMock

    from cogniverse_runtime.agent_dispatcher import (
        CONVERSATION_PERSIST_FAILURE_CAPACITY,
        CONVERSATION_SAVE_LEASE_S,
        AgentDispatcher,
    )
    from cogniverse_runtime.session_state import ConversationLedger
    from cogniverse_runtime.shared_state import connect_shared_state_redis

    async def main():
        mm = _build_manager(shared_memory_vespa=vespa_ports, shared_denseon=denseon_url)
        redis = await connect_shared_state_redis(redis_url)
        dispatcher = AgentDispatcher(
            agent_registry=MagicMock(),
            config_manager=MagicMock(),
            schema_loader=MagicMock(),
            conversation_ledger=ConversationLedger(
                redis,
                save_lease_s=CONVERSATION_SAVE_LEASE_S,
                failure_capacity=CONVERSATION_PERSIST_FAILURE_CAPACITY,
                key_prefix=prefix,
            ),
        )
        dispatcher._conversation_store_factory = lambda tenant_id: ConversationStore(
            mm, tenant_id
        )
        endpoint = MagicMock()
        endpoint.capabilities = {"search"}
        dispatcher._registry.refresh = AsyncMock()
        dispatcher._registry.get_agent.return_value = endpoint

        async def skip_wiki(*_args, **_kwargs):
            return None

        seen = []
        delay = {"s": 0.0}

        async def answer(query, tenant_id, top_k, conversation_history=None, **kw):
            seen.append(list(conversation_history or []))
            await asyncio.sleep(delay["s"])
            return {"message": REPLIES.get(query, f"answer to {query}"), "entities": []}

        dispatcher._maybe_auto_file_wiki = skip_wiki
        dispatcher._execute_search_task = answer

        async def dispatch(query, context_id):
            result = await dispatcher.dispatch(
                agent_name="search_agent",
                query=query,
                context={"tenant_id": TENANT, "context_id": context_id},
            )
            return {
                "pid": os.getpid(),
                "answer": result["answer"],
                "conversation": result["conversation"],
                "seen": seen[-1],
            }

        conn.send(("ready", os.getpid()))
        try:
            while True:
                command, *args = await asyncio.to_thread(conn.recv)
                if command == "dispatch":
                    delay["s"] = 0.0
                    conn.send(await dispatch(*args))
                elif command == "together":
                    delay["s"] = TOGETHER_AGENT_S
                    await asyncio.to_thread(barrier.wait, 60)
                    conn.send(await dispatch(*args))
                elif command == "drain":
                    conn.send(await dispatcher.drain_conversation_saves())
                else:
                    return
        finally:
            await redis.aclose()

    asyncio.run(main())


class _Process:
    def __init__(self, process, conn):
        self.process = process
        self.conn = conn

    def call(self, *command, timeout=4 * CONVERSATION_SAVE_TIMEOUT_S):
        self.conn.send(command)
        assert self.conn.poll(timeout), f"{command[0]} got no answer in {timeout}s"
        return self.conn.recv()


@pytest.fixture
def two_processes(shared_memory_vespa, shared_denseon, workflow_state_redis_url):
    context = multiprocessing.get_context("spawn")
    prefix = f"test:conversation:{uuid.uuid4().hex}"
    barrier = context.Barrier(2)
    vespa_ports = {
        "http_port": shared_memory_vespa["http_port"],
        "config_port": shared_memory_vespa["config_port"],
    }
    started = []
    for _ in range(2):
        parent, child = context.Pipe()
        process = context.Process(
            target=_serve_turns,
            args=(
                vespa_ports,
                shared_denseon,
                workflow_state_redis_url,
                prefix,
                barrier,
                child,
            ),
        )
        process.start()
        started.append(_Process(process, parent))
    try:
        for served in started:
            assert served.conn.poll(300), "a process never became ready"
            assert served.conn.recv() == ("ready", served.process.pid)
        yield started
    finally:
        for served in started:
            if served.process.exitcode is None:
                served.conn.send(("stop",))
        for served in started:
            served.process.join(timeout=60)
            if served.process.exitcode is None:
                served.process.kill()


def test_alternating_processes_each_read_every_earlier_turn(
    two_processes, shared_memory_vespa, shared_denseon
):
    first, second = two_processes
    ctx = f"chat{uuid.uuid4().hex[:10]}"
    served = []
    for index, query in enumerate(QUERIES):
        process = (first, second)[index % 2]
        served.append(process.call("dispatch", query, ctx))

    expected_turns = []
    for index, query in enumerate(QUERIES):
        # Each turn read exactly the turns answered before it, whichever
        # process answered them and whether or not their saves had landed.
        assert served[index]["seen"] == expected_turns
        assert served[index]["conversation"] == {
            "state": CONVERSATION_HISTORY_LOADED,
            "turn_count": len(expected_turns),
            "reason": None,
        }
        assert served[index]["answer"] == REPLIES[query]
        expected_turns = expected_turns + [
            {"role": "user", "content": query},
            {"role": "assistant", "content": REPLIES[query]},
        ]
    assert [turn["pid"] for turn in served] == [
        first.process.pid,
        second.process.pid,
    ] * 2

    assert first.call("drain") is True
    assert second.call("drain") is True
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    assert ConversationStore(mm, TENANT).get_history(ctx) == expected_turns


def test_one_context_answered_by_both_processes_at_once_stores_whole_turns(
    two_processes, shared_memory_vespa, shared_denseon
):
    """Both processes read the context before either is accepted, so both
    answer from no history; both turns are stored, each user turn next to its
    own reply."""
    first, second = two_processes
    ctx = f"chat{uuid.uuid4().hex[:10]}"
    # Warm both processes' stores on another context so the timing below is
    # the dispatch, not a first Mem0 build.
    warm = f"warm{uuid.uuid4().hex[:10]}"
    first.call("dispatch", "warm first", warm)
    second.call("dispatch", "warm second", warm)

    first.conn.send(("together", "in first", ctx))
    second.conn.send(("together", "in second", ctx))
    answers = []
    for served in (first, second):
        assert served.conn.poll(4 * CONVERSATION_SAVE_TIMEOUT_S)
        answers.append(served.conn.recv())

    assert [answer["seen"] for answer in answers] == [[], []]
    assert first.call("drain") is True
    assert second.call("drain") is True
    mm = _build_manager(
        shared_memory_vespa=shared_memory_vespa, shared_denseon=shared_denseon
    )
    in_first = [
        {"role": "user", "content": "in first"},
        {"role": "assistant", "content": "answer to in first"},
    ]
    in_second = [
        {"role": "user", "content": "in second"},
        {"role": "assistant", "content": "answer to in second"},
    ]
    assert ConversationStore(mm, TENANT).get_history(ctx) in (
        in_first + in_second,
        in_second + in_first,
    )
