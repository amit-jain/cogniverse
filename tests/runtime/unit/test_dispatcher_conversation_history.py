"""Server-managed conversation history in the agent dispatcher.

The messaging gateway sends only a ``context_id`` (the chat id) and no
history; the runtime loads that context's recent turns before the agent
runs and appends the two new turns after — so multi-turn memory works in
the deployed chart without the gateway holding a Mem0 connection. A caller
that passes its own ``conversation_history`` (the A2A coding path) is
respected as-is and never double-managed.

These drive the REAL ``AgentDispatcher.dispatch`` with a stubbed agent
execution and a partition-faithful in-memory Mem0 double injected through
the store factory seam. History is enrichment: a Mem0 outage degrades to
no-history and still returns the agent's answer.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from cogniverse_core.conversation import ConversationStore
from cogniverse_runtime.agent_dispatcher import AgentDispatcher

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TENANT = "acme:acme"


class _PartitionedMemory:
    """In-memory Mem0 double keyed by (tenant_id, agent_name), preserving
    insertion order and metadata (including the seq the store writes)."""

    def __init__(self):
        self.store = {}
        self.fail_reads = False
        self.fail_writes = False

    def add_memory(self, content, tenant_id, agent_name, metadata=None, **kwargs):
        if self.fail_writes:
            raise ConnectionError("mem0 down")
        self.store.setdefault((tenant_id, agent_name), []).append(
            {"memory": content, "metadata": metadata or {}}
        )
        return "mem_1"

    def get_all_memories(self, tenant_id, agent_name):
        if self.fail_reads:
            raise ConnectionError("mem0 down")
        return list(self.store.get((tenant_id, agent_name), []))


def _dispatcher(memory):
    sys_cfg = MagicMock()
    config_manager = MagicMock()
    config_manager.get_system_config.return_value = sys_cfg

    d = AgentDispatcher(
        agent_registry=MagicMock(),
        config_manager=config_manager,
        schema_loader=MagicMock(),
    )
    d._conversation_store_factory = lambda tenant_id: ConversationStore(
        memory, tenant_id
    )
    # Route "search" capability to a stub that records the history it saw.
    agent = MagicMock()
    agent.capabilities = {"search"}
    d._registry.get_agent.return_value = agent
    d._spawn_background = lambda coro: coro.close()
    return d


def _stub_execute(dispatcher, reply="the answer"):
    seen = {}

    async def _fake(query, tenant_id, top_k, conversation_history=None, **kwargs):
        seen["history"] = conversation_history
        seen["query"] = query
        return {"message": reply, "entities": []}

    dispatcher._execute_search_task = _fake
    return seen


async def _dispatch(dispatcher, query, context_id="99", tenant=TENANT, **extra):
    context = {"tenant_id": tenant}
    if context_id is not None:
        context["context_id"] = context_id
    context.update(extra)
    return await dispatcher.dispatch(
        agent_name="search_agent", query=query, context=context
    )


@pytest.mark.asyncio
async def test_history_loads_empty_first_then_carries_prior_turns():
    memory = _PartitionedMemory()
    d = _dispatcher(memory)
    seen = _stub_execute(d, reply="A1")

    result = await _dispatch(d, "Q1")
    assert result["message"] == "A1"
    assert seen["history"] == []  # first turn, nothing prior

    _stub_execute(d, reply="A2")
    seen2 = _stub_execute(d, reply="A2")
    await _dispatch(d, "Q2")
    assert seen2["history"] == [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
    ]


@pytest.mark.asyncio
async def test_contexts_are_isolated():
    memory = _PartitionedMemory()
    d = _dispatcher(memory)
    _stub_execute(d, reply="A")
    await _dispatch(d, "in chat 1", context_id="1")

    seen = _stub_execute(d, reply="B")
    await _dispatch(d, "in chat 12", context_id="12")
    assert seen["history"] == []  # chat 12 must not see chat 1's turn


@pytest.mark.asyncio
async def test_explicit_history_is_respected_and_not_persisted():
    """A caller passing its own conversation_history bypasses server
    management entirely — the runtime neither overwrites nor saves it."""
    memory = _PartitionedMemory()
    d = _dispatcher(memory)
    seen = _stub_execute(d)

    supplied = [{"role": "user", "content": "prior"}]
    await _dispatch(d, "Q", conversation_history=supplied)

    assert seen["history"] == supplied
    assert memory.store == {}  # nothing managed, nothing stored


@pytest.mark.asyncio
async def test_no_context_id_means_no_history_management():
    memory = _PartitionedMemory()
    d = _dispatcher(memory)
    seen = _stub_execute(d)

    await _dispatch(d, "Q", context_id=None)

    assert seen["history"] == []
    assert memory.store == {}


@pytest.mark.asyncio
async def test_mem0_read_outage_degrades_to_no_history_but_still_answers():
    memory = _PartitionedMemory()
    memory.fail_reads = True
    d = _dispatcher(memory)
    seen = _stub_execute(d, reply="still answered")

    result = await _dispatch(d, "Q")

    assert result["message"] == "still answered"
    assert seen["history"] == []  # outage degraded, agent still ran


@pytest.mark.asyncio
async def test_mem0_write_outage_does_not_fail_the_reply():
    memory = _PartitionedMemory()
    memory.fail_writes = True
    d = _dispatcher(memory)
    _stub_execute(d, reply="answered")

    result = await _dispatch(d, "Q")

    assert result["message"] == "answered"  # save failure swallowed


@pytest.mark.asyncio
async def test_unconfigured_memory_disables_history():
    d = _dispatcher(_PartitionedMemory())
    d._conversation_store_factory = lambda tenant_id: None
    seen = _stub_execute(d, reply="ok")

    result = await _dispatch(d, "Q")

    assert result["message"] == "ok"
    assert seen["history"] == []


@pytest.mark.asyncio
async def test_assistant_turn_skipped_when_no_message():
    memory = _PartitionedMemory()
    d = _dispatcher(memory)

    async def _fake(query, tenant_id, top_k, conversation_history=None, **kwargs):
        return {"entities": []}  # no message field

    d._execute_search_task = _fake
    await _dispatch(d, "Q")

    turns = memory.store.get((TENANT, "_conversation"), [])
    roles = [t["metadata"]["role"] for t in turns]
    assert roles == ["user"]  # user turn stored, no empty assistant turn


@pytest.mark.asyncio
async def test_concurrent_dispatches_same_context_persist_all_turns():
    """Two concurrent messages for one chat both complete and both persist
    their user+assistant turns — no lost writes, no exception."""
    memory = _PartitionedMemory()
    d = _dispatcher(memory)

    async def _fake(query, tenant_id, top_k, conversation_history=None, **kwargs):
        await asyncio.sleep(0)  # yield so the two dispatches interleave
        return {"message": f"reply to {query}", "entities": []}

    d._execute_search_task = _fake

    await asyncio.gather(
        _dispatch(d, "first"),
        _dispatch(d, "second"),
    )

    turns = memory.store.get((TENANT, "_conversation"), [])
    contents = sorted(t["memory"] for t in turns)
    assert contents == [
        "[ctx:99] [assistant] reply to first",
        "[ctx:99] [assistant] reply to second",
        "[ctx:99] [user] first",
        "[ctx:99] [user] second",
    ]
