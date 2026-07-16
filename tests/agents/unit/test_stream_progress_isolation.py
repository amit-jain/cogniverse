"""Concurrent streams on a dispatcher-shared agent must not cross-talk.

The dispatcher caches one agent instance per (tenant/profile) and shares it
across requests. When _stream_with_progress kept its queue + sentinel on the
instance (self._progress_queue), a second concurrent stream overwrote the
first's queue: the first's events and its raw loop-local sentinel object landed
in the second's stream, and the first hung forever with no sentinel. A
per-invocation ContextVar isolates each stream.
"""

import asyncio

import pytest

from cogniverse_core.agents.base import AgentBase, AgentDeps, AgentInput, AgentOutput

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _Input(AgentInput):
    tag: str = ""


class _Output(AgentOutput):
    tag: str = ""


class _Deps(AgentDeps):
    pass


class _StreamingAgent(AgentBase[_Input, _Output, _Deps]):
    """Emits one progress event tagged with the request, gated on an event so
    two invocations can be forced to interleave on the shared instance."""

    def __init__(self, deps, gate: asyncio.Event):
        super().__init__(deps=deps)
        self._gate = gate

    async def _process_impl(self, input: _Input) -> _Output:
        self.emit_progress("phase", f"progress-{input.tag}")
        # Hold here so the second stream starts (and, pre-fix, overwrote the
        # shared queue) before this one finishes.
        await self._gate.wait()
        return _Output(tag=input.tag)


async def _drain(agent, tag: str) -> list:
    events = []
    stream = await agent.process(_Input(tag=tag), stream=True)
    async for event in stream:
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_concurrent_streams_do_not_cross_talk():
    gate = asyncio.Event()
    agent = _StreamingAgent(_Deps(), gate)

    task_a = asyncio.create_task(_drain(agent, "A"))
    await asyncio.sleep(0.02)  # let A emit + reach the gate
    task_b = asyncio.create_task(_drain(agent, "B"))
    await asyncio.sleep(0.02)  # let B start (pre-fix: overwrites A's queue)
    gate.set()

    events_a, events_b = await asyncio.wait_for(
        asyncio.gather(task_a, task_b), timeout=5.0
    )

    # Each stream sees ONLY its own progress event and its own final payload —
    # no foreign events, no leaked sentinel surfacing as an event.
    a_progress = [e for e in events_a if e.get("type") == "status"]
    b_progress = [e for e in events_b if e.get("type") == "status"]
    assert [e["message"] for e in a_progress] == ["progress-A"]
    assert [e["message"] for e in b_progress] == ["progress-B"]

    a_final = [e for e in events_a if e.get("type") == "final"]
    b_final = [e for e in events_b if e.get("type") == "final"]
    assert len(a_final) == 1 and a_final[0]["data"]["tag"] == "A"
    assert len(b_final) == 1 and b_final[0]["data"]["tag"] == "B"

    # No raw sentinel (a bare object(), not a dict) ever leaked into a stream.
    for event in events_a + events_b:
        assert isinstance(event, dict)
