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


def _memory_config_manager():
    """The ConfigManager the runtime binds into this agent, over an in-memory store."""
    from cogniverse_foundation.config.manager import ConfigManager
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)


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
        self.bind_config_manager(_memory_config_manager())
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


class _AnswerInput(AgentInput):
    query: str = ""


class _AnswerOutput(AgentOutput):
    answer: str = ""


class _Module:
    """Stands in for a compiled DSPy program on the non-streaming path only;
    the streaming path never calls it because streamify is replaced."""

    def __init__(self, answer: str):
        self._answer = answer

    def __call__(self, **kwargs):
        import dspy

        return dspy.Prediction(answer=self._answer)


def _streamify_yielding(chunks):
    """A streamify replacement that yields exactly what dspy's yields: real
    StreamResponse / StatusMessage objects, then the final Prediction."""

    def fake_streamify(program, **_):
        async def run(**kwargs):
            for item in chunks:
                yield item

        return run

    return fake_streamify


class _StreamingAnswerAgent(AgentBase[_AnswerInput, _AnswerOutput, _Deps]):
    def __init__(self, deps, module, output_field="answer"):
        super().__init__(deps=deps)
        self.bind_config_manager(_memory_config_manager())
        self._module = module
        self._output_field = output_field

    async def _process_impl(self, input: _AnswerInput) -> _AnswerOutput:
        prediction = await self.call_dspy(
            self._module, output_field=self._output_field, query=input.query
        )
        return _AnswerOutput(answer=getattr(prediction, self._output_field))


class _OuterAgent(AgentBase[_AnswerInput, _AnswerOutput, _Deps]):
    """Streams its own answer and, mid-turn, awaits an inner agent's
    non-streamed process() — the orchestrator → sub-agent shape."""

    def __init__(self, deps, module, inner):
        super().__init__(deps=deps)
        self.bind_config_manager(_memory_config_manager())
        self._module = module
        self._inner = inner

    async def _process_impl(self, input: _AnswerInput) -> _AnswerOutput:
        inner_out = await self._inner.process(_AnswerInput(query="inner"))
        prediction = await self.call_dspy(
            self._module, output_field="answer", query=input.query
        )
        return _AnswerOutput(answer=f"{prediction.answer}|{inner_out.answer}")


def _token_messages(events) -> list:
    return [e["message"] for e in events if e.get("phase") == "token"]


@pytest.mark.asyncio
@pytest.mark.parametrize("output_field", ["answer", "questions"])
async def test_token_events_carry_the_chunk_text(monkeypatch, output_field):
    import dspy
    from dspy.streaming import StatusMessage, StreamResponse

    monkeypatch.setattr(
        dspy,
        "streamify",
        _streamify_yielding(
            [
                StreamResponse("p", output_field, "Hel", False),
                StatusMessage("thinking"),
                StreamResponse("p", output_field, "lo", True),
                dspy.Prediction(**{output_field: "Hello"}),
            ]
        ),
    )
    agent = _StreamingAnswerAgent(_Deps(), _Module("unused"), output_field=output_field)

    events = await _drain_answer(agent)

    assert _token_messages(events) == ["Hel", "lo"]
    assert [e["data"]["accumulated"] for e in events if e.get("phase") == "token"] == [
        "Hel",
        "Hello",
    ]
    assert [e["data"] for e in events if e.get("phase") == "token"] == [
        {"accumulated": "Hel", "output_field": output_field},
        {"accumulated": "Hello", "output_field": output_field},
    ]
    finals = [e for e in events if e.get("type") == "final"]
    assert [f["data"]["answer"] for f in finals] == ["Hello"]


@pytest.mark.asyncio
async def test_nested_agent_does_not_stream_onto_the_outer_stream(monkeypatch):
    import dspy
    from dspy.streaming import StreamResponse

    monkeypatch.setattr(
        dspy,
        "streamify",
        _streamify_yielding(
            [
                StreamResponse("p", "answer", "OUTER", True),
                dspy.Prediction(answer="OUTER"),
            ]
        ),
    )
    inner = _StreamingAnswerAgent(_Deps(), _Module("inner-full"))
    outer = _OuterAgent(_Deps(), _Module("unused"), inner)

    events = await _drain_answer(outer)

    assert _token_messages(events) == ["OUTER"]
    finals = [e for e in events if e.get("type") == "final"]
    assert [f["data"]["answer"] for f in finals] == ["OUTER|inner-full"]


async def _drain_answer(agent) -> list:
    events = []
    stream = await agent.process(_AnswerInput(query="q"), stream=True)
    async for event in stream:
        events.append(event)
    return events
