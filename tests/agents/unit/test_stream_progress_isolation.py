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


class TestStreamedFieldText:
    """What a token event carries: the field value as the adapter parses it."""

    @pytest.mark.parametrize(
        "raw, decoded",
        [
            ("", ""),
            ('"', ""),
            ('"Ice', "Ice"),
            ('"Ice \\"floats\\"', 'Ice "floats"'),
            ('"one\\n\\ntwo', "one\n\ntwo"),
            ('"cut at a lone \\', "cut at a lone "),
            ('"partial \\u00', "partial "),
            ('"caf\\u00e9', "café"),
            ('"pair \\ud83d', "pair "),
            ('"pair \\ud83d\\ude00', "pair \U0001f600"),
            ('"done",\n  "key_points": "x', "done"),
            ('  "leading space kept inside"', "leading space kept inside"),
            ("[1, 2]", ""),
        ],
    )
    def test_json_field_prefix_decodes_exactly(self, raw, decoded):
        from cogniverse_core.agents.base import _json_string_prefix

        assert _json_string_prefix(raw) == decoded

    def test_every_json_prefix_begins_the_parsed_value(self):
        import json

        from cogniverse_core.agents.base import _json_string_prefix

        value = 'Say "less dense".\n\nThen café \U0001f600 and a \\ backslash.'
        raw = json.dumps(value) + ',\n  "key_points": "a, b"}'
        views = [_json_string_prefix(raw[:end]) for end in range(len(raw) + 1)]
        assert [view for view in views if not value.startswith(view)] == []
        assert views[-1] == value

    def test_adapter_selects_the_decoding(self):
        import dspy

        from cogniverse_core.agents.base import _field_text
        from cogniverse_foundation.dspy import LenientJSONAdapter

        raw = '\n"Ice \\"floats\\"",'
        assert _field_text(raw, LenientJSONAdapter()) == 'Ice "floats"'
        assert _field_text(raw, dspy.ChatAdapter()) == '"Ice \\"floats\\"",'
        assert _field_text("\n\nIce floats\n", None) == "Ice floats\n"


class TestRejectedStatus:
    """Only a status the LM actually answered with reaches the client."""

    def test_a_4xx_leaf_names_its_status(self):
        from cogniverse_core.agents.base import _rejected_status

        class Rejected(Exception):
            status_code = 413

        assert _rejected_status([ValueError("x"), Rejected("too large")]) == 413

    def test_a_synthesized_5xx_names_none(self):
        from cogniverse_core.agents.base import _rejected_status

        class Unreachable(Exception):
            status_code = 500

        assert _rejected_status([Unreachable("connection refused")]) is None


def test_concurrent_first_touches_build_one_lm_stream_loop(monkeypatch):
    import threading

    from cogniverse_core.agents import base

    def loop_threads() -> set:
        return {t for t in threading.enumerate() if t.name == "lm-stream-loop"}

    monkeypatch.setattr(base, "_LM_STREAM_LOOP", None)
    before = loop_threads()
    built: list = []
    real_new_loop = base.asyncio.new_event_loop

    def recording_new_loop():
        loop = real_new_loop()
        built.append(loop)
        return loop

    monkeypatch.setattr(base.asyncio, "new_event_loop", recording_new_loop)
    barrier = threading.Barrier(16)
    returned: list = []

    def first_touch() -> None:
        barrier.wait()
        returned.append(base._lm_stream_loop())

    threads = [threading.Thread(target=first_touch) for _ in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    started = loop_threads() - before

    for loop in built:
        loop.call_soon_threadsafe(loop.stop)
    assert len(returned) == 16
    assert len(built) == 1
    assert {id(loop) for loop in returned} == {id(built[0])}
    assert len(started) == 1


class _SseChatServer:
    """A real OpenAI-compatible endpoint that streams one fixed completion."""

    def __init__(self, content: str, piece: int = 3) -> None:
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.requests: list = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                return

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                pieces = [content[i : i + piece] for i in range(0, len(content), piece)]
                for index, text in enumerate(pieces + [""]):
                    chunk = {
                        "id": "chatcmpl-stub",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "stub-model",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text} if text else {},
                                "finish_reason": None if text else "stop",
                            }
                        ],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.api_base = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


class _AnswerOnlyOutput(AgentOutput):
    answer: str = ""


class _ChatStreamingAgent(AgentBase[_Input, _AnswerOnlyOutput, _Deps]):
    async def _process_impl(self, input: _Input) -> _AnswerOnlyOutput:
        import dspy

        prediction = await self.call_dspy(
            dspy.Predict("question -> answer"),
            output_field="answer",
            question=input.tag,
        )
        return _AnswerOnlyOutput(answer=prediction.answer)


async def test_chat_format_tokens_are_the_stripped_answer_over_a_real_stream():
    import uuid

    import dspy

    server = _SseChatServer(
        "[[ ## answer ## ]]\n\n  Ice floats because its lattice is open.\n\n"
        "[[ ## completed ## ]]\n"
    )
    agent = _ChatStreamingAgent(_Deps())
    agent.bind_config_manager(_memory_config_manager())
    lm = dspy.LM(
        "openai/stub-model", api_base=server.api_base, api_key="stub", cache=False
    )
    try:
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
            events = [
                event
                async for event in await agent.process(
                    _Input(tag=f"why does ice float {uuid.uuid4().hex}"), stream=True
                )
            ]
    finally:
        server.close()

    tokens = [event for event in events if event.get("phase") == "token"]
    assert "".join(event["message"] for event in tokens) == (
        "Ice floats because its lattice is open."
    )
    assert tokens[-1]["data"] == {
        "accumulated": "Ice floats because its lattice is open.",
        "output_field": "answer",
    }
    assert events[-1] == {
        "type": "final",
        "data": {"answer": "Ice floats because its lattice is open."},
    }
    assert [request.get("stream") for request in server.requests] == [True]
