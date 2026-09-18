"""The DSPy span lookup's diagnostics and the span names it queries."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import dspy
import pandas as pd
import phoenix.client as phoenix_client
import pytest
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import tests.e2e.test_inbound_dspy_span_e2e as e2e
from cogniverse_foundation.config.semantic_router import create_routed_lm
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
)
from cogniverse_foundation.dspy import LenientJSONAdapter


@dataclass
class _FakePhoenixSpans:
    chain_frame: pd.DataFrame
    lm_frame: pd.DataFrame
    chain_error: Exception | None = None
    lm_error: Exception | None = None
    chain_calls: int = 0
    lm_calls: int = 0

    def get_spans_dataframe(self, **kwargs):
        condition = getattr(getattr(kwargs["query"], "_filter", None), "condition", "")
        if "ChainOfThought.forward" in condition:
            self.chain_calls += 1
            if self.chain_error is not None:
                raise self.chain_error
            return self.chain_frame
        if e2e.LM_SPAN_NAME in condition:
            self.lm_calls += 1
            if self.lm_error is not None:
                raise self.lm_error
            return self.lm_frame
        if self.lm_error is not None:
            raise self.lm_error
        raise AssertionError(f"unexpected Phoenix query: {kwargs['query']!r}")


class _FakePhoenixClient:
    def __init__(self, base_url: str, spans: _FakePhoenixSpans):
        self.base_url = base_url
        self.spans = spans


def _frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _run_case(
    monkeypatch: pytest.MonkeyPatch,
    spans: _FakePhoenixSpans,
    text: str = "needle",
    timeout_s: float = 1.0,
) -> str:
    clock = {"now": 0.0}

    def fake_time() -> float:
        return clock["now"]

    def fake_sleep(seconds: float) -> None:
        clock["now"] += seconds

    def fake_client(base_url: str) -> _FakePhoenixClient:
        return _FakePhoenixClient(base_url, spans)

    monkeypatch.setattr(e2e.time, "time", fake_time)
    monkeypatch.setattr(e2e.time, "sleep", fake_sleep)
    monkeypatch.setattr(phoenix_client, "Client", fake_client)

    with pytest.raises(AssertionError) as excinfo:
        e2e._query_dspy_lm_spans_with_text(text, timeout_s=timeout_s)
    return str(excinfo.value)


def test_query_dspy_lm_spans_reports_three_distinct_failure_messages(
    monkeypatch: pytest.MonkeyPatch,
):
    messages = {
        "phoenix": _run_case(
            monkeypatch,
            _FakePhoenixSpans(
                chain_frame=_frame([]),
                lm_frame=_frame([]),
                chain_error=TimeoutError("phoenix busy"),
            ),
        ),
        "chain": _run_case(
            monkeypatch,
            _FakePhoenixSpans(
                chain_frame=_frame(
                    [
                        {
                            "attributes.input.value": "other query one",
                            "context.trace_id": "trace-a",
                            "context.span_id": "span-a",
                            "name": "ChainOfThought.forward",
                            "attributes.output.value": "{}",
                        },
                        {
                            "attributes.input.value": "other query two",
                            "context.trace_id": "trace-b",
                            "context.span_id": "span-b",
                            "name": "ChainOfThought.forward",
                            "attributes.output.value": "{}",
                        },
                    ]
                ),
                lm_frame=_frame([]),
            ),
        ),
        "lm": _run_case(
            monkeypatch,
            _FakePhoenixSpans(
                chain_frame=_frame(
                    [
                        {
                            "attributes.input.value": "needle in the first chain row",
                            "context.trace_id": "trace-c",
                            "context.span_id": "span-c",
                            "name": "ChainOfThought.forward",
                            "attributes.output.value": "{}",
                        },
                        {
                            "attributes.input.value": "another chain row",
                            "context.trace_id": "trace-d",
                            "context.span_id": "span-d",
                            "name": "ChainOfThought.forward",
                            "attributes.output.value": "{}",
                        },
                    ]
                ),
                lm_frame=_frame(
                    [
                        {
                            "attributes.input.value": "lm child prompt one",
                            "context.trace_id": "trace-c",
                            "context.span_id": "lm-span-1",
                            "name": e2e.LM_SPAN_NAME,
                            "attributes.output.value": "{}",
                        },
                        {
                            "attributes.input.value": "lm child prompt two",
                            "context.trace_id": "trace-c",
                            "context.span_id": "lm-span-2",
                            "name": e2e.LM_SPAN_NAME,
                            "attributes.output.value": "{}",
                        },
                    ]
                ),
            ),
        ),
    }

    assert len(set(messages.values())) == 3
    assert (
        "Phoenix error while querying ChainOfThought.forward spans"
        in messages["phoenix"]
    )
    assert "TimeoutError: phoenix busy" in messages["phoenix"]
    assert "chain_span_count=2" in messages["chain"]
    assert "matching_chain_count=0" in messages["chain"]
    assert "trace_ids=['trace-c']" in messages["lm"]
    assert "lm_child_count=2" in messages["lm"]
    assert "matching_lm_count=0" in messages["lm"]


_QUESTION = "what does the reformulator ask the served model"
_ANSWER = "it asks the routed model the reformulated question"
_REASONING = "the routed model answers the reformulated question"


class _ChatCompletionsServer:
    """Real OpenAI-compatible boundary the routed LM completes against."""

    def __init__(self, content: str):
        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                payload = json.dumps(
                    {
                        "id": "chatcmpl-span-name",
                        "object": "chat.completion",
                        "model": "served-model",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": content},
                            }
                        ],
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                return None

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def __enter__(self):
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def exported_spans():
    """Real OpenInference DSPy instrumentation writing to a real OTel exporter."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    DSPyInstrumentor().instrument(tracer_provider=provider)
    try:
        yield exporter
    finally:
        DSPyInstrumentor().uninstrument()


def test_the_routed_lm_emits_the_span_name_the_lookup_queries(exported_spans):
    """The served LM's own class names its span, so the lookup must follow it.

    ``DSPyInstrumentor`` names an LM span ``type(lm).__name__ + ".__call__"``;
    every LM a served agent runs under comes from
    ``semantic_router.routed_lm_context_for``, never from a bare ``dspy.LM``.
    """
    completion = json.dumps({"reasoning": _REASONING, "answer": _ANSWER})
    with _ChatCompletionsServer(completion) as server:
        router = SemanticRouterConfig(enabled=True, semantic_router_url=server.base_url)
        lm = create_routed_lm(
            LLMEndpointConfig(
                model="openai/served-model",
                api_base=server.base_url,
                api_key="test-key",
                num_retries=0,
                request_timeout=30.0,
            ),
            router,
            tenant_id="spanname_org:t1",
            tier="free",
            call_site="orchestrator_agent",
        )
        with dspy.context(lm=lm, adapter=LenientJSONAdapter()):
            prediction = dspy.ChainOfThought("question -> answer")(question=_QUESTION)

    assert prediction.answer == _ANSWER

    spans = exported_spans.get_finished_spans()
    lm_spans = [
        s for s in spans if s.attributes.get("openinference.span.kind") == "LLM"
    ]
    assert [s.name for s in lm_spans] == [f"{type(lm).__name__}.__call__"]
    assert [s.name for s in lm_spans] == [e2e.LM_SPAN_NAME]

    chain_spans = [
        s for s in spans if s.name == f"{dspy.ChainOfThought.__name__}.forward"
    ]
    assert len(chain_spans) == 1
    # The lookup keeps LM spans by the chain span's trace id, so they share one.
    assert lm_spans[0].context.trace_id == chain_spans[0].context.trace_id

    recorded_input = json.loads(lm_spans[0].attributes["input.value"])
    assert sorted(recorded_input) == ["kwargs", "messages", "prompt"]
    assert [m["role"] for m in recorded_input["messages"]] == ["system", "user"]
    assert _QUESTION in recorded_input["messages"][1]["content"]
    assert json.loads(lm_spans[0].attributes["output.value"]) == [completion]
