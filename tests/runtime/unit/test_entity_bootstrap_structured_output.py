"""The entity bootstrap survives a teacher that answers `{}` without a schema.

A real chat-completions server stands in for the teacher: it returns a bare
`{}` — valid JSON carrying no output field — unless the request constrains the
response with a `json_schema`. The bootstrap walk runs against it through the
production optimizer functions, so the schema either reaches the wire or the
walk records the error the run that motivated this saw.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import dspy
import pytest

from cogniverse_agents.entity_extraction_agent import (
    EntityExtractionModule,
    EntityMention,
)
from cogniverse_foundation.config.llm_factory import create_budgeted_dspy_lm
from cogniverse_foundation.config.unified_config import LLMEndpointConfig
from cogniverse_foundation.dspy import signature_response_format
from cogniverse_runtime.optimization_cli import (
    BootstrapMetricRecorder,
    _bootstrap_report,
    _create_teleprompter,
    _entity_bootstrap_threshold,
    _entity_extraction_example,
    _entity_extraction_quality,
    bootstrap_error_log,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TEACHER_MODEL = "Qwen/Qwen3-14B-AWQ"
QUERY = "What are the people doing behind the car at the beginning of the video?"
RECORDED_ENTITIES = [
    {"text": "people", "type": "PERSON"},
    {"text": "car", "type": "CONCEPT"},
]
SCHEMA_HONOURING_CONTENT = json.dumps(
    {
        "reasoning": "people appears first and is a PERSON; car follows and is a CONCEPT.",
        "entities": [
            {"text": "people", "type": "PERSON"},
            {"text": "car", "type": "CONCEPT"},
        ],
    }
)
UNCONSTRAINED_CONTENT = "{}"


class _TeacherServer:
    """Chat-completions server that only answers usefully under a json_schema."""

    def __init__(self) -> None:
        self.chat_requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler contract
                self._send(
                    {
                        "object": "list",
                        "data": [{"id": TEACHER_MODEL, "max_model_len": 4096}],
                    }
                )

            def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler contract
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length) or b"{}")
                outer.chat_requests.append(request)
                response_format = request.get("response_format") or {}
                content = (
                    SCHEMA_HONOURING_CONTENT
                    if response_format.get("type") == "json_schema"
                    else UNCONSTRAINED_CONTENT
                )
                self._send(
                    {
                        "id": "chatcmpl-stub",
                        "object": "chat.completion",
                        "created": 0,
                        "model": TEACHER_MODEL,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": content},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                )

            def log_message(self, *args):
                return

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> "_TeacherServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    @property
    def api_base(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"


@pytest.fixture
def teacher_server():
    with _TeacherServer() as server:
        yield server


@pytest.fixture(autouse=True)
def _no_dspy_cache():
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)


def _teacher_lm(api_base: str):
    return create_budgeted_dspy_lm(
        LLMEndpointConfig(
            model=f"openai/{TEACHER_MODEL}",
            api_base=api_base,
            api_key="stub-key",
            temperature=0.7,
            max_tokens=2048,
            context_window=4096,
            request_timeout=30.0,
            num_retries=0,
        )
    )


def _run_bootstrap(module: EntityExtractionModule, api_base: str) -> dict:
    """The compile step of run_entity_extraction_optimization, same functions."""
    lm = _teacher_lm(api_base)
    trainset = [
        _entity_extraction_example({"query": QUERY, "entities": RECORDED_ENTITIES})
    ]
    recorder = BootstrapMetricRecorder(
        _entity_extraction_quality,
        tenant="test:unit",
        threshold=_entity_bootstrap_threshold(0.0, None),
    )
    teleprompter = _create_teleprompter(
        len(trainset),
        teacher_settings={"lm": lm},
        metric=recorder,
        metric_threshold=recorder.threshold,
    )
    with dspy.context(lm=lm), bootstrap_error_log() as error_log:
        compiled = teleprompter.compile(module, trainset=trainset)
    return _bootstrap_report(
        recorder,
        teleprompter,
        compiled,
        len(trainset),
        error_causes=error_log.causes,
    )


class TestEntityBootstrapAgainstAnEmptyObjectTeacher:
    def test_bootstrap_sends_the_schema_and_accepts_the_example(self, teacher_server):
        module = EntityExtractionModule()

        report = _run_bootstrap(module, teacher_server.api_base)

        assert report["errors"] == 0
        assert report["error_causes"] == []
        assert report["attempts"] == 1
        assert report["accepted"] == 1
        assert report["metric_values"] == [1.0]
        assert report["bootstrapped_demos"] == 1
        assert report["examples_walked"] == 1

        assert len(teacher_server.chat_requests) == 1
        assert teacher_server.chat_requests[0][
            "response_format"
        ] == signature_response_format(module.extractor.predict.signature)

    def test_without_the_schema_the_same_teacher_costs_the_example(
        self, teacher_server
    ):
        """Control: the stock adapter this replaced, on the same server."""
        module = EntityExtractionModule()
        module.dspy_adapter = dspy.JSONAdapter()

        report = _run_bootstrap(module, teacher_server.api_base)

        assert teacher_server.chat_requests[0]["response_format"] == {
            "type": "json_object"
        }
        assert report["errors"] == 1
        assert report["bootstrapped_demos"] == 0
        assert report["attempts"] == 0
        assert len(report["error_causes"]) == 1
        cause = report["error_causes"][0]
        assert "Adapter JSONAdapter failed to parse the LM response" in cause
        assert "LM Response: {}" in cause
        assert QUERY in cause

    def test_the_bootstrapped_demo_carries_the_teacher_answer(self, teacher_server):
        module = EntityExtractionModule()
        lm = _teacher_lm(teacher_server.api_base)
        trainset = [
            _entity_extraction_example({"query": QUERY, "entities": RECORDED_ENTITIES})
        ]
        recorder = BootstrapMetricRecorder(
            _entity_extraction_quality,
            tenant="test:unit",
            threshold=_entity_bootstrap_threshold(0.0, None),
        )
        teleprompter = _create_teleprompter(
            len(trainset),
            teacher_settings={"lm": lm},
            metric=recorder,
            metric_threshold=recorder.threshold,
        )
        with dspy.context(lm=lm):
            compiled = teleprompter.compile(module, trainset=trainset)

        demos = [
            demo
            for _, predictor in compiled.named_predictors()
            for demo in predictor.demos
            if demo.get("augmented", False)
        ]
        assert len(demos) == 1
        assert demos[0]["query"] == QUERY
        assert demos[0]["entities"] == [
            EntityMention(text="people", type="PERSON"),
            EntityMention(text="car", type="CONCEPT"),
        ]
        persisted = json.loads(json.dumps(compiled.dump_state(), default=str))
        assert [
            demo["entities"]
            for demo in persisted["extractor.predict"]["demos"]
            if demo.get("augmented", False)
        ] == [RECORDED_ENTITIES]
        assert demos[0]["reasoning"] == (
            "people appears first and is a PERSON; car follows and is a CONCEPT."
        )
