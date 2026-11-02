"""Gateway classifier examples read from a real Phoenix project.

Routing spans are logged in the shape the gateway records, reviews are stored
as ``routing_annotation``s the way AnnotationStorage stores them, and
``load_gateway_label_examples`` must return exactly the examples those
reviews settle. A Phoenix answering 504 must raise, never read as no data.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from uuid import uuid4

import httpx
import pytest

from cogniverse_finetuning.dataset.gateway_labels import (
    QueryLabelExample,
    load_gateway_label_examples,
)
from cogniverse_foundation.telemetry.config import SPAN_NAME_ROUTING
from cogniverse_telemetry_phoenix.provider import PhoenixProvider

pytestmark = [pytest.mark.integration]

ANNOTATION = "routing_annotation"


def _provider(http_endpoint: str, grpc_endpoint: str) -> PhoenixProvider:
    provider = PhoenixProvider()
    provider.initialize(
        {
            "tenant_id": "gateway-labels-tests",
            "http_endpoint": http_endpoint,
            "grpc_endpoint": grpc_endpoint,
        }
    )
    return provider


def _routing_span(query: str, decision: dict, start: datetime) -> tuple[dict, str]:
    span_id = uuid4().hex[:16]
    return (
        {
            "name": SPAN_NAME_ROUTING,
            "context": {"trace_id": uuid4().hex, "span_id": span_id},
            "span_kind": "CHAIN",
            "start_time": start.isoformat(),
            "end_time": (start + timedelta(milliseconds=3)).isoformat(),
            "status_code": "OK",
            "attributes": {
                "input.value": query,
                "output.value": json.dumps(decision),
                "operation": "routing",
            },
        },
        span_id,
    )


def _decision(complexity: str, modality: str, generation_type: str, agent: str):
    return {
        "chosen_agent": agent,
        "recommended_agent": agent,
        "confidence": 0.5,
        "reasoning": "",
        "complexity": complexity,
        "modality": modality,
        "generation_type": generation_type,
        "fast_path_confidence_threshold": 0.4,
        "gliner_threshold": 0.3,
        "entity_extraction_failed": False,
        "shadow_classification": None,
    }


@pytest.mark.asyncio
async def test_reviewed_spans_in_phoenix_become_exact_examples(phoenix_container):
    from phoenix.client import AsyncClient as PhoenixAsyncClient

    http = phoenix_container["http_endpoint"]
    provider = _provider(http, phoenix_container["grpc_endpoint"])
    project = f"gateway-labels-{uuid4().hex}"
    start = datetime.now(timezone.utc) - timedelta(minutes=2)
    rows = [
        _routing_span(
            "show me cat videos",
            _decision("simple", "video", "raw_results", "search_agent"),
            start,
        ),
        _routing_span(
            "compare the launches",
            _decision("complex", "video", "raw_results", "orchestrator_agent"),
            start + timedelta(seconds=1),
        ),
        _routing_span(
            "find dog photos",
            _decision("simple", "image", "raw_results", "image_search_agent"),
            start + timedelta(seconds=2),
        ),
    ]
    async with httpx.AsyncClient(base_url=http, timeout=120) as http_client:
        client = PhoenixAsyncClient(base_url=http, http_client=http_client)
        await client.spans.log_spans(
            project_identifier=project, spans=[span for span, _ in rows], timeout=120
        )
    deadline = time.monotonic() + 60
    while len(await provider.traces.get_all_spans(project=project)) != len(rows):
        if time.monotonic() > deadline:
            pytest.fail(f"Phoenix never exposed {len(rows)} spans in {project}")
        await asyncio.sleep(0.25)

    (_, cat), (_, compare), (_, dog) = rows
    await provider.annotations.add_annotation(
        span_id=cat,
        name=ANNOTATION,
        label="correct",
        score=1.0,
        metadata={},
        project=project,
    )
    await provider.annotations.add_annotation(
        span_id=compare,
        name=ANNOTATION,
        label="wrong",
        score=0.0,
        metadata={"suggested_agent": "summarizer_agent"},
        project=project,
    )
    await provider.annotations.add_annotation(
        span_id=dog,
        name=ANNOTATION,
        label="ambiguous",
        score=0.5,
        metadata={},
        project=project,
    )

    expected = [
        QueryLabelExample(
            cat,
            "show me cat videos",
            {
                "complexity": "simple",
                "modality": "video",
                "generation_type": "raw_results",
            },
        ),
        QueryLabelExample(
            compare,
            "compare the launches",
            {"complexity": "simple", "generation_type": "summary"},
        ),
    ]
    deadline = time.monotonic() + 60
    while True:
        examples = await load_gateway_label_examples(
            provider,
            project,
            ANNOTATION,
            start - timedelta(seconds=1),
            datetime.now(timezone.utc),
        )
        if sorted(examples, key=lambda e: e.text) == sorted(
            expected, key=lambda e: e.text
        ):
            break
        if time.monotonic() > deadline:
            assert sorted(examples, key=lambda e: e.text) == sorted(
                expected, key=lambda e: e.text
            )
        await asyncio.sleep(0.25)


class _GatewayTimeout(BaseHTTPRequestHandler):
    def _timeout(self) -> None:
        body = b'{"detail":"upstream Phoenix timed out"}'
        self.send_response(504)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _timeout
    do_POST = _timeout

    def log_message(self, format: str, *args) -> None:
        return


@contextmanager
def _timeout_endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GatewayTimeout)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_phoenix_timeout_raises_instead_of_an_empty_training_set():
    project = "gateway-labels-timeout"
    now = datetime.now(timezone.utc)
    with _timeout_endpoint() as endpoint:
        provider = _provider(endpoint, "http://127.0.0.1:1")
        with pytest.raises(RuntimeError) as raised:
            await load_gateway_label_examples(
                provider, project, ANNOTATION, now - timedelta(hours=1), now
            )

    assert str(raised.value) == (
        f"Failed to query every span from Phoenix project {project}"
    )
    cause = raised.value.__cause__
    assert isinstance(cause, httpx.HTTPStatusError)
    assert cause.response.status_code == 504
