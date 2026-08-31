"""HTTP-level coverage for the POST /{agent_name}/process success body.

The process route's 2xx path — merging the task fields into a dispatch
context, seeding a request_id, calling the dispatcher, and returning its
result verbatim — was only ever run behind a skip-gated e2e. A direct
coroutine call bypasses request-body binding and JSON serialization. These
drive the mounted FastAPI app via ASGITransport with a faithful dispatcher
double (async ``dispatch`` returning the dict a real dispatch returns), so
the exact wire body and the args the route forwards are both asserted.
"""

from __future__ import annotations

import pytest
from cogniverse_messaging.runtime_client import RuntimeClient
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from cogniverse_runtime.routers import agents as agents_router

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]


class _StubDispatcher:
    """Faithful double for AgentDispatcher.dispatch: an async method taking
    agent_name/query/context/top_k and returning the result dict the route
    passes through verbatim."""

    def __init__(self, result):
        self._result = result
        self.received = None

    async def dispatch(self, agent_name, query, context, top_k):
        self.received = {
            "agent_name": agent_name,
            "query": query,
            "context": dict(context),
            "top_k": top_k,
        }
        return dict(self._result)


class _TracingDispatcher:
    """Dispatcher double that emits a real exported span for the agent."""

    def __init__(self, tracer):
        self._tracer = tracer
        self.received = None

    async def dispatch(self, agent_name, query, context, top_k):
        self.received = {
            "agent_name": agent_name,
            "query": query,
            "context": dict(context),
            "top_k": top_k,
        }
        with self._tracer.start_as_current_span(
            "EntityExtractionAgent.process"
        ) as span:
            span.set_attribute("tenant.id", context["tenant_id"])
            span.set_attribute("agent_name", agent_name)
            return {
                "status": "ok",
                "agent": agent_name,
                "query": query,
                "top_k": top_k,
            }


def _build_app(result, monkeypatch):
    stub = _StubDispatcher(result)
    # The route resolves the dispatcher via _ensure_dispatcher(), which returns
    # the module global when already set — inject the double there.
    monkeypatch.setattr(agents_router, "_dispatcher", stub, raising=False)
    app = FastAPI()
    app.include_router(agents_router.router, prefix="/agents")
    return app, stub


def _build_tracing_app(dispatcher, monkeypatch):
    monkeypatch.setattr(agents_router, "_dispatcher", dispatcher, raising=False)
    app = FastAPI()
    app.include_router(agents_router.router, prefix="/agents")
    return app


def _trace_capture():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("trace-context-propagation-test")
    return exporter, tracer


async def _post(app, agent_name, body, headers=None):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        return await client.post(
            f"/agents/{agent_name}/process", json=body, headers=headers
        )


async def _dispatch_with_runtime_client(app, agent_name, query, tenant_id):
    client = RuntimeClient("http://runtime")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://runtime"
    ) as http_client:
        client._client = http_client
        return await client.dispatch_agent(
            agent_name=agent_name,
            query=query,
            tenant_id=tenant_id,
            context={"source": "trace-test"},
        )


@pytest.mark.asyncio
async def test_process_returns_dispatcher_result_verbatim(monkeypatch):
    result = {
        "agent": "video_search",
        "status": "success",
        "results": [
            {"video_id": "v1", "score": 0.91, "title": "Robot arm demo"},
        ],
        "metadata": {"backend": "vespa", "top_k": 5},
    }
    app, stub = _build_app(result, monkeypatch)

    resp = await _post(
        app,
        "video_search",
        {
            "agent_name": "video_search",
            "query": "robot arm",
            "context": {"tenant_id": "acme:acme"},
            "top_k": 5,
            "session_id": "sess-42",
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == result

    # Path agent_name and body fields reach the dispatcher; a request_id is
    # seeded from session_id for canary bucketing.
    assert stub.received["agent_name"] == "video_search"
    assert stub.received["query"] == "robot arm"
    assert stub.received["top_k"] == 5
    assert stub.received["context"]["tenant_id"] == "acme:acme"
    assert stub.received["context"]["session_id"] == "sess-42"
    assert stub.received["context"]["request_id"] == "sess-42"


@pytest.mark.asyncio
async def test_process_seeds_request_id_when_no_session(monkeypatch):
    app, stub = _build_app({"status": "ok"}, monkeypatch)

    resp = await _post(
        app,
        "text_agent",
        {
            "agent_name": "text_agent",
            "query": "hi",
            "context": {"tenant_id": "acme:acme"},
        },
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "ok"}
    # top_k defaults to 10 when omitted; a fresh 32-hex request_id is minted.
    assert stub.received["top_k"] == 10
    req_id = stub.received["context"]["request_id"]
    assert len(req_id) == 32
    int(req_id, 16)


@pytest.mark.asyncio
async def test_process_missing_query_is_rejected(monkeypatch):
    app, _ = _build_app({"status": "ok"}, monkeypatch)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        resp = await client.post(
            "/agents/text_agent/process", json={"agent_name": "text_agent"}
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_runtime_client_and_route_share_trace_id(monkeypatch):
    exporter, tracer = _trace_capture()
    dispatcher = _TracingDispatcher(tracer)
    app = _build_tracing_app(dispatcher, monkeypatch)

    with tracer.start_as_current_span("cogniverse.orchestration") as caller_span:
        result = await _dispatch_with_runtime_client(
            app,
            "entity_extraction_agent",
            "extract entities",
            "acme:acme",
        )

    assert result == {
        "status": "ok",
        "agent": "entity_extraction_agent",
        "query": "extract entities",
        "top_k": 10,
    }
    assert dispatcher.received["agent_name"] == "entity_extraction_agent"
    assert dispatcher.received["query"] == "extract entities"
    assert dispatcher.received["top_k"] == 10
    assert dispatcher.received["context"]["source"] == "trace-test"
    assert dispatcher.received["context"]["tenant_id"] == "acme:acme"
    assert set(dispatcher.received["context"]) == {
        "request_id",
        "source",
        "tenant_id",
    }
    assert len(dispatcher.received["context"]["request_id"]) == 32
    int(dispatcher.received["context"]["request_id"], 16)

    spans = exporter.get_finished_spans()
    assert {span.name for span in spans} == {
        "cogniverse.orchestration",
        "EntityExtractionAgent.process",
    }
    caller_exported = next(span for span in spans if span.name == caller_span.name)
    agent_span = next(
        span for span in spans if span.name == "EntityExtractionAgent.process"
    )
    assert agent_span.context.trace_id == caller_exported.context.trace_id
    assert agent_span.parent is not None
    assert agent_span.parent.span_id == caller_exported.context.span_id


@pytest.mark.asyncio
async def test_runtime_client_without_traceparent_starts_root_trace(monkeypatch):
    exporter, tracer = _trace_capture()
    dispatcher = _TracingDispatcher(tracer)
    app = _build_tracing_app(dispatcher, monkeypatch)

    result = await _dispatch_with_runtime_client(
        app,
        "entity_extraction_agent",
        "extract entities",
        "acme:acme",
    )

    assert result == {
        "status": "ok",
        "agent": "entity_extraction_agent",
        "query": "extract entities",
        "top_k": 10,
    }
    spans = exporter.get_finished_spans()
    assert {span.name for span in spans} == {"EntityExtractionAgent.process"}
    agent_span = spans[0]
    assert agent_span.parent is None


@pytest.mark.asyncio
async def test_malformed_traceparent_does_not_crash_route(monkeypatch):
    exporter, tracer = _trace_capture()
    dispatcher = _TracingDispatcher(tracer)
    app = _build_tracing_app(dispatcher, monkeypatch)

    resp = await _post(
        app,
        "entity_extraction_agent",
        {
            "agent_name": "entity_extraction_agent",
            "query": "extract entities",
            "context": {"tenant_id": "acme:acme"},
        },
        headers={"traceparent": "00-bad-trace-id-bad-span-id-01"},
    )

    assert resp.status_code == 200, resp.text
    spans = exporter.get_finished_spans()
    assert {span.name for span in spans} == {"EntityExtractionAgent.process"}
    agent_span = spans[0]
    assert agent_span.parent is None
