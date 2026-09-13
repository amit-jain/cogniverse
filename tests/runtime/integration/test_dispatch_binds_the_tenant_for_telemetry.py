"""A dispatched agent runs with the request's tenant bound for telemetry.

The tenant-routing tracer provider files an instrumented span under the
tenant in context and drops one emitted with no tenant. The dispatch route is
where every agent call enters, so it is where the tenant is bound: without
it, every DSPy-instrumented span of a dispatched agent - the LM call and the
served model it records - is dropped on the deployed cluster.
"""

from __future__ import annotations

import asyncio
from typing import Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from cogniverse_agents.summarizer_agent import (
    SummarizerAgent,
    SummaryResult,
    ThinkingPhase,
)
from cogniverse_core.registries.agent_registry import AgentEndpoint, AgentRegistry
from cogniverse_foundation.common.tenant_utils import canonical_tenant_id
from cogniverse_foundation.telemetry.tenant_context import (
    current_tenant_id,
    tenant_span_context,
)
from cogniverse_foundation.telemetry.tenant_routing import (
    TenantRoutingTracerProvider,
)
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.routers import agents

pytestmark = pytest.mark.integration

TENANT_A = "bind_a:production"
TENANT_B = "bind_b:production"


@pytest.fixture(scope="module")
def dispatcher(config_manager, schema_loader):
    registry = AgentRegistry(tenant_id=TENANT_A, config_manager=config_manager)
    registry.register_agent(
        AgentEndpoint(
            name="summarizer_agent",
            url="http://localhost:8004",
            capabilities=["summarization"],
            health_endpoint="/health",
        )
    )
    return AgentDispatcher(
        agent_registry=registry,
        config_manager=config_manager,
        schema_loader=schema_loader,
    )


@pytest.fixture
def route_client(dispatcher):
    agents._dispatcher = dispatcher
    app = FastAPI()
    app.include_router(agents.router, prefix="/agents")
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    agents._dispatcher = None


@pytest.fixture
def tenant_exporters():
    exporters: Dict[str, InMemorySpanExporter] = {}
    tracers: Dict[str, object] = {}

    def resolve(tenant_id: str):
        if tenant_id not in tracers:
            exporter = InMemorySpanExporter()
            provider = TracerProvider()
            provider.add_span_processor(SimpleSpanProcessor(exporter))
            exporters[tenant_id] = exporter
            tracers[tenant_id] = provider.get_tracer(f"cogniverse-{tenant_id}")
        return tracers[tenant_id]

    return TenantRoutingTracerProvider(resolve_tracer=resolve), exporters


@pytest.fixture
def recording_summarizer(monkeypatch, tenant_exporters):
    """The summarizer's body becomes: record the tenant in context, emit one
    instrumented-style span, answer. Dispatcher, router and agent stay real."""
    provider, _ = tenant_exporters
    tracer = provider.get_tracer("dspy")
    seen: list[tuple[str, str | None]] = []
    gate = asyncio.Event()

    async def _record(self, request):
        with tracer.start_as_current_span("RoutedLM.__call__"):
            seen.append((request.query, current_tenant_id()))
            await gate.wait()
            if request.query == "fail":
                raise ValueError("summary refused")
        return SummaryResult(
            summary="recorded",
            key_points=[],
            visual_insights=[],
            confidence_score=1.0,
            thinking_phase=ThinkingPhase(
                key_themes=[],
                content_categories=[],
                relevance_scores={},
                visual_elements=[],
                reasoning="recorded",
            ),
            metadata={},
        )

    monkeypatch.setattr(SummarizerAgent, "summarize", _record)
    return seen, gate


def _task(tenant: str, query: str) -> dict:
    return {
        "agent_name": "summarizer_agent",
        "query": query,
        "context": {
            "tenant_id": tenant,
            "search_results": [{"id": "recorded-source", "text": "recorded"}],
        },
    }


def test_the_agent_runs_with_the_requests_tenant_bound(
    route_client, recording_summarizer, tenant_exporters
):
    seen, gate = recording_summarizer
    gate.set()
    _, exporters = tenant_exporters
    response = route_client.post(
        "/agents/summarizer_agent/process", json=_task(TENANT_A, "q-a")
    )
    assert response.status_code == 200, response.text[:300]
    assert response.json()["result"] == {
        "summary": "recorded",
        "key_points": [],
        "visual_insights": [],
        "confidence_score": 1.0,
        "thinking_phase": {
            "key_themes": [],
            "content_categories": [],
            "relevance_scores": {},
            "visual_elements": [],
            "reasoning": "recorded",
            "entity_insights": None,
            "relationship_patterns": None,
            "contextual_connections": None,
        },
        "metadata": {},
        "relationship_summary": None,
        "entity_analysis": None,
        "enhancement_applied": False,
    }
    assert seen == [("q-a", canonical_tenant_id(TENANT_A))]
    assert {
        t: [s.name for s in e.get_finished_spans()] for t, e in exporters.items()
    } == {canonical_tenant_id(TENANT_A): ["RoutedLM.__call__"]}


async def test_concurrent_tenants_each_see_only_their_own(
    dispatcher, recording_summarizer, tenant_exporters
):
    """Two dispatches held open together: each agent body sees its own tenant
    and each span files under its own project."""
    seen, gate = recording_summarizer
    _, exporters = tenant_exporters
    first = asyncio.create_task(
        dispatcher.dispatch(
            "summarizer_agent", "q-a", _task(TENANT_A, "q-a")["context"]
        )
    )
    second = asyncio.create_task(
        dispatcher.dispatch(
            "summarizer_agent", "q-b", _task(TENANT_B, "q-b")["context"]
        )
    )
    while len(seen) < 2:
        await asyncio.sleep(0.01)
    gate.set()
    await asyncio.gather(first, second)
    assert sorted(seen) == [
        ("q-a", canonical_tenant_id(TENANT_A)),
        ("q-b", canonical_tenant_id(TENANT_B)),
    ]
    assert {
        t: [s.name for s in e.get_finished_spans()] for t, e in exporters.items()
    } == {
        canonical_tenant_id(TENANT_A): ["RoutedLM.__call__"],
        canonical_tenant_id(TENANT_B): ["RoutedLM.__call__"],
    }


async def test_a_failed_summary_restores_the_enclosing_tenant(
    dispatcher, recording_summarizer, tenant_exporters
):
    seen, gate = recording_summarizer
    gate.set()
    _, exporters = tenant_exporters
    with tenant_span_context("outer:production"):
        with pytest.raises(ValueError, match="^summary refused$"):
            await dispatcher.dispatch(
                "summarizer_agent", "fail", _task(TENANT_A, "fail")["context"]
            )
        assert current_tenant_id() == "outer:production"
    assert current_tenant_id() is None
    assert seen == [("fail", canonical_tenant_id(TENANT_A))]
    assert {
        t: [s.name for s in e.get_finished_spans()] for t, e in exporters.items()
    } == {canonical_tenant_id(TENANT_A): ["RoutedLM.__call__"]}
