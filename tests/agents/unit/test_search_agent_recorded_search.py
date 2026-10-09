"""Every text search the search agent runs is a recorded search.

The evaluation (golden set and dataset modes) and the optimization
framework's search annotations, golden dataset and span analysis read only
``search_service.search`` spans. The agent searched its backend directly, so
a search made in the web Search workspace, which runs through the agent,
never reached any of them. Each backend
search the agent runs for a query now records that span, carrying the user's
query rather than the agent's rewrite of it, since golden queries are matched
by their text.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from cogniverse_agents.search_agent import SearchContext, SearchInput
from cogniverse_evaluation.recorded_searches import SEARCH_SPAN_NAME
from cogniverse_foundation.telemetry.span_contract import search_result_row
from tests.utils.stub_search import StubBackend, build_stub_search_agent

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

TENANT = "acme:recorded"
ORIGINAL = "athlete throwing discus"
REWRITTEN = "video of athlete throwing discus in stadium"
HITS = [
    ("discus_seg_1", 0.91, {"source_title": "Discus Final", "video_id": "discus"}),
    ("relay_seg_4", 0.42, {"source_title": "Relay Heats", "video_id": "relay"}),
]
PROFILE = "video_colpali_smol500_mv_frame"
OTHER_PROFILE = "video_colqwen_omni_mv_chunk_30s"


@pytest.fixture
def exported(monkeypatch):
    """The spans the search agent records, through a telemetry manager that
    keeps them in memory."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("recorded-search")

    class _Manager:
        @contextmanager
        def span(self, name, tenant_id, project_name=None, attributes=None, **_):
            with tracer.start_as_current_span(name) as span:
                span.set_attribute("tenant.id", tenant_id)
                for key, value in (attributes or {}).items():
                    span.set_attribute(key, value)
                yield span

    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.context.get_telemetry_manager",
        lambda: _Manager(),
    )
    return exporter


def _recorded(exporter):
    return [
        span for span in exporter.get_finished_spans() if span.name == SEARCH_SPAN_NAME
    ]


def _rows(hits):
    return [
        search_result_row(
            SimpleNamespace(
                document=SimpleNamespace(id=doc_id, metadata=dict(meta)), score=score
            )
        )
        for doc_id, score, meta in hits
    ]


def _attributes(span):
    attributes = dict(span.attributes)
    attributes.pop("latency_ms")
    return attributes


def _expected(query, profile, top_k, rows, **extra):
    return {
        "tenant.id": TENANT,
        "openinference.span.kind": "CHAIN",
        "operation.name": "search",
        "backend": "vespa",
        "query": query,
        "strategy": "default",
        "top_k": top_k,
        "profile": profile,
        "input.value": query,
        "operation": "search",
        "num_results": len(rows),
        "output.value": json.dumps(rows),
        "top_score": rows[0]["score"],
        **extra,
    }


@pytest.mark.asyncio
async def test_a_rewritten_text_search_records_the_users_query(exported):
    agent = build_stub_search_agent(TENANT, HITS)

    out = await agent.process(
        SearchInput(query=ORIGINAL, tenant_id=TENANT, enhanced_query=REWRITTEN, top_k=5)
    )

    [span] = _recorded(exported)
    assert _attributes(span) == _expected(
        ORIGINAL, PROFILE, 5, _rows(HITS), enhanced_query=REWRITTEN
    )
    assert span.status.status_code == StatusCode.OK
    assert [r["id"] for r in out.results] == ["discus_seg_1", "relay_seg_4"]


def test_an_unrewritten_text_search_records_no_rewrite(exported):
    agent = build_stub_search_agent(TENANT, HITS)

    agent.search_by_text(ORIGINAL, tenant_id=TENANT, top_k=3)

    [span] = _recorded(exported)
    assert _attributes(span) == _expected(ORIGINAL, PROFILE, 3, _rows(HITS))


@pytest.mark.asyncio
async def test_an_ensemble_records_each_profiles_search_of_the_users_query(
    exported,
):
    agent = build_stub_search_agent(TENANT, HITS)

    await agent.process(
        SearchInput(
            query=ORIGINAL,
            tenant_id=TENANT,
            enhanced_query=REWRITTEN,
            profiles=[PROFILE, OTHER_PROFILE],
            top_k=4,
        )
    )

    recorded = sorted(_recorded(exported), key=lambda span: span.attributes["profile"])
    assert [_attributes(span) for span in recorded] == [
        _expected(ORIGINAL, profile, 8, _rows(HITS), enhanced_query=REWRITTEN)
        for profile in sorted([PROFILE, OTHER_PROFILE])
    ]


@pytest.mark.asyncio
async def test_a_multi_query_fusion_records_the_fused_results_once(exported):
    agent = build_stub_search_agent(TENANT, HITS)

    out = await agent.process(
        SearchInput(
            query=ORIGINAL,
            tenant_id=TENANT,
            query_variants=[
                {"name": "original", "query": ORIGINAL},
                {"name": "expansion", "query": REWRITTEN},
            ],
            top_k=5,
        )
    )

    [span] = _recorded(exported)
    fused = agent._fuse_results_rrf(
        {
            "original": [
                {"id": doc_id, "score": score, **meta} for doc_id, score, meta in HITS
            ],
            "expansion": [
                {"id": doc_id, "score": score, **meta} for doc_id, score, meta in HITS
            ],
        },
        k=60,
        top_k=5,
    )
    rows = [search_result_row(result) for result in fused]
    assert _attributes(span) == _expected(ORIGINAL, PROFILE, 5, rows)
    assert [r["id"] for r in out.results] == ["discus_seg_1", "relay_seg_4"]


def test_a_relationship_search_records_the_original_query(exported):
    agent = build_stub_search_agent(TENANT, HITS)

    agent.search_with_relationship_context(
        SearchContext(
            original_query=ORIGINAL,
            enhanced_query=REWRITTEN,
            entities=[],
            relationships=[],
            routing_metadata={},
            confidence=0.9,
        ),
        tenant_id=TENANT,
        top_k=2,
    )

    [span] = _recorded(exported)
    assert _attributes(span) == _expected(
        ORIGINAL, PROFILE, 2, _rows(HITS), enhanced_query=REWRITTEN
    )


def test_a_failed_backend_search_is_recorded_as_failed_and_still_raises(exported):
    agent = build_stub_search_agent(TENANT, HITS)

    class _DeadBackend:
        def search(self, query_dict):
            raise ConnectionError("vespa is down")

    agent._get_backend = lambda: _DeadBackend()

    with pytest.raises(ConnectionError, match="^vespa is down$"):
        agent.search_by_text(ORIGINAL, tenant_id=TENANT, top_k=3)

    [span] = _recorded(exported)
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "ConnectionError: vespa is down"
    assert span.attributes["query"] == ORIGINAL


def test_results_that_cannot_be_recorded_still_answer_the_search(
    exported, monkeypatch, caplog
):
    agent = build_stub_search_agent(TENANT, HITS)

    def unrecordable(span, results, output_value=None):
        raise TypeError("result rows are not serializable")

    monkeypatch.setattr(
        "cogniverse_agents.search_agent.add_search_results_to_span", unrecordable
    )

    with caplog.at_level(logging.WARNING, logger="cogniverse_agents.search_agent"):
        results = agent.search_by_text(ORIGINAL, tenant_id=TENANT, top_k=3)

    assert [r["id"] for r in results] == ["discus_seg_1", "relay_seg_4"]
    [span] = _recorded(exported)
    assert span.status.status_code == StatusCode.OK
    assert "output.value" not in span.attributes
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "cogniverse_agents.search_agent"
    ] == [
        f"search span of {PROFILE!r} for tenant {TENANT!r} recorded no results: "
        "result rows are not serializable"
    ]


def test_the_backend_seam_is_unchanged_for_callers_that_bind_it(exported):
    """``_search_backend(query_dict)`` stays the one-argument seam tests and
    callers bind a backend at."""
    agent = build_stub_search_agent(TENANT, HITS)
    seen = []
    backend = StubBackend(HITS)

    def bound(query_dict):
        seen.append(query_dict["query"])
        return backend.search(query_dict)

    agent._search_backend = bound

    agent.search_by_text(ORIGINAL, tenant_id=TENANT, top_k=3)

    assert seen == [ORIGINAL]
    assert len(_recorded(exported)) == 1


def test_a_search_is_recorded_under_the_canonical_tenant(exported):
    """The evaluation reads the canonical tenant's project; a raw tenant id on
    the request records there too."""
    agent = build_stub_search_agent("acme", HITS)

    agent.search_by_text(ORIGINAL, tenant_id="acme", top_k=3)

    [span] = _recorded(exported)
    assert span.attributes["tenant.id"] == "acme:acme"


def test_telemetry_that_cannot_be_set_up_leaves_the_search_unrecorded(
    exported, monkeypatch, caplog
):
    agent = build_stub_search_agent(TENANT, HITS)

    def unavailable():
        raise ConnectionError("config store refused the telemetry read")

    monkeypatch.setattr(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager", unavailable
    )

    with caplog.at_level(logging.WARNING, logger="cogniverse_agents.search_agent"):
        results = agent.search_by_text(ORIGINAL, tenant_id=TENANT, top_k=3)

    assert [r["id"] for r in results] == ["discus_seg_1", "relay_seg_4"]
    assert _recorded(exported) == []
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "cogniverse_agents.search_agent"
    ] == [
        f"search of {PROFILE!r} for tenant {TENANT!r} not recorded: telemetry "
        "unavailable: config store refused the telemetry read"
    ]
