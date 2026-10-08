"""A search agent's search is a recorded search the evaluation scores.

Real Vespa, real Phoenix, the real agent. The agent searches its rewrite of
the user's query; the ``search_service.search`` span it records carries the
user's query, so the tenant's golden set, matched by query text, scores it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

from cogniverse_agents.search_agent import SearchAgent, SearchAgentDeps, SearchInput
from cogniverse_core.common.tenant_utils import canonical_tenant_id
from cogniverse_core.query.encoders import QueryEncoderFactory
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_evaluation.recorded_searches import (
    SEARCH_SPAN_NAME,
    score_recorded_searches,
)
from cogniverse_foundation.config.unified_config import BackendProfileConfig
from cogniverse_foundation.telemetry.span_contract import read_span_io
from tests.utils.vespa_test_helpers import deploy_tenant_schema, make_config_manager

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]

BASE_SCHEMA = "video_colpali_smol500_mv_frame"
SHIPPED_PROFILE = json.loads(Path("configs/config.json").read_text())["backend"][
    "profiles"
][BASE_SCHEMA]
PROFILE = "agentrec_frames"
RUN = uuid.uuid4().hex[:8]
TENANT_A = f"agentrec{RUN}:a"
TENANT_B = f"agentrec{RUN}:b"
ORIGINAL = "discus throw"
REWRITTEN = "athlete discus throw stadium"
# (video_id, source title, segment description) per tenant.
CORPUS = {
    TENANT_A: [
        ("discus_final", "discus_final.mp4", "an athlete winds up a discus throw"),
        ("relay_heats", "relay_heats.mp4", "relay runners pass a baton at a stadium"),
    ],
    TENANT_B: [
        ("discus_clinic", "discus_clinic.mp4", "a coach breaks down the discus throw"),
        ("pool_laps", "pool_laps.mp4", "swimmers turn at the end of the pool"),
    ],
}


@pytest.fixture(scope="module")
def corpus(shared_vespa, real_telemetry):
    BackendRegistry.clear_instances()
    QueryEncoderFactory._encoder_cache.clear()
    config_manager = make_config_manager(shared_vespa)
    for tenant_id, segments in CORPUS.items():
        config_manager.add_backend_profile(
            BackendProfileConfig.from_dict(PROFILE, SHIPPED_PROFILE),
            tenant_id=tenant_id,
        )
        # A text-only default strategy: the search needs no query encoder.
        backend_config = config_manager.get_backend_config(tenant_id)
        backend_config.default_profiles = {
            "video": {"profile": PROFILE, "strategy": "bm25_only"}
        }
        config_manager.set_backend_config(backend_config)
        schema = deploy_tenant_schema(
            shared_vespa,
            tenant_id=tenant_id,
            base_schema_name=BASE_SCHEMA,
            config_manager=config_manager,
        )
        for video_id, title, description in segments:
            response = requests.post(
                f"http://localhost:{shared_vespa['http_port']}/document/v1/"
                f"content/{schema}/docid/{video_id}_seg_0",
                json={
                    "fields": {
                        "video_id": video_id,
                        "segment_id": 0,
                        "video_title": title,
                        "segment_description": description,
                    }
                },
                timeout=30,
            )
            assert response.status_code == 200, response.text
    yield {"vespa": shared_vespa, "config_manager": config_manager}
    BackendRegistry.clear_instances()
    QueryEncoderFactory._encoder_cache.clear()


def _agent(corpus, tenant_id, port):
    vespa = corpus["vespa"]
    return SearchAgent(
        deps=SearchAgentDeps(
            profile=PROFILE,
            tenant_id=tenant_id,
            backend_url="http://localhost",
            backend_port=vespa["http_port"],
            backend_config_port=vespa["config_port"],
            auto_create_memory_schema=False,
        ),
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        config_manager=corpus["config_manager"],
        port=port,
    )


async def _search_spans(telemetry, tenant_id, count, since):
    """The tenant's ``search_service.search`` spans since ``since``, once
    ``count`` of them are readable."""
    telemetry.force_flush(timeout_millis=10000)
    canonical = canonical_tenant_id(tenant_id)
    project = telemetry.config.get_project_name(canonical)
    provider = telemetry.get_provider(tenant_id=canonical, project_name=project)
    deadline = time.monotonic() + 90
    while True:
        spans = await provider.traces.get_all_spans(
            project=project,
            start_time=since,
            end_time=datetime.now(timezone.utc) + timedelta(minutes=1),
            filters={"name": SEARCH_SPAN_NAME},
        )
        if len(spans) >= count or time.monotonic() > deadline:
            return spans
        await asyncio.sleep(2)


def _rows(span):
    return [
        (row["video_id"], row["source_title"], row["score"])
        for row in read_span_io(span)["output"]
    ]


def _video_ids(out):
    return [r["metadata"]["video_id"] for r in out.results]


def _answered(out):
    return [
        (r["metadata"]["video_id"], r["metadata"]["source_title"], r["score"])
        for r in out.results
    ]


@pytest.mark.asyncio
async def test_a_rewritten_agent_search_is_recorded_and_scored(corpus, real_telemetry):
    since = datetime.now(timezone.utc) - timedelta(seconds=5)
    agent = _agent(corpus, TENANT_A, 8071)

    out = await agent.process(
        SearchInput(
            query=ORIGINAL, tenant_id=TENANT_A, enhanced_query=REWRITTEN, top_k=5
        )
    )
    spans = await _search_spans(real_telemetry, TENANT_A, 1, since)

    assert _video_ids(out) == ["discus_final", "relay_heats"]
    assert len(spans) == 1
    span = spans.iloc[0]
    attributes = {
        key: span[f"attributes.{key}"]
        for key in ("query", "enhanced_query", "profile", "strategy", "top_k")
    }
    assert attributes == {
        "query": ORIGINAL,
        "enhanced_query": REWRITTEN,
        "profile": PROFILE,
        "strategy": "default",
        "top_k": 5,
    }
    assert span["status_code"] == "OK"
    assert _rows(span) == _answered(out)

    scored = score_recorded_searches(
        spans,
        [
            {"query": ORIGINAL, "expected_videos": ["discus_final"]},
            {"query": "pool laps", "expected_videos": ["pool_laps"]},
        ],
    )
    [query] = scored["queries"]
    assert (query["profile"], query["strategy"], query["query"]) == (
        PROFILE,
        "default",
        ORIGINAL,
    )
    assert (query["expected"], query["retrieved"]) == (
        ["discus_final"],
        ["discus_final", "relay_heats"],
    )
    assert {
        name: query[name]
        for name in ("mrr", "ndcg", "recall_at_1", "recall_at_5", "precision_at_5")
    } == {
        "mrr": 1.0,
        "ndcg": 1.0,
        "recall_at_1": 1.0,
        "recall_at_5": 1.0,
        "precision_at_5": 0.5,
    }
    assert [
        (s["profile"], s["strategy"], s["queries"], s["success_rate"])
        for s in scored["strategies"]
    ] == [(PROFILE, "default", 1, 1.0)]
    assert scored["unsearched_queries"] == ["pool laps"]
    assert (scored["failed_searches"], scored["unscored_searches"]) == (0, 0)


@pytest.mark.asyncio
async def test_two_tenants_searching_at_once_each_record_their_own_search(
    corpus, real_telemetry
):
    """Both searches run inside their spans at the same moment: each tenant's
    project holds exactly its own search, with its own results."""
    since = datetime.now(timezone.utc) - timedelta(seconds=5)
    agents = {
        TENANT_A: _agent(corpus, TENANT_A, 8072),
        TENANT_B: _agent(corpus, TENANT_B, 8073),
    }
    together = threading.Barrier(2, timeout=60)
    inside = []
    for agent in agents.values():
        search = agent._search_backend

        def held(query_dict, search=search):
            inside.append(query_dict["tenant_id"])
            together.wait()
            return search(query_dict)

        agent._search_backend = held

    outs = await asyncio.gather(
        *(
            agent.process(
                SearchInput(
                    query=ORIGINAL,
                    tenant_id=tenant_id,
                    enhanced_query=REWRITTEN,
                    top_k=5,
                )
            )
            for tenant_id, agent in agents.items()
        )
    )
    recorded = {
        tenant_id: await _search_spans(real_telemetry, tenant_id, 1, since)
        for tenant_id in agents
    }

    assert sorted(inside) == sorted(agents)
    assert [_video_ids(out) for out in outs] == [
        ["discus_final", "relay_heats"],
        ["discus_clinic"],
    ]
    for (tenant_id, spans), out in zip(recorded.items(), outs):
        assert len(spans) == 1, tenant_id
        span = spans.iloc[0]
        assert (
            span["attributes.query"],
            span["attributes.tenant.id"],
            _rows(span),
        ) == (ORIGINAL, canonical_tenant_id(tenant_id), _answered(out))


@pytest.mark.asyncio
class TestTelemetryOutageLeavesTheSearchAlone:
    """Last in the module: it replaces the process's telemetry manager."""

    async def test_an_unreachable_collector_still_answers_and_logs_the_loss(
        self, corpus, real_telemetry, caplog
    ):
        import cogniverse_foundation.telemetry.manager as telemetry_module
        from cogniverse_foundation.telemetry.config import (
            BatchExportConfig,
            TelemetryConfig,
        )
        from cogniverse_foundation.telemetry.manager import TelemetryManager

        agent = _agent(corpus, TENANT_B, 8074)
        TelemetryManager.reset()
        dead = TelemetryManager(
            config=TelemetryConfig(
                otlp_endpoint="http://127.0.0.1:9",
                provider_config={
                    "http_endpoint": "http://127.0.0.1:9",
                    "grpc_endpoint": "http://127.0.0.1:9",
                },
                batch_config=BatchExportConfig(use_sync_export=False),
            )
        )
        telemetry_module._telemetry_manager = dead
        try:
            started = time.monotonic()
            out = await agent.process(
                SearchInput(
                    query="pool laps",
                    tenant_id=TENANT_B,
                    enhanced_query="pool laps",
                    top_k=5,
                )
            )
            elapsed = time.monotonic() - started
            with caplog.at_level(logging.WARNING):
                dead.force_flush(timeout_millis=30000)
        finally:
            TelemetryManager.reset()

        assert _video_ids(out) == ["pool_laps"]
        assert elapsed < 10, f"search took {elapsed:.1f}s with the collector down"
        # The span is lost, and the loss is logged where the export failed.
        assert [
            (record.name, record.getMessage())
            for record in caplog.records
            if record.levelno >= logging.ERROR
        ] == [
            (
                "opentelemetry.exporter.otlp.proto.grpc.exporter",
                "Failed to export traces to 127.0.0.1:9, error code: "
                "StatusCode.UNAVAILABLE",
            )
        ]
