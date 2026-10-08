"""Every agent run the dispatcher serves is traced in its process span.

The summarizer, the detailed report and deep research ran without a telemetry
manager, so no span of theirs reached Phoenix or the analytics built on it; a
document search ran outside any span, so its hits carried no span id a client
could rate them against. These tests drive the real dispatcher over a real
Vespa (the tenant's document corpus fed with the served ColBERT encoder), the
LM on Modal and a real Phoenix, and read every span back from Phoenix.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import dspy
import pytest

from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.telemetry.config import BatchExportConfig, TelemetryConfig
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_foundation.telemetry.span_contract import (
    OP_SEARCH,
    RESULT_RELEVANCE,
    persist_result_relevance,
    read_span_io,
)
from cogniverse_runtime.admin import tenant_manager
from cogniverse_runtime.agent_dispatcher import GROUNDING_THREADED, AgentDispatcher
from tests.utils.vespa_test_helpers import feed_text_documents

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHIPPED = json.loads((_REPO_ROOT / "configs" / "config.json").read_text())
COLBERT_MODEL = _SHIPPED["backend"]["profiles"]["document_text_semantic"][
    "embedding_model"
]

TENANT = f"agentspans{uuid.uuid4().hex[:8]}:main"
CORPUS_ID = "agent_spans_corpus"
CORPUS = (
    {
        "id": "harbour_dredging",
        "title": "Harbour Dredging Survey",
        "text": (
            "The dredging survey recorded silt accumulation across the tidal "
            "basin and recommends removing sediment from the northern berth "
            "before the winter shipping season begins."
        ),
    },
    {
        "id": "winter_beekeeping",
        "title": "Winter Beekeeping Manual",
        "text": (
            "Overwintering colonies need ventilated hives and candy board "
            "feeding, because the cluster warms itself and opening a hive "
            "during frost chills the brood."
        ),
    },
)
HARBOUR_QUERY = "silt accumulation across the tidal basin"
BEEKEEPING_QUERY = "candy board feeding for overwintering colonies"
THREADED_HITS = [
    {
        "id": "harbour_dredging",
        "title": "Harbour Dredging Survey",
        "content": CORPUS[0]["text"],
        "score": 0.92,
    }
]


@pytest.fixture(scope="module")
def corpus_tenant(vespa_instance, config_manager, schema_loader, pylate_server):
    """The tenant's document_text schema, fed through its ingestion backend,
    with the ColBERT encoder served for search."""
    from cogniverse_runtime.ingestion.processors.embedding_generator.backend_factory import (  # noqa: E501
        BackendFactory,
    )

    system_config = config_manager.get_system_config()
    system_config.inference_service_urls = {
        **system_config.inference_service_urls,
        "colbert_pylate": pylate_server,
    }
    config_manager.set_system_config(system_config)
    tenant_manager.set_config_manager(config_manager)
    tenant_manager.set_schema_loader(schema_loader)
    backend = BackendFactory.create(
        "vespa",
        TENANT,
        {},
        config_manager=config_manager,
        schema_loader=schema_loader,
    )
    backend.schema_registry.deploy_schemas(TENANT, ["document_text"])
    fed = feed_text_documents(
        backend_client=backend,
        schema_name="document_text",
        inference_url=pylate_server,
        model_name=COLBERT_MODEL,
        documents=CORPUS,
        corpus_id=CORPUS_ID,
    )
    assert (fed.documents_fed, fed.documents_processed, fed.errors) == (2, 2, [])
    time.sleep(3)
    yield TENANT
    tenant_manager.set_config_manager(None)
    tenant_manager.set_schema_loader(None)


@pytest.fixture(scope="module")
def dispatcher(config_manager, schema_loader, real_telemetry, corpus_tenant):
    return AgentDispatcher(
        agent_registry=AgentRegistry(tenant_id=TENANT, config_manager=config_manager),
        config_manager=config_manager,
        schema_loader=schema_loader,
    )


def _provider(telemetry, tenant_id):
    project = telemetry.config.get_project_name(tenant_id)
    return telemetry.get_provider(tenant_id=tenant_id, project_name=project), project


async def _spans(telemetry, tenant_id, name, count, timeout=90.0):
    """The tenant's spans named ``name`` once ``count`` are readable."""
    telemetry.force_flush(timeout_millis=10000)
    provider, project = _provider(telemetry, tenant_id)
    deadline = time.monotonic() + timeout
    while True:
        end = datetime.now(timezone.utc)
        spans = await provider.traces.get_spans(
            project=project,
            start_time=end - timedelta(hours=1),
            end_time=end,
            filters={"name": name},
        )
        if len(spans) >= count or time.monotonic() > deadline:
            return spans
        await asyncio.sleep(2)


def _rows_by_span_id(spans) -> dict:
    return {row["context.span_id"]: row for _, row in spans.iterrows()}


async def _spans_with_ids(telemetry, tenant_id, name, span_ids, timeout=90.0):
    """The tenant's spans named ``name``, by id, once every one of
    ``span_ids`` is readable."""
    deadline = time.monotonic() + timeout
    while True:
        rows = _rows_by_span_id(
            await _spans(telemetry, tenant_id, name, len(span_ids), timeout=0)
        )
        if set(span_ids) <= set(rows) or time.monotonic() > deadline:
            return rows
        await asyncio.sleep(2)


@pytest.mark.asyncio
class TestDocumentHitsCarryTheirSearchSpan:
    async def test_the_envelope_names_the_span_that_recorded_its_hits(
        self, dispatcher, real_telemetry
    ):
        result = await dispatcher._execute_document_search_task(
            HARBOUR_QUERY, TENANT, 2
        )

        hit_ids = [hit["document_id"] for hit in result["results"]]
        assert hit_ids == [
            "harbour_dredging",
            "winter_beekeeping",
        ]
        spans = await _spans_with_ids(
            real_telemetry, TENANT, "DocumentAgent.process", [result["span_id"]]
        )
        recorded = read_span_io(spans[result["span_id"]])
        assert (
            recorded["input"],
            recorded["operation"],
            recorded["modality"],
            [row["document_id"] for row in recorded["output"]],
            [row["source_title"] for row in recorded["output"]],
        ) == (
            HARBOUR_QUERY,
            OP_SEARCH,
            "document",
            hit_ids,
            ["Harbour Dredging Survey", "Winter Beekeeping Manual"],
        )

    async def test_a_hit_is_rated_on_that_span(self, dispatcher, real_telemetry):
        result = await dispatcher._execute_document_search_task(
            BEEKEEPING_QUERY, TENANT, 2
        )
        top = result["results"][0]["document_id"]
        await _spans_with_ids(
            real_telemetry, TENANT, "DocumentAgent.process", [result["span_id"]]
        )
        provider, project = _provider(real_telemetry, TENANT)

        score = await persist_result_relevance(
            provider, project, result["span_id"], top, "Highly Relevant"
        )

        assert (top, score) == ("winter_beekeeping", 1.0)
        deadline = time.monotonic() + 90
        while True:
            spans = await _spans(real_telemetry, TENANT, "DocumentAgent.process", 2)
            annotations = await provider.annotations.get_annotations(
                spans_df=spans[spans["context.span_id"] == result["span_id"]],
                project=project,
                annotation_names=[RESULT_RELEVANCE],
            )
            if len(annotations) or time.monotonic() > deadline:
                break
            await asyncio.sleep(2)
        assert [
            (row["result.label"], row["result.score"])
            for _, row in annotations.iterrows()
        ] == [("Highly Relevant", 1.0)]

    async def test_concurrent_searches_each_record_their_own_query(
        self, dispatcher, real_telemetry
    ):
        """Eight searches in flight together: each envelope's span recorded
        that envelope's own query and hits, and no two share a span."""
        queries = [HARBOUR_QUERY, BEEKEEPING_QUERY] * 4

        results = await asyncio.gather(
            *(
                dispatcher._execute_document_search_task(query, TENANT, 2)
                for query in queries
            )
        )

        span_ids = [result["span_id"] for result in results]
        assert len(set(span_ids)) == len(queries)
        spans = await _spans_with_ids(
            real_telemetry, TENANT, "DocumentAgent.process", span_ids
        )
        assert [
            (
                read_span_io(spans[result["span_id"]])["input"],
                [
                    row["document_id"]
                    for row in read_span_io(spans[result["span_id"]])["output"]
                ],
            )
            for result in results
        ] == [
            (query, [hit["document_id"] for hit in result["results"]])
            for query, result in zip(queries, results)
        ]


@pytest.mark.asyncio
class TestAnswerAgentsAreTraced:
    @pytest.fixture(autouse=True)
    def _lm(self, ensure_host_ollama):
        """The served runtime binds its LM at startup; deep research plans
        through it. The root conftest clears it after each test."""
        from tests.fixtures.llm import make_dspy_lm

        dspy.configure(lm=make_dspy_lm())

    async def test_a_summary_leaves_its_summarizer_span(
        self, dispatcher, real_telemetry
    ):
        result = await dispatcher._execute_summarization_task(
            HARBOUR_QUERY, TENANT, {"search_results": THREADED_HITS}
        )

        spans = await _spans(real_telemetry, TENANT, "SummarizerAgent.process", 1)
        assert (result["status"], result["grounding"]["state"]) == (
            "success",
            GROUNDING_THREADED,
        )
        assert (len(spans), spans.iloc[0]["status_code"]) == (1, "UNSET")

    async def test_a_detailed_report_leaves_its_report_span(
        self, dispatcher, real_telemetry
    ):
        result = await dispatcher._execute_detailed_report_task(
            HARBOUR_QUERY, TENANT, {"search_results": THREADED_HITS}
        )

        spans = await _spans(real_telemetry, TENANT, "DetailedReportAgent.process", 1)
        assert (result["status"], result["grounding"]["state"]) == (
            "success",
            GROUNDING_THREADED,
        )
        assert (len(spans), spans.iloc[0]["status_code"]) == (1, "UNSET")

    async def test_deep_research_leaves_its_research_span(
        self, dispatcher, real_telemetry
    ):
        result = await dispatcher._execute_deep_research_task(
            "research the harbour dredging survey documents",
            TENANT,
            {"tenant_id": TENANT, "max_iterations": 1},
        )

        spans = await _spans(real_telemetry, TENANT, "DeepResearchAgent.process", 1)
        assert result["status"] == "success"
        assert (len(spans), spans.iloc[0]["status_code"]) == (1, "UNSET")


@pytest.mark.asyncio
class TestTelemetryOutageLeavesServingAlone:
    """Last in the module: it replaces the process's telemetry manager."""

    async def test_an_unreachable_collector_still_answers_the_search(
        self, dispatcher, real_telemetry
    ):
        """Serving never waits on telemetry: with the span exporter pointed at
        a port nothing listens on, the search answers the same hits and still
        names the span it ran in."""
        import cogniverse_foundation.telemetry.manager as telemetry_module

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
            result = await dispatcher._execute_document_search_task(
                HARBOUR_QUERY, TENANT, 2
            )
            elapsed = time.monotonic() - started
        finally:
            TelemetryManager.reset()

        assert [hit["document_id"] for hit in result["results"]] == [
            "harbour_dredging",
            "winter_beekeeping",
        ]
        assert re.fullmatch(r"[0-9a-f]{16}", result["span_id"])
        assert elapsed < 10, f"search took {elapsed:.1f}s with the collector down"
