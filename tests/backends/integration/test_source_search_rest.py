"""POST /search reports an incomplete source search, against real Vespa.

The search router runs over the real SearchService, VespaSearchBackend, X-CLIP
text encoder (the video_embed sidecar) and Vespa. A dominant source whose
segments fill the nearest-neighbor candidate budget answers one source with
``source_search_incomplete`` true in the response body and on the request,
search and backend spans; a top_k whose budget covers the corpus answers
false.
"""

import json
import threading
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

import cogniverse_foundation.telemetry.manager as telemetry_manager_module
from cogniverse_agents.search.service import SearchService
from cogniverse_core.query.encoders import QueryEncoderFactory
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.utils import get_config
from cogniverse_foundation.inference_specs import get_inference_service_spec
from cogniverse_foundation.telemetry.config import (
    BatchExportConfig,
    TelemetryConfig,
    TelemetryLevel,
)
from cogniverse_foundation.telemetry.manager import TelemetryManager
from cogniverse_runtime.routers import search as search_router
from tests.backends.integration.test_source_grouped_search import (
    FAST_RETRY,
    _fault_backend,
    _feed,
    _grouped_body,
    _is_backend_search,
    _search_requests,
    _video_fields,
    _wait_for_count,
)
from tests.fixtures.inference import LocalEndpointProvider
from tests.utils.vespa_test_helpers import deploy_tenant_schema, make_config_manager

pytestmark = [pytest.mark.integration]

PROFILE = "video_xclip_sv_chunk_6s"
TENANT = f"srcrest{uuid.uuid4().hex[:8]}:rest"
QUERY = "a harbour ferry docking at the pier"
DOMINANT_SEGMENTS = 60
MINORITY = [f"restminor{j}" for j in range(5)]


@pytest.fixture(scope="module")
def video_embed_url():
    provider = LocalEndpointProvider()
    try:
        yield provider.resolve(get_inference_service_spec("video_embed")).base_url
    finally:
        provider.close()


@pytest.fixture(scope="module")
def rest_corpus(vespa_instance, video_embed_url):
    """Segments placed around the served query embedding.

    Every dominant segment lies nearer the query than any minority source, so
    the 40 nearest candidates of a top_k=10 search all belong to one source.
    """
    config_manager = make_config_manager(
        vespa_instance, inference_service_urls={"video_embed": video_embed_url}
    )
    schema = deploy_tenant_schema(
        vespa_instance,
        tenant_id=TENANT,
        base_schema_name=PROFILE,
        config_manager=config_manager,
    )
    QueryEncoderFactory._encoder_cache.clear()
    encoder = QueryEncoderFactory.create_encoder(
        PROFILE,
        json.loads(Path("configs/config.json").read_text())["backend"]["profiles"][
            PROFILE
        ]["embedding_model"],
        config=get_config(tenant_id=TENANT, config_manager=config_manager),
    )
    query = np.asarray(encoder.encode(QUERY), dtype=np.float32).reshape(-1)
    assert query.shape == (768,)
    query = query / np.linalg.norm(query)
    rng = np.random.default_rng(7)

    def near(scale: float) -> tuple:
        offset = rng.standard_normal(768).astype(np.float32)
        offset -= offset.dot(query) * query
        vector = query + scale * offset / np.linalg.norm(offset)
        binary = np.packbits((vector > 0).astype(np.uint8)).astype(np.int8)
        return {"values": vector.tolist()}, {"values": binary.tolist()}

    port = vespa_instance["http_port"]
    for i in range(DOMINANT_SEGMENTS):
        _feed(
            port,
            schema,
            f"restdom_{i:03d}",
            _video_fields("restdom", "restcorpus", i, *near(0.01 + 0.001 * i)),
        )
    for j, source_id in enumerate(MINORITY):
        _feed(
            port,
            schema,
            f"{source_id}_000",
            _video_fields(source_id, "restcorpus", 0, *near(0.5 + 0.05 * j)),
        )
    _wait_for_count(port, schema, DOMINANT_SEGMENTS + len(MINORITY))
    try:
        yield config_manager
    finally:
        QueryEncoderFactory._encoder_cache.clear()


@pytest.fixture
def spans():
    """The process telemetry manager, recording into memory for this test."""
    TelemetryManager.reset()
    manager = TelemetryManager(
        TelemetryConfig(
            enabled=True,
            otlp_endpoint="http://localhost:4317",
            service_name="source-search-rest-test",
            environment="test",
            level=TelemetryLevel.VERBOSE,
            batch_config=BatchExportConfig(use_sync_export=True),
        )
    )
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("source-search-rest-test")
    manager._get_tracer_for_project = lambda tenant_id, project_name=None: tracer
    telemetry_manager_module._telemetry_manager = manager
    try:
        yield exporter
    finally:
        TelemetryManager.reset()
        provider.shutdown()


@pytest.fixture
def client(rest_corpus):
    app = FastAPI()
    app.include_router(search_router.router, prefix="/search")
    app.dependency_overrides[search_router.get_config_manager_dependency] = lambda: (
        rest_corpus
    )
    app.dependency_overrides[search_router.get_schema_loader_dependency] = lambda: (
        FilesystemSchemaLoader(Path("configs/schemas"))
    )

    async def tenant_is_registered(_tenant_id: str) -> None:
        return None

    with patch.object(search_router, "assert_tenant_exists", tenant_is_registered):
        with TestClient(app) as test_client:
            yield test_client


def _post(client, tag: str, **extra):
    return client.post(
        "/search/",
        json={
            "query": QUERY,
            "profile": PROFILE,
            "strategy": "float_float",
            "tenant_id": TENANT,
            "top_k": 10,
            "result_granularity": "source",
            "filters": {"video_title": tag},
            **extra,
        },
    )


def _flags(exporter) -> dict:
    return {
        span.name: span.attributes.get("source_search_incomplete")
        for span in exporter.get_finished_spans()
        if span.name
        in {"api.search.request", "search_service.search", "search.execute"}
    }


def test_a_saturated_budget_reports_an_incomplete_search(client, spans):
    response = _post(client, "restcorpus")

    assert response.status_code == 200, response.text
    body = response.json()
    assert [row["source_id"] for row in body["results"]] == ["restdom"]
    assert [row["document_id"] for row in body["results"][0]["matched_segments"]] == [
        "restdom_000",
        "restdom_001",
        "restdom_002",
        "restdom_003",
    ]
    assert body["source_search_incomplete"] is True
    assert _flags(spans) == {
        "api.search.request": True,
        "search_service.search": True,
        "search.execute": True,
    }


def test_matches_within_the_budget_report_a_complete_search(client, spans):
    response = _post(client, "restcorpus", top_k=20)

    assert response.status_code == 200, response.text
    body = response.json()
    assert [row["source_id"] for row in body["results"]] == ["restdom", *MINORITY]
    assert body["source_search_incomplete"] is False
    assert _flags(spans) == {
        "api.search.request": False,
        "search_service.search": False,
        "search.execute": False,
    }


def test_the_stream_final_event_carries_the_flag(client, spans):
    response = _post(client, "restcorpus", stream=True)

    assert response.status_code == 200, response.text
    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [event["type"] for event in events] == ["status", "final"]
    assert events[1]["data"]["source_search_incomplete"] is True
    assert [row["source_id"] for row in events[1]["data"]["results"]] == ["restdom"]


def test_concurrent_requests_each_report_their_own_flag(client, spans):
    top_ks = [10, 20] * 4
    barrier = threading.Barrier(len(top_ks))
    outcomes: dict = {}

    def run(slot: int, top_k: int) -> None:
        barrier.wait()
        try:
            body = _post(client, "restcorpus", top_k=top_k).json()
            outcomes[slot] = (
                [row["source_id"] for row in body["results"]],
                body["source_search_incomplete"],
            )
        except BaseException as exc:
            outcomes[slot] = exc

    threads = [
        threading.Thread(target=run, args=(slot, top_k))
        for slot, top_k in enumerate(top_ks)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert outcomes == {
        slot: ((["restdom"], True) if top_k == 10 else (["restdom", *MINORITY], False))
        for slot, top_k in enumerate(top_ks)
    }


DEGRADED = _grouped_body(
    60,
    [{"id": "group:root:0", "relevance": 1.0}],
    coverage={"coverage": 40, "degraded": {"timeout": True}},
)
GROUPS_LOST = _grouped_body(60, [{"id": "group:root:0", "relevance": 1.0}])


@pytest.fixture(params=["degraded", "groups_lost"])
def faulted(request, vespa_instance, rest_corpus, client):
    """The request's SearchService searches through a real backend whose
    Vespa answers every search with an unusable grouped response."""
    body, message = {
        "degraded": (DEGRADED, "Vespa query coverage degraded"),
        "groups_lost": (
            GROUPS_LOST,
            "Vespa grouped 60 matched segments into no source groups",
        ),
    }[request.param]

    def intercept(method, path, payload):
        return (200, body) if _is_backend_search(path, payload) else None

    backend, proxy = _fault_backend(rest_corpus, vespa_instance, intercept)
    try:
        with patch.object(SearchService, "_get_backend", lambda *_: backend):
            yield proxy, message
    finally:
        backend.close()
        proxy.__exit__(None, None, None)


def test_a_faulted_search_is_a_server_error_not_an_empty_answer(client, faulted):
    proxy, message = faulted

    response = _post(client, "restcorpus")

    assert response.status_code == 500, response.text
    assert message in response.json()["detail"], response.text
    assert "source_search_incomplete" not in response.text
    assert _search_requests(proxy) == FAST_RETRY.max_attempts


def test_a_faulted_stream_ends_in_an_error_event_not_a_final_one(client, faulted):
    proxy, message = faulted

    response = _post(client, "restcorpus", stream=True)

    assert response.status_code == 200, response.text
    events = [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [event["type"] for event in events] == ["status", "error"]
    assert message in events[1]["error"]
    assert events[1]["error_type"] == "VespaError"
    assert _search_requests(proxy) == FAST_RETRY.max_attempts
