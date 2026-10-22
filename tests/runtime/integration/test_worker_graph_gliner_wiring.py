"""The ingestion-worker graph path uses its captured GLiNER endpoint."""

from __future__ import annotations

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.graph.graph_schema import Mention
from cogniverse_core.common.models.model_loaders import RemoteColBERTLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime.ingestion_worker import queue, worker
from cogniverse_runtime.ingestion_worker.worker import _build_worker_graph_factory
from tests.utils.memory_store import InMemoryConfigStore


@pytest.fixture(autouse=True)
def _minio_outage_env(monkeypatch):
    """MinIO endpoint that refuses connections, single boto3 attempt: the
    worker's original-filename lookup exercises its real outage fallback
    ("") instead of failing the job on unset env."""
    monkeypatch.setenv("MINIO_ENDPOINT", "http://127.0.0.1:29071")
    monkeypatch.setenv("MINIO_ACCESS_KEY", "test")
    monkeypatch.setenv("MINIO_SECRET_KEY", "test")
    monkeypatch.setenv("AWS_RETRY_MODE", "standard")
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "1")


pytestmark = [
    pytest.mark.integration,
    pytest.mark.ci_fast,
    pytest.mark.no_shared_vespa,
]


def _gliner_server(entity_text: str, label: str):
    requests: list[tuple[str, dict]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            requests.append((self.path, payload))
            body = json.dumps(
                {
                    "entities": [
                        {
                            "text": entity_text,
                            "label": label,
                            "score": 0.97,
                            "start": 0,
                            "end": len(entity_text),
                        }
                    ]
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_port}", requests


def _manager(endpoint: str, tenant_id: str):
    backend = MagicMock()
    backend.get_tenant_schema_name.return_value = (
        f"knowledge_graph_{tenant_id.replace(':', '_')}"
    )
    config_manager = SimpleNamespace(
        get_system_config=lambda: SimpleNamespace(
            inference_service_urls={
                "colbert_pylate": "http://colbert.invalid:9000",
                "gliner": endpoint,
            }
        )
    )
    return _build_worker_graph_factory(backend, config_manager)(tenant_id)


def test_concurrent_worker_graph_extraction_uses_each_captured_real_http_endpoint(
    monkeypatch,
):
    monkeypatch.setattr(
        RemoteColBERTLoader,
        "load_model",
        lambda self: (object(), None),
    )
    first_server, first_url, first_requests = _gliner_server("Marie Curie", "Person")
    second_server, second_url, second_requests = _gliner_server("radium", "Substance")
    try:
        first = _manager(first_url, "acme:first")
        second = _manager(second_url, "acme:second")
        anchor = Mention(
            source_doc_id="science.txt",
            segment_id="segment-1",
            ts_start=0.0,
            ts_end=1.0,
            modality="document",
            evidence_span="Marie Curie discovered radium.",
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_result, second_result = executor.map(
                lambda pair: pair[0]._doc_extractor.extract_from_text(
                    "Marie Curie discovered radium.",
                    tenant_id=pair[1],
                    source_doc_id="science.txt",
                    segment_anchor=anchor,
                ),
                [(first, "acme:first"), (second, "acme:second")],
            )
    finally:
        first_server.shutdown()
        second_server.shutdown()

    assert [(node.name, node.label) for node in first_result.nodes] == [
        ("Marie Curie", "Person")
    ]
    assert [(node.name, node.label) for node in second_result.nodes] == [
        ("radium", "Substance")
    ]
    assert first_result.edges == []
    assert second_result.edges == []
    assert len(first_requests) == len(second_requests) == 1
    for requests, expected_url in (
        (first_requests, first_url),
        (second_requests, second_url),
    ):
        path, payload = requests[0]
        assert path == "/predict_entities"
        assert payload["text"] == "Marie Curie discovered radium."
        assert payload["threshold"] == 0.3
        assert payload["model"] == "urchade/gliner_large-v2.1"
        assert payload["labels"] == [
            "Person",
            "Organization",
            "Location",
            "Date",
            "Substance",
            "Award",
            "Field",
            "Event",
            "Concept",
            "Technology",
            "Product",
            "Algorithm",
            "Model",
            "Framework",
            "Language",
        ]
        assert expected_url.startswith("http://127.0.0.1:")


def _job_config_manager(gliner_url: str | None) -> ConfigManager:
    urls = {"colbert_pylate": "http://colbert.invalid:9000"}
    if gliner_url is not None:
        urls["gliner"] = gliner_url
    manager = ConfigManager(store=InMemoryConfigStore())
    manager.set_system_config(
        SystemConfig(
            telemetry_url="",
            telemetry_collector_endpoint="",
            inference_service_urls=urls,
        )
    )
    return manager


class _NoClaims:
    def __init__(self, **kwargs):
        pass

    def extract(self, **kwargs):
        return []


@pytest.fixture
def ingest_graph_stage(monkeypatch):
    """The ingest path's KG stage with a recording graph store: runs the real
    ``_extract_graph_per_segment_inner`` with the real DocExtractor and
    CrossModalLinker for one transcript segment, and returns the extraction
    each tenant's GraphManager was asked to upsert."""
    from cogniverse_agents.graph.cross_modal_linker import CrossModalLinker
    from cogniverse_agents.graph.doc_extractor import DocExtractor
    from cogniverse_agents.graph.graph_manager import GraphManager
    from cogniverse_agents.graph.graph_schema import ExtractionResult
    from cogniverse_runtime.routers import ingestion

    async def _no_backrefs(**kwargs):
        return None

    monkeypatch.setattr(ingestion, "_write_backrefs_to_content", _no_backrefs)
    upserted: dict[str, list] = {}

    def _manager_for(tenant_id: str):
        mgr = MagicMock(spec=GraphManager)
        type(mgr).backend_resolver = property(lambda self: SimpleNamespace)
        mgr.upsert.side_effect = lambda linked: (
            upserted.setdefault(tenant_id, []).append(linked)
            or {
                "nodes_upserted": len(linked.nodes),
                "edges_upserted": len(linked.edges),
                "failed_ids": [],
            }
        )
        return mgr

    graph_router = SimpleNamespace(_graph_manager_factory=_manager_for)

    async def _run(tenant_id: str, config_manager, text: str) -> dict:
        return await ingestion._extract_graph_per_segment_inner(
            processing_results={
                "transcript": {"segments": [{"start": 0.0, "end": 2.0, "text": text}]}
            },
            source_doc_id=f"video-{tenant_id}",
            tenant_id=tenant_id,
            config_manager=config_manager,
            DocExtractor=DocExtractor,
            ClaimExtractor=_NoClaims,
            CrossModalLinker=CrossModalLinker,
            ExtractionResult=ExtractionResult,
            graph_router=graph_router,
        )

    return SimpleNamespace(run=_run, upserted=upserted)


@pytest.fixture
def process_config_names_another_gliner(monkeypatch):
    """Bind the process-wide ConfigManager singleton to a GLiNER service no
    job names: anything that reads it instead of the job's config lands here."""
    import cogniverse_foundation.config.utils as config_utils

    server, url, requests = _gliner_server("process singleton", "Organization")
    monkeypatch.setattr(
        config_utils, "_config_manager_singleton", _job_config_manager(url)
    )
    yield requests
    server.shutdown()


@pytest.mark.asyncio
async def test_ingest_graph_stage_extracts_entities_with_the_jobs_gliner(
    ingest_graph_stage, process_config_names_another_gliner
):
    server, url, requests = _gliner_server("Marie Curie", "Person")
    try:
        result = await ingest_graph_stage.run(
            "acme:curie",
            _job_config_manager(url),
            "Marie Curie discovered radium.",
        )
    finally:
        server.shutdown()

    assert [(path, payload["text"]) for path, payload in requests] == [
        ("/predict_entities", "Marie Curie discovered radium.")
    ]
    assert process_config_names_another_gliner == []
    (linked,) = ingest_graph_stage.upserted["acme:curie"]
    assert [(node.name, node.label) for node in linked.nodes] == [
        ("Marie Curie", "Person")
    ]
    assert result["nodes_upserted"] == 1
    assert result["backrefs_by_segment"] == {
        "seg_0": {
            "entity_ids": [linked.nodes[0].node_id],
            "relation_ids": [],
            "claim_ids": [],
        }
    }


@pytest.mark.asyncio
async def test_concurrent_ingest_graph_stages_each_reach_their_own_gliner(
    ingest_graph_stage, process_config_names_another_gliner
):
    first_server, first_url, first_requests = _gliner_server("Marie Curie", "Person")
    second_server, second_url, second_requests = _gliner_server("radium", "Substance")
    try:
        await asyncio.gather(
            ingest_graph_stage.run(
                "acme:first", _job_config_manager(first_url), "Marie Curie spoke."
            ),
            ingest_graph_stage.run(
                "acme:second", _job_config_manager(second_url), "Radium glowed."
            ),
        )
    finally:
        first_server.shutdown()
        second_server.shutdown()

    assert [payload["text"] for _, payload in first_requests] == ["Marie Curie spoke."]
    assert [payload["text"] for _, payload in second_requests] == ["Radium glowed."]
    assert process_config_names_another_gliner == []
    assert {
        tenant: [(node.name, node.label) for linked in batches for node in linked.nodes]
        for tenant, batches in ingest_graph_stage.upserted.items()
    } == {
        "acme:first": [("Marie Curie", "Person")],
        "acme:second": [("radium", "Substance")],
    }


@pytest.mark.asyncio
async def test_ingest_graph_stage_without_a_configured_gliner_fails_the_stage(
    ingest_graph_stage, process_config_names_another_gliner
):
    with pytest.raises(RuntimeError) as exc_info:
        await ingest_graph_stage.run(
            "acme:nogliner", _job_config_manager(None), "Marie Curie spoke."
        )

    assert str(exc_info.value) == (
        "knowledge_graph requires gliner in INFERENCE_SERVICE_URLS. "
        "Available: ['colbert_pylate']"
    )
    assert process_config_names_another_gliner == []
    assert ingest_graph_stage.upserted == {}


@pytest.mark.asyncio
async def test_worker_marks_graph_stage_retryable_when_boundary_fails(
    monkeypatch,
):
    scratch = []

    class Pipeline:
        def __init__(self, **kwargs):
            scratch.append(("retain", kwargs["retain_job_scratch"]))

        def release_retained_scratch(self):
            scratch.append(("release", None))

        async def process_video_async(self, path, source_uri=None):
            return {
                "status": "success",
                "video_id": "video-graph-fault",
                "results": {"video_id": "video-graph-fault"},
            }

    class Locator:
        def __init__(self, tenant_id, config):
            pass

        def localize(self, url):
            return "/tmp/video-graph-fault.mp4"

    async def graph_failure(**kwargs):
        raise ConnectionError("GLiNER sidecar reset the connection")

    monkeypatch.setattr(
        "cogniverse_runtime.ingestion.pipeline.VideoIngestionPipeline", Pipeline
    )
    monkeypatch.setattr("cogniverse_core.common.media.MediaLocator", Locator)
    monkeypatch.setattr(
        "cogniverse_runtime.routers.ingestion._extract_graph_per_segment",
        graph_failure,
    )
    monkeypatch.setattr(
        worker,
        "_prepare_job_context",
        lambda service_urls: (object(), object()),
    )
    job = queue.IngestJob(
        message_id="0-1",
        ingest_id="ing-graph-fault",
        source_url="s3://media/video-graph-fault.mp4",
        profile="video",
        tenant_id="acme:graph",
        sha="sha-graph-fault",
    )
    marked = []

    async def mark_graph_pending(pending_job):
        marked.append((pending_job.message_id, pending_job.ingest_id))

    with pytest.raises(
        worker.GraphStageIncomplete,
        match="graph extraction failed for ingest ing-graph-fault",
    ) as exc_info:
        await worker._default_processor(
            job,
            service_urls=None,
            mark_graph_pending=mark_graph_pending,
            graph_deadline_s=1,
        )

    assert marked == [("0-1", "ing-graph-fault")]
    # The keyframe files stay for the graph stage and go when it ends, even
    # when it fails.
    assert scratch == [("retain", True), ("release", None)]
    assert str(exc_info.value) == (
        "graph extraction failed for ingest ing-graph-fault: "
        "ConnectionError: GLiNER sidecar reset the connection"
    )
    assert isinstance(exc_info.value.__cause__, ConnectionError)
    assert str(exc_info.value.__cause__) == "GLiNER sidecar reset the connection"


@pytest.mark.asyncio
async def test_worker_treats_graph_marker_write_failure_as_retryable(monkeypatch):
    class Pipeline:
        def __init__(self, **kwargs):
            pass

        def release_retained_scratch(self):
            pass

        async def process_video_async(self, path, source_uri=None):
            return {
                "status": "success",
                "video_id": "video-marker-fault",
                "results": {"video_id": "video-marker-fault"},
            }

    class Locator:
        def __init__(self, tenant_id, config):
            pass

        def localize(self, url):
            return "/tmp/video-marker-fault.mp4"

    graph_calls = 0

    async def graph_extraction(**kwargs):
        nonlocal graph_calls
        graph_calls += 1
        return {"nodes_upserted": 1, "edges_upserted": 0, "graph_failed": 0}

    async def marker_failure(job):
        raise ConnectionError("Redis disconnected after content feed")

    monkeypatch.setattr(
        "cogniverse_runtime.ingestion.pipeline.VideoIngestionPipeline", Pipeline
    )
    monkeypatch.setattr("cogniverse_core.common.media.MediaLocator", Locator)
    monkeypatch.setattr(
        "cogniverse_runtime.routers.ingestion._extract_graph_per_segment",
        graph_extraction,
    )
    monkeypatch.setattr(
        worker,
        "_prepare_job_context",
        lambda service_urls: (object(), object()),
    )
    job = queue.IngestJob(
        message_id="0-2",
        ingest_id="ing-marker-fault",
        source_url="s3://media/video-marker-fault.mp4",
        profile="video",
        tenant_id="acme:graph",
        sha="sha-marker-fault",
    )

    with pytest.raises(worker.GraphStageIncomplete) as exc_info:
        await worker._default_processor(
            job,
            service_urls=None,
            mark_graph_pending=marker_failure,
            graph_deadline_s=1,
        )

    assert str(exc_info.value) == (
        "graph marker write failed after content feed for ingest ing-marker-fault"
    )
    assert isinstance(exc_info.value.__cause__, ConnectionError)
    assert str(exc_info.value.__cause__) == ("Redis disconnected after content feed")
    assert graph_calls == 0


@pytest.mark.asyncio
async def test_worker_retries_partial_graph_write_result(monkeypatch):
    class Pipeline:
        def __init__(self, **kwargs):
            pass

        def release_retained_scratch(self):
            pass

        async def process_video_async(self, path, source_uri=None):
            return {
                "status": "success",
                "video_id": "video-partial-graph",
                "results": {"video_id": "video-partial-graph"},
            }

    class Locator:
        def __init__(self, tenant_id, config):
            pass

        def localize(self, url):
            return "/tmp/video-partial-graph.mp4"

    async def partial_graph_write(**kwargs):
        return {"nodes_upserted": 2, "edges_upserted": 1, "graph_failed": 1}

    monkeypatch.setattr(
        "cogniverse_runtime.ingestion.pipeline.VideoIngestionPipeline", Pipeline
    )
    monkeypatch.setattr("cogniverse_core.common.media.MediaLocator", Locator)
    monkeypatch.setattr(
        "cogniverse_runtime.routers.ingestion._extract_graph_per_segment",
        partial_graph_write,
    )
    monkeypatch.setattr(
        worker,
        "_prepare_job_context",
        lambda service_urls: (object(), object()),
    )
    job = queue.IngestJob(
        message_id="0-3",
        ingest_id="ing-partial-graph",
        source_url="s3://media/video-partial-graph.mp4",
        profile="video",
        tenant_id="acme:graph",
        sha="sha-partial-graph",
    )
    marked = []

    async def mark_graph_pending(pending_job):
        marked.append((pending_job.message_id, pending_job.ingest_id))

    with pytest.raises(worker.GraphStageIncomplete) as exc_info:
        await worker._default_processor(
            job,
            service_urls=None,
            mark_graph_pending=mark_graph_pending,
            graph_deadline_s=1,
        )

    assert str(exc_info.value) == (
        "graph extraction left 1 failed writes for ingest ing-partial-graph"
    )
    assert marked == [("0-3", "ing-partial-graph")]


def _held_deploy_lease_wait():
    """Wait for the deploy lease a live peer holds, as a nested schema deploy
    does, until the wait runs out."""
    from cogniverse_core.registries.schema_deploy_lease import SchemaDeployLease
    from tests.utils.memory_store import InMemoryConfigStore

    store = InMemoryConfigStore()
    peer = SchemaDeployLease(store)
    assert peer.acquire() is peer
    try:
        SchemaDeployLease(store, wait_seconds=0.5).acquire()
    finally:
        peer.release()


@pytest.mark.asyncio
async def test_a_deploy_lease_wait_in_the_graph_stage_is_not_read_as_its_deadline(
    monkeypatch,
):
    class Pipeline:
        def __init__(self, **kwargs):
            pass

        def release_retained_scratch(self):
            pass

        async def process_video_async(self, path, source_uri=None):
            return {
                "status": "success",
                "video_id": "video-graph-lease",
                "results": {"video_id": "video-graph-lease"},
            }

    class Locator:
        def __init__(self, tenant_id, config):
            pass

        def localize(self, url):
            return "/tmp/video-graph-lease.mp4"

    async def graph_waits_for_the_deploy_lease(**kwargs):
        _held_deploy_lease_wait()

    monkeypatch.setattr(
        "cogniverse_runtime.ingestion.pipeline.VideoIngestionPipeline", Pipeline
    )
    monkeypatch.setattr("cogniverse_core.common.media.MediaLocator", Locator)
    monkeypatch.setattr(
        "cogniverse_runtime.routers.ingestion._extract_graph_per_segment",
        graph_waits_for_the_deploy_lease,
    )
    monkeypatch.setattr(
        worker,
        "_prepare_job_context",
        lambda service_urls: (object(), object()),
    )
    job = queue.IngestJob(
        message_id="0-2",
        ingest_id="ing-graph-lease",
        source_url="s3://media/video-graph-lease.mp4",
        profile="video",
        tenant_id="acme:graph",
        sha="sha-graph-lease",
    )

    async def mark_graph_pending(pending_job):
        return None

    with pytest.raises(
        worker.GraphStageIncomplete,
        match="graph extraction failed for ingest ing-graph-lease",
    ) as exc_info:
        await worker._default_processor(
            job,
            service_urls=None,
            mark_graph_pending=mark_graph_pending,
            graph_deadline_s=30,
        )

    cause = exc_info.value.__cause__
    assert isinstance(cause, TimeoutError)
    assert type(cause).__name__ == "LeaseWaitTimeout"
    assert "Vespa deployment lease still held by" in str(cause)
    assert str(exc_info.value) == (
        f"graph extraction failed for ingest ing-graph-lease: LeaseWaitTimeout: {cause}"
    )
