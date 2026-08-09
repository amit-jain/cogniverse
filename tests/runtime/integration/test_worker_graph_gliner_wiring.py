"""The ingestion-worker graph path uses its captured GLiNER endpoint."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cogniverse_agents.graph.graph_schema import Mention
from cogniverse_core.common.models.model_loaders import RemoteColBERTLoader
from cogniverse_runtime.ingestion_worker import queue, worker
from cogniverse_runtime.ingestion_worker.worker import _build_worker_graph_factory

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


@pytest.mark.asyncio
async def test_worker_marks_graph_stage_retryable_when_boundary_fails(
    monkeypatch,
):
    class Pipeline:
        def __init__(self, **kwargs):
            pass

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
    assert isinstance(exc_info.value.__cause__, ConnectionError)
    assert str(exc_info.value.__cause__) == "GLiNER sidecar reset the connection"


@pytest.mark.asyncio
async def test_worker_treats_graph_marker_write_failure_as_retryable(monkeypatch):
    class Pipeline:
        def __init__(self, **kwargs):
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
