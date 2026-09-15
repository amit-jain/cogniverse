"""Token-window ingestion and source retrieval against owned PyLate and Vespa."""

import json
import logging
import os
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import requests
from vespa.application import Vespa

from cogniverse_core.common.models.model_loaders import RemoteColBERTLoader
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_runtime.ingestion.processors.embedding_generator.embedding_generator_impl import (
    EmbeddingGeneratorImpl,
)
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.ingestion.integration.test_upload_via_queue import (
    _deploy_metadata_schemas,
    _install_test_vespa_disk_limit,
    _paired_free_ports,
    _wait_for_config_port,
    _wait_for_data_port,
    _wait_for_schema_ready,
)

pytestmark = pytest.mark.integration
MODEL = "lightonai/LateOn"
REVISION = "c01907b70557ee5c7753680d4819a5cce1674b83"


@contextmanager
def owned_container(image, args):
    name = f"ingestion-windows-{uuid.uuid4().hex[:10]}"
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--label",
            f"cogniverse-test-owner-pid={os.getpid()}",
            *args,
            image,
        ],
        check=True,
        capture_output=True,
    )
    try:
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=True, capture_output=True)


@pytest.fixture(scope="module")
def window_encoder():
    server = Path("libs/cli/cogniverse_cli/modal_inference/servers/pylate.py").resolve()
    cache = Path.home() / ".cache/cogniverse-tests/huggingface"
    with owned_container(
        "cogniverse/pylate:0.1.0-dev",
        [
            "-p",
            "127.0.0.1::8080",
            "-v",
            f"{server}:/app/server.py:ro",
            "-v",
            f"{cache}:/hf:ro",
            "-e",
            "HF_HOME=/hf",
            "-e",
            "HF_HUB_OFFLINE=1",
            "-e",
            f"MODEL_NAME={MODEL}",
            "-e",
            f"MODEL_REVISION={REVISION}",
            "-e",
            "DEVICE=cpu",
            "-e",
            "OMP_NUM_THREADS=2",
        ],
    ) as name:
        port = (
            subprocess.check_output(["docker", "port", name, "8080"], text=True)
            .strip()
            .rsplit(":", 1)[1]
        )
        endpoint = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                response = requests.get(f"{endpoint}/health", timeout=5)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)
        else:
            logs = subprocess.check_output(["docker", "logs", name], text=True)
            pytest.fail(f"PyLate did not become ready: {logs}")
        model, _ = RemoteColBERTLoader(
            MODEL, {"remote_inference_url": endpoint}, _resolved_headers={}
        ).load_model()
        try:
            yield model
        finally:
            model._close()


@pytest.fixture(scope="module")
def window_backend():
    http_port = _paired_free_ports()
    config_port = http_port + 10991
    with owned_container(
        "vespaengine/vespa:8.668.5",
        [
            "-p",
            f"127.0.0.1:{http_port}:8080",
            "-p",
            f"127.0.0.1:{config_port}:19071",
            "--memory",
            "6g",
        ],
    ):
        assert _wait_for_config_port(config_port) is True
        _deploy_metadata_schemas(config_port)
        assert _wait_for_data_port(http_port) is True
        assert _wait_for_schema_ready(http_port, "tenant_metadata") is True
        with pytest.MonkeyPatch.context() as patch:
            _install_test_vespa_disk_limit(patch)
            cm = ConfigManager(
                store=VespaConfigStore(
                    backend_url="http://localhost",
                    backend_port=http_port,
                )
            )
            cm.set_system_config(
                SystemConfig(backend_url="http://localhost", backend_port=http_port)
            )
            tenant = f"window{uuid.uuid4().hex[:8]}"
            config = json.loads(Path("configs/config.json").read_text())["backend"]
            config.update(
                url="http://localhost", port=http_port, config_port=config_port
            )
            backend = BackendRegistry.get_instance().get_ingestion_backend(
                name="vespa",
                tenant_id=tenant,
                config={"backend": config},
                config_manager=cm,
                schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
            )
            backend.schema_registry.deploy_schema(
                tenant_id=tenant, base_schema_name="document_text"
            )
            schema = backend.get_tenant_schema_name(tenant, "document_text")
            assert _wait_for_schema_ready(http_port, schema) is True
            yield (
                backend,
                schema,
                Vespa(url=f"http://127.0.0.1:{http_port}"),
                config["profiles"]["document_text_semantic"],
                tenant,
            )


def generator(backend, profile, model):
    config = dict(profile, remote_inference_url=model.endpoint_url)
    result = EmbeddingGeneratorImpl(config, logging.getLogger(__name__), backend)
    result.colbert_model = model
    return result


def ingest(gen, source, text):
    return gen._process_document_segments(
        {"video_id": source, "source_url": f"s3://corpus/{source}.txt"},
        [
            {
                "document_id": source,
                "filename": f"{source}.txt",
                "path": f"/{source}.txt",
                "document_type": "txt",
                "extracted_text": text,
            }
        ],
    )


def rows(app, schema, source):
    response = app.query(
        body={
            "yql": f'select * from {schema} where document_id contains "{source}"',
            "hits": 400,
        }
    )
    assert response.is_successful() is True
    return sorted(
        [hit["fields"] for hit in response.hits], key=lambda row: row["chunk_start"]
    )


def model_windows(encoder, text):
    """The served model's own split of ``text``, and its document window."""
    response = requests.post(
        f"{encoder.endpoint_url}/windows",
        json={"input": [text], "model": encoder.model_name},
        timeout=120,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    return (
        [(int(start), int(end)) for start, end in payload["data"][0]["spans"]],
        payload["document_length"],
    )


def test_document_windows_cover_the_source_and_suffix_search_ranks_it(
    window_backend, window_encoder
):
    backend, schema, app, profile, tenant = window_backend
    gen = generator(backend, profile, window_encoder)
    prefix = "background " * 650
    sources = {
        "orchard": prefix
        + "Orchard tractors harvest ripe apples from the fruit trees.",
        "orbit": prefix
        + "Orbital rockets launch astronauts aboard spacecraft to the moon.",
    }
    for source, text in sources.items():
        spans, document_length = model_windows(window_encoder, text)
        # The corpus has to exceed one window or the defect cannot appear:
        # encoding the whole text returns exactly the model's capped sequence.
        assert len(window_encoder.encode([text], is_query=False)[0]) == document_length
        assert spans[0][0] == 0
        assert spans[-1][1] == len(text)
        assert [end for _, end in spans[:-1]] == [start for start, _ in spans[1:]]

        result = ingest(gen, source, text)
        assert (
            result.total_documents,
            result.documents_processed,
            result.documents_fed,
            result.errors,
        ) == (len(spans), len(spans), len(spans), [])

        stored = rows(app, schema, source)
        assert [(row["chunk_start"], row["chunk_end"]) for row in stored] == spans
        assert [row["chunk_index"] for row in stored] == list(range(len(spans)))
        assert [row["chunk_count"] for row in stored] == [len(spans)] * len(spans)
        assert [row["document_id"] for row in stored] == [source] * len(spans)
        assert "".join(row["full_text"] for row in stored) == text

    for query, expected in [
        ("harvesting apples with a tractor", "orchard"),
        ("astronauts flying a rocket to the moon", "orbit"),
    ]:
        embeddings = np.asarray(
            window_encoder.encode([query], is_query=True)[0], dtype=np.float32
        )
        results = backend.search(
            {
                "query": query,
                "type": profile["type"],
                "query_embeddings": embeddings,
                "profile": "document_text_semantic",
                "tenant_id": tenant,
                "top_k": 2,
            }
        )
        assert [hit.document.metadata["source_id"] for hit in results] == [
            expected,
            "orbit" if expected == "orchard" else "orchard",
        ]
        assert results.result_granularity == "source"


def test_concurrent_long_sources_keep_their_own_windows(window_backend, window_encoder):
    backend, schema, app, profile, tenant = window_backend
    gen = generator(backend, profile, window_encoder)
    sources = {
        "parallelone": "background " * 650 + "orchard tractors harvest apples",
        "paralleltwo": "background " * 650 + "orbital rockets carry astronauts",
    }
    expected = {
        source: model_windows(window_encoder, text)[0]
        for source, text in sources.items()
    }
    barrier = threading.Barrier(2, timeout=60)

    def run(item):
        barrier.wait()
        return ingest(gen, *item)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, sources.items()))
    assert [(r.documents_fed, r.errors) for r in results] == [
        (len(expected[source]), []) for source in sources
    ]
    for source, text in sources.items():
        stored = rows(app, schema, source)
        assert [(row["chunk_start"], row["chunk_end"]) for row in stored] == expected[
            source
        ]
        assert "".join(row["full_text"] for row in stored) == text


@contextmanager
def stub_windows_service(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def stub_generator(endpoint):
    gen = EmbeddingGeneratorImpl(
        {
            "embedding_model": MODEL,
            "embedding_type": "multi_vector",
            "model_loader": "colbert",
            "schema_name": "document_text",
            "inference_services": {"embedding": "colbert_pylate"},
            "remote_inference_url": endpoint,
        },
        logging.getLogger(__name__),
        None,
    )
    return gen


def _json_handler(status, body):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

    return Handler


def test_window_service_failure_fails_the_document_instead_of_truncating():
    with stub_windows_service(
        _json_handler(500, {"detail": "pylate: windowing failed"})
    ) as endpoint:
        gen = stub_generator(endpoint)
        result = gen._process_document_segments(
            {"video_id": "faulted"},
            [{"document_id": "faulted", "extracted_text": "background " * 650}],
        )
        assert (
            result.total_documents,
            result.documents_processed,
            result.documents_fed,
        ) == (1, 0, 0)
        assert result.errors == [
            "Document 0: remote ColBERT windowing failed for model "
            f"{MODEL!r} at {endpoint}"
        ]


def test_partial_window_coverage_fails_the_document():
    text = "background " * 650
    with stub_windows_service(
        _json_handler(
            200,
            {
                "object": "list",
                "data": [{"index": 0, "spans": [[0, 100]]}],
                "model": MODEL,
                "document_length": 300,
                "window_tokens": 293,
            },
        )
    ) as endpoint:
        gen = stub_generator(endpoint)
        result = gen._process_document_segments(
            {"video_id": "clipped"},
            [{"document_id": "clipped", "extracted_text": text}],
        )
        assert (
            result.total_documents,
            result.documents_processed,
            result.documents_fed,
        ) == (1, 0, 0)
        assert result.errors == [
            "Document 0: remote ColBERT windowing returned spans covering "
            f"0..100 of {len(text)} characters from model {MODEL!r} at {endpoint}"
        ]
