"""
Pytest configuration and fixtures for evaluation framework tests.
"""

import json
import logging

# Add project root to path
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pandas as pd
import pytest
import requests
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

eval_logger = logging.getLogger(__name__)

# Path to schema JSON files
EVAL_SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "configs" / "schemas"
EVAL_COLPALI_MODEL = "vidore/colsmol-500m"
EVAL_TENANT_SCHEMA = "video_colpali_smol500_mv_frame_default"


@pytest.fixture(scope="function")
def phoenix_test_server():
    """Start Phoenix Docker container for integration tests."""
    import os
    import subprocess
    import tempfile
    import time

    from cogniverse_evaluation.providers.registry import get_evaluation_registry
    from cogniverse_foundation.telemetry.manager import TelemetryManager

    container_name = f"phoenix_eval_test_{int(time.time() * 1000)}"

    # Clear evaluation provider cache before starting so stale instances don't leak
    get_evaluation_registry().clear_cache()

    # Clean up old containers
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "-q", "--filter", "name=phoenix_eval_test"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.stdout.strip():
            old_containers = result.stdout.strip().split("\n")
            for container_id in old_containers:
                subprocess.run(
                    ["docker", "rm", "-f", container_id],
                    capture_output=True,
                    timeout=10,
                )
    except Exception:
        pass

    try:
        # Create temporary directory for Phoenix data
        test_data_dir = os.path.join(
            tempfile.gettempdir(), f"phoenix_test_{int(time.time())}"
        )
        os.makedirs(test_data_dir, exist_ok=True)

        # Start Phoenix container
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container_name,
                "-p",
                "26006:6006",  # HTTP port
                "-p",
                "24317:4317",  # gRPC port
                "-v",
                f"{test_data_dir}:/phoenix_data",
                "-e",
                "PHOENIX_WORKING_DIR=/phoenix_data",
                "-e",
                "PHOENIX_SQL_DATABASE_URL=sqlite:////phoenix_data/phoenix.db",
                "arizephoenix/phoenix:14.2.1",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Wait for Phoenix to be ready
        import requests

        for _ in range(60):
            try:
                response = requests.get("http://localhost:26006", timeout=1)
                if response.status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.5)

        # Create server object
        class PhoenixServer:
            base_url = "http://localhost:26006"
            port = 26006

        yield PhoenixServer()

    finally:
        # Cleanup
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
        TelemetryManager.reset()
        from cogniverse_evaluation.providers.registry import get_evaluation_registry

        get_evaluation_registry().clear_cache()


@pytest.fixture
def phoenix_client(phoenix_test_server):
    """Get Phoenix client connected to test server."""
    from phoenix.client import Client

    return Client(base_url=phoenix_test_server.base_url)


@pytest.fixture
def mock_phoenix_client():
    """Mock Phoenix client for testing.

    Patches phoenix.client.Client (used in task.py) to return
    mock datasets with to_dataframe() returning a proper DataFrame.
    """
    with patch("phoenix.client.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock dataset with to_dataframe() returning DataFrame
        # task.py calls: sync_client.datasets.get_dataset(dataset=...).to_dataframe()
        mock_dataset = MagicMock()
        mock_dataset.id = "test_dataset_id"
        mock_dataset.name = "test_dataset"
        mock_dataset.examples = [
            MagicMock(
                id="example1",
                input={"query": "test query 1", "category": "test"},
                output={
                    "expected_videos": ["item1", "item2"],
                    "expected_items": ["item1", "item2"],
                },
            ),
            MagicMock(
                id="example2",
                input={"query": "test query 2", "category": "test"},
                output={"expected_videos": ["item3"], "expected_items": ["item3"]},
            ),
        ]
        # to_dataframe() returns Phoenix nested input format
        mock_dataset.to_dataframe.return_value = pd.DataFrame(
            [
                {
                    "input": {
                        "query": "test query 1",
                        "expected_videos": "item1,item2",
                        "query_type": "test",
                    }
                },
                {
                    "input": {
                        "query": "test query 2",
                        "expected_videos": "item3",
                        "query_type": "test",
                    }
                },
            ]
        )
        mock_client.datasets.get_dataset.return_value = mock_dataset
        mock_client.datasets.create_dataset.return_value = mock_dataset

        # Mock spans dataframe
        mock_df = pd.DataFrame(
            [
                {
                    "trace_id": "trace1",
                    "attributes.input.value": "test query 1",
                    "attributes.output.value": [{"item_id": "item1", "score": 0.9}],
                    "attributes.metadata.profile": "test_profile",
                    "attributes.metadata.strategy": "test_strategy",
                    "timestamp": datetime.now().isoformat(),
                    "duration_ms": 100,
                }
            ]
        )
        mock_client.spans.get_spans_dataframe.return_value = mock_df

        yield mock_client


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        {
            "query": "person wearing red shirt",
            "expected_items": ["item1", "item2"],
            "category": "visual",
        },
        {
            "query": "what happened after the meeting",
            "expected_items": ["item3"],
            "category": "temporal",
        },
        {
            "query": "dog playing in the park",
            "expected_items": ["item4", "item5"],
            "category": "activity",
        },
    ]


@pytest.fixture
def sample_results():
    """Sample search results for testing."""
    return [
        {
            "item_id": "item1",
            "score": 0.9,
            "rank": 1,
            "document_id": "item1_part_0",
            "content": "person in red shirt walking",
            "temporal_info": {"timestamp": 100},
            "metadata": {},
        },
        {
            "item_id": "item2",
            "score": 0.8,
            "rank": 2,
            "document_id": "item2_part_1",
            "content": "another person with red clothing",
            "temporal_info": {"timestamp": 200},
            "metadata": {},
        },
        {
            "item_id": "item1",  # Duplicate for diversity testing
            "score": 0.7,
            "rank": 3,
            "document_id": "item1_part_5",
            "content": "same person later",
            "temporal_info": {"timestamp": 150},
            "metadata": {},
        },
    ]


@pytest.fixture
def mock_inspect_state():
    """Mock Inspect AI state for testing scorers."""
    state = MagicMock()
    state.input = {"query": "test query"}
    state.outputs = {
        "test_profile_test_strategy": {
            "results": [
                {"video_id": "video1", "score": 0.9, "content": "test content"},
                {"video_id": "video2", "score": 0.8, "content": "more content"},
                {"video_id": "video1", "score": 0.7, "content": "duplicate"},
            ],
            "profile": "test_profile",
            "strategy": "test_strategy",
            "success": True,
        }
    }
    return state


@pytest.fixture
def temp_csv_file(tmp_path):
    """Create a temporary CSV file for testing."""
    csv_file = tmp_path / "test_queries.csv"
    csv_content = """query,expected_videos,category
person wearing red shirt,"video1,video2",visual
what happened after the meeting,video3,temporal
dog playing in the park,"video4,video5",activity"""
    csv_file.write_text(csv_content)
    return str(csv_file)


@pytest.fixture
def temp_json_file(tmp_path, sample_queries):
    """Create a temporary JSON file for testing."""
    json_file = tmp_path / "test_queries.json"
    json_file.write_text(json.dumps(sample_queries))
    return str(json_file)


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "use_ragas": True,
        "ragas_metrics": ["context_relevancy"],
        "use_custom": True,
        "custom_metrics": ["diversity", "temporal_coherence"],
        "use_visual": False,
        "top_k": 10,
    }


@pytest.fixture
def mock_get_config():
    """Mock get_config function."""
    with patch("cogniverse_agents.tools.config.get_config") as mock_config:
        mock_config.return_value = {
            "backend_url": "http://localhost",
            "backend_port": 8080,
            "schema_name": "video_frame",
        }
        yield mock_config


@pytest.fixture
def mock_evaluator_provider(mock_phoenix_client):
    """Mock evaluator provider for testing."""
    with patch(
        "cogniverse_evaluation.providers.get_evaluation_provider"
    ) as mock_get_provider:
        # Create mock provider structure
        mock_provider = MagicMock()

        # Mock telemetry provider with datasets and traces
        mock_telemetry = MagicMock()

        # Mock datasets interface - async method
        async def mock_get_dataset(name):
            dataset = mock_phoenix_client.get_dataset(name)
            if dataset is None:
                return None
            return {
                "id": dataset.id,
                "name": dataset.name,
                "examples": [
                    {
                        "id": ex.id,
                        "input": ex.input,
                        "output": ex.output,
                    }
                    for ex in dataset.examples
                ],
            }

        mock_datasets = MagicMock()
        mock_datasets.get_dataset = mock_get_dataset
        mock_telemetry.datasets = mock_datasets

        # Mock traces interface - async method
        async def mock_get_spans(**kwargs):
            return mock_phoenix_client.get_spans_dataframe(**kwargs)

        mock_traces = MagicMock()
        mock_traces.get_spans = mock_get_spans
        mock_telemetry.traces = mock_traces

        # Attach to provider
        mock_provider.telemetry = mock_telemetry

        # Set return value
        mock_get_provider.return_value = mock_provider

        yield mock_provider


@pytest.fixture(autouse=True)
def mock_provider_for_unit_tests(request):
    """Auto-mock provider for unit tests that don't explicitly need a real provider."""
    # Only apply to unit tests, skip integration tests
    if "unit" in request.keywords or (
        "integration" not in request.keywords and "phoenix" not in request.keywords
    ):
        from unittest.mock import AsyncMock, MagicMock, patch

        # Mock the provider to prevent initialization errors
        with patch(
            "cogniverse_evaluation.providers.get_evaluation_provider"
        ) as mock_get:
            mock_provider = MagicMock()

            # Mock Phoenix evaluator base class

            class MockPhoenixEvaluator:
                pass

            # Mock framework
            mock_framework = MagicMock()
            mock_framework.get_evaluator_base_class.return_value = MockPhoenixEvaluator
            mock_framework.get_evaluation_result_type.return_value = dict

            def mock_create_result(score, label, explanation, metadata=None):
                return MagicMock(
                    score=score,
                    label=label,
                    explanation=explanation,
                    metadata=metadata or {},
                )

            mock_framework.create_evaluation_result = mock_create_result
            mock_provider.framework = mock_framework

            mock_telemetry = MagicMock()

            mock_datasets = MagicMock()

            async def mock_get_dataset(name):
                return None

            mock_datasets.get_dataset = AsyncMock(side_effect=mock_get_dataset)
            mock_telemetry.datasets = mock_datasets

            mock_traces = MagicMock()

            async def mock_get_spans(**kwargs):
                return pd.DataFrame()

            mock_traces.get_spans = AsyncMock(side_effect=mock_get_spans)
            mock_telemetry.traces = mock_traces

            mock_provider.telemetry = mock_telemetry

            mock_get.return_value = mock_provider
            yield
    else:
        yield


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    yield


def _embeddings_to_vespa_tensors(embeddings: np.ndarray):
    """Convert (num_patches, 128) float32 embeddings to Vespa tensor dict format."""
    float_dict = {str(idx): vector.tolist() for idx, vector in enumerate(embeddings)}
    binarized = np.packbits(
        np.where(embeddings > 0, 1, 0).astype(np.uint8), axis=1
    ).astype(np.int8)
    binary_dict = {str(idx): vector.tolist() for idx, vector in enumerate(binarized)}
    return float_dict, binary_dict


@pytest.fixture(scope="module")
def eval_vespa_instance():
    """Start Vespa Docker container for evaluation integration tests.

    Deploys metadata + data schemas in a single application package.
    Module-scoped to share across all tests in the integration module.
    """
    # Import lazily to avoid loading for unit tests
    import cogniverse_vespa  # noqa: F401
    from cogniverse_core.registries.backend_registry import BackendRegistry
    from tests.utils.vespa_docker import VespaDockerManager

    manager = VespaDockerManager()

    BackendRegistry._instance = None
    BackendRegistry._backend_instances.clear()

    try:
        container_info = manager.start_container(
            module_name="evaluation_integration_tests",
            use_module_ports=True,
        )
        manager.wait_for_config_ready(container_info, timeout=180)

        eval_logger.info("Waiting 15s for Vespa internal services...")
        time.sleep(15)

        # Deploy metadata + data schemas
        from vespa.package import ApplicationPackage

        from cogniverse_vespa.json_schema_parser import JsonSchemaParser
        from cogniverse_vespa.metadata_schemas import (
            create_adapter_registry_schema,
            create_config_metadata_schema,
            create_organization_metadata_schema,
            create_tenant_metadata_schema,
        )

        metadata_schemas = [
            create_organization_metadata_schema(),
            create_tenant_metadata_schema(),
            create_config_metadata_schema(),
            create_adapter_registry_schema(),
        ]

        schema_file = EVAL_SCHEMAS_DIR / "video_colpali_smol500_mv_frame_schema.json"
        with open(schema_file) as f:
            schema_json = json.load(f)
        schema_json["name"] = EVAL_TENANT_SCHEMA
        schema_json["document"]["name"] = EVAL_TENANT_SCHEMA

        parser = JsonSchemaParser()
        data_schema = parser.parse_schema(schema_json)

        all_schemas = metadata_schemas + [data_schema]
        app_package = ApplicationPackage(name="cogniverse", schema=all_schemas)

        from cogniverse_vespa.vespa_schema_manager import VespaSchemaManager

        schema_manager = VespaSchemaManager(
            backend_endpoint="http://localhost",
            backend_port=container_info["config_port"],
        )
        schema_manager._deploy_package(app_package)

        manager.wait_for_application_ready(container_info, timeout=120)

        eval_logger.info("Evaluation Vespa ready")
        yield container_info

    except Exception as e:
        eval_logger.error(f"Failed to start evaluation Vespa: {e}")
        pytest.skip(f"Failed to start evaluation Vespa: {e}")

    finally:
        manager.stop_container()
        try:
            BackendRegistry._instance = None
            BackendRegistry._backend_instances.clear()
        except Exception as cleanup_err:
            eval_logger.warning(f"BackendRegistry cleanup failed: {cleanup_err}")


@pytest.fixture(scope="module")
def eval_colpali_model():
    """Load ColPali model once for evaluation integration tests."""
    from cogniverse_core.common.models import get_or_load_model
    from cogniverse_core.query.encoders import QueryEncoderFactory

    config = {
        "colpali_model": EVAL_COLPALI_MODEL,
        "embedding_type": "multi_vector",
        "model_loader": "colpali",
    }
    model, processor = get_or_load_model(EVAL_COLPALI_MODEL, config, eval_logger)
    device = next(model.parameters()).device

    yield model, processor, device

    QueryEncoderFactory._encoder_cache.clear()


@pytest.fixture(scope="module")
def eval_seeded_documents(eval_vespa_instance, eval_colpali_model):
    """Feed real ColPali-embedded documents into Vespa for evaluation tests."""
    model, processor, device = eval_colpali_model

    test_docs = [
        {
            "color": (255, 0, 0),
            "title": "Red sunset landscape",
            "video_id": "sunset_vid",
        },
        {
            "color": (0, 0, 255),
            "title": "Ocean waves coastal scene",
            "video_id": "ocean_vid",
        },
        {
            "color": (0, 128, 0),
            "title": "Forest trail nature walk",
            "video_id": "forest_vid",
        },
    ]

    http_port = eval_vespa_instance["http_port"]

    for i, doc_info in enumerate(test_docs):
        img = Image.new("RGB", (224, 224), color=doc_info["color"])
        batch_inputs = processor.process_images([img]).to(device)
        with torch.no_grad():
            doc_embeddings = model(**batch_inputs)
        embeddings_np = doc_embeddings.squeeze(0).cpu().float().numpy()

        float_dict, binary_dict = _embeddings_to_vespa_tensors(embeddings_np)

        doc_id = f"eval_test_doc_{i}"
        vespa_doc = {
            "fields": {
                "video_id": doc_info["video_id"],
                "video_title": doc_info["title"],
                "segment_id": 0,
                "start_time": 0.0,
                "end_time": 5.0,
                "segment_description": doc_info["title"],
                "audio_transcript": "",
                "embedding": float_dict,
                "embedding_binary": binary_dict,
            }
        }

        resp = requests.post(
            f"http://localhost:{http_port}/document/v1/video/{EVAL_TENANT_SCHEMA}/docid/{doc_id}",
            json=vespa_doc,
            timeout=10,
        )
        assert resp.status_code in [200, 201], (
            f"Failed to feed eval doc {doc_id}: {resp.status_code}: {resp.text[:200]}"
        )

    time.sleep(5)

    yield test_docs

    for i in range(len(test_docs)):
        doc_id = f"eval_test_doc_{i}"
        try:
            requests.delete(
                f"http://localhost:{http_port}/document/v1/video/{EVAL_TENANT_SCHEMA}/docid/{doc_id}",
                timeout=5,
            )
        except Exception:
            pass


@pytest.fixture(scope="module")
def eval_search_client(eval_vespa_instance, eval_seeded_documents, phoenix_container):
    """FastAPI TestClient with real search router wired to test Vespa.

    Provides a TestClient that can execute real searches using ColPali encoder
    against the test Vespa instance with seeded documents.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import (
        BackendProfileConfig,
        SystemConfig,
    )
    from cogniverse_runtime.routers import search
    from cogniverse_vespa.config.config_store import VespaConfigStore

    store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=eval_vespa_instance["http_port"],
    )
    cm = ConfigManager(store=store)
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=eval_vespa_instance["http_port"],
        )
    )
    cm.add_backend_profile(
        BackendProfileConfig(
            profile_name="test_colpali",
            type="video",
            schema_name="video_colpali_smol500_mv_frame",
            embedding_model=EVAL_COLPALI_MODEL,
        ),
    )

    schema_loader = FilesystemSchemaLoader(EVAL_SCHEMAS_DIR)

    app = FastAPI()
    app.include_router(search.router, prefix="/search")
    app.dependency_overrides[search.get_config_manager_dependency] = lambda: cm
    app.dependency_overrides[search.get_schema_loader_dependency] = lambda: (
        schema_loader
    )

    import os

    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()

    telemetry_config = TelemetryConfig(
        otlp_endpoint=os.getenv("TELEMETRY_OTLP_ENDPOINT", "localhost:4317"),
        provider_config={
            "http_endpoint": "http://localhost:16006",
            "grpc_endpoint": "http://localhost:14317",
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    tm = TelemetryManager(config=telemetry_config)
    telemetry_manager_module._telemetry_manager = tm

    with TestClient(app) as client:
        yield client

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()


@contextmanager
def intercept_search_calls(test_client):
    """Context manager that routes httpx.post calls through a TestClient.

    The evaluation solver makes httpx.post calls to the runtime search API.
    This interceptor routes those calls through the TestClient so they hit
    the real search router (with real encoder + real Vespa) without needing
    a running HTTP server.
    """
    original_post = httpx.post

    def patched_post(url, **kwargs):
        if "/search" in url:
            resp = test_client.post("/search/", json=kwargs.get("json"))
            return httpx.Response(
                status_code=resp.status_code,
                content=resp.content,
                headers=dict(resp.headers),
                request=httpx.Request("POST", url),
            )
        return original_post(url, **kwargs)

    with patch("httpx.post", side_effect=patched_post):
        yield


@pytest.fixture(scope="module")
def search_evaluator_provider(phoenix_container):
    """Real evaluator provider backed by Phoenix Docker.

    Uploads a test dataset to real Phoenix, then configures
    get_evaluation_provider() to return a real PhoenixEvaluationProvider
    wired to the test Phoenix instance.

    Sets up TelemetryManager singleton with test Phoenix endpoints so that
    PhoenixEvaluationProvider.initialize() picks up the right endpoints
    (instead of falling back to localhost:6006 when VespaConfigStore is
    unreachable at localhost:8080).
    """
    import os

    from phoenix.client import Client

    import cogniverse_foundation.telemetry.manager as telemetry_manager_module
    from cogniverse_evaluation.providers.registry import (
        get_evaluation_registry,
        set_evaluation_provider,
    )
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry
    from cogniverse_telemetry_phoenix.evaluation.evaluation_provider import (
        PhoenixEvaluationProvider,
    )

    phoenix_endpoint = "http://localhost:16006"
    grpc_endpoint = "http://localhost:14317"

    # Set up TelemetryManager singleton with test Phoenix endpoints.
    # PhoenixEvaluationProvider.initialize() calls get_telemetry_manager()
    # and reads provider_config for endpoint resolution. Without this,
    # it falls back to localhost:6006 when VespaConfigStore is unreachable.
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
    get_evaluation_registry().clear_cache()

    telemetry_config = TelemetryConfig(
        otlp_endpoint=os.getenv("TELEMETRY_OTLP_ENDPOINT", grpc_endpoint),
        provider_config={
            "http_endpoint": phoenix_endpoint,
            "grpc_endpoint": grpc_endpoint,
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    tm = TelemetryManager(config=telemetry_config)
    telemetry_manager_module._telemetry_manager = tm

    sync_client = Client(base_url=phoenix_endpoint)

    # Upload real test dataset to Phoenix (idempotent — skip if already exists)
    test_df = pd.DataFrame(
        [
            {
                "query": "sunset landscape mountains",
                "expected_videos": "sunset_vid",
                "query_type": "visual",
            },
            {
                "query": "ocean waves coastal",
                "expected_videos": "ocean_vid",
                "query_type": "visual",
            },
        ]
    )
    try:
        sync_client.datasets.create_dataset(
            name="test_dataset",
            dataframe=test_df,
            input_keys=["query", "expected_videos", "query_type"],
            output_keys=[],
        )
    except Exception as upload_err:
        if "already exists" not in str(upload_err):
            raise

    provider = PhoenixEvaluationProvider()
    provider.initialize(
        {
            "tenant_id": "test:unit",
            "http_endpoint": phoenix_endpoint,
            "grpc_endpoint": grpc_endpoint,
        }
    )
    set_evaluation_provider(provider)

    yield provider

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()
    get_evaluation_registry().clear_cache()
