"""
Pytest configuration and fixtures for evaluation framework tests.
"""

import json

# Add project root to path
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Standardized Phoenix Docker container fixture for integration tests
@pytest.fixture(scope="function")
def phoenix_test_server():
    """Start Phoenix Docker container for integration tests."""
    import os
    import subprocess
    import tempfile
    import time

    from cogniverse_foundation.telemetry.manager import TelemetryManager

    container_name = f"phoenix_eval_test_{int(time.time() * 1000)}"

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
                subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=10)
    except Exception:
        pass

    try:
        # Create temporary directory for Phoenix data
        test_data_dir = os.path.join(tempfile.gettempdir(), f"phoenix_test_{int(time.time())}")
        os.makedirs(test_data_dir, exist_ok=True)

        # Start Phoenix container
        subprocess.run(
            [
                "docker", "run", "-d",
                "--name", container_name,
                "-p", "26006:6006",  # HTTP port
                "-p", "24317:4317",  # gRPC port
                "-v", f"{test_data_dir}:/phoenix_data",
                "-e", "PHOENIX_WORKING_DIR=/phoenix_data",
                "-e", "PHOENIX_SQL_DATABASE_URL=sqlite:////phoenix_data/phoenix.db",
                "arizephoenix/phoenix:latest",
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


@pytest.fixture
def phoenix_client(phoenix_test_server):
    """Get Phoenix client connected to test server."""
    import phoenix as px
    return px.Client(endpoint=phoenix_test_server.base_url)


@pytest.fixture
def mock_phoenix_client():
    """Mock Phoenix client for testing."""
    with patch("phoenix.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Mock dataset
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
        mock_client.get_dataset.return_value = mock_dataset
        mock_client.upload_dataset.return_value = mock_dataset

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
        mock_client.get_spans_dataframe.return_value = mock_df

        yield mock_client


@pytest.fixture
def mock_search_service():
    """Mock search service for testing."""
    with patch("cogniverse_agents.search.service.SearchService") as mock_service_class:
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service

        # Mock search results - use Mock with proper to_dict method
        mock_result1 = Mock()
        mock_result1.to_dict.return_value = {
            "document_id": "item1_part_0",
            "source_id": "item1",
            "score": 0.9,
            "content": "test content 1",
        }

        mock_result2 = Mock()
        mock_result2.to_dict.return_value = {
            "document_id": "item2_part_1",
            "source_id": "item2",
            "score": 0.8,
            "content": "test content 2",
        }

        mock_results = [mock_result1, mock_result2]
        mock_service.search.return_value = mock_results

        yield mock_service


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
            "vespa_url": "http://localhost",
            "vespa_port": 8080,
            "schema_name": "video_frame",
        }
        yield mock_config


@pytest.fixture
def mock_evaluator_provider(mock_phoenix_client):
    """Mock evaluator provider for testing."""
    with patch("cogniverse_evaluation.providers.get_evaluation_provider") as mock_get_provider:
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
        with patch("cogniverse_evaluation.providers.get_evaluation_provider") as mock_get:
            mock_provider = MagicMock()

            # Mock Phoenix evaluator base class

            class MockPhoenixEvaluator:
                """Mock Phoenix evaluator base"""
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
                    metadata=metadata or {}
                )

            mock_framework.create_evaluation_result = mock_create_result
            mock_provider.framework = mock_framework

            # Mock telemetry provider with async methods
            mock_telemetry = MagicMock()

            # Mock datasets with async get_dataset
            mock_datasets = MagicMock()
            async def mock_get_dataset(name):
                return None  # Default return, tests can override
            mock_datasets.get_dataset = AsyncMock(side_effect=mock_get_dataset)
            mock_telemetry.datasets = mock_datasets

            # Mock traces with async get_spans
            mock_traces = MagicMock()
            async def mock_get_spans(**kwargs):
                return pd.DataFrame()  # Default empty dataframe
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
    # Add any singleton resets here if needed
    yield
