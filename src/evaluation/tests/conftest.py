"""
Pytest configuration and fixtures for evaluation framework tests.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
import pandas as pd
import json
from pathlib import Path
import tempfile

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import Phoenix test server fixtures for integration tests
try:
    from .fixtures.phoenix_test_server import phoenix_test_server, phoenix_client
except ImportError:
    # If Phoenix is not available, provide skip fixtures
    @pytest.fixture(scope="session")
    def phoenix_test_server():
        pytest.skip("Phoenix not available")
    
    @pytest.fixture
    def phoenix_client():
        pytest.skip("Phoenix not available")


@pytest.fixture
def mock_phoenix_client():
    """Mock Phoenix client for testing."""
    with patch('phoenix.Client') as mock_client_class:
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
                output={"expected_videos": ["item1", "item2"], "expected_items": ["item1", "item2"]}
            ),
            MagicMock(
                id="example2",
                input={"query": "test query 2", "category": "test"},
                output={"expected_videos": ["item3"], "expected_items": ["item3"]}
            )
        ]
        mock_client.get_dataset.return_value = mock_dataset
        mock_client.upload_dataset.return_value = mock_dataset
        
        # Mock spans dataframe
        mock_df = pd.DataFrame([
            {
                "trace_id": "trace1",
                "attributes.input.value": "test query 1",
                "attributes.output.value": [{"item_id": "item1", "score": 0.9}],
                "attributes.metadata.profile": "test_profile",
                "attributes.metadata.strategy": "test_strategy",
                "timestamp": datetime.now().isoformat(),
                "duration_ms": 100
            }
        ])
        mock_client.get_spans_dataframe.return_value = mock_df
        
        yield mock_client


@pytest.fixture
def mock_search_service():
    """Mock search service for testing."""
    with patch('src.app.search.service.SearchService') as mock_service_class:
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        
        # Mock search results - use Mock with proper to_dict method
        mock_result1 = Mock()
        mock_result1.to_dict.return_value = {
            "document_id": "item1_part_0",
            "source_id": "item1",
            "score": 0.9,
            "content": "test content 1"
        }
        
        mock_result2 = Mock()
        mock_result2.to_dict.return_value = {
            "document_id": "item2_part_1",
            "source_id": "item2",
            "score": 0.8,
            "content": "test content 2"
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
            "category": "visual"
        },
        {
            "query": "what happened after the meeting",
            "expected_items": ["item3"],
            "category": "temporal"
        },
        {
            "query": "dog playing in the park",
            "expected_items": ["item4", "item5"],
            "category": "activity"
        }
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
            "metadata": {}
        },
        {
            "item_id": "item2",
            "score": 0.8,
            "rank": 2,
            "document_id": "item2_part_1",
            "content": "another person with red clothing",
            "temporal_info": {"timestamp": 200},
            "metadata": {}
        },
        {
            "item_id": "item1",  # Duplicate for diversity testing
            "score": 0.7,
            "rank": 3,
            "document_id": "item1_part_5",
            "content": "same person later",
            "temporal_info": {"timestamp": 150},
            "metadata": {}
        }
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
                {"video_id": "video1", "score": 0.7, "content": "duplicate"}
            ],
            "profile": "test_profile",
            "strategy": "test_strategy",
            "success": True
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
        "top_k": 10
    }


@pytest.fixture
def mock_get_config():
    """Mock get_config function."""
    with patch('src.tools.config.get_config') as mock_config:
        mock_config.return_value = {
            "vespa_url": "http://localhost",
            "vespa_port": 8080,
            "vespa_schema": "video_frame"
        }
        yield mock_config


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    # Add any singleton resets here if needed
    yield