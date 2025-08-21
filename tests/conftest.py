"""
Shared pytest fixtures and configuration for routing tests.
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return {
        "simple_entity": [
            "show me videos about cats",
            "find documents about machine learning",
            "search for python tutorials"
        ],
        "relationship": [
            "compare the video with the document",
            "show me content related to yesterday's search",
            "what's the connection between these topics"
        ],
        "structured_extraction": [
            "extract timestamps from video",
            "get all product IDs in JSON format",
            "list specific dates and times"
        ],
        "summary": [
            "summarize the main points",
            "give me the key takeaways",
            "brief overview of the content"
        ],
        "detailed_report": [
            "detailed analysis of the data",
            "comprehensive breakdown of issues",
            "in-depth performance review"
        ]
    }


@pytest.fixture
def mock_gliner_model():
    """Mock GLiNER model for unit tests."""
    with patch('src.app.routing.strategies.GLiNER') as mock_gliner:
        mock_model = Mock()
        mock_model.predict_entities = Mock(return_value=[])
        mock_gliner.from_pretrained.return_value = mock_model
        yield mock_model


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response."""
    def _mock_response(search_modality="both", generation_type="raw_results", reasoning="Test"):
        return {
            "response": json.dumps({
                "search_modality": search_modality,
                "generation_type": generation_type,
                "reasoning": reasoning
            })
        }
    return _mock_response


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for API calls."""
    with patch('httpx.AsyncClient') as mock_client:
        mock_instance = Mock()
        mock_client.return_value.__aenter__.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def routing_config():
    """Standard routing configuration for tests."""
    return {
        "routing_mode": "tiered",
        "tier_config": {
            "enable_fast_path": True,
            "enable_slow_path": True,
            "enable_langextract": True,
            "enable_fallback": True,
            "fast_path_confidence_threshold": 0.7,
            "slow_path_confidence_threshold": 0.6,
            "langextract_confidence_threshold": 0.5
        },
        "gliner_config": {
            "gliner_model": "urchade/gliner_multi-v2.1",
            "gliner_threshold": 0.3,
            "gliner_labels": [
                "video_content", "document_content",
                "summary_request", "detailed_analysis"
            ]
        },
        "llm_config": {
            "provider": "local",
            "model": "smollm2:1.7b",
            "endpoint": "http://localhost:11434",
            "temperature": 0.1
        },
        "langextract_config": {
            "langextract_model": "qwen2.5:7b",
            "ollama_url": "http://localhost:11434"
        },
        "keyword_config": {
            "video_keywords": ["video", "watch", "show"],
            "text_keywords": ["document", "text", "article"],
            "summary_keywords": ["summary", "summarize", "brief"],
            "report_keywords": ["detailed", "comprehensive", "full"]
        },
        "cache_config": {
            "enable_caching": True,
            "cache_ttl_seconds": 300,
            "max_cache_size": 1000
        }
    }


@pytest.fixture
def assert_routing_decision():
    """Helper to assert routing decision properties."""
    def _assert(decision, expected_modality=None, expected_type=None, 
                min_confidence=0.0, max_confidence=1.0):
        from src.app.routing.base import RoutingDecision, SearchModality, GenerationType
        
        assert isinstance(decision, RoutingDecision)
        
        if expected_modality:
            assert decision.search_modality == expected_modality
        
        if expected_type:
            assert decision.generation_type == expected_type
        
        assert min_confidence <= decision.confidence_score <= max_confidence
        assert decision.routing_method is not None
        
        return True
    
    return _assert


# Markers for conditional test execution
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "requires_ollama: mark test as requiring Ollama to be running"
    )
    config.addinivalue_line(
        "markers", "requires_gliner: mark test as requiring GLiNER models"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Skip tests if dependencies not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to skip tests based on availability."""
    import httpx
    
    # Check if Ollama is available
    ollama_available = False
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=1)
        ollama_available = response.status_code == 200
    except:
        pass
    
    skip_ollama = pytest.mark.skip(reason="Ollama not available")
    skip_slow = pytest.mark.skip(reason="Slow test skipped (use --slow to run)")
    
    for item in items:
        if "requires_ollama" in item.keywords and not ollama_available:
            item.add_marker(skip_ollama)
        
        if "slow" in item.keywords and not config.getoption("--slow", default=False):
            item.add_marker(skip_slow)