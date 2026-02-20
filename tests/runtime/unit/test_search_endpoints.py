"""
Unit tests for search router endpoints.

Tests HTTP routing, request validation, and response structure
without requiring backend infrastructure or ML models.
"""

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.routers import search
from cogniverse_runtime.routers.search import SearchRequest
from cogniverse_runtime.search.base import SearchResult
from cogniverse_sdk.document import ContentType, Document
from cogniverse_sdk.interfaces.schema_loader import SchemaLoader
from tests.utils.memory_store import InMemoryConfigStore


def _make_config_manager() -> ConfigManager:
    """Create ConfigManager with in-memory store for testing."""
    store = InMemoryConfigStore()
    store.initialize()
    return ConfigManager(store=store)


def _make_mock_schema_loader() -> SchemaLoader:
    """Create a mock SchemaLoader."""
    loader = MagicMock(spec=SchemaLoader)
    loader.load_schema.return_value = {"fields": []}
    loader.list_available_schemas.return_value = []
    loader.schema_exists.return_value = True
    return loader


def _make_noop_telemetry_manager():
    """Create a no-op telemetry manager that returns dummy context managers."""
    manager = MagicMock()

    @contextmanager
    def _noop_span(*args, **kwargs):
        span = MagicMock()
        yield span

    manager.span = _noop_span
    manager.session_span = _noop_span
    return manager


def _make_search_result(doc_id: str = "doc-1", score: float = 0.95) -> SearchResult:
    """Create a SearchResult for test assertions."""
    doc = Document(
        id=doc_id,
        content_type=ContentType.VIDEO,
        metadata={"source_id": "video-1", "start_time": 0.0, "end_time": 5.0},
    )
    return SearchResult(document=doc, score=score)


@pytest.fixture
def search_client():
    """
    TestClient with search router mounted and dependencies overridden.

    Patches SearchService to avoid backend/model initialization and
    patches get_telemetry_manager to avoid Phoenix/OTLP dependencies.
    """
    test_app = FastAPI()
    test_app.include_router(search.router, prefix="/search")

    config_manager = _make_config_manager()
    schema_loader = _make_mock_schema_loader()

    test_app.dependency_overrides[search.get_config_manager_dependency] = (
        lambda: config_manager
    )
    test_app.dependency_overrides[search.get_schema_loader_dependency] = (
        lambda: schema_loader
    )

    with patch(
        "cogniverse_runtime.routers.search.get_telemetry_manager",
        return_value=_make_noop_telemetry_manager(),
    ):
        with TestClient(test_app) as client:
            yield client


# ── GET /search/strategies ──────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.ci_fast
class TestListStrategies:
    def test_list_strategies(self, search_client):
        """GET /search/strategies returns strategy list."""
        resp = search_client.get("/search/strategies")
        assert resp.status_code == 200
        data = resp.json()
        assert "strategies" in data
        assert len(data["strategies"]) == 5

    def test_list_strategies_format(self, search_client):
        """Each strategy has name and description fields."""
        resp = search_client.get("/search/strategies")
        for strategy in resp.json()["strategies"]:
            assert "name" in strategy
            assert "description" in strategy
            assert isinstance(strategy["name"], str)
            assert isinstance(strategy["description"], str)

    def test_list_strategies_names(self, search_client):
        """Verify the exact strategy names returned."""
        resp = search_client.get("/search/strategies")
        names = {s["name"] for s in resp.json()["strategies"]}
        assert names == {"semantic", "bm25", "hybrid", "learned", "multi_modal"}


# ── GET /search/profiles ────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.ci_fast
class TestListProfiles:
    def test_list_profiles_default_tenant(self, search_client):
        """GET /search/profiles returns profiles for default tenant."""
        resp = search_client.get("/search/profiles")
        assert resp.status_code == 200
        data = resp.json()
        assert "tenant_id" in data
        assert "count" in data
        assert "profiles" in data
        assert data["tenant_id"] == "default"

    def test_list_profiles_custom_tenant(self, search_client):
        """GET /search/profiles?tenant_id=acme scopes to that tenant."""
        resp = search_client.get("/search/profiles?tenant_id=acme")
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == "acme"

    def test_list_profiles_response_structure(self, search_client):
        """Profile list entries have name, model, and type fields."""
        resp = search_client.get("/search/profiles")
        data = resp.json()
        assert isinstance(data["count"], int)
        assert isinstance(data["profiles"], list)
        # Each profile entry should have the expected keys
        for profile in data["profiles"]:
            assert "name" in profile
            assert "model" in profile
            assert "type" in profile


# ── POST /search ─────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.ci_fast
class TestSearchEndpoint:
    def test_search_missing_query(self, search_client):
        """POST /search without query field returns 422."""
        resp = search_client.post("/search", json={"top_k": 5})
        assert resp.status_code == 422

    def test_search_request_defaults(self):
        """SearchRequest model has correct defaults."""
        req = SearchRequest(query="test query")
        assert req.strategy == "hybrid"
        assert req.top_k == 10
        assert req.stream is False
        assert req.filters == {}
        assert req.tenant_id is None
        assert req.session_id is None

    @patch("cogniverse_runtime.routers.search.SearchService")
    def test_search_success(self, mock_service_cls, search_client):
        """POST /search with valid query returns SearchResponse."""
        mock_instance = MagicMock()
        mock_instance.search.return_value = [_make_search_result()]
        mock_service_cls.return_value = mock_instance

        resp = search_client.post(
            "/search",
            json={"query": "find sunset scenes", "top_k": 5},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "find sunset scenes"
        assert data["results_count"] == 1
        assert len(data["results"]) == 1

    @patch("cogniverse_runtime.routers.search.SearchService")
    def test_search_empty_results(self, mock_service_cls, search_client):
        """POST /search returns results_count=0 when no matches."""
        mock_instance = MagicMock()
        mock_instance.search.return_value = []
        mock_service_cls.return_value = mock_instance

        resp = search_client.post("/search", json={"query": "nonexistent content"})

        assert resp.status_code == 200
        assert resp.json()["results_count"] == 0
        assert resp.json()["results"] == []

    @patch("cogniverse_runtime.routers.search.SearchService")
    def test_search_service_error(self, mock_service_cls, search_client):
        """POST /search returns 500 when SearchService raises."""
        mock_service_cls.side_effect = RuntimeError("Backend unavailable")

        resp = search_client.post("/search", json={"query": "test"})
        assert resp.status_code == 500
        assert "Backend unavailable" in resp.json()["detail"]

    @patch("cogniverse_runtime.routers.search.SearchService")
    def test_search_with_session_id(self, mock_service_cls, search_client):
        """POST /search with session_id uses session_span for telemetry."""
        mock_instance = MagicMock()
        mock_instance.search.return_value = []
        mock_service_cls.return_value = mock_instance

        resp = search_client.post(
            "/search",
            json={"query": "test", "session_id": "sess-abc123"},
        )

        assert resp.status_code == 200
        assert resp.json()["session_id"] == "sess-abc123"


# ── POST /search (stream=True) ──────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.ci_fast
class TestSearchStreaming:
    @patch("cogniverse_runtime.routers.search.SearchService")
    def test_search_stream_success(self, mock_service_cls, search_client):
        """POST /search with stream=True returns SSE with status + final events."""
        mock_instance = MagicMock()
        mock_instance.search.return_value = [_make_search_result()]
        mock_service_cls.return_value = mock_instance

        resp = search_client.post(
            "/search",
            json={"query": "sunset", "stream": True},
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        # Parse SSE events
        events = []
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        assert len(events) == 2
        assert events[0]["type"] == "status"
        assert events[0]["query"] == "sunset"
        assert events[1]["type"] == "final"
        assert events[1]["data"]["results_count"] == 1

    @patch("cogniverse_runtime.routers.search.SearchService")
    def test_search_stream_error(self, mock_service_cls, search_client):
        """POST /search with stream=True emits SSE error event on failure."""
        mock_instance = MagicMock()
        mock_instance.search.side_effect = RuntimeError("encoder crashed")
        mock_service_cls.return_value = mock_instance

        resp = search_client.post(
            "/search",
            json={"query": "test", "stream": True},
        )

        assert resp.status_code == 200  # SSE always returns 200

        events = []
        for line in resp.text.strip().split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))

        # status event + error event
        assert len(events) == 2
        assert events[1]["type"] == "error"
        assert "encoder crashed" in events[1]["error"]


# ── POST /search/rerank ──────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.ci_fast
class TestRerankEndpoint:
    def test_rerank_missing_query(self, search_client):
        """POST /search/rerank without query returns 400."""
        resp = search_client.post(
            "/search/rerank",
            json={"results": [{"id": "1", "score": 0.5}]},
        )
        assert resp.status_code == 400

    def test_rerank_missing_results(self, search_client):
        """POST /search/rerank without results returns 400."""
        resp = search_client.post(
            "/search/rerank",
            json={"query": "test"},
        )
        assert resp.status_code == 400

    def test_rerank_unknown_strategy(self, search_client):
        """POST /search/rerank with unknown strategy returns 400."""
        resp = search_client.post(
            "/search/rerank",
            json={
                "query": "test",
                "results": [{"id": "1", "score": 0.5}],
                "strategy": "nonexistent_strategy",
            },
        )
        # Unknown strategy → 400 from the endpoint (before import failure)
        assert resp.status_code == 400
        assert "Unknown strategy" in resp.json()["detail"]
