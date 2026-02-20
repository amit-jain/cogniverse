"""
Integration tests for search router with real Vespa backend.

Tests verify the full wiring between routers, ConfigManager (VespaConfigStore),
BackendRegistry, and SchemaLoader. Only QueryEncoder is mocked to avoid
loading ML models.
"""

import json
import logging
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

logger = logging.getLogger(__name__)


def _make_noop_telemetry_manager():
    """Create a no-op telemetry manager for integration tests."""
    manager = MagicMock()

    @contextmanager
    def _noop_span(*args, **kwargs):
        span = MagicMock()
        yield span

    manager.span = _noop_span
    manager.session_span = _noop_span
    return manager


def _make_dummy_encoder(embedding_dim: int = 128):
    """Create a dummy query encoder that returns zero-vectors."""
    encoder = MagicMock()
    encoder.encode.return_value = np.zeros((1, embedding_dim), dtype=np.float32)
    encoder.get_embedding_dim.return_value = embedding_dim
    return encoder


# ── GET /search/profiles ──────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.ci_fast
@pytest.mark.requires_vespa
class TestListProfilesIntegration:
    def test_list_profiles_from_vespa_config(self, search_client):
        """GET /search/profiles returns seeded profiles from real VespaConfigStore.

        Profile list includes both system profiles (from configs/config.json)
        and tenant-specific profiles seeded via ConfigManager.add_backend_profile().
        """
        resp = search_client.get("/search/profiles")

        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == "default"
        # Count includes system profiles merged with seeded test profiles
        assert data["count"] >= 2

        profile_names = {p["name"] for p in data["profiles"]}

        # Seeded test profiles must be present
        assert "test_colpali" in profile_names
        assert "test_videoprism" in profile_names

        # Verify seeded profile details match
        profiles_by_name = {p["name"]: p for p in data["profiles"]}
        assert profiles_by_name["test_colpali"]["model"] == "vidore/colpali-v1.2"
        assert profiles_by_name["test_colpali"]["type"] == "video"
        assert profiles_by_name["test_videoprism"]["model"] == "google/videoprism-base"
        assert profiles_by_name["test_videoprism"]["type"] == "video"

    def test_list_profiles_tenant_scoping(self, search_client):
        """GET /search/profiles?tenant_id=tenant_b returns that tenant's profiles.

        tenant_b has its own seeded profile plus system profiles from config.json.
        The key assertion is that default-only profiles (test_colpali, test_videoprism)
        are NOT present for tenant_b.
        """
        resp = search_client.get("/search/profiles?tenant_id=tenant_b")

        assert resp.status_code == 200
        data = resp.json()
        assert data["tenant_id"] == "tenant_b"
        assert data["count"] >= 1

        profile_names = {p["name"] for p in data["profiles"]}
        assert "tenant_b_profile" in profile_names

        # Profiles seeded exclusively for default tenant should not appear
        assert "test_colpali" not in profile_names
        assert "test_videoprism" not in profile_names


# ── POST /search ──────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.ci_fast
@pytest.mark.requires_vespa
class TestSearchIntegration:
    @patch("cogniverse_runtime.routers.search.get_telemetry_manager")
    @patch(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
    )
    @patch("cogniverse_runtime.search.service.QueryEncoderFactory.create_encoder")
    def test_search_with_mocked_encoder(
        self,
        mock_create_encoder,
        mock_foundation_telemetry,
        mock_router_telemetry,
        search_client,
    ):
        """POST /search with real ConfigManager + Vespa, only QueryEncoder mocked.

        Verifies SearchService initializes correctly through the full chain:
        ConfigManager -> get_config -> profile lookup -> encoder creation -> backend init.
        Uses strategy="default" which exists in the real schema ranking strategies.
        """
        mock_create_encoder.return_value = _make_dummy_encoder()
        noop_tm = _make_noop_telemetry_manager()
        mock_router_telemetry.return_value = noop_tm
        mock_foundation_telemetry.return_value = noop_tm

        resp = search_client.post(
            "/search",
            json={
                "query": "test search query",
                "profile": "test_colpali",
                "strategy": "default",
                "top_k": 5,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "test search query"
        assert data["profile"] == "test_colpali"
        assert isinstance(data["results"], list)
        assert data["results_count"] == len(data["results"])

        # Verify the encoder factory was called with the correct profile and model
        mock_create_encoder.assert_called_once()
        call_args = mock_create_encoder.call_args
        assert call_args[0][0] == "test_colpali"
        assert call_args[0][1] == "vidore/colpali-v1.2"

    @patch("cogniverse_runtime.routers.search.get_telemetry_manager")
    @patch(
        "cogniverse_foundation.telemetry.manager.get_telemetry_manager",
    )
    @patch("cogniverse_runtime.search.service.QueryEncoderFactory.create_encoder")
    def test_search_stream_with_mocked_encoder(
        self,
        mock_create_encoder,
        mock_foundation_telemetry,
        mock_router_telemetry,
        search_client,
    ):
        """POST /search (stream=True) with real wiring — verifies SSE path."""
        mock_create_encoder.return_value = _make_dummy_encoder()
        noop_tm = _make_noop_telemetry_manager()
        mock_router_telemetry.return_value = noop_tm
        mock_foundation_telemetry.return_value = noop_tm

        resp = search_client.post(
            "/search",
            json={
                "query": "streaming test query",
                "profile": "test_colpali",
                "strategy": "default",
                "top_k": 3,
                "stream": True,
            },
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
        assert events[0]["query"] == "streaming test query"
        assert events[1]["type"] == "final"
        assert events[1]["data"]["query"] == "streaming test query"
        assert isinstance(events[1]["data"]["results"], list)
