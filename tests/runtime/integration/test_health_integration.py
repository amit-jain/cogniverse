"""
Integration tests for health router with real Vespa backend.

Tests verify health endpoints with real BackendRegistry state
and real ConfigManager backed by VespaConfigStore.
"""

import logging
from unittest.mock import patch

import pytest

import cogniverse_vespa  # noqa: F401 - triggers self-registration
from cogniverse_core.registries.backend_registry import BackendRegistry

logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.ci_fast
@pytest.mark.requires_vespa
class TestHealthIntegration:
    def test_liveness_always_works(self, health_client):
        """GET /health/live returns alive even during integration test with Vespa running."""
        resp = health_client.get("/health/live")

        assert resp.status_code == 200
        assert resp.json() == {"status": "alive"}

    def test_readiness_with_registered_vespa(self, health_client):
        """GET /health/ready reports ready when Vespa backend is registered.

        The `import cogniverse_vespa` at module level triggers self-registration
        of the Vespa backend into BackendRegistry.
        """
        # Verify Vespa is actually registered (precondition)
        registry = BackendRegistry.get_instance()
        assert registry.is_registered("vespa"), (
            "Vespa backend should be registered after importing cogniverse_vespa"
        )

        resp = health_client.get("/health/ready")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["backends"] >= 1

    @patch("cogniverse_runtime.routers.health.create_default_config_manager")
    def test_health_full_with_real_config(
        self, mock_create_cm, health_client, config_manager
    ):
        """GET /health with real ConfigManager — backends and agents sections populated.

        Patches create_default_config_manager to return the test ConfigManager
        (which is backed by real VespaConfigStore) instead of trying to
        discover environment variables.
        """
        mock_create_cm.return_value = config_manager

        resp = health_client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "cogniverse-runtime"

        # Backends section should have at least vespa registered
        assert "backends" in data
        assert data["backends"]["registered"] >= 1
        assert "vespa" in data["backends"]["backends"]

        # Agents section should exist (may be empty in integration test)
        assert "agents" in data
        assert "registered" in data["agents"]
