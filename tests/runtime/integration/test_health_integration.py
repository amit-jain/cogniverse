"""
Integration tests for health router with real Vespa backend.

Tests verify health endpoints with real BackendRegistry state
and real ConfigManager backed by VespaConfigStore.
"""

import logging
import os

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

    def test_health_full_with_real_config(
        self, health_client, vespa_instance
    ):
        """GET /health with real ConfigManager — backends and agents sections populated.

        Sets BACKEND_URL and BACKEND_PORT env vars to point at the test Vespa
        so create_default_config_manager() discovers the real VespaConfigStore.
        """
        original_url = os.environ.get("BACKEND_URL")
        original_port = os.environ.get("BACKEND_PORT")
        os.environ["BACKEND_URL"] = "http://localhost"
        os.environ["BACKEND_PORT"] = str(vespa_instance["http_port"])

        try:
            resp = health_client.get("/health")
        finally:
            if original_url is not None:
                os.environ["BACKEND_URL"] = original_url
            elif "BACKEND_URL" in os.environ:
                del os.environ["BACKEND_URL"]
            if original_port is not None:
                os.environ["BACKEND_PORT"] = original_port
            elif "BACKEND_PORT" in os.environ:
                del os.environ["BACKEND_PORT"]

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "cogniverse-runtime"

        assert "backends" in data
        assert data["backends"]["registered"] >= 1
        assert "vespa" in data["backends"]["backends"]

        # Agents section should exist (may be empty in integration test)
        assert "agents" in data
        assert "registered" in data["agents"]
