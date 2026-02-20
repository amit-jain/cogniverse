"""
Unit tests for health router endpoints.

All external dependencies (BackendRegistry, AgentRegistry, ConfigManager)
are patched at the module boundary so each endpoint is tested in isolation.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime.routers import health


@pytest.fixture
def health_client():
    """TestClient with health router mounted."""
    test_app = FastAPI()
    test_app.include_router(health.router)

    with TestClient(test_app) as client:
        yield client


# ── GET /health/live ─────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.ci_fast
class TestLivenessProbe:
    def test_liveness_probe(self, health_client):
        """GET /health/live returns alive status."""
        resp = health_client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json() == {"status": "alive"}

    def test_liveness_no_deps(self, health_client):
        """Liveness probe succeeds with no setup — zero external calls."""
        resp = health_client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"


# ── GET /health/ready ────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.ci_fast
class TestReadinessProbe:
    @patch("cogniverse_runtime.routers.health.BackendRegistry")
    def test_readiness_no_backends(self, mock_registry_cls, health_client):
        """GET /health/ready returns not_ready when no backends registered."""
        mock_registry_cls.get_instance.return_value.list_backends.return_value = []

        resp = health_client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_ready"
        assert "reason" in data

    @patch("cogniverse_runtime.routers.health.BackendRegistry")
    def test_readiness_with_backends(self, mock_registry_cls, health_client):
        """GET /health/ready returns ready when backends are registered."""
        mock_registry_cls.get_instance.return_value.list_backends.return_value = [
            "vespa"
        ]

        resp = health_client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["backends"] == 1

    @patch("cogniverse_runtime.routers.health.BackendRegistry")
    def test_readiness_backend_count(self, mock_registry_cls, health_client):
        """Readiness probe count matches number of registered backends."""
        mock_registry_cls.get_instance.return_value.list_backends.return_value = [
            "vespa",
            "elasticsearch",
        ]

        resp = health_client.get("/health/ready")
        data = resp.json()
        assert data["status"] == "ready"
        assert data["backends"] == 2


# ── GET /health ──────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.ci_fast
class TestHealthCheckFull:
    @patch("cogniverse_runtime.routers.health.AgentRegistry")
    @patch("cogniverse_runtime.routers.health.BackendRegistry")
    @patch("cogniverse_runtime.routers.health.create_default_config_manager")
    def test_health_check_full(
        self, mock_create_cm, mock_backend_cls, mock_agent_cls, health_client
    ):
        """GET /health returns full system status."""
        mock_backend_cls.get_instance.return_value.list_backends.return_value = [
            "vespa"
        ]
        mock_agent_cls.return_value.list_agents.return_value = ["search_agent"]

        resp = health_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "cogniverse-runtime"
        assert "backends" in data
        assert "agents" in data

    @patch("cogniverse_runtime.routers.health.AgentRegistry")
    @patch("cogniverse_runtime.routers.health.BackendRegistry")
    @patch("cogniverse_runtime.routers.health.create_default_config_manager")
    def test_health_response_structure(
        self, mock_create_cm, mock_backend_cls, mock_agent_cls, health_client
    ):
        """GET /health response has all expected keys."""
        mock_backend_cls.get_instance.return_value.list_backends.return_value = []
        mock_agent_cls.return_value.list_agents.return_value = []

        resp = health_client.get("/health")
        data = resp.json()
        assert set(data.keys()) == {"status", "service", "backends", "agents"}
        assert "registered" in data["backends"]
        assert "backends" in data["backends"]
        assert "registered" in data["agents"]
        assert "agents" in data["agents"]
