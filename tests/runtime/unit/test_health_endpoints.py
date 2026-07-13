"""
Unit tests for health router endpoints.

The registry dependencies (BackendRegistry, AgentRegistry, ConfigManager)
are patched at the module boundary so each endpoint is tested in isolation.
Backend reachability is NOT mocked: the fixture points the probes at a real
local stub HTTP server that answers ``/ApplicationStatus`` 200, so the
ready/healthy assertions reflect a genuinely reachable backend.
"""

import http.server
import threading
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_runtime.routers import health


class _OKHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/ApplicationStatus":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass


@contextmanager
def _stub_backend():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _OKHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()


@pytest.fixture(autouse=True)
def _clear_health_cache():
    """The AgentRegistry is cached process-wide; reset between tests so each
    test's patched dependencies are actually re-read."""
    health._get_agent_registry.cache_clear()
    yield
    health._get_agent_registry.cache_clear()


@pytest.fixture
def health_client():
    """TestClient with health router mounted and a reachable stub backend."""
    test_app = FastAPI()
    test_app.include_router(health.router)

    with _stub_backend() as base_url:
        test_app.state.backend_base_url = base_url
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
        """GET /health/ready returns 503 + not_ready when no backends registered."""
        mock_registry_cls.get_instance.return_value.list_backends.return_value = []

        resp = health_client.get("/health/ready")
        assert resp.status_code == 503
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

    @patch("cogniverse_runtime.routers.health.AgentRegistry")
    @patch("cogniverse_runtime.routers.health.BackendRegistry")
    @patch("cogniverse_runtime.routers.health.create_default_config_manager")
    def test_config_stack_built_once_across_probes(
        self, mock_create_cm, mock_backend_cls, mock_agent_cls, health_client
    ):
        """Repeated probes must reuse one config build + AgentRegistry (one
        httpx client), not rebuild the stack and leak a client per hit."""
        mock_backend_cls.get_instance.return_value.list_backends.return_value = [
            "vespa"
        ]
        mock_agent_cls.return_value.list_agents.return_value = []

        for _ in range(3):
            assert health_client.get("/health").status_code == 200

        assert mock_create_cm.call_count == 1
        assert mock_agent_cls.call_count == 1

    @patch("cogniverse_runtime.routers.health.create_default_config_manager")
    def test_health_returns_503_not_500_on_config_error(
        self, mock_create_cm, health_client
    ):
        """A config failure (e.g. unset BACKEND_URL) must surface as 503
        unhealthy, not a 500 server crash a probe reads as an outage bug."""
        mock_create_cm.side_effect = ValueError(
            "BACKEND_URL environment variable is required"
        )

        resp = health_client.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["service"] == "cogniverse-runtime"
        assert "BACKEND_URL" in data["reason"]
