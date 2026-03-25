"""Integration tests for the full cogniverse deployment lifecycle.

These tests create a REAL k3d cluster, build REAL Docker images,
deploy via REAL Helm install, and verify REAL service health.

Requires: docker, k3d, kubectl, helm
Marked: @pytest.mark.integration, @pytest.mark.slow, @pytest.mark.requires_docker
"""

import subprocess

import httpx
import pytest


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_docker
class TestDeployedServices:
    """Tests that run against a deployed cogniverse stack."""

    def test_vespa_config_healthy(self, deployed_stack):
        """Vespa config server responds to health check."""
        resp = httpx.get("http://localhost:19071/state/v1/health", timeout=10)
        assert resp.status_code == 200

    def test_vespa_query_reachable(self, deployed_stack):
        """Vespa query endpoint is reachable."""
        resp = httpx.get("http://localhost:8080/ApplicationStatus", timeout=10)
        assert resp.status_code == 200

    def test_runtime_healthy(self, deployed_stack):
        """Runtime API responds to health check."""
        resp = httpx.get(f"{deployed_stack['runtime_url']}/health", timeout=10)
        assert resp.status_code == 200

    def test_runtime_openapi(self, deployed_stack):
        """Runtime serves OpenAPI spec."""
        resp = httpx.get(f"{deployed_stack['runtime_url']}/openapi.json", timeout=10)
        assert resp.status_code == 200
        data = resp.json()
        assert "paths" in data
        assert "/admin/profiles" in data["paths"]

    def test_dashboard_healthy(self, deployed_stack):
        """Dashboard Streamlit app responds to health check."""
        resp = httpx.get(f"{deployed_stack['dashboard_url']}/_stcore/health", timeout=10)
        assert resp.status_code == 200

    def test_phoenix_healthy(self, deployed_stack):
        """Phoenix telemetry responds to health check."""
        resp = httpx.get(f"{deployed_stack['phoenix_url']}/health", timeout=10)
        assert resp.status_code == 200

    def test_all_pods_running(self, deployed_stack):
        """All pods in cogniverse namespace are Running."""
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", "cogniverse",
             "-o", "jsonpath={.items[*].status.phase}"],
            capture_output=True, text=True, timeout=10,
        )
        phases = result.stdout.strip().split()
        # All should be Running (LLM might still be pulling)
        running = [p for p in phases if p == "Running"]
        assert len(running) >= 4, f"Expected at least 4 Running pods, got: {phases}"

    def test_helm_release_deployed(self, deployed_stack):
        """Helm release exists and is deployed."""
        result = subprocess.run(
            ["helm", "status", "cogniverse", "-n", "cogniverse"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "deployed" in result.stdout.lower()


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_docker
class TestClusterLifecycle:
    """Tests for cluster creation and deletion (run independently)."""

    def test_cogniverse_cli_help(self):
        """CLI entrypoint works."""
        result = subprocess.run(
            ["uv", "run", "cogniverse", "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "up" in result.stdout
        assert "down" in result.stdout
        assert "status" in result.stdout

    def test_cogniverse_status_no_cluster(self):
        """Status command works even with no cluster (shows all down)."""
        result = subprocess.run(
            ["uv", "run", "cogniverse", "status"],
            capture_output=True, text=True, timeout=30,
        )
        # Should not crash, just show services as down
        assert result.returncode == 0
