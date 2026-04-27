"""Integration tests for the full cogniverse deployment lifecycle.

These tests create a REAL k3d cluster, build REAL Docker images,
deploy via REAL Helm install, and verify REAL service health.

Requires: docker, k3d, kubectl, helm
Marked: @pytest.mark.integration, @pytest.mark.slow, @pytest.mark.requires_docker
"""

import json
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
        from tests.e2e.deployment.conftest import PORTS

        resp = httpx.get(
            f"http://localhost:{PORTS['vespa_config']}/state/v1/health", timeout=10
        )
        assert resp.status_code == 200

    def test_vespa_query_reachable(self, deployed_stack):
        """Vespa query endpoint is reachable."""
        resp = httpx.get(f"{deployed_stack['vespa_url']}/ApplicationStatus", timeout=10)
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
        resp = httpx.get(
            f"{deployed_stack['dashboard_url']}/_stcore/health", timeout=10
        )
        assert resp.status_code == 200

    def test_phoenix_healthy(self, deployed_stack):
        """Phoenix telemetry responds to health check."""
        resp = httpx.get(f"{deployed_stack['phoenix_url']}/health", timeout=10)
        assert resp.status_code == 200

    def test_whisper_sidecar_healthy(self, deployed_stack):
        """Whisper ASR sidecar responds with the engine + model identifier.

        Proves three things at once:
        - The chart's ``whisper.enabled=true`` path actually deploys the
          pod (otherwise the port-forward at conftest:pf_specs would fail).
        - The image cogniverse/whisper-fw:dev was built and imported into
          the k3d cluster (otherwise: ImagePullBackOff and pod never Ready).
        - faster-whisper loaded the configured model (model name comes
          from /health, sourced from the WHISPER_ENGINE/MODEL_NAME envs
          the chart sets from values.k3s.yaml).
        """
        resp = httpx.get(f"{deployed_stack['whisper_url']}/health", timeout=10)
        assert resp.status_code == 200, (
            f"whisper /health returned {resp.status_code}: {resp.text[:200]}"
        )
        body = resp.json()
        assert body["status"] == "ok", body
        # values.k3s.yaml pins engine=faster-whisper, model=tiny.
        assert body["engine"] == "faster-whisper", body
        assert body["model"] == "tiny", body

    def test_runtime_pod_sees_whisper_in_inference_service_urls(self, deployed_stack):
        """Runtime pod's ``INFERENCE_SERVICE_URLS`` env carries whisper.

        The chart populates that env from a ``whisper.enabled``-gated
        template at ``charts/cogniverse/templates/all-resources.yaml:330``.
        Profiles whose transcription strategy sets
        ``inference_services.transcription: "whisper"`` route through the sidecar via this
        map. This assertion locks the wiring: the runtime container must
        receive ``whisper`` in the env, not a stale missing value, so the
        agent and ingestion-side AudioProcessor remote paths both find
        the pod. Reads via ``kubectl exec`` so we don't need a new
        runtime API endpoint solely for testability.
        """
        from tests.e2e.deployment.conftest import NAMESPACE

        # Find the runtime pod's name (single-replica deployment in tests).
        # Chart selector: app.kubernetes.io/component=runtime + name=cogniverse.
        pod_lookup = subprocess.run(
            [
                "kubectl",
                "get",
                "pods",
                "-n",
                NAMESPACE,
                "-l",
                "app.kubernetes.io/component=runtime,app.kubernetes.io/name=cogniverse",
                "-o",
                "jsonpath={.items[0].metadata.name}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert pod_lookup.returncode == 0, pod_lookup.stderr[:200]
        pod_name = pod_lookup.stdout.strip()
        assert pod_name, "no runtime pod found"

        env_dump = subprocess.run(
            [
                "kubectl",
                "exec",
                "-n",
                NAMESPACE,
                pod_name,
                "--",
                "printenv",
                "INFERENCE_SERVICE_URLS",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert env_dump.returncode == 0, env_dump.stderr[:200]

        urls = json.loads(env_dump.stdout)
        assert "vllm_asr" in urls, (
            f"whisper service URL missing from runtime pod env; got keys "
            f"{sorted(urls.keys())!r}"
        )
        assert urls["vllm_asr"].startswith("http://"), urls["vllm_asr"]

    def test_all_pods_running(self, deployed_stack):
        """All pods in test namespace are Running."""
        from tests.e2e.deployment.conftest import NAMESPACE

        result = subprocess.run(
            [
                "kubectl",
                "get",
                "pods",
                "-n",
                NAMESPACE,
                "-o",
                "jsonpath={.items[*].status.phase}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        phases = result.stdout.strip().split()
        running = [p for p in phases if p == "Running"]
        assert len(running) >= 4, f"Expected at least 4 Running pods, got: {phases}"

    def test_helm_release_deployed(self, deployed_stack):
        """Helm release exists and is deployed."""
        from tests.e2e.deployment.conftest import NAMESPACE

        result = subprocess.run(
            ["helm", "status", "cogniverse", "-n", NAMESPACE],
            capture_output=True,
            text=True,
            timeout=10,
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
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "up" in result.stdout
        assert "down" in result.stdout
        assert "status" in result.stdout

    def test_cogniverse_status_no_cluster(self):
        """Status command works even with no cluster (shows all down)."""
        result = subprocess.run(
            ["uv", "run", "cogniverse", "status"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should not crash, just show services as down
        assert result.returncode == 0
