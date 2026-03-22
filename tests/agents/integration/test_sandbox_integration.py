"""
Integration tests for OpenShell sandbox execution.

Uses the OpenShell Python SDK (protobuf conflict resolved by upgrading
mem0ai and grpcio-status to allow protobuf 6.x).

Tests manage their own OpenShell gateway lifecycle — start on setup,
destroy on teardown. No pre-existing gateway required.
"""

import subprocess
import time

import pytest

from cogniverse_runtime.sandbox_manager import SandboxManager

GATEWAY_NAME = "cogniverse-test-gw"
GATEWAY_PORT = 19090


def _openshell_cli_available():
    try:
        result = subprocess.run(
            ["openshell", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _docker_available():
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.skipif(
        not _openshell_cli_available(), reason="openshell CLI not installed"
    ),
    pytest.mark.skipif(
        not _docker_available(), reason="Docker not running"
    ),
]


@pytest.fixture(scope="module")
def openshell_gateway():
    """
    Start an OpenShell gateway for integration tests.

    Uses a unique name and port to avoid conflicts with user's dev gateway.
    Destroys on teardown.
    """
    subprocess.run(
        ["openshell", "gateway", "destroy", "--name", GATEWAY_NAME],
        capture_output=True, timeout=30, check=False,
    )

    result = subprocess.run(
        ["openshell", "gateway", "start", "--name", GATEWAY_NAME,
         "--port", str(GATEWAY_PORT)],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        pytest.skip(f"Failed to start OpenShell gateway: {result.stderr}")

    subprocess.run(
        ["openshell", "gateway", "select", GATEWAY_NAME],
        capture_output=True, timeout=10, check=False,
    )

    yield GATEWAY_NAME

    subprocess.run(
        ["openshell", "gateway", "destroy", "--name", GATEWAY_NAME],
        capture_output=True, timeout=60, check=False,
    )


class TestSandboxExecutionSDK:
    """Test sandbox creation and run using the Python SDK."""

    def test_create_run_delete_via_sdk(self, openshell_gateway):
        """Create sandbox via SDK, wait ready, run command, verify output, delete."""
        from openshell import SandboxClient

        client = SandboxClient.from_active_cluster()
        session = client.create_session()
        client.wait_ready(session.sandbox.name, timeout_seconds=120)

        result = session.exec(["echo", "hello-from-sdk"])
        assert result.exit_code == 0, f"Run failed: {result.stderr}"
        assert "hello-from-sdk" in result.stdout

        session.delete()
        client.close()

    def test_sandbox_network_isolation_via_sdk(self, openshell_gateway):
        """Sandbox blocks arbitrary egress by default."""
        from openshell import SandboxClient

        client = SandboxClient.from_active_cluster()
        session = client.create_session()
        client.wait_ready(session.sandbox.name, timeout_seconds=120)

        result = session.exec(
            ["python3", "-c",
             "import urllib.request; urllib.request.urlopen('http://example.com', timeout=5)"],
            timeout_seconds=30,
        )
        assert result.exit_code != 0 or "Error" in result.stderr

        session.delete()
        client.close()


class TestSandboxManagerIntegration:
    """Test SandboxManager with real gateway."""

    def test_manager_connects_and_reports_available(self, openshell_gateway):
        manager = SandboxManager(
            policy_dir="configs/openshell",
            enabled=True,
        )
        assert manager.available, "SandboxManager should detect running gateway"
        assert len(manager._policies) >= 4
        manager.close()

    def test_run_in_sandbox_via_manager(self, openshell_gateway):
        manager = SandboxManager(
            policy_dir="configs/openshell",
            enabled=True,
        )
        assert manager.available

        result = manager.exec_in_sandbox(
            "search_agent",
            ["echo", "sandbox-run-test"],
            timeout_seconds=30,
        )
        assert result is not None
        assert result["exit_code"] == 0
        assert "sandbox-run-test" in result["stdout"]
        manager.close()

    def test_policy_egress_rules(self, openshell_gateway):
        manager = SandboxManager(policy_dir="configs/openshell", enabled=False)
        manager._load_policies()

        search_policy = manager.get_policy("search_agent")
        egress = search_policy["network_policies"]["egress"]
        ports = {rule["port"] for rule in egress}
        assert 8080 in ports, "Search agent must reach Vespa"
        assert 11434 in ports, "Search agent must reach Ollama"
        assert search_policy["network_policies"]["deny_all_other"] is True

        summarizer_policy = manager.get_policy("summarizer_agent")
        summarizer_ports = {r["port"] for r in summarizer_policy["network_policies"]["egress"]}
        assert 8080 not in summarizer_ports, "Summarizer should NOT reach Vespa"
