"""
Integration tests for OpenShell sandbox execution.

Uses the openshell CLI (subprocess) since the Python SDK has a protobuf
version conflict with opentelemetry-proto (gencode 6.x vs runtime 5.x).

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
    # Destroy any leftover gateway from a previous failed run
    subprocess.run(
        ["openshell", "gateway", "destroy", "--name", GATEWAY_NAME],
        capture_output=True, timeout=30, check=False,
    )

    # Start gateway on non-default port
    result = subprocess.run(
        ["openshell", "gateway", "start", "--name", GATEWAY_NAME,
         "--port", str(GATEWAY_PORT)],
        capture_output=True, text=True, timeout=180,
    )
    if result.returncode != 0:
        pytest.skip(f"Failed to start OpenShell gateway: {result.stderr}")

    # Select this gateway as active
    subprocess.run(
        ["openshell", "gateway", "select", GATEWAY_NAME],
        capture_output=True, timeout=10, check=False,
    )

    yield GATEWAY_NAME

    # Teardown: destroy gateway
    subprocess.run(
        ["openshell", "gateway", "destroy", "--name", GATEWAY_NAME],
        capture_output=True, timeout=60, check=False,
    )


class TestSandboxExecution:
    """Test real sandbox creation and command execution via CLI."""

    def test_create_and_exec_in_sandbox(self, openshell_gateway):
        """Create a sandbox, run a command inside it, verify output, delete."""
        sandbox_name = f"test-exec-{int(time.time())}"

        result = subprocess.run(
            ["openshell", "sandbox", "create", "--name", sandbox_name,
             "--no-keep", "--", "echo", "hello-cogniverse"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"Failed: {result.stderr}"
        assert "hello-cogniverse" in result.stdout

        subprocess.run(
            ["openshell", "sandbox", "delete", sandbox_name],
            capture_output=True, timeout=30, check=False,
        )

    def test_sandbox_network_isolation(self, openshell_gateway):
        """Sandbox cannot reach arbitrary external hosts."""
        sandbox_name = f"test-netiso-{int(time.time())}"

        result = subprocess.run(
            ["openshell", "sandbox", "create", "--name", sandbox_name,
             "--no-keep", "--",
             "python3", "-c",
             "import urllib.request; urllib.request.urlopen('http://example.com', timeout=5)"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode != 0 or "Error" in result.stderr + result.stdout

        subprocess.run(
            ["openshell", "sandbox", "delete", sandbox_name],
            capture_output=True, timeout=30, check=False,
        )


class TestSandboxManagerPolicies:
    """Test SandboxManager policy loading and validation."""

    def test_manager_loads_all_policies(self, openshell_gateway):
        manager = SandboxManager(
            policy_dir="configs/openshell",
            enabled=True,
        )
        assert len(manager._policies) >= 4
        assert "search_agent" in manager._policies
        assert "routing_agent" in manager._policies
        assert "orchestrator_agent" in manager._policies
        assert "summarizer_agent" in manager._policies

    def test_search_agent_policy_allows_vespa_and_ollama(self, openshell_gateway):
        manager = SandboxManager(policy_dir="configs/openshell", enabled=False)
        manager._load_policies()

        policy = manager.get_policy("search_agent")
        egress = policy["network_policies"]["egress"]
        ports = {rule["port"] for rule in egress}
        assert 8080 in ports, "Search agent must reach Vespa (8080)"
        assert 11434 in ports, "Search agent must reach Ollama (11434)"
        assert policy["network_policies"]["deny_all_other"] is True

    def test_summarizer_agent_blocked_from_vespa(self, openshell_gateway):
        manager = SandboxManager(policy_dir="configs/openshell", enabled=False)
        manager._load_policies()

        policy = manager.get_policy("summarizer_agent")
        egress = policy["network_policies"]["egress"]
        ports = {rule["port"] for rule in egress}
        assert 8080 not in ports, "Summarizer should NOT reach Vespa"
        assert 11434 in ports, "Summarizer must reach Ollama"
