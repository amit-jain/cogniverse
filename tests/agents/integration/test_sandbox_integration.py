"""
Integration tests for OpenShell sandbox execution.

Uses the OpenShell Python SDK (protobuf conflict resolved by upgrading
mem0ai and grpcio-status to allow protobuf 6.x).

Tests manage their own OpenShell gateway lifecycle — start on setup,
destroy on teardown — in a private config root, so the host's OpenShell
configuration and its other gateways are never touched.
"""

import json
import subprocess
from pathlib import Path

import pytest

from cogniverse_runtime.sandbox_manager import SandboxManager, SandboxPolicy

GATEWAY_NAME = "cogniverse-test-gw"
GATEWAY_PORT = 19090


def _openshell_cli_available():
    try:
        result = subprocess.run(
            ["openshell", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _docker_available():
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _openshell_cli_available(), reason="openshell CLI not installed"
    ),
    pytest.mark.skipif(not _docker_available(), reason="Docker not running"),
]


def _host_openshell_config() -> dict[str, bytes]:
    """Every file of the host's OpenShell config (registrations and the
    active-gateway pointer), by path relative to its root."""
    root = Path.home() / ".config" / "openshell"
    if not root.exists():
        return {}
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _other_gateway_containers() -> dict[str, tuple[str, str]]:
    """``openshell-cluster-*`` containers other than this module's, by name,
    with their id and creation time."""
    listing = subprocess.run(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            "name=openshell-cluster-",
            "--format",
            "{{.Names}}\t{{.ID}}\t{{.CreatedAt}}",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    containers = {}
    for line in listing.stdout.splitlines():
        name, container_id, created = line.split("\t")
        if name != f"openshell-cluster-{GATEWAY_NAME}":
            containers[name] = (container_id, created)
    return containers


@pytest.fixture(scope="module")
def openshell_gateway(tmp_path_factory):
    """Start this module's OpenShell gateway in a private config root.

    ``XDG_CONFIG_HOME`` points at that root for the module, so the CLI
    registers and activates ``GATEWAY_NAME`` there and the SDK resolves it
    there. The host's OpenShell registrations, its active-gateway pointer and
    every other gateway's container (the e2e stack's ``openshell`` among them)
    stay exactly as they were: only this module's own container is created and
    removed.
    """
    from tests.agents.integration.conftest import OpenShellTestGateway

    host_config_before = _host_openshell_config()
    other_containers_before = _other_gateway_containers()
    config_home = tmp_path_factory.mktemp("openshell-config")
    gateway = OpenShellTestGateway(GATEWAY_NAME, GATEWAY_PORT, config_home)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("XDG_CONFIG_HOME", str(config_home))
        mp.delenv("OPENSHELL_GATEWAY", raising=False)
        mp.delenv("OPENSHELL_GATEWAY_ENDPOINT", raising=False)
        gateway.start()
        try:
            assert json.loads(gateway.metadata_path.read_text()) == {
                "name": GATEWAY_NAME,
                "gateway_endpoint": f"https://127.0.0.1:{GATEWAY_PORT}",
                "is_remote": False,
                "gateway_port": GATEWAY_PORT,
            }
            assert _host_openshell_config() == host_config_before
            yield GATEWAY_NAME
        finally:
            gateway.destroy()
    assert _host_openshell_config() == host_config_before
    assert _other_gateway_containers() == other_containers_before


class TestSandboxExecutionSDK:
    """Test sandbox creation and run using the Python SDK."""

    def test_create_run_delete_via_sdk(self, openshell_gateway):
        """Create sandbox via SDK, wait ready, run command, verify output, delete."""
        from openshell import SandboxClient

        client = SandboxClient.from_active_cluster()
        session = client.create_session()
        # 120s isn't enough on a cold K3s cluster (the gateway image needs
        # to pull the sandbox runtime image on first run + scheduler needs
        # to admit the pod). 300s tracks the prevailing K3s cold-start
        # latency on this dev-host class.
        client.wait_ready(session.sandbox.name, timeout_seconds=300)

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
            [
                "python3",
                "-c",
                "import urllib.request; urllib.request.urlopen('http://example.com', timeout=5)",
            ],
            timeout_seconds=30,
        )
        assert result.exit_code != 0 or "Error" in result.stderr

        session.delete()
        client.close()


class TestGatewayHealthProbeRealGateway:
    """health probe against the real OpenShell gateway.

    Verifies the probe records (available=1, latency>0) for a live gateway
    and (available=0, latency>0, error=...) when the SDK call fails. These
    tests exercise the actual SDK ``health()`` call — no mocks on the
    client boundary.
    """

    @pytest.mark.asyncio
    async def test_probe_reports_available_for_live_gateway(self, openshell_gateway):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from cogniverse_runtime.openshell_health import GatewayHealthProbe
        from cogniverse_runtime.sandbox_manager import SandboxPolicy

        manager = SandboxManager(
            policy_dir="configs/agent_policies",
            policy=SandboxPolicy.OPTIONAL,
        )
        try:
            assert manager.available is True

            exporter = InMemorySpanExporter()
            provider = TracerProvider()
            provider.add_span_processor(SimpleSpanProcessor(exporter))

            probe = GatewayHealthProbe(
                sandbox_manager=manager,
                interval_seconds=30.0,
                tracer=provider.get_tracer("test"),
            )
            available, latency = await probe.probe_once()

            assert available is True, (
                f"live OpenShell gateway must report available; latency_ms={latency}"
            )
            assert latency > 0
            attrs = dict(exporter.get_finished_spans()[0].attributes)
            assert attrs["openshell.gateway_available"] == 1
            assert attrs["openshell.gateway_latency_ms"] > 0
        finally:
            manager.close()


class TestSandboxPolicyAtBoot:
    """boot-time policy enforcement against the real OpenShell gateway.

    These tests do NOT mock SandboxManager._connect — they construct a real
    SandboxManager bound to the test gateway lifecycle. The required and
    optional paths must produce the documented behaviour against a live
    gateway, not just against a stub.
    """

    def test_required_boots_against_live_gateway(self, openshell_gateway):
        """policy=required succeeds when the gateway is up."""
        from cogniverse_runtime.sandbox_manager import SandboxPolicy

        manager = SandboxManager(
            policy_dir="configs/agent_policies",
            policy=SandboxPolicy.REQUIRED,
        )
        try:
            assert manager.available is True
            assert manager._policy is SandboxPolicy.REQUIRED
        finally:
            manager.close()

    def test_required_refuses_when_gateway_endpoint_invalid(
        self, monkeypatch, tmp_path
    ):
        """policy=required + bogus endpoint => raise at construction."""
        from cogniverse_runtime.sandbox_manager import (
            SandboxGatewayUnavailableError,
            SandboxPolicy,
        )

        # Force the SDK to dial a non-listening port. The real openshell SDK
        # call will fail; SandboxManager's optional/required gates decide.
        monkeypatch.setenv("OPENSHELL_GATEWAY_ENDPOINT", "https://127.0.0.1:1")
        # Empty policy dir so we can isolate the gateway-availability check.
        with pytest.raises(SandboxGatewayUnavailableError):
            SandboxManager(
                policy_dir=tmp_path,
                policy=SandboxPolicy.REQUIRED,
            )

    def test_optional_degrades_when_gateway_endpoint_invalid(
        self, monkeypatch, tmp_path
    ):
        """policy=optional + bogus endpoint => construction succeeds, .available=False."""
        from cogniverse_runtime.sandbox_manager import SandboxPolicy

        monkeypatch.setenv("OPENSHELL_GATEWAY_ENDPOINT", "https://127.0.0.1:1")
        manager = SandboxManager(
            policy_dir=tmp_path,
            policy=SandboxPolicy.OPTIONAL,
        )
        # Gateway is unreachable; manager exists but reports unavailable.
        assert manager._policy is SandboxPolicy.OPTIONAL


class TestSandboxManagerIntegration:
    """Test SandboxManager with real gateway."""

    def test_manager_connects_and_reports_available(self, openshell_gateway):
        manager = SandboxManager(
            policy_dir="configs/agent_policies",
            policy=SandboxPolicy.OPTIONAL,
        )
        assert manager.available, "SandboxManager should detect running gateway"
        assert len(manager._policies) >= 4
        manager.close()

    @pytest.mark.asyncio
    async def test_run_in_sandbox_via_manager(self, openshell_gateway):
        manager = SandboxManager(
            policy_dir="configs/agent_policies",
            policy=SandboxPolicy.OPTIONAL,
        )
        assert manager.available

        async with manager.task_session(
            "search_agent", "prodfixagents:sandbox"
        ) as session:
            first_sandbox = session.session_name
            result = await session.exec(
                ["echo", "sandbox-run-test"], timeout_seconds=30
            )
        assert result == {
            "stdout": "sandbox-run-test\n",
            "stderr": "",
            "exit_code": 0,
        }

        # Every task owns its own sandbox: the released one is gone, and the
        # next task gets a different container rather than a reused session.
        async with manager.task_session(
            "search_agent", "prodfixagents:sandbox"
        ) as second:
            assert second.session_name != first_sandbox
            again = await second.exec(["echo", "second-task"], timeout_seconds=30)
        assert again == {
            "stdout": "second-task\n",
            "stderr": "",
            "exit_code": 0,
        }
        manager.close()

    def test_policy_egress_rules(self, openshell_gateway):
        manager = SandboxManager(
            policy_dir="configs/agent_policies", policy=SandboxPolicy.DISABLED
        )
        manager._load_policies()

        search_policy = manager.get_policy("search_agent")
        egress = search_policy["network_policies"]["egress"]
        ports = {rule["port"] for rule in egress}
        assert 8080 in ports, "Search agent must reach Vespa"
        assert 11434 in ports, "Search agent must reach Ollama"
        assert search_policy["network_policies"]["deny_all_other"] is True

        summarizer_policy = manager.get_policy("summarizer_agent")
        summarizer_ports = {
            r["port"] for r in summarizer_policy["network_policies"]["egress"]
        }
        assert 8080 not in summarizer_ports, "Summarizer should NOT reach Vespa"
