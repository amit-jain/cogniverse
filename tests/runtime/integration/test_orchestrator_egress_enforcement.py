"""Dispatcher's policy-enforcing httpx client denies non-allowlisted egress."""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from cogniverse_agents.orchestrator_agent import AgentStep, OrchestrationPlan
from cogniverse_core.registries.agent_registry import AgentEndpoint, AgentRegistry
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.sandbox_http import (
    EgressDeniedError,
    PolicyEnforcingTransport,
)
from cogniverse_runtime.sandbox_manager import (
    _DEFAULT_POLICY_DIR,
    SandboxManager,
    SandboxPolicy,
)
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = pytest.mark.integration


_ORCHESTRATOR_POLICY_YAML = """\
network_policies:
  egress:
    - host: "localhost"
      port: 8000
      protocol: "tcp"
      comment: "A2A subagent dispatch (allowed)"
  deny_all_other: true
"""


@pytest.fixture
def policy_dir(tmp_path: Path) -> Path:
    (tmp_path / "orchestrator_agent.yaml").write_text(_ORCHESTRATOR_POLICY_YAML)
    return tmp_path


@pytest.fixture
def sandbox_mgr(policy_dir: Path) -> SandboxManager:
    # OPTIONAL loads policies; DISABLED short-circuits _load_policies.
    return SandboxManager(policy_dir=policy_dir, policy=SandboxPolicy.OPTIONAL)


@pytest.fixture
def sandbox_mgr_disabled(policy_dir: Path) -> SandboxManager:
    return SandboxManager(policy_dir=policy_dir, policy=SandboxPolicy.DISABLED)


class TestPolicyEnforcingClient:
    def test_returned_client_carries_policy_enforcing_transport(self, sandbox_mgr):
        client = sandbox_mgr.make_http_client("orchestrator_agent")
        try:
            assert isinstance(client._transport, PolicyEnforcingTransport), (
                f"expected PolicyEnforcingTransport; got "
                f"{type(client._transport).__name__}"
            )
        finally:
            del client

    @pytest.mark.asyncio
    async def test_off_allowlist_request_raises_egress_denied(self, sandbox_mgr):
        client = sandbox_mgr.make_http_client("orchestrator_agent")
        try:
            with pytest.raises(EgressDeniedError) as excinfo:
                await client.post(
                    "http://evil.example.com:9999/agents/x/process",
                    json={"query": "subagent call"},
                )
            err = excinfo.value
            assert err.host == "evil.example.com"
            assert err.port == 9999
            assert "evil.example.com:9999" in str(err)
            assert "Allow-listed" in str(err), (
                f"deny message must surface what IS allowed so the operator "
                f"can fix the policy YAML; got: {err}"
            )
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_allowlisted_request_passes_through_to_inner_transport(
        self, sandbox_mgr
    ):
        # Nothing listens on localhost:8000 → ConnectError proves we
        # got past the policy check (deny would raise EgressDeniedError).
        client = sandbox_mgr.make_http_client(
            "orchestrator_agent", timeout=httpx.Timeout(2.0, connect=1.0)
        )
        try:
            with pytest.raises(httpx.ConnectError):
                await client.post(
                    "http://localhost:8000/agents/search/process",
                    json={"query": "subagent call"},
                )
        except EgressDeniedError as exc:  # pragma: no cover - regression guard
            pytest.fail(
                f"localhost:8000 IS on the allowlist but the policy denied it: {exc}"
            )
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_agent_with_no_policy_file_is_unwrapped(
        self, sandbox_mgr, tmp_path: Path
    ):
        client = sandbox_mgr.make_http_client("agent_with_no_policy_file")
        try:
            assert not isinstance(client._transport, PolicyEnforcingTransport), (
                f"agent without policy must get a bare httpx client; "
                f"wrapping with an empty allowlist would deny everything. "
                f"got transport={type(client._transport).__name__}"
            )
        finally:
            del client

    @pytest.mark.asyncio
    async def test_disabled_sandbox_returns_bare_client(self, sandbox_mgr_disabled):
        client = sandbox_mgr_disabled.make_http_client("orchestrator_agent")
        try:
            assert not isinstance(client._transport, PolicyEnforcingTransport), (
                f"policy=disabled must hand back a bare client; "
                f"got transport={type(client._transport).__name__}"
            )
        finally:
            del client


class TestEnforcementEnvVarOverride:
    @pytest.mark.asyncio
    async def test_env_var_disabled_returns_bare_client(self, sandbox_mgr, monkeypatch):
        monkeypatch.setenv("COGNIVERSE_OPENSHELL_HTTP_ENFORCEMENT", "disabled")
        client = sandbox_mgr.make_http_client("orchestrator_agent")
        try:
            assert not isinstance(client._transport, PolicyEnforcingTransport), (
                f"COGNIVERSE_OPENSHELL_HTTP_ENFORCEMENT=disabled must hand "
                f"back a bare client; got transport={type(client._transport).__name__}"
            )
        finally:
            del client


PLAN_AGENTS = ("query_enhancement_agent", "search_agent", "summarizer_agent")
DEPLOYED_TENANTS = ("egressdeployed:alpha", "egressdeployed:beta")


class ChildRuntime:
    """The runtime's agent routes on a real socket, served at an address other
    than the policy rule's ``localhost:8000``."""

    def __init__(self):
        self.lock = threading.Lock()
        self.requests: list[tuple[str, dict]] = []
        runtime = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
                with runtime.lock:
                    runtime.requests.append((self.path, payload))
                encoded = json.dumps(
                    {
                        "status": "success",
                        "agent": payload["agent_name"],
                        "tenant": payload["context"]["tenant_id"],
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"


@pytest.fixture
def child_runtime():
    runtime = ChildRuntime()
    runtime.thread.start()
    try:
        yield runtime
    finally:
        runtime.server.shutdown()
        runtime.server.server_close()
        runtime.thread.join(5)


@pytest.fixture
def telemetry_off():
    from cogniverse_foundation.telemetry import manager as telemetry_manager_module
    from cogniverse_foundation.telemetry.manager import (
        TelemetryConfig,
        TelemetryManager,
    )

    installed = None
    if telemetry_manager_module._telemetry_manager is None:
        installed = TelemetryManager(TelemetryConfig(enabled=False))
        telemetry_manager_module._telemetry_manager = installed
    yield
    if (
        installed is not None
        and telemetry_manager_module._telemetry_manager is installed
    ):
        telemetry_manager_module._telemetry_manager = None


def deployed_dispatcher(runtime_url: str) -> AgentDispatcher:
    """A dispatcher configured the way the chart deploys the runtime: the
    shipped agent policies, and ``agent_registry_url`` at the runtime service."""
    manager = ConfigManager(store=InMemoryConfigStore())
    manager.set_system_config(
        replace(manager.get_system_config(), agent_registry_url=runtime_url)
    )
    registry = AgentRegistry(tenant_id=DEPLOYED_TENANTS[0], config_manager=manager)
    for name in PLAN_AGENTS:
        registry.register_agent(
            AgentEndpoint(
                name=name,
                url=runtime_url,
                capabilities=[name.removesuffix("_agent")],
                health_endpoint="/health",
                process_endpoint=f"/agents/{name}/process",
            )
        )
    return AgentDispatcher(
        agent_registry=registry,
        config_manager=manager,
        schema_loader=None,
        sandbox_manager=SandboxManager(
            policy_dir=_DEFAULT_POLICY_DIR, policy=SandboxPolicy.OPTIONAL
        ),
    )


def sequential_plan(query: str) -> OrchestrationPlan:
    return OrchestrationPlan(
        query=query,
        steps=[
            AgentStep(
                agent_name=name,
                input_data={"query": query},
                depends_on=[index - 1] if index else [],
                reasoning=f"Step {index + 1}: {name} processing",
            )
            for index, name in enumerate(PLAN_AGENTS)
        ],
    )


def dispatched(query: str, tenant: str) -> list[tuple[str, dict]]:
    return [
        (
            f"/agents/{name}/process",
            {"agent_name": name, "query": query, "context": {"tenant_id": tenant}},
        )
        for name in PLAN_AGENTS
    ]


def answered(tenant: str) -> dict:
    return {
        name: {"status": "success", "agent": name, "tenant": tenant}
        for name in PLAN_AGENTS
    }


class TestDeployedRuntimeDispatch:
    """The orchestrator reaches its sub-agents at the configured runtime address
    under the shipped ``orchestrator_agent`` policy."""

    async def test_every_plan_step_reaches_the_configured_runtime(
        self, child_runtime, telemetry_off
    ):
        dispatcher = deployed_dispatcher(child_runtime.url)
        orchestrator = await dispatcher._get_or_build_orchestrator(DEPLOYED_TENANTS[0])
        query = "Find videos about marathon training and summarize the key advice"

        results = await orchestrator._execute_plan(
            sequential_plan(query), tenant_id=DEPLOYED_TENANTS[0]
        )

        assert results == answered(DEPLOYED_TENANTS[0])
        assert child_runtime.requests == dispatched(query, DEPLOYED_TENANTS[0])

    async def test_other_addresses_stay_denied_under_the_binding(
        self, child_runtime, telemetry_off
    ):
        dispatcher = deployed_dispatcher(child_runtime.url)
        orchestrator = await dispatcher._get_or_build_orchestrator(DEPLOYED_TENANTS[0])
        port = child_runtime.server.server_port

        with pytest.raises(EgressDeniedError) as excinfo:
            await orchestrator._http_client_override.post(
                f"http://127.0.0.2:{port}/agents/search_agent/process", json={}
            )

        assert (excinfo.value.host, excinfo.value.port) == ("127.0.0.2", port)
        assert (
            f"Allow-listed: [localhost:8000/tcp (bound to 127.0.0.1:{port}), "
            "localhost:11434/tcp]" in str(excinfo.value)
        )
        assert child_runtime.requests == []

    async def test_runtime_down_fails_each_step_with_its_connect_error(
        self, child_runtime, telemetry_off
    ):
        dispatcher = deployed_dispatcher(child_runtime.url)
        orchestrator = await dispatcher._get_or_build_orchestrator(DEPLOYED_TENANTS[0])
        child_runtime.server.shutdown()
        child_runtime.server.server_close()

        results = await orchestrator._execute_plan(
            sequential_plan("runtime down"), tenant_id=DEPLOYED_TENANTS[0]
        )

        assert {
            name: result["status"] for name, result in results.items()
        } == dict.fromkeys(PLAN_AGENTS, "error")
        assert {
            name: result["message"].split(":", 1)[0] for name, result in results.items()
        } == dict.fromkeys(PLAN_AGENTS, "ConnectError")

    async def test_concurrent_tenants_dispatch_to_the_runtime_as_themselves(
        self, child_runtime, telemetry_off
    ):
        dispatcher = deployed_dispatcher(child_runtime.url)
        orchestrators = await asyncio.gather(
            *[dispatcher._get_or_build_orchestrator(t) for t in DEPLOYED_TENANTS]
        )
        queries = [f"marathon advice for {tenant}" for tenant in DEPLOYED_TENANTS]

        results = await asyncio.gather(
            *[
                orchestrator._execute_plan(sequential_plan(query), tenant_id=tenant)
                for orchestrator, query, tenant in zip(
                    orchestrators, queries, DEPLOYED_TENANTS
                )
            ]
        )

        assert results == [answered(tenant) for tenant in DEPLOYED_TENANTS]
        by_tenant = {
            tenant: [
                request
                for request in child_runtime.requests
                if request[1]["context"]["tenant_id"] == tenant
            ]
            for tenant in DEPLOYED_TENANTS
        }
        assert by_tenant == {
            tenant: dispatched(query, tenant)
            for query, tenant in zip(queries, DEPLOYED_TENANTS)
        }
        assert len(child_runtime.requests) == 2 * len(PLAN_AGENTS)
