"""Dispatcher's policy-enforcing httpx client denies non-allowlisted egress."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from cogniverse_runtime.sandbox_http import (
    EgressDeniedError,
    PolicyEnforcingTransport,
)
from cogniverse_runtime.sandbox_manager import SandboxManager, SandboxPolicy

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
