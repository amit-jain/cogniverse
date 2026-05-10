"""Integration: dispatcher's policy-enforcing client denies non-allowlisted egress.

Closes the audit's PARTIAL on D.1 (orchestrator wire). The unit test in
``test_sandbox_policy.py:105`` asserts that ``make_http_client`` returns a
``PolicyEnforcingTransport``-wrapped client; this test asserts the
returned client actually denies a real outbound call when the request
targets a host:port outside the policy's egress allowlist, and conversely
lets allowlisted traffic through.

Architectural division (see also docs/operations/multi-tenant-ops.md):

  * **CodingAgent** → container isolation via ``exec_in_sandbox``
    (the only agent that runs LLM-generated code).
  * **OrchestratorAgent** → application-layer egress check via this
    policy-enforcing httpx client (the only non-coding agent that makes
    direct httpx calls — for A2A subagent dispatch).
  * **Search / Summarizer / Gateway** → CNI NetworkPolicy at the kernel
    boundary (those agents reach Vespa + the LLM via DSPy / pyvespa,
    libraries that do not expose hookable transports).
  * **All agents** → dispatch-time validation
    (``AgentDispatcher.validate_dispatch_endpoints``) catches operator
    config drift at boot, regardless of the runtime enforcement layer.
"""

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
    """Tmp dir holding a single orchestrator_agent.yaml policy."""
    (tmp_path / "orchestrator_agent.yaml").write_text(_ORCHESTRATOR_POLICY_YAML)
    return tmp_path


@pytest.fixture
def sandbox_mgr(policy_dir: Path) -> SandboxManager:
    """SandboxManager pointed at the tmp policy dir.

    ``policy=optional`` loads the policy file AND attempts to connect
    to the OpenShell gateway. The connect fails silently in CI (no
    openshell binary), but ``make_http_client`` does not require the
    gateway — only the loaded policy dict. ``optional`` is the right
    posture for the tests that need wrapping behaviour;
    ``disabled`` short-circuits ``_load_policies`` and so cannot be
    re-armed mid-test.
    """
    return SandboxManager(policy_dir=policy_dir, policy=SandboxPolicy.OPTIONAL)


@pytest.fixture
def sandbox_mgr_disabled(policy_dir: Path) -> SandboxManager:
    """SandboxManager constructed with ``policy=disabled``.

    Used to verify the disabled-mode contract: ``make_http_client``
    must hand back a bare httpx client (no wrapping) so operators
    can opt out of application-layer enforcement.
    """
    return SandboxManager(policy_dir=policy_dir, policy=SandboxPolicy.DISABLED)


class TestPolicyEnforcingClient:
    """The client returned by ``make_http_client`` enforces the YAML policy."""

    def test_returned_client_carries_policy_enforcing_transport(self, sandbox_mgr):
        client = sandbox_mgr.make_http_client("orchestrator_agent")
        try:
            assert isinstance(client._transport, PolicyEnforcingTransport), (
                "make_http_client must wrap the transport when an "
                f"agent has a policy; got {type(client._transport).__name__}"
            )
        finally:
            # AsyncClient.aclose is async; the sync .close() is a no-op
            # but we drop the reference for GC.
            del client

    @pytest.mark.asyncio
    async def test_off_allowlist_request_raises_egress_denied(self, sandbox_mgr):
        """An A2A POST to a host:port outside the allowlist is blocked."""
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
                "the deny message must surface what IS allowed so the "
                "operator can fix the policy YAML; got: " + str(err)
            )
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_allowlisted_request_passes_through_to_inner_transport(
        self, sandbox_mgr
    ):
        """An allowlisted endpoint reaches the inner transport.

        Nobody is listening on localhost:8000 in this test, so the
        inner transport raises ``ConnectError``. That is the *correct*
        outcome: it proves we got past the policy check (a deny would
        raise ``EgressDeniedError`` with no inner transport call).
        """
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
        """An agent without a policy file gets a bare httpx client.

        The dispatcher still calls ``make_http_client`` for every
        memory-aware agent; when there is no YAML, the call must
        return an unwrapped client (no surprise wrapping with an
        empty allowlist that would deny everything).
        """
        client = sandbox_mgr.make_http_client("agent_with_no_policy_file")
        try:
            assert not isinstance(client._transport, PolicyEnforcingTransport), (
                "an agent without a policy file must get a bare httpx "
                "client — wrapping with an empty policy would deny all "
                "egress and break unrelated agents"
            )
        finally:
            del client

    @pytest.mark.asyncio
    async def test_disabled_sandbox_returns_bare_client(self, sandbox_mgr_disabled):
        """When ``sandbox.policy=disabled``, no wrapping (operator opt-out)."""
        client = sandbox_mgr_disabled.make_http_client("orchestrator_agent")
        try:
            assert not isinstance(client._transport, PolicyEnforcingTransport), (
                "policy=disabled must hand back a bare httpx client so "
                "operators can flip enforcement off without code changes"
            )
        finally:
            del client


class TestEnforcementEnvVarOverride:
    """Operators can disable application-layer enforcement at boot."""

    @pytest.mark.asyncio
    async def test_env_var_disabled_returns_bare_client(self, sandbox_mgr, monkeypatch):
        """COGNIVERSE_OPENSHELL_HTTP_ENFORCEMENT=disabled → bare client.

        Useful for dev iteration: the operator can hot-toggle
        application-layer enforcement off (CNI still enforces) without
        editing every agent's policy.
        """
        monkeypatch.setenv("COGNIVERSE_OPENSHELL_HTTP_ENFORCEMENT", "disabled")
        client = sandbox_mgr.make_http_client("orchestrator_agent")
        try:
            assert not isinstance(client._transport, PolicyEnforcingTransport), (
                "COGNIVERSE_OPENSHELL_HTTP_ENFORCEMENT=disabled must "
                "hand back a bare client; otherwise the env var is a lie"
            )
        finally:
            del client
