"""dispatch-time egress policy enforcement for shared agents.

Audit found search/summarizer/routing only logged the egress policy at
dispatch; nothing actually verified that the destinations the agent
would reach are on the allowlist. The plan's verification was: "(b) a
non-allow-listed outbound is denied at the sandbox boundary."

Per-agent enforcement matrix today:

  * **CodingAgent** → container isolation (``exec_in_sandbox``). The
    only agent that runs LLM-generated code; container is the right
    boundary regardless of egress policy.
  * **OrchestratorAgent** → application-layer egress check via
    ``SandboxManager.make_http_client("orchestrator_agent")`` →
    ``PolicyEnforcingTransport``. Wired and tested in
    ``test_orchestrator_egress_enforcement.py``.
  * **Search / Summarizer / Gateway** → egress goes through
    ``pyvespa.Vespa()`` and DSPy's LM client. Both expose hookable
    transports (``Vespa(client=httpx.AsyncClient(...))``,
    ``litellm`` accepts a custom client) but the cogniverse-vespa
    backend factory and the dspy.LM construction sites do not yet
    plumb them through. Until they do, CNI NetworkPolicy is the
    only enforcement layer for these agents at runtime.
  * **All agents** → dispatch-time validation (this test) catches
    operator config drift at boot regardless of the runtime layer.

The CNI / kernel layer (the plan's unified-runtime NetworkPolicy) is
the real deny mechanism in production for any agent — packets leaving
the pod are dropped by Cilium / Calico before the destination is
reached. The application layer is defence-in-depth: it catches dev /
test misconfiguration (where NetworkPolicy doesn't apply) and produces
sub-millisecond ``EgressDeniedError`` exceptions with the policy file
named, instead of 30-second TCP timeouts.

This test exercises ``validate_dispatch_endpoints`` directly with two
realistic policies: one where the endpoints align (no violations)
and one where they drift (violation reported with reason).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher

pytestmark = pytest.mark.integration


class _StubSandboxManager:
    """Minimal SandboxManager stand-in that returns a fixed policy."""

    def __init__(self, policies: Dict[str, Dict[str, Any]]):
        self._policies = policies

    def get_policy(self, agent_name: str) -> Optional[Dict[str, Any]]:
        return self._policies.get(agent_name)


def _build_dispatcher(policies: Dict[str, Dict[str, Any]]) -> AgentDispatcher:
    cm = create_default_config_manager()
    from cogniverse_core.registries.agent_registry import AgentRegistry

    return AgentDispatcher(
        agent_registry=AgentRegistry(tenant_id="h12_test", config_manager=cm),
        config_manager=cm,
        schema_loader=None,
        sandbox_manager=_StubSandboxManager(policies),
    )


class TestEndpointValidation:
    def test_allowed_endpoints_produce_no_violations(self):
        policy = {
            "network_policies": {
                "egress": [
                    {"host": "localhost", "port": 8080, "protocol": "tcp"},
                    {"host": "localhost", "port": 11434, "protocol": "tcp"},
                ],
                "deny_all_other": True,
            }
        }
        dispatcher = _build_dispatcher({"search_agent": policy})
        violations = dispatcher.validate_dispatch_endpoints(
            "search_agent",
            [
                {"host": "localhost", "port": 8080},
                {"host": "localhost", "port": 11434},
            ],
        )
        assert violations == [], (
            f"endpoints on the allowlist must not produce violations; "
            f"got {violations!r}"
        )

    def test_off_allowlist_endpoint_is_reported(self):
        policy = {
            "network_policies": {
                "egress": [
                    {"host": "localhost", "port": 8080, "protocol": "tcp"},
                ],
                "deny_all_other": True,
            }
        }
        dispatcher = _build_dispatcher({"search_agent": policy})
        violations = dispatcher.validate_dispatch_endpoints(
            "search_agent",
            [
                {"host": "localhost", "port": 8080},  # ok
                {"host": "evil.example.com", "port": 443},  # not allowed
            ],
        )
        assert len(violations) == 1, (
            f"expected exactly one violation for the off-allowlist host; "
            f"got {violations!r}"
        )
        assert violations[0]["host"] == "evil.example.com"
        assert violations[0]["port"] == 443
        assert "not in egress allowlist" in violations[0]["reason"]

    def test_protocol_mismatch_is_a_violation(self):
        # The allowlist says TCP-only; a UDP call to the same port is
        # still off-allowlist and must be flagged.
        policy = {
            "network_policies": {
                "egress": [
                    {"host": "localhost", "port": 53, "protocol": "tcp"},
                ],
                "deny_all_other": True,
            }
        }
        dispatcher = _build_dispatcher({"search_agent": policy})
        violations = dispatcher.validate_dispatch_endpoints(
            "search_agent",
            [{"host": "localhost", "port": 53, "protocol": "udp"}],
        )
        assert len(violations) == 1, (
            f"protocol mismatch must produce a violation; got {violations!r}"
        )

    def test_no_policy_means_no_violations(self):
        # Agent has no policy at all → validate is a no-op (the audit
        # path is opt-in per agent; absence of policy isn't itself a
        # violation).
        dispatcher = _build_dispatcher(policies={})
        violations = dispatcher.validate_dispatch_endpoints(
            "unregistered_agent",
            [{"host": "anywhere.example.com", "port": 9999}],
        )
        assert violations == []

    def test_missing_sandbox_manager_short_circuits(self):
        cm = create_default_config_manager()
        from cogniverse_core.registries.agent_registry import AgentRegistry

        dispatcher = AgentDispatcher(
            agent_registry=AgentRegistry(tenant_id="h12_test", config_manager=cm),
            config_manager=cm,
            schema_loader=None,
            sandbox_manager=None,
        )
        violations = dispatcher.validate_dispatch_endpoints(
            "search_agent",
            [{"host": "x", "port": 1}],
        )
        assert violations == [], (
            "without a sandbox_manager, validation must be a no-op so "
            "dev / test setups that don't wire OpenShell don't fail "
            "every dispatch"
        )


class TestRealPolicyShape:
    """Use the actual configs/agent_policies/search_agent.yaml shape so
    the test catches a regression in the YAML format."""

    def test_actual_search_agent_policy_validates(self):
        from pathlib import Path

        import yaml

        policy_path = Path("configs/agent_policies/search_agent.yaml")
        with open(policy_path) as f:
            policy = yaml.safe_load(f)
        dispatcher = _build_dispatcher({"search_agent": policy})

        # The shipped policy allows localhost:8080 (Vespa) + localhost:11434 (LLM).
        violations = dispatcher.validate_dispatch_endpoints(
            "search_agent",
            [
                {"host": "localhost", "port": 8080, "protocol": "tcp"},
                {"host": "localhost", "port": 11434, "protocol": "tcp"},
            ],
        )
        assert violations == [], (
            f"the shipped search_agent policy must allow Vespa+LLM by "
            f"default; got violations={violations!r}"
        )
