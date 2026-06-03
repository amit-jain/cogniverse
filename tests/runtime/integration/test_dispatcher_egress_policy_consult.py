"""dispatcher consults egress policies for search/summarizer/routing.

Without this wire, the YAMLs in ``configs/agent_policies/`` for these three
agents were dead config: the dispatcher loaded them at boot but never
referenced them at dispatch. Now every dispatch path that hits one of
those three agents calls ``consult_egress_policy`` so the lookup is
observable and the lookup-side wire is alive.

Full egress *enforcement* (each agent routing its outbound HTTP through
the policy-enforcing transport) is a follow-up wire that requires
agent-internal HTTP rewiring; this commit is the first half — making
the policies discoverable at dispatch time and surfacing them via the
SandboxManager API the dispatcher already owns.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from cogniverse_runtime.sandbox_manager import SandboxManager, SandboxPolicy

pytestmark = pytest.mark.integration


@pytest.fixture
def sandbox_manager_with_real_policies() -> SandboxManager:
    """Real SandboxManager loading every YAML in configs/agent_policies/.

    Uses ``policy=OPTIONAL`` so the policies are loaded (DISABLED would
    short-circuit before the load step) but the boot gracefully degrades
    if the OpenShell gateway is unreachable — which is the expected
    state in test environments.
    """
    policies_dir = Path("configs/agent_policies")
    assert policies_dir.exists(), "configs/agent_policies/ must exist for this test"
    return SandboxManager(
        policy_dir=policies_dir,
        policy=SandboxPolicy.OPTIONAL,
    )


@pytest.fixture
def dispatcher_with_sandbox(
    sandbox_manager_with_real_policies: SandboxManager,
) -> AgentDispatcher:
    cm = create_default_config_manager()
    registry = AgentRegistry(tenant_id="p3_tenant", config_manager=cm)
    return AgentDispatcher(
        agent_registry=registry,
        config_manager=cm,
        schema_loader=None,
        sandbox_manager=sandbox_manager_with_real_policies,
    )


@pytest.fixture
def dispatcher_without_sandbox() -> AgentDispatcher:
    """Back-compat: deployments without a sandbox manager keep working."""
    cm = create_default_config_manager()
    registry = AgentRegistry(tenant_id="p3_tenant_nosandbox", config_manager=cm)
    return AgentDispatcher(
        agent_registry=registry,
        config_manager=cm,
        schema_loader=None,
        sandbox_manager=None,
    )


class TestConsultHelper:
    @pytest.mark.parametrize(
        "agent_name",
        ["search_agent", "summarizer_agent", "routing_agent"],
    )
    def test_existing_policy_yamls_are_consulted(
        self, dispatcher_with_sandbox, agent_name: str
    ):
        # The 3 YAMLs are shipped in configs/agent_policies/. The lookup must
        # surface them rather than returning None — that was the dead
        # state before this wire.
        policy = dispatcher_with_sandbox.consult_egress_policy(agent_name)
        if not (Path("configs/agent_policies") / f"{agent_name}.yaml").exists():
            pytest.skip(f"{agent_name}.yaml not shipped (deployment-specific)")
        assert policy is not None, (
            f"{agent_name} has a policy YAML but the dispatcher's lookup "
            "returned None — the wire is dead"
        )
        assert isinstance(policy, dict)

    def test_unknown_agent_returns_none(self, dispatcher_with_sandbox):
        # No YAML for this agent → lookup returns None (correct: no
        # policy is also a valid state, distinct from "policy of zero
        # rules").
        policy = dispatcher_with_sandbox.consult_egress_policy(
            "agent_with_no_policy_xyz"
        )
        assert policy is None

    def test_no_sandbox_manager_returns_none(self, dispatcher_without_sandbox):
        # Back-compat: deployments without a sandbox manager get None
        # for every lookup, never raise.
        policy = dispatcher_without_sandbox.consult_egress_policy("search_agent")
        assert policy is None


class TestExecutionPathInvocation:
    """Each of the 3 execute_* methods must consult the egress policy through
    the real SandboxManager boundary before constructing the agent.

    Rather than patch the dispatcher's own consult helper (which would only
    prove the dispatcher calls its own method), this wraps the real
    SandboxManager.get_policy collaborator so the assertion observes the
    dispatch path reaching the actual policy store AND surfacing the real
    YAML for the shipped agents.
    """

    @pytest.mark.parametrize(
        "method_name,agent_name",
        [
            ("_execute_search_task", "search_agent"),
            ("_execute_gateway_task", "routing_agent"),
            ("_execute_summarization_task", "summarizer_agent"),
        ],
    )
    @pytest.mark.asyncio
    async def test_dispatch_method_consults_policy(
        self,
        dispatcher_with_sandbox,
        sandbox_manager_with_real_policies,
        method_name: str,
        agent_name: str,
    ):
        seen: list[tuple[str, object]] = []
        real_get_policy = sandbox_manager_with_real_policies.get_policy

        def _recording_get_policy(name: str):
            policy = real_get_policy(name)
            seen.append((name, policy))
            return policy

        sandbox_manager_with_real_policies.get_policy = _recording_get_policy  # type: ignore[method-assign]

        method = getattr(dispatcher_with_sandbox, method_name)
        # The downstream agent execution fails (no Vespa/LLM wired here), but
        # the policy consult is the first line of each method, so it reaches
        # the SandboxManager before the failure point.
        try:
            if method_name == "_execute_search_task":
                await method("q", "p3_tenant", 1)
            elif method_name == "_execute_gateway_task":
                await method("q", {}, "p3_tenant")
            else:  # _execute_summarization_task
                await method("q", "p3_tenant")
        except Exception:
            pass

        consulted = {name: policy for name, policy in seen}
        assert agent_name in consulted, (
            f"{method_name} did not reach SandboxManager.get_policy for "
            f"{agent_name} before agent construction. Calls observed: "
            f"{list(consulted)}"
        )
        if (Path("configs/agent_policies") / f"{agent_name}.yaml").exists():
            policy = consulted[agent_name]
            assert isinstance(policy, dict), (
                f"{agent_name} ships a policy YAML but the dispatch-time "
                f"consult surfaced {policy!r} instead of the parsed policy dict"
            )
            egress = policy.get("network_policies", {}).get("egress")
            assert isinstance(egress, list) and egress, (
                f"{agent_name} policy surfaced no network_policies.egress "
                f"allow-list: {policy!r}"
            )
