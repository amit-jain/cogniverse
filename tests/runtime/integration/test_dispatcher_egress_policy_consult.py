"""P3 — dispatcher consults egress policies for search/summarizer/routing.

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
    """Each of the 3 execute_* methods must call consult_egress_policy.

    We patch the helper to a recorder, fire the dispatch path's pre-flight
    setup (the helper is called BEFORE the agent is constructed so we can
    short-circuit by having the registry not contain the agent — the
    consult call still fires).
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
        self, dispatcher_with_sandbox, method_name: str, agent_name: str
    ):
        seen: list = []
        original = dispatcher_with_sandbox.consult_egress_policy

        def _spy(name: str):
            seen.append(name)
            return original(name)

        dispatcher_with_sandbox.consult_egress_policy = _spy  # type: ignore[method-assign]

        method = getattr(dispatcher_with_sandbox, method_name)
        # Best-effort call; the downstream agent execution will likely fail
        # because we haven't set up Vespa/LLM/etc., but the policy consult
        # happens in the very first line of each method — before the failure
        # point.
        try:
            if method_name == "_execute_search_task":
                await method("q", "p3_tenant", 1)
            elif method_name == "_execute_gateway_task":
                await method("q", {}, "p3_tenant")
            else:  # _execute_summarization_task
                await method("q", "p3_tenant")
        except Exception:
            pass  # downstream failures are expected; the wire ran first

        assert agent_name in seen, (
            f"{method_name} did not consult the {agent_name} egress policy "
            f"before agent construction. Calls observed: {seen}"
        )
