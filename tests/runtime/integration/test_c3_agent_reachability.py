"""F7.1 — C3 agents are reachable through the orchestrator's planner.

Audit caught: the 9 C3 agents were registered in ``AGENT_CLASSES`` and
appeared in ``configs/config.json`` but every one was ``enabled: false``,
so ``ConfigLoader.load_agents`` skipped all of them. The orchestrator's
planner queries ``registry.list_agents()`` for its
``available_agents`` field — agents that never registered never appear,
so an end-user query could not reach them.

This test verifies, against the real ConfigLoader + AgentRegistry +
orchestrator-discovery path:

  * ``audit_explanation_agent`` ships ``enabled=true`` by default
    (read-only, safe for production) so operators get one C3 agent
    reachable out of the box;
  * after the loader runs, ``audit_explanation_agent`` is in
    ``registry.list_agents()`` — i.e. the orchestrator's planner will
    see it as a routing target;
  * the other 8 C3 agents stay ``enabled=false`` by default (they
    write knowledge or require admin actor_role) — operators flip
    each individually;
  * flipping ``enabled=true`` on a previously-disabled C3 agent in
    config causes it to register on the next ConfigLoader.load_agents
    call (proves the opt-in path works) AND it shows up in the
    orchestrator's planner's available-agents list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_runtime.config_loader import ConfigLoader

pytestmark = pytest.mark.integration


_C3_AGENTS: List[Tuple[str, bool]] = [
    # (name, expected_default_enabled)
    ("citation_tracing_agent", False),
    ("contradiction_reconciliation_agent", False),
    ("multi_document_synthesis_agent", False),
    ("kg_traversal_agent", False),
    ("cross_tenant_comparison_agent", False),
    ("federated_query_agent", False),
    ("temporal_reasoning_agent", False),
    ("knowledge_summarization_agent", False),
    ("audit_explanation_agent", True),  # F7.1 — default enabled
]


@pytest.fixture(scope="module")
def config_json() -> dict:
    path = Path(__file__).resolve().parents[3] / "configs" / "config.json"
    with open(path) as f:
        return json.load(f)


class TestDefaultReachability:
    """Audit_explanation_agent must register out-of-the-box."""

    def test_audit_explanation_enabled_by_default(self, config_json: dict):
        entry = config_json["agents"]["audit_explanation_agent"]
        assert entry["enabled"] is True, (
            "audit_explanation_agent must default to enabled=true so "
            "an operator gets at least one C3 agent reachable through "
            "the orchestrator without any per-agent opt-in. Without "
            "this, every C3 agent stays orphan after the original "
            "registration commit."
        )

    def test_other_c3_agents_default_disabled(self, config_json: dict):
        for name, expected_enabled in _C3_AGENTS:
            if name == "audit_explanation_agent":
                continue  # covered above
            entry = config_json["agents"][name]
            assert entry["enabled"] is expected_enabled, (
                f"{name} default enabled={entry['enabled']!r}; expected "
                f"{expected_enabled!r}. Write-capable + admin-gated C3 "
                "agents must stay disabled-by-default so existing "
                "deployments don't suddenly serve new endpoints on "
                "upgrade."
            )

    def test_load_agents_registers_audit_agent(self, config_json: dict):
        # Use the real ConfigLoader path with the real AgentRegistry.
        cm = create_default_config_manager()
        registry = AgentRegistry(tenant_id="f71_default", config_manager=cm)
        loader = ConfigLoader.__new__(ConfigLoader)
        loader.config = {"agents": dict(config_json["agents"])}
        loader.config_manager = None
        loader.load_agents(agent_registry=registry)
        registered = set(registry.list_agents())
        assert "audit_explanation_agent" in registered, (
            "after the real loader runs against the shipped config.json, "
            "audit_explanation_agent must be registered so the "
            "orchestrator's planner discovers it"
        )


class TestOptInPattern:
    """When an operator flips a C3 agent to enabled, registration follows."""

    def test_flipping_enabled_registers_the_agent(self, config_json: dict):
        # Take the shipped config and overlay enabled=true on a
        # currently-disabled C3 agent.
        cm = create_default_config_manager()
        registry = AgentRegistry(tenant_id="f71_optin", config_manager=cm)
        overlay = dict(config_json["agents"])
        overlay["temporal_reasoning_agent"] = dict(
            overlay["temporal_reasoning_agent"], enabled=True
        )
        loader = ConfigLoader.__new__(ConfigLoader)
        loader.config = {"agents": overlay}
        loader.config_manager = None
        loader.load_agents(agent_registry=registry)
        registered = set(registry.list_agents())
        assert "temporal_reasoning_agent" in registered, (
            "operator flipped temporal_reasoning_agent to enabled=true "
            "but ConfigLoader still skipped it — the opt-in path is "
            "broken"
        )
        # And the disabled ones stay out — opt-in is per-agent.
        assert "knowledge_summarization_agent" not in registered

    def test_planner_sees_enabled_c3_in_available_agents(self, config_json: dict):
        """The orchestrator's planner builds its available_agents string
        from registry.list_agents(). Test that an enabled C3 agent
        appears in that list (i.e. the planner CAN consider routing
        to it). The planner's actual routing decision is DSPy-driven
        and out of scope here — what matters is discoverability."""
        cm = create_default_config_manager()
        registry = AgentRegistry(tenant_id="f71_planner", config_manager=cm)
        # Enable a few C3 agents to mirror an operator who's opted in.
        overlay = dict(config_json["agents"])
        for name in (
            "audit_explanation_agent",  # default-enabled
            "citation_tracing_agent",
            "kg_traversal_agent",
        ):
            overlay[name] = dict(overlay[name], enabled=True)
        loader = ConfigLoader.__new__(ConfigLoader)
        loader.config = {"agents": overlay}
        loader.config_manager = None
        loader.load_agents(agent_registry=registry)

        # Mimic what the orchestrator's _generate_orchestration_plan does:
        #   registered_agents = self.registry.list_agents()
        #   available_agents = ", ".join(registered_agents)
        registered = registry.list_agents()
        available_agents_str = ", ".join(registered)
        for required in (
            "audit_explanation_agent",
            "citation_tracing_agent",
            "kg_traversal_agent",
        ):
            assert required in available_agents_str, (
                f"{required} must appear in the orchestrator's "
                "available_agents list when enabled; otherwise the "
                "DSPy planner can't consider routing to it. Got: "
                f"{available_agents_str!r}"
            )


class TestDocumentationOfPattern:
    """A test of the docstring contract: the rest of the C3 agents are
    opt-in by design. This makes the disabled-by-default policy
    explicit and discoverable from the test name alone, so future
    contributors don't read it as a bug."""

    def test_write_capable_c3_agents_require_explicit_opt_in(self, config_json: dict):
        # These four are write-capable / admin-gated and must NEVER ship
        # default-enabled — they would change tenant or org-trunk state
        # the moment the runtime starts.
        for name in (
            "contradiction_reconciliation_agent",  # writes resolutions
            "multi_document_synthesis_agent",  # writes synthesis
            "knowledge_summarization_agent",  # writes to org trunk
            "cross_tenant_comparison_agent",  # admin actor_role
            "federated_query_agent",  # admin actor_role
        ):
            entry = config_json["agents"][name]
            assert entry["enabled"] is False, (
                f"{name} ships enabled=true but it is write-capable / "
                "admin-gated. Default-enabling it would change tenant "
                "state or expose admin-gated reads on first deploy. "
                "Operators must opt in explicitly per agent."
            )
