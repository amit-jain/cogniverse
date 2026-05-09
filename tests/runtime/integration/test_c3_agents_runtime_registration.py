"""C3.x runtime registration — real ConfigLoader.load_agents round-trip.

This is the wire-up test for the 9 knowledge-system agents
(C3.1–C3.9). It verifies that:

  * each agent's class path is resolvable from ``ConfigLoader.AGENT_CLASSES``
    using the *real* importlib path (not a stubbed mapping);
  * each agent appears in ``configs/config.json`` under the ``agents``
    section so operators can flip ``enabled: true`` per deployment;
  * when their config flag is enabled, ``load_agents`` registers them in
    a real ``AgentRegistry`` with the right capability strings;
  * the registered URL matches the documented per-agent port (8019–8027)
    so an A2A caller hitting that port reaches the right agent.

Without this test, the C3 agents were shipped as orphan classes —
present in the source tree, never reachable through the runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pytest

from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_runtime.config_loader import ConfigLoader

pytestmark = pytest.mark.integration


_C3_AGENTS: Dict[str, Tuple[int, str]] = {
    # name -> (port, primary_capability)
    "citation_tracing_agent": (8019, "citation_tracing"),
    "contradiction_reconciliation_agent": (8020, "contradiction_reconciliation"),
    "multi_document_synthesis_agent": (8021, "multi_document_synthesis"),
    "kg_traversal_agent": (8022, "knowledge_graph_traversal"),
    "cross_tenant_comparison_agent": (8023, "cross_tenant_comparison"),
    "federated_query_agent": (8024, "federated_query"),
    "temporal_reasoning_agent": (8025, "temporal_reasoning"),
    "knowledge_summarization_agent": (8026, "knowledge_summarization"),
    "audit_explanation_agent": (8027, "audit_explanation"),
}


@pytest.fixture(scope="module")
def config_json() -> dict:
    path = Path(__file__).resolve().parents[3] / "configs" / "config.json"
    with open(path) as f:
        return json.load(f)


class TestAgentClassesMapping:
    @pytest.mark.parametrize("agent_name", sorted(_C3_AGENTS))
    def test_class_path_present_and_importable(self, agent_name: str):
        # AGENT_CLASSES must list the agent.
        class_path = ConfigLoader.AGENT_CLASSES.get(agent_name)
        assert class_path is not None, (
            f"{agent_name} missing from ConfigLoader.AGENT_CLASSES — "
            "agent will be silently skipped at runtime startup"
        )
        # The class must actually import — no fictional paths.
        module_path, class_name = class_path.split(":")
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name, None)
        assert cls is not None, (
            f"{class_path} resolves but the class {class_name!r} is missing"
        )


class TestConfigJsonEntries:
    @pytest.mark.parametrize("agent_name,port_cap", sorted(_C3_AGENTS.items()))
    def test_entry_present_with_correct_port_and_capability(
        self, config_json: dict, agent_name: str, port_cap: Tuple[int, str]
    ):
        port, primary_cap = port_cap
        entry = config_json.get("agents", {}).get(agent_name)
        assert entry is not None, (
            f"{agent_name} missing from configs/config.json — operators "
            "have no way to enable it without editing source"
        )
        # URL must match the documented port so A2A routing reaches the right port.
        assert entry["url"] == f"http://localhost:{port}", (
            f"{agent_name} url={entry['url']} does not match documented port {port}"
        )
        assert primary_cap in entry.get("capabilities", []), (
            f"{agent_name} config is missing primary capability {primary_cap!r}; "
            "the orchestrator's planner relies on capability strings to "
            "select the right agent"
        )

    def test_default_disabled(self, config_json: dict):
        # New agents ship disabled-by-default so existing deployments
        # don't suddenly start serving 9 new endpoints on upgrade.
        for agent_name in _C3_AGENTS:
            entry = config_json["agents"][agent_name]
            assert entry.get("enabled") is False, (
                f"{agent_name} should default to enabled=False; existing "
                "deployments should opt in explicitly"
            )


class TestLoadAgentsRoundTrip:
    """Execute the real load path with the real registry."""

    def _build_loader_with_overlay(self, overlay_agents: dict) -> ConfigLoader:
        """Build a ConfigLoader whose `agents` section is replaced by overlay."""
        loader = ConfigLoader.__new__(ConfigLoader)
        # We bypass __init__ to avoid the SystemConfig dependency; load_agents
        # only reads self.config["agents"], so a minimal config dict suffices.
        loader.config = {"agents": overlay_agents}
        loader.config_manager = None
        return loader

    def test_disabled_agents_skipped(self, config_json: dict):
        # When all 9 are disabled (the default), none get registered.
        registry = AgentRegistry(
            tenant_id="c3_runtime_registration_test",
            config_manager=create_default_config_manager(),
        )
        loader = self._build_loader_with_overlay(
            {
                name: {**config_json["agents"][name], "enabled": False}
                for name in _C3_AGENTS
            }
        )
        loader.load_agents(agent_registry=registry)
        for name in _C3_AGENTS:
            assert registry.get_agent(name) is None

    def test_enabled_agents_registered_with_correct_metadata(self, config_json: dict):
        registry = AgentRegistry(
            tenant_id="c3_runtime_registration_test",
            config_manager=create_default_config_manager(),
        )
        loader = self._build_loader_with_overlay(
            {
                name: {**config_json["agents"][name], "enabled": True}
                for name in _C3_AGENTS
            }
        )
        loader.load_agents(agent_registry=registry)

        for name, (port, primary_cap) in _C3_AGENTS.items():
            ep = registry.get_agent(name)
            assert ep is not None, (
                f"{name} was enabled in config but ConfigLoader.load_agents "
                "did not register it in the AgentRegistry"
            )
            assert ep.url == f"http://localhost:{port}"
            assert primary_cap in ep.capabilities
            assert ep.health_endpoint == "/health"

    def test_all_nine_register_in_one_pass(self, config_json: dict):
        """Single-shot: enable all 9, verify list_agents returns them."""
        registry = AgentRegistry(
            tenant_id="c3_runtime_registration_test",
            config_manager=create_default_config_manager(),
        )
        loader = self._build_loader_with_overlay(
            {
                name: {**config_json["agents"][name], "enabled": True}
                for name in _C3_AGENTS
            }
        )
        loader.load_agents(agent_registry=registry)
        registered = set(registry.list_agents())
        missing = set(_C3_AGENTS) - registered
        assert not missing, (
            f"After enabling all 9 C3 agents, these did not register: {missing}"
        )
