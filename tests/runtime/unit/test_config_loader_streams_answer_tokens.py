"""The shipped config's token-streaming declarations reach the agent registry.

``ConfigLoader.load_agents`` is the only writer of ``streams_answer_tokens`` on
a runtime endpoint: an agent that declares it in ``configs/config.json`` must
come out of the registry claiming token streaming, and every other shipped
agent must not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_runtime.config_loader import ConfigLoader

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

# The agents whose answers are produced token by token by their DSPy call, so a
# streaming client receives tokens rather than one final chunk.
_STREAMING_AGENTS = frozenset(
    {"deep_research_agent", "detailed_report_agent", "summarizer_agent"}
)


def _shipped_agents() -> dict:
    root = Path(__file__).resolve().parents[3]
    return json.loads((root / "configs" / "config.json").read_text())["agents"]


def _load(agents: dict) -> AgentRegistry:
    registry = AgentRegistry(
        tenant_id="acme:config-loader-streaming", config_manager=object()
    )
    loader = ConfigLoader.__new__(ConfigLoader)
    loader.config = {"agents": agents}
    loader.config_manager = None
    loader.load_agents(agent_registry=registry)
    return registry


def test_shipped_config_declares_streaming_for_exactly_these_agents():
    declared = {
        name
        for name, entry in _shipped_agents().items()
        if entry.get("streams_answer_tokens") is True
    }

    assert declared == set(_STREAMING_AGENTS)


def test_every_shipped_agent_resolves_its_declared_flag():
    agents = {
        name: {**entry, "enabled": True} for name, entry in _shipped_agents().items()
    }

    registry = _load(agents)

    resolved = {
        name: registry.get_agent(name).streams_answer_tokens
        for name in registry.list_agents()
    }
    assert resolved == {
        name: name in _STREAMING_AGENTS
        for name in agents
        if name in ConfigLoader.AGENT_CLASSES
    }


def test_declaration_is_not_inferred_from_capabilities():
    """Only the config key decides; an agent with the same capabilities but no
    declaration stays off."""
    shipped = _shipped_agents()
    summarizer = {**shipped["summarizer_agent"], "enabled": True}
    twin = {k: v for k, v in summarizer.items() if k != "streams_answer_tokens"}

    registry = _load({"summarizer_agent": summarizer, "text_analysis_agent": twin})

    assert registry.get_agent("summarizer_agent").streams_answer_tokens is True
    assert registry.get_agent("text_analysis_agent").streams_answer_tokens is False
