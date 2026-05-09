"""F2.1 — dispatcher hands the per-request artefact overlay to the agent.

The previous audit caught: dispatcher computes the canary/variant
decision, stashes it in ``context["_artefact_overlay"]``, but no agent
ever reads that key — the overlay was a write-only black hole.

This test verifies the consumer wire (in two halves):
  * ``MemoryAwareMixin.set_dispatched_artefact`` is the public hook
    every memory-aware agent now exposes;
  * ``AgentDispatcher._apply_artefact_overlay`` calls that hook with
    the context overlay; the generic agent execution path invokes it
    after constructing the agent.

End result: an agent dispatched with a canary-on or variant-selected
context now has ``self._dispatched_artefact`` populated, and
``self.get_dispatched_prompts()`` returns the overlay's prompts.
Agents migrate from default-only to overlay-aware loading at their
own pace; the wire is observable today.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
from cogniverse_runtime.agent_dispatcher import AgentDispatcher

pytestmark = pytest.mark.integration


class _Agent(MemoryAwareMixin):
    """Bare memory-aware agent — exposes the new hook."""


class _NonMemoryAgent:
    """Plain object — no MemoryAwareMixin. Must silently no-op."""


class TestSetDispatchedArtefact:
    def test_overlay_stored_on_agent(self):
        agent = _Agent()
        overlay = {
            "served_from": "canary",
            "version": 3,
            "variant_id": "with_jurisdiction",
            "prompts": {"system": "VARIANT_PROMPT"},
        }
        agent.set_dispatched_artefact(overlay)
        assert agent._dispatched_artefact == overlay

    def test_get_dispatched_prompts_returns_overlay_prompts(self):
        agent = _Agent()
        agent.set_dispatched_artefact(
            {"prompts": {"system": "X"}, "served_from": "canary"}
        )
        assert agent.get_dispatched_prompts() == {"system": "X"}

    def test_get_dispatched_prompts_returns_none_when_unset(self):
        agent = _Agent()
        assert agent.get_dispatched_prompts() is None

    def test_get_dispatched_prompts_none_when_overlay_has_no_prompts(self):
        agent = _Agent()
        agent.set_dispatched_artefact({"served_from": "default", "version": None})
        assert agent.get_dispatched_prompts() is None


class TestDispatcherInjection:
    """Dispatcher's _apply_artefact_overlay wires the hook to the context."""

    def test_overlay_reaches_memory_aware_agent(self):
        agent = _Agent()
        context = {
            "_artefact_overlay": {
                "prompts": {"system": "FROM_OVERLAY"},
                "served_from": "canary",
                "version": 2,
                "variant_id": "default",
            }
        }
        AgentDispatcher._apply_artefact_overlay(agent, context)
        assert agent._dispatched_artefact["served_from"] == "canary"
        assert agent.get_dispatched_prompts() == {"system": "FROM_OVERLAY"}

    def test_no_overlay_in_context_is_no_op(self):
        agent = _Agent()
        AgentDispatcher._apply_artefact_overlay(agent, {})
        # Hook never called → attribute never set.
        assert getattr(agent, "_dispatched_artefact", None) is None

    def test_no_context_at_all_is_no_op(self):
        agent = _Agent()
        AgentDispatcher._apply_artefact_overlay(agent, None)
        assert getattr(agent, "_dispatched_artefact", None) is None

    def test_non_memory_agent_silently_skipped(self):
        # Plain object without the mixin must not crash the dispatcher.
        agent = _NonMemoryAgent()
        context = {
            "_artefact_overlay": {
                "prompts": {"system": "X"},
                "served_from": "canary",
            }
        }
        # Should not raise.
        AgentDispatcher._apply_artefact_overlay(agent, context)
        # Attribute should not be set on the non-mixin object either.
        assert not hasattr(agent, "_dispatched_artefact")

    def test_setter_exception_does_not_propagate(self):
        # An agent whose setter raises must not break the dispatch.
        class _BoomAgent(_Agent):
            def set_dispatched_artefact(self, overlay):
                raise RuntimeError("intentional")

        agent = _BoomAgent()
        context = {
            "_artefact_overlay": {
                "prompts": {"system": "X"},
                "served_from": "canary",
            }
        }
        # Must not raise — overlay injection is best-effort.
        AgentDispatcher._apply_artefact_overlay(agent, context)


class TestEndToEndWire:
    """Trace dispatcher → resolve_artefact_for_request → context → agent."""

    @pytest.mark.asyncio
    async def test_resolve_then_apply_round_trip(self):
        """The full chain: dispatcher resolves the overlay AND injects it."""
        from cogniverse_agents.optimizer.signature_variants import (
            DEFAULT_VARIANT_ID,
        )

        # Build a fake ArtifactManager whose load_for_request returns a
        # canary decision. We don't need real Phoenix here — that's
        # covered by test_dispatcher_canary_routing.py. This test
        # exclusively covers the F2.1 inject-into-agent step.
        class _StubAM:
            async def load_for_request(self, agent_type, *, request_seed, variant_id):
                return {
                    "served_from": "canary",
                    "version": 7,
                    "variant_id": variant_id,
                    "prompts": {"system": "FROM_RESOLVED_CANARY"},
                }

        from cogniverse_core.registries.agent_registry import AgentRegistry
        from cogniverse_foundation.config.utils import (
            create_default_config_manager,
        )

        cm = create_default_config_manager()
        registry = AgentRegistry(tenant_id="f21_test", config_manager=cm)
        dispatcher = AgentDispatcher(
            agent_registry=registry,
            config_manager=cm,
            schema_loader=None,
            artifact_manager_factory=lambda _t: _StubAM(),
        )
        agent = _Agent()
        # Mimic what dispatch() does: resolve the overlay, stash in context,
        # then inject onto the agent.
        overlay = await dispatcher.resolve_artefact_for_request(
            "search_agent", "f21_test", request_seed="seed_1"
        )
        context: Dict[str, Any] = {"_artefact_overlay": overlay}
        AgentDispatcher._apply_artefact_overlay(agent, context)

        assert agent._dispatched_artefact["served_from"] == "canary"
        assert agent._dispatched_artefact["version"] == 7
        assert agent._dispatched_artefact["variant_id"] == DEFAULT_VARIANT_ID
        assert agent.get_dispatched_prompts() == {"system": "FROM_RESOLVED_CANARY"}
