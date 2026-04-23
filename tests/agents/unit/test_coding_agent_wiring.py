"""Unit tests for CodingAgent wiring (audit fix #9).

Verifies that CodingAgent inherits MemoryAwareMixin and that its
``_process_impl`` calls ``inject_context_into_prompt`` to enrich the
coding task with the FULL context stack (instructions + learned
strategies + tenant memories) — same pattern as SearchAgent and
SummarizerAgent.

Before this fix CodingAgent had RLM but no memory wiring, so coding
tasks ran with only the raw user query and no learned context.
"""

import inspect

import pytest


@pytest.mark.unit
@pytest.mark.ci_fast
class TestCodingAgentMemoryWiring:
    def test_inherits_memory_aware_mixin(self):
        """CodingAgent must inherit from cogniverse_agents.memory_aware_mixin
        — the EXTENDED mixin with get_strategies, not the deleted base."""
        from cogniverse_agents.coding_agent import CodingAgent
        from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin

        assert issubclass(CodingAgent, MemoryAwareMixin)
        # Confirm it's the extended one with strategies.
        assert hasattr(MemoryAwareMixin, "get_strategies")

    def test_inherits_rlm_aware_mixin_too(self):
        """CodingAgent should retain its RLM wiring after the memory addition.
        Multi-mixin inheritance is the whole point of MRO, so confirm the
        new addition didn't accidentally drop a base class."""
        from cogniverse_agents.coding_agent import CodingAgent
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        assert issubclass(CodingAgent, RLMAwareMixin)

    def test_process_impl_calls_inject_context_into_prompt(self):
        """Pin the wiring at the source level. A regression where someone
        removes the inject_context_into_prompt call would silently drop
        learned strategies for the coding agent."""
        from cogniverse_agents.coding_agent import CodingAgent

        source = inspect.getsource(CodingAgent._process_impl)
        assert "inject_context_into_prompt" in source, (
            "CodingAgent._process_impl must call self.inject_context_into_prompt() "
            "to inject the FULL context stack. Audit fix #9 requires this — see "
            "docs/superpowers/audits/2026-04-07-orphan-and-wiring-audit.md"
        )
        assert "set_tenant_for_context" in source, (
            "CodingAgent._process_impl must call self.set_tenant_for_context() "
            "before inject_context_into_prompt() so the instructions/strategies "
            "are loaded for the right tenant"
        )

    def test_enriched_task_is_passed_to_planner(self):
        """The enriched task (output of inject_context_into_prompt) must
        flow into the planning step, not the raw input.task. Otherwise the
        wiring exists but is dead code."""
        from cogniverse_agents.coding_agent import CodingAgent

        source = inspect.getsource(CodingAgent._process_impl)
        # The enriched task must be passed to _plan(), not the raw input.task.
        # We look for "self._plan(enriched_task" or similar.
        assert "_plan(enriched_task" in source, (
            "_process_impl must pass `enriched_task` (the output of "
            "inject_context_into_prompt) to self._plan(), not the raw "
            "input.task. Otherwise the wiring is dead code."
        )
