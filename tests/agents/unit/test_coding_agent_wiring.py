"""Unit tests for CodingAgent memory/context wiring (audit fix #9).

CodingAgent must inherit MemoryAwareMixin (the extended one with strategies)
and its ``_process_impl`` must enrich the task via inject_context_into_prompt
and pass the ENRICHED task to planning — not the raw input. The wiring is
verified by *executing* ``_process_impl`` with the calls spied, not by grepping
the source (which would pass even if the call were dead).
"""

from unittest.mock import AsyncMock, Mock

import pytest

from cogniverse_agents.coding_agent import CodingAgent, CodingInput


@pytest.mark.unit
@pytest.mark.ci_fast
class TestCodingAgentMemoryWiring:
    def test_inherits_memory_aware_mixin(self):
        """CodingAgent must inherit the EXTENDED MemoryAwareMixin (with
        get_strategies), not the deleted base."""
        from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin

        assert issubclass(CodingAgent, MemoryAwareMixin)
        assert hasattr(MemoryAwareMixin, "get_strategies")

    def test_inherits_rlm_aware_mixin_too(self):
        """Adding the memory mixin must not drop the RLM base (MRO check)."""
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        assert issubclass(CodingAgent, RLMAwareMixin)

    @pytest.mark.asyncio
    async def test_process_impl_enriches_task_and_passes_it_to_plan(self):
        """Run _process_impl with the context wiring spied: it must set the
        tenant, call inject_context_into_prompt(task), and pass the ENRICHED
        result — not the raw task — into _plan. Aborts at _plan to stay focused
        on the wiring rather than the downstream code-generation loop.
        """
        agent = object.__new__(CodingAgent)  # bare instance: exercise wiring only
        captured: dict = {}

        agent.set_tenant_for_context = Mock(
            side_effect=lambda t: captured.__setitem__("tenant", t)
        )
        agent.inject_context_into_prompt = Mock(return_value="ENRICHED_TASK")
        agent.emit_progress = Mock()
        agent._search_code_context = AsyncMock(return_value=[])

        class _StopAfterPlan(Exception):
            pass

        async def _capture_plan(enriched, code_context, language):
            captured["plan_task"] = enriched
            raise _StopAfterPlan

        agent._plan = _capture_plan

        inp = CodingInput(task="write a quicksort", tenant_id="acme:prod")
        with pytest.raises(_StopAfterPlan):
            await agent._process_impl(inp)

        # Tenant set for context; raw task enriched; the ENRICHED task (not the
        # raw one) is what reaches the planner — proving the wiring is live.
        assert captured["tenant"] == "acme:prod"
        agent.inject_context_into_prompt.assert_called_once_with(
            "write a quicksort", "write a quicksort"
        )
        assert captured["plan_task"] == "ENRICHED_TASK"
