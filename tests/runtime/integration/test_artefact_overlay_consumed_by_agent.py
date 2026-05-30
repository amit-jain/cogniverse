"""dispatcher overlay reaches the agent AND shapes the DSPy call.

Two halves verified end-to-end:

  * **Wire** — ``MemoryAwareMixin.set_dispatched_artefact`` /
    ``get_dispatched_prompts`` plumb the overlay; the dispatcher's
    ``_apply_artefact_overlay`` calls the setter from search /
    orchestration / summarization paths (not just the generic path).
  * **Consumer** — ``AgentBase.call_dspy`` consults
    ``get_dispatched_prompts()`` and applies the overlay's
    instructions to the matching predictor for the duration of the
    call, then restores. This is what makes per-tenant canary +
    signature-variant selection actually shape agent output instead
    of sitting in a context dict no one reads.
"""

from __future__ import annotations

from typing import Any, Dict

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
        assert agent.get_dispatched_artefact() == overlay

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
        assert agent.get_dispatched_artefact()["served_from"] == "canary"
        assert agent.get_dispatched_prompts() == {"system": "FROM_OVERLAY"}

    def test_no_overlay_in_context_is_no_op(self):
        agent = _Agent()
        AgentDispatcher._apply_artefact_overlay(agent, {})
        # Hook never called → attribute never set.
        assert agent.get_dispatched_artefact() is None

    def test_no_context_at_all_is_no_op(self):
        agent = _Agent()
        AgentDispatcher._apply_artefact_overlay(agent, None)
        assert agent.get_dispatched_artefact() is None

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
        # exclusively covers the inject-into-agent step.
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

        assert agent.get_dispatched_artefact()["served_from"] == "canary"
        assert agent.get_dispatched_artefact()["version"] == 7
        assert agent.get_dispatched_artefact()["variant_id"] == DEFAULT_VARIANT_ID
        assert agent.get_dispatched_prompts() == {"system": "FROM_RESOLVED_CANARY"}


class TestPromptOverlayShapesDSPyCall:
    """the overlay must change what the LM actually sees, not just
    what's stored on the agent. Without this assertion, the entire
    canary + variant pipeline is a write-only black hole even after
    the wire reaches the agent.
    """

    @pytest.mark.asyncio
    async def test_overlay_swaps_predictor_instructions_during_call(self):
        """A real DSPy module + a recording fake LM proves the overlay
        instructions land in the prompt sent to the LM, and that the
        original instructions are restored after the call.
        """
        import dspy

        from cogniverse_core.agents.base import (
            AgentBase,
            AgentDeps,
            AgentInput,
            AgentOutput,
        )

        class _Sig(dspy.Signature):
            """ORIGINAL_INSTRUCTIONS"""

            query: str = dspy.InputField()
            answer: str = dspy.OutputField()

        class _Module(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predictor = dspy.Predict(_Sig)

            def forward(self, query: str):
                return self.predictor(query=query)

        class _RecordingLM(dspy.LM):
            """Captures every prompt sent to it; returns a stub answer."""

            def __init__(self):
                super().__init__(model="stub/test", api_base="none", api_key="none")
                self.captured_messages: list = []

            def __call__(self, prompt=None, messages=None, **kwargs):
                if messages is not None:
                    self.captured_messages.append(messages)
                else:
                    self.captured_messages.append(prompt)
                return ["[[ ## answer ## ]]\nstub_answer\n[[ ## completed ## ]]"]

        class _NullDeps(AgentDeps):
            pass

        class _NullInput(AgentInput):
            pass

        class _NullOutput(AgentOutput):
            pass

        class _OverlayAgent(AgentBase[_NullInput, _NullOutput, _NullDeps]):
            _input_type = _NullInput
            _output_type = _NullOutput

            async def _process_impl(self, input_data: _NullInput) -> _NullOutput:
                raise NotImplementedError

            # Stand in for MemoryAwareMixin's hook so call_dspy can find it.
            def get_dispatched_prompts(self):
                return self._overlay

            def __init__(self, deps):
                super().__init__(deps=deps)
                self._overlay = None

        agent = _OverlayAgent(deps=_NullDeps())
        module = _Module()
        lm = _RecordingLM()

        # Sanity: pre-overlay, the predictor's instructions are the original.
        assert module.predictor.signature.instructions == "ORIGINAL_INSTRUCTIONS"

        # Set the overlay so the call sees it.
        agent._overlay = {"predictor": "OVERLAY_INSTRUCTIONS_FOR_CANARY"}

        with dspy.context(lm=lm):
            await agent.call_dspy(module, output_field="answer", query="hello")

        # 1) The LM must have received the overlay text in its prompt.
        assert lm.captured_messages, "fake LM was never called"
        seen_text = ""
        for msgs in lm.captured_messages:
            if isinstance(msgs, list):
                for m in msgs:
                    if isinstance(m, dict):
                        seen_text += str(m.get("content", ""))
                    else:
                        seen_text += str(m)
            else:
                seen_text += str(msgs)
        assert "OVERLAY_INSTRUCTIONS_FOR_CANARY" in seen_text, (
            "overlay instructions did not appear in the LM prompt; the "
            "consumer wire from get_dispatched_prompts() to "
            "predictor.signature.instructions is broken. "
            f"Captured text head: {seen_text[:500]!r}"
        )
        assert "ORIGINAL_INSTRUCTIONS" not in seen_text, (
            "original instructions still in the LM prompt — the swap "
            "didn't take effect for this call. "
            f"Captured text head: {seen_text[:500]!r}"
        )

        # 2) After the call, the predictor's instructions must be restored.
        assert module.predictor.signature.instructions == "ORIGINAL_INSTRUCTIONS", (
            "predictor instructions were not restored after the overlay "
            "call — subsequent requests on this module would leak the "
            "canary prompt instead of returning to the active artefact."
        )

    @pytest.mark.asyncio
    async def test_no_overlay_leaves_predictor_untouched(self):
        """When no overlay is in scope, call_dspy must not mutate anything."""
        import dspy

        from cogniverse_core.agents.base import (
            AgentBase,
            AgentDeps,
            AgentInput,
            AgentOutput,
        )

        class _Sig(dspy.Signature):
            """BASELINE"""

            q: str = dspy.InputField()
            a: str = dspy.OutputField()

        class _Module(dspy.Module):
            def __init__(self):
                super().__init__()
                self.p = dspy.Predict(_Sig)

            def forward(self, q):
                return self.p(q=q)

        class _LM(dspy.LM):
            def __init__(self):
                super().__init__(model="stub/test", api_base="none", api_key="none")

            def __call__(self, prompt=None, messages=None, **kwargs):
                return ["[[ ## a ## ]]\nx\n[[ ## completed ## ]]"]

        class _D(AgentDeps):
            pass

        class _I(AgentInput):
            pass

        class _O(AgentOutput):
            pass

        class _A(AgentBase[_I, _O, _D]):
            _input_type = _I
            _output_type = _O

            async def _process_impl(self, input_data):
                raise NotImplementedError

            def get_dispatched_prompts(self):
                return None

        agent = _A(deps=_D())
        module = _Module()
        with dspy.context(lm=_LM()):
            await agent.call_dspy(module, output_field="a", q="x")
        assert module.p.signature.instructions == "BASELINE"
