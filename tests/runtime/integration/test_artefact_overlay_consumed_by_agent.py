"""dispatcher overlay reaches the agent AND shapes the DSPy call.

Two halves verified end-to-end:

  * **Wire** — ``MemoryAwareMixin.set_dispatched_artefact`` /
    ``get_dispatched_prompts`` plumb the overlay; the dispatcher's
    ``_apply_artefact_overlay`` calls the setter from search /
    orchestration / summarization paths (not just the generic path).
  * **Consumer** — ``AgentBase.call_dspy`` consults
    ``get_dispatched_prompts()`` and applies the overlay's instructions
    to the matching predictor on a per-call DEEP COPY of the module, so
    the shared dispatcher-cached module is never mutated across the
    ``await`` (no cross-request/tenant prompt bleed). The overlay reaches
    a ``ChainOfThought`` via ``.predict.signature``, not just a bare
    ``dspy.Predict``. This is what makes per-tenant canary +
    signature-variant selection actually shape agent output instead of
    sitting in a context dict no one reads.
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

    Every optimization-served agent (search_optimizer / summarizer /
    report_generator) is a ``dspy.ChainOfThought``, which keeps its
    signature on ``.predict.signature`` — NOT ``.signature``. A test that
    only exercised a bare ``dspy.Predict`` (which does expose
    ``.signature``) would pass while the overlay silently no-ops for
    every real agent, so the ChainOfThought shape is covered explicitly
    below.
    """

    @pytest.mark.asyncio
    async def test_overlay_applies_to_chain_of_thought_predictor(self):
        """Production shape: a ChainOfThought predictor. The overlay must
        reach ``.predict.signature`` and land in the LM prompt, and the
        SHARED module must never be mutated (a per-call clone carries the
        overlay).
        """
        import dspy

        from cogniverse_core.agents.base import (
            AgentBase,
            AgentDeps,
            AgentInput,
            AgentOutput,
            _signature_predictor,
        )

        class _Sig(dspy.Signature):
            """ORIGINAL_INSTRUCTIONS"""

            query: str = dspy.InputField()
            answer: str = dspy.OutputField()

        class _Module(dspy.Module):
            def __init__(self):
                super().__init__()
                # ChainOfThought — the shape every served agent uses.
                self.search_optimizer = dspy.ChainOfThought(_Sig)

            def forward(self, query: str):
                return self.search_optimizer(query=query)

        class _RecordingLM(dspy.LM):
            def __init__(self):
                super().__init__(model="stub/test", api_base="none", api_key="none")
                self.captured_messages: list = []

            def __call__(self, prompt=None, messages=None, **kwargs):
                self.captured_messages.append(
                    messages if messages is not None else prompt
                )
                return [
                    "[[ ## reasoning ## ]]\nok\n[[ ## answer ## ]]\n"
                    "stub_answer\n[[ ## completed ## ]]"
                ]

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

            def get_dispatched_prompts(self):
                return self._overlay

            def __init__(self, deps):
                super().__init__(deps=deps)
                self._overlay = None

        agent = _OverlayAgent(deps=_NullDeps())
        module = _Module()
        lm = _RecordingLM()

        # The ChainOfThought keeps instructions on .predict.signature.
        assert (
            _signature_predictor(module.search_optimizer).signature.instructions
            == "ORIGINAL_INSTRUCTIONS"
        )

        agent._overlay = {"search_optimizer": "OVERLAY_INSTRUCTIONS_FOR_CANARY"}

        with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
            await agent.call_dspy(module, output_field="answer", query="hello")

        assert lm.captured_messages, "fake LM was never called"
        seen_text = ""
        for msgs in lm.captured_messages:
            if isinstance(msgs, list):
                for m in msgs:
                    seen_text += str(m.get("content", "") if isinstance(m, dict) else m)
            else:
                seen_text += str(msgs)
        assert "OVERLAY_INSTRUCTIONS_FOR_CANARY" in seen_text, (
            "overlay instructions did not reach the LM for a ChainOfThought "
            "predictor — the overlay silently skipped it (the bug: it gated on "
            f"hasattr(predictor, 'signature')). Captured head: {seen_text[:500]!r}"
        )
        assert "ORIGINAL_INSTRUCTIONS" not in seen_text

        # The SHARED module must be untouched — the overlay ran on a clone,
        # so there is nothing to "restore"; concurrent requests cannot leak.
        assert (
            _signature_predictor(module.search_optimizer).signature.instructions
            == "ORIGINAL_INSTRUCTIONS"
        ), "shared module was mutated by the overlay — cross-request bleed risk"

    @pytest.mark.asyncio
    async def test_overlay_applies_to_direct_predict(self):
        """A bare dspy.Predict (has .signature directly) also works, on a
        per-call clone that leaves the shared module untouched.
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
            def __init__(self):
                super().__init__(model="stub/test", api_base="none", api_key="none")
                self.captured_messages: list = []

            def __call__(self, prompt=None, messages=None, **kwargs):
                self.captured_messages.append(
                    messages if messages is not None else prompt
                )
                return ["[[ ## answer ## ]]\nstub_answer\n[[ ## completed ## ]]"]

        class _D(AgentDeps):
            pass

        class _I(AgentInput):
            pass

        class _O(AgentOutput):
            pass

        class _A(AgentBase[_I, _O, _D]):
            _input_type = _I
            _output_type = _O

            def __init__(self, deps):
                super().__init__(deps=deps)
                self._overlay = {"predictor": "OVERLAY_INSTRUCTIONS_FOR_CANARY"}

            async def _process_impl(self, input_data):
                raise NotImplementedError

            def get_dispatched_prompts(self):
                return self._overlay

        agent = _A(deps=_D())
        module = _Module()
        lm = _RecordingLM()
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
            await agent.call_dspy(module, output_field="answer", query="hi")
        seen = "".join(
            str(m.get("content", "") if isinstance(m, dict) else m)
            for msgs in lm.captured_messages
            for m in (msgs if isinstance(msgs, list) else [msgs])
        )
        assert "OVERLAY_INSTRUCTIONS_FOR_CANARY" in seen
        assert module.predictor.signature.instructions == "ORIGINAL_INSTRUCTIONS"

    @pytest.mark.asyncio
    async def test_concurrent_overlays_do_not_bleed_across_shared_module(self):
        """Two requests running concurrently on ONE shared, dispatcher-cached
        module — each with its own canary prompt — must each observe only its
        own instructions, and the shared module's baseline must survive both.

        A ``threading.Barrier(2)`` inside forward forces both worker threads to
        be mid-invocation simultaneously, so a shared-state mutation would be
        caught as cross-observation. On the pre-fix code the overlay both
        no-ops for ChainOfThought AND mutates the shared module across the
        await, so this assertion fails.
        """
        import asyncio
        import contextvars
        import threading

        import dspy

        from cogniverse_core.agents.base import (
            AgentBase,
            AgentDeps,
            AgentInput,
            AgentOutput,
            _signature_predictor,
        )

        overlay_var = contextvars.ContextVar("test_overlay", default=None)
        barrier = threading.Barrier(2)
        sink: Dict[str, str] = {}

        class _Sig(dspy.Signature):
            """ORIGINAL"""

            query: str = dspy.InputField()
            answer: str = dspy.OutputField()

        class _Module(dspy.Module):
            def __init__(self):
                super().__init__()
                self.search_optimizer = dspy.ChainOfThought(_Sig)

            def forward(self, query, tag):
                # Runs in the to_thread worker. Both requests meet here, so a
                # mutation of shared predictor state is observable as bleed.
                barrier.wait(timeout=10)
                sink[tag] = _signature_predictor(
                    self.search_optimizer
                ).signature.instructions
                return dspy.Prediction(answer="stub")

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
                return overlay_var.get()

        agent = _A(deps=_D())
        module = _Module()  # ONE shared instance, as the dispatcher caches it

        async def run(tag: str, prompt: str):
            overlay_var.set({"search_optimizer": prompt})
            await agent.call_dspy(module, output_field="answer", query="q", tag=tag)

        await asyncio.gather(run("A", "PROMPT_ALPHA"), run("B", "PROMPT_BETA"))

        assert sink["A"] == "PROMPT_ALPHA", f"request A saw {sink.get('A')!r}"
        assert sink["B"] == "PROMPT_BETA", f"request B saw {sink.get('B')!r}"
        assert (
            _signature_predictor(module.search_optimizer).signature.instructions
            == "ORIGINAL"
        ), "shared module baseline was corrupted by a concurrent overlay call"

    @pytest.mark.asyncio
    async def test_no_overlay_leaves_predictor_untouched(self):
        """When no overlay is in scope, call_dspy must not mutate anything."""
        import dspy

        from cogniverse_core.agents.base import (
            AgentBase,
            AgentDeps,
            AgentInput,
            AgentOutput,
            _signature_predictor,
        )

        class _Sig(dspy.Signature):
            """BASELINE"""

            q: str = dspy.InputField()
            a: str = dspy.OutputField()

        class _Module(dspy.Module):
            def __init__(self):
                super().__init__()
                self.p = dspy.ChainOfThought(_Sig)

            def forward(self, q):
                return self.p(q=q)

        class _LM(dspy.LM):
            def __init__(self):
                super().__init__(model="stub/test", api_base="none", api_key="none")

            def __call__(self, prompt=None, messages=None, **kwargs):
                return [
                    "[[ ## reasoning ## ]]\nok\n[[ ## a ## ]]\nx\n[[ ## completed ## ]]"
                ]

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
        with dspy.context(lm=_LM(), adapter=dspy.ChatAdapter()):
            await agent.call_dspy(module, output_field="a", q="x")
        assert _signature_predictor(module.p).signature.instructions == "BASELINE"


class TestServedAgentModulesAreOverlayReachable:
    """Every agent wired into the optimization/canary serve pipeline
    (``_SERVE_TARGET``) must expose, on the exact DSPy module it hands to
    ``call_dspy``, a predictor the overlay resolver can reach by the
    attribute name the canary path writes (``candidate_prompts={attr: ...}``).

    This is the guard that was missing for seven audits: the original
    end-to-end overlay test used a bare ``dspy.Predict`` (which exposes
    ``.signature`` directly), so it never noticed that every real served
    agent is a ``dspy.ChainOfThought`` whose signature lives on
    ``.predict.signature`` — the overlay silently no-op'd for all of them.
    Driving the REAL served modules here fails the instant a served agent
    uses a predictor wrapper the resolver can't reach.
    """

    def test_every_serve_target_predictor_is_overlay_reachable(self):
        from cogniverse_agents.detailed_report_agent import ReportGenerationModule
        from cogniverse_agents.search_agent import SearchOptimizationModule
        from cogniverse_agents.summarizer_agent import SummarizationModule
        from cogniverse_core.agents.base import _signature_predictor
        from cogniverse_runtime.optimization_cli import _SERVE_TARGET

        # served_agent -> the DSPy module class it actually passes to call_dspy
        module_for = {
            "search_agent": SearchOptimizationModule,
            "summarizer_agent": SummarizationModule,
            "detailed_report_agent": ReportGenerationModule,
        }

        for _mode, (served_agent, predictor_attr) in _SERVE_TARGET.items():
            assert served_agent in module_for, (
                f"_SERVE_TARGET wires served agent {served_agent!r} but this "
                "guard has no module for it — add its call_dspy module so "
                "overlay-reachability stays covered."
            )
            module = module_for[served_agent]()
            attr = getattr(module, predictor_attr, None)
            assert attr is not None, (
                f"{served_agent}: module exposes no attribute {predictor_attr!r} "
                "(the key the canary path writes) — the overlay can never target it."
            )
            predictor = _signature_predictor(attr)
            assert predictor is not None and hasattr(predictor, "signature"), (
                f"{served_agent}.{predictor_attr}: the overlay resolver cannot "
                "reach a .signature, so a promoted canary/variant prompt would "
                "silently serve the un-optimized prompt."
            )
