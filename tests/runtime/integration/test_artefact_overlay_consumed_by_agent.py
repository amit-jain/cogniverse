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


class TestOverlayIsolationOnRealModule:
    """The overlay must run on a per-call COPY of the real served module and
    never mutate the shared, dispatcher-cached instance — otherwise a canary
    prompt bleeds across concurrent requests/tenants and pins the module.

    These drive the REAL ``SearchOptimizationModule`` (the exact object the
    search agent hands to ``call_dspy``), not a stand-in: a fabricated module
    can drift from production exactly the way the ChainOfThought bug did. The
    end-to-end proof that a canary prompt reaches the served output lives in
    ``tests/agents/integration/test_canary_overlay_e2e.py`` (real agent, real
    dispatch, real LM); this pins the concurrency isolation that a sequential
    e2e can't.
    """

    def test_no_overlay_returns_the_shared_module_itself(self):
        from cogniverse_agents.search_agent import SearchOptimizationModule
        from cogniverse_core.agents.base import _dispatched_prompt_overlay

        class _NoOverlayAgent:
            def get_dispatched_prompts(self):
                return None

        module = SearchOptimizationModule()
        with _dispatched_prompt_overlay(_NoOverlayAgent(), module) as call_module:
            # No overlay in scope -> the shared module is used directly, no copy.
            assert call_module is module

    def test_concurrent_overlays_clone_and_leave_shared_module_untouched(self):
        """Two requests, each with its own canary prompt, entering the overlay
        on ONE shared real module simultaneously (a threading.Barrier forces
        both inside their overlay at once, mirroring the two to_thread workers
        call_dspy spawns) must each see ONLY their own prompt on their own
        clone, and must leave the shared module's baseline intact.
        """
        import threading

        from cogniverse_agents.search_agent import SearchOptimizationModule
        from cogniverse_core.agents.base import (
            _dispatched_prompt_overlay,
            _signature_predictor,
        )

        module = SearchOptimizationModule()  # ONE shared instance
        baseline = _signature_predictor(module.search_optimizer).signature.instructions

        class _Agent:
            def __init__(self, prompt):
                self._prompt = prompt

            def get_dispatched_prompts(self):
                return {"search_optimizer": self._prompt}

        barrier = threading.Barrier(2)
        seen: Dict[str, str] = {}
        errors: Dict[str, BaseException] = {}

        def run(tag: str, prompt: str):
            try:
                with _dispatched_prompt_overlay(_Agent(prompt), module) as clone:
                    # Both threads are inside their overlay at once here.
                    barrier.wait(timeout=10)
                    seen[tag] = _signature_predictor(
                        clone.search_optimizer
                    ).signature.instructions
            except BaseException as exc:  # noqa: BLE001 — surface in the assert
                errors[tag] = exc

        threads = [
            threading.Thread(target=run, args=("A", "PROMPT_ALPHA")),
            threading.Thread(target=run, args=("B", "PROMPT_BETA")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert not errors, f"overlay raised under concurrency: {errors}"
        assert seen["A"] == "PROMPT_ALPHA", f"request A saw {seen.get('A')!r}"
        assert seen["B"] == "PROMPT_BETA", f"request B saw {seen.get('B')!r}"
        assert (
            _signature_predictor(module.search_optimizer).signature.instructions
            == baseline
        ), "the shared real module was mutated by a concurrent overlay call"


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


class TestOverlayValueFaultContract:
    """A malformed overlay value (non-string, empty, None) must degrade to the
    active prompt for that predictor: no crash at LM-call time, no silent
    blanking of the served instructions.

    Drives the REAL served ``SearchOptimizationModule`` (a ChainOfThought whose
    signature lives on ``.predict.signature``) — the exact object the search
    agent hands to ``call_dspy``. A non-string value fed to
    ``signature.with_instructions`` produces a corrupted signature whose later
    ``.instructions`` access raises ``AttributeError`` inside the to_thread /
    streamify worker, crashing the whole dispatch; an empty/None value silently
    blanks the served instructions.
    """

    class _Agent:
        def __init__(self, prompts):
            self._prompts = prompts

        def get_dispatched_prompts(self):
            return self._prompts

    def _base_instructions(self, module):
        from cogniverse_core.agents.base import _signature_predictor

        return _signature_predictor(module.search_optimizer).signature.instructions

    def test_non_string_overlay_value_serves_active_prompt(self):
        from cogniverse_agents.search_agent import SearchOptimizationModule
        from cogniverse_core.agents.base import (
            _dispatched_prompt_overlay,
            _signature_predictor,
        )

        module = SearchOptimizationModule()
        base_instr = self._base_instructions(module)

        with _dispatched_prompt_overlay(
            self._Agent({"search_optimizer": 123}), module
        ) as clone:
            served = _signature_predictor(clone.search_optimizer).signature
            # Accessing .instructions must not raise (the corrupted-signature bug
            # deferred the AttributeError to LM-call time inside to_thread).
            assert served.instructions == base_instr

        assert self._base_instructions(module) == base_instr

    def test_empty_string_overlay_value_serves_active_prompt(self):
        from cogniverse_agents.search_agent import SearchOptimizationModule
        from cogniverse_core.agents.base import (
            _dispatched_prompt_overlay,
            _signature_predictor,
        )

        module = SearchOptimizationModule()
        base_instr = self._base_instructions(module)

        with _dispatched_prompt_overlay(
            self._Agent({"search_optimizer": ""}), module
        ) as clone:
            served = _signature_predictor(clone.search_optimizer).signature
            assert served.instructions == base_instr

        assert self._base_instructions(module) == base_instr

    def test_none_overlay_value_serves_active_prompt(self):
        from cogniverse_agents.search_agent import SearchOptimizationModule
        from cogniverse_core.agents.base import (
            _dispatched_prompt_overlay,
            _signature_predictor,
        )

        module = SearchOptimizationModule()
        base_instr = self._base_instructions(module)

        with _dispatched_prompt_overlay(
            self._Agent({"search_optimizer": None}), module
        ) as clone:
            served = _signature_predictor(clone.search_optimizer).signature
            assert served.instructions == base_instr

        assert self._base_instructions(module) == base_instr

    def test_valid_string_overlay_value_applies_and_leaves_base_untouched(self):
        from cogniverse_agents.search_agent import SearchOptimizationModule
        from cogniverse_core.agents.base import (
            _dispatched_prompt_overlay,
            _signature_predictor,
        )

        module = SearchOptimizationModule()
        base_instr = self._base_instructions(module)

        with _dispatched_prompt_overlay(
            self._Agent({"search_optimizer": "OPTIMIZED_PROMPT"}), module
        ) as clone:
            served = _signature_predictor(clone.search_optimizer).signature
            assert served.instructions == "OPTIMIZED_PROMPT"

        # The shared cached module's base is never mutated.
        assert self._base_instructions(module) == base_instr
