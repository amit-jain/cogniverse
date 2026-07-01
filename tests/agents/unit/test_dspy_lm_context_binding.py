"""Regression tests for per-tenant DSPy LM context binding.

Two sites invoked a DSPy module without an enclosing
``dspy.context(lm=...)`` wrap, so the call silently fell back to
``dspy.settings.lm`` (the global runtime LM or none on a standalone
endpoint), ignoring the tenant's configured LM:

  * ``knowledge_summarization_agent._summarise_without_rlm`` — built
    ``_llm_config`` in ``__init__`` then dropped it on the non-RLM
    happy path. The sibling ``multi_document_synthesis_agent`` wraps
    correctly; that's the reference shape.

  * ``text_analysis_agent.analyze_text`` — when ``_dspy_lm`` is
    ``None``, the ``else`` branch invoked the module without context.
    Silent fall-through; fix raises ``RuntimeError`` so a partially-
    constructed agent surfaces immediately instead of running against
    the wrong global LM.

Tests verify the wrap happens (knowledge_summarization) and that the
misconfig is surfaced (text_analysis) by stubbing the DSPy module +
``create_dspy_lm`` boundary. The contract being asserted is the
LM-binding code path, not the LM's own behavior.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import dspy
import pytest

from cogniverse_agents.text_analysis_agent import TextAnalysisAgent


class _CapturingDSPyModule:
    """Stand-in for ``dspy.ChainOfThought(_SummarizationSignature)``.

    When invoked, records the currently-active ``dspy.settings.lm`` so
    the test can verify the wrap entered ``dspy.context(lm=...)`` with
    the per-tenant LM (not the ambient one).
    """

    def __init__(self) -> None:
        self.captured_lm: object = None
        self.call_count: int = 0

    def __call__(self, **kwargs):
        self.captured_lm = dspy.settings.lm
        self.call_count += 1
        return MagicMock(summary="STUBBED_SUMMARY")


class TestKnowledgeSummarizationLMContext:
    """The non-RLM path must run inside ``dspy.context(lm=tenant_lm)`` when
    ``_llm_config`` is set, and only fall through to the ambient LM when it
    is None."""

    def _make_agent_with_llm_config(self, llm_config) -> object:
        """Construct a minimal agent skipping super().__init__ side-effects.

        ``KnowledgeSummarizationAgent.__init__`` runs the A2A registration
        chain, builds a real KnowledgeRegistry, etc. — none of that is
        needed to exercise the LM-binding logic in ``_summarise_without_rlm``.
        Bypass __init__, set only the attributes the method reads.
        """
        from cogniverse_agents.knowledge_summarization_agent import (
            KnowledgeSummarizationAgent,
        )

        agent = object.__new__(KnowledgeSummarizationAgent)
        agent._llm_config = llm_config
        agent._config_manager = None
        agent._memory_tenant_id = None
        agent._dspy_module = _CapturingDSPyModule()
        return agent

    def test_per_tenant_lm_bound_when_llm_config_set(self) -> None:
        """The synthesis call must run with the per-tenant LM active, NOT the
        ambient ``dspy.settings.lm``."""
        sentinel_lm = MagicMock(name="per_tenant_lm")
        ambient_lm = MagicMock(name="ambient_global_lm")
        agent = self._make_agent_with_llm_config(llm_config=MagicMock())

        # Place an ambient LM globally; the wrap must override it. The
        # non-RLM path binds its LM via routed_lm_context_for, which builds
        # it through gateway_routing's own create_dspy_lm binding — patch
        # there (routing is disabled by default, so it takes the direct path).
        with dspy.context(lm=ambient_lm):
            with patch(
                "cogniverse_foundation.config.gateway_routing.create_dspy_lm",
                return_value=sentinel_lm,
            ) as mock_factory:
                result = agent._summarise_without_rlm(title="t", block="b")

        assert agent._dspy_module.call_count == 1
        # Strong assertion: the LM seen at module-invocation time is the
        # per-tenant sentinel, not the ambient one.
        assert agent._dspy_module.captured_lm is sentinel_lm, (
            f"Expected per-tenant LM {sentinel_lm!r}; "
            f"got {agent._dspy_module.captured_lm!r} (ambient: {ambient_lm!r})"
        )
        # create_dspy_lm must have been called with the per-agent llm_config.
        assert mock_factory.call_count == 1
        assert result == "STUBBED_SUMMARY"

    def test_ambient_lm_used_when_llm_config_is_none(self) -> None:
        """No per-agent override → fall through to ``dspy.settings.lm``."""
        ambient_lm = MagicMock(name="ambient_global_lm")
        agent = self._make_agent_with_llm_config(llm_config=None)

        with dspy.context(lm=ambient_lm):
            result = agent._summarise_without_rlm(title="t", block="b")

        assert agent._dspy_module.captured_lm is ambient_lm, (
            f"Expected ambient {ambient_lm!r}; got {agent._dspy_module.captured_lm!r}"
        )
        assert result == "STUBBED_SUMMARY"

    def test_dspy_failure_returns_fallback_text(self) -> None:
        """Synthesis failure path still surfaces SOME artefact (not silent empty)."""
        agent = self._make_agent_with_llm_config(llm_config=MagicMock())

        def raise_on_call(**_):
            raise RuntimeError("synth blew up")

        agent._dspy_module = raise_on_call

        with patch(
            "cogniverse_foundation.config.gateway_routing.create_dspy_lm",
            return_value=MagicMock(),
        ):
            result = agent._summarise_without_rlm(title="t", block="some content here")

        assert result.startswith("[FALLBACK: synthesis failed]")
        assert "some content here" in result


class TestTextAnalysisAgentLMGuard:
    """``analyze_text`` must raise when ``_dspy_lm`` is missing, instead of
    silently using the ambient ``dspy.settings.lm``."""

    def _make_agent_without_dspy_lm(self) -> TextAnalysisAgent:
        """Bypass __init__ so we can hit the ``getattr(self, "_dspy_lm", None)``
        guard in ``analyze_text``."""
        agent = object.__new__(TextAnalysisAgent)
        # The guard sits before any other attribute access; set just enough for
        # the preceding `get_or_create_module` + `inject_context_into_prompt`
        # not to crash.
        agent._dynamic_modules = {}
        agent._signatures = {}
        return agent

    def test_missing_dspy_lm_raises_runtime_error(self) -> None:
        """Missing ``_dspy_lm`` must raise, not silently use the global LM."""
        agent = self._make_agent_without_dspy_lm()
        # Patch the surrounding helpers so we reach the LM guard cleanly.
        with patch.object(
            agent,
            "get_or_create_module",
            return_value=MagicMock(return_value=MagicMock(result="x", confidence=0.5)),
        ):
            with patch.object(agent, "inject_context_into_prompt", return_value="t"):
                with pytest.raises(RuntimeError) as excinfo:
                    agent.analyze_text("some text", analysis_type="summary")
        # Strong assertion: the error names the attribute and the remedy so
        # operators can fix the misconfig.
        assert "_dspy_lm" in str(excinfo.value)
        assert "initialize_dynamic_dspy" in str(excinfo.value)
        # And the error explicitly explains why we don't silently fall through.
        assert (
            "silently" in str(excinfo.value).lower()
            or "fall back" in str(excinfo.value).lower()
        )

    def test_per_tenant_lm_bound_when_dspy_lm_set(self) -> None:
        """Happy path: ``_dspy_lm`` is set → the module runs under that LM."""
        sentinel_lm = MagicMock(name="per_tenant_lm")
        ambient_lm = MagicMock(name="ambient_global_lm")

        captured_lm = []

        def capture_module(**_):
            captured_lm.append(dspy.settings.lm)
            return MagicMock(result="OK", confidence=0.9)

        agent = self._make_agent_without_dspy_lm()
        agent._dspy_lm = sentinel_lm
        agent.config = MagicMock()
        agent.config.module_config.module_type.value = "predict"

        with dspy.context(lm=ambient_lm):
            with patch.object(
                agent, "get_or_create_module", return_value=capture_module
            ):
                with patch.object(
                    agent, "inject_context_into_prompt", return_value="text"
                ):
                    out = agent.analyze_text("text", analysis_type="summary")

        assert len(captured_lm) == 1
        assert captured_lm[0] is sentinel_lm, (
            f"Expected per-tenant {sentinel_lm!r}; got {captured_lm[0]!r} "
            f"(ambient was {ambient_lm!r})"
        )
        assert out["result"] == "OK"
        assert out["confidence"] == 0.9
