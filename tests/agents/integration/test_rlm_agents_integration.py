"""Integration tests for RLM — real Ollama calls, no mocks on the LLM boundary."""

import shutil

import pytest

from tests.agents.integration.conftest import is_ollama_available

pytestmark = pytest.mark.integration

skip_if_no_ollama = pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama not available at http://localhost:11434",
)

def _deno_available() -> bool:
    """Check if Deno is available, including ~/.deno/bin."""
    import os
    from pathlib import Path

    if shutil.which("deno"):
        return True
    deno_home = Path.home() / ".deno" / "bin" / "deno"
    if deno_home.exists():
        # Add to PATH so subprocess calls also find it
        os.environ["PATH"] = f"{deno_home.parent}:{os.environ.get('PATH', '')}"
        return True
    return False


skip_if_no_deno = pytest.mark.skipif(
    not _deno_available(),
    reason="Deno not installed — DSPy RLM REPL requires Deno for code execution",
)

_OLLAMA_MODEL = "ollama/llama3.2"
_OLLAMA_API_BASE = "http://localhost:11434"

# Short contexts so each test runs in under 30 s.
_FRANCE_CONTEXT = (
    "France is a country in Western Europe. Its capital city is Paris. "
    "Paris is home to many famous landmarks including the Eiffel Tower and the Louvre. "
    "The city has been the capital since the 10th century."
)

_PYTHON_OLD = (
    "Python is a high-level programming language created by Guido van Rossum in 1991. "
    "It supports multiple programming paradigms including procedural, object-oriented, "
    "and functional programming. Python is widely used in data science and web development."
)

_PYTHON_NEW = (
    "Python 3.12 was released in October 2023. Key improvements include faster error "
    "messages, a new type parameter syntax (PEP 695), and performance gains of up to 5% "
    "over Python 3.11 in benchmark suites."
)


@skip_if_no_ollama
class TestRLMInferenceDirect:
    """RLMInference.process() called with real Ollama."""

    @skip_if_no_deno
    def test_process_returns_answer_with_content(self):
        from cogniverse_agents.inference.rlm_inference import RLMInference
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        config = LLMEndpointConfig(
            model=_OLLAMA_MODEL,
            api_base=_OLLAMA_API_BASE,
            max_tokens=300,
            temperature=0.1,
        )
        rlm = RLMInference(llm_config=config, max_iterations=3, timeout_seconds=120)
        result = rlm.process(
            query="What is the capital of France?",
            context=_FRANCE_CONTEXT,
        )

        assert result.answer, "answer must be a non-empty string"
        assert "Paris" in result.answer, f"expected 'Paris' in answer, got: {result.answer!r}"

    def test_process_returns_positive_latency(self):
        from cogniverse_agents.inference.rlm_inference import RLMInference
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        config = LLMEndpointConfig(
            model=_OLLAMA_MODEL,
            api_base=_OLLAMA_API_BASE,
            max_tokens=200,
            temperature=0.1,
        )
        rlm = RLMInference(llm_config=config, max_iterations=3, timeout_seconds=120)
        result = rlm.process(
            query="Name the capital city mentioned in the text.",
            context=_FRANCE_CONTEXT,
        )

        assert result.latency_ms > 0, "latency_ms must be positive"
        assert result.depth_reached >= 1, "depth_reached must be at least 1"

    def test_process_result_has_all_fields(self):
        from cogniverse_agents.inference.rlm_inference import RLMInference, RLMResult
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        config = LLMEndpointConfig(
            model=_OLLAMA_MODEL,
            api_base=_OLLAMA_API_BASE,
            max_tokens=200,
            temperature=0.1,
        )
        rlm = RLMInference(llm_config=config, max_iterations=3, timeout_seconds=120)
        result = rlm.process(
            query="What programming paradigms does Python support?",
            context=_PYTHON_OLD,
        )

        assert isinstance(result, RLMResult)
        assert isinstance(result.answer, str) and result.answer
        assert isinstance(result.depth_reached, int) and result.depth_reached >= 1
        assert isinstance(result.total_calls, int) and result.total_calls >= 1
        assert isinstance(result.latency_ms, float) and result.latency_ms > 0
        assert isinstance(result.metadata, dict)


@skip_if_no_ollama
class TestRLMAwareMixinProcess:
    """RLMAwareMixin.process_with_rlm() called with real Ollama."""

    def _make_agent(self):
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin

        class _Agent(RLMAwareMixin):
            pass

        return _Agent()

    @skip_if_no_deno
    def test_process_with_rlm_returns_rlm_result(self):
        from cogniverse_agents.inference.rlm_inference import RLMResult
        from cogniverse_core.agents.rlm_options import RLMOptions

        opts = RLMOptions(
            enabled=True,
            backend="ollama",
            model="llama3.2",
            max_iterations=3,
            timeout_seconds=120,
        )
        result = self._make_agent().process_with_rlm(
            query="What is the capital of France?",
            context=_FRANCE_CONTEXT,
            rlm_options=opts,
        )

        assert isinstance(result, RLMResult)
        assert "Paris" in result.answer, f"expected 'Paris', got: {result.answer!r}"

    def test_process_with_rlm_telemetry_dict_contains_rlm_enabled(self):
        from cogniverse_core.agents.rlm_options import RLMOptions

        opts = RLMOptions(
            enabled=True,
            backend="ollama",
            model="llama3.2",
            max_iterations=3,
            timeout_seconds=120,
        )
        result = self._make_agent().process_with_rlm(
            query="When was Python created and by whom?",
            context=_PYTHON_OLD,
            rlm_options=opts,
        )

        telemetry = result.to_telemetry_dict()
        assert telemetry["rlm_enabled"] is True
        assert telemetry["rlm_latency_ms"] > 0
        assert telemetry["rlm_depth_reached"] >= 1

    def test_process_with_rlm_bad_model_raises(self):
        """process_with_rlm propagates the exception when the model is unavailable."""
        from cogniverse_core.agents.rlm_options import RLMOptions

        opts = RLMOptions(
            enabled=True,
            backend="ollama",
            model="nonexistent-model-xyz-9999",
            max_iterations=2,
            timeout_seconds=30,
        )
        with pytest.raises(Exception) as exc_info:
            self._make_agent().process_with_rlm(
                query="test",
                context="test context",
                rlm_options=opts,
            )

        # The underlying error should mention the missing model
        assert "nonexistent-model-xyz-9999" in str(exc_info.value) or "not found" in str(
            exc_info.value
        ), f"unexpected error message: {exc_info.value}"


@skip_if_no_ollama
class TestWikiManagerMergeWithRLM:
    """WikiManager._merge_with_rlm() integrates old and new content via real RLM."""

    def _make_wiki_manager(self):
        from cogniverse_agents.wiki.wiki_manager import WikiManager

        wm = WikiManager.__new__(WikiManager)
        return wm

    def test_merge_combines_old_and_new_facts(self):
        wm = self._make_wiki_manager()
        merged = wm._merge_with_rlm(_PYTHON_OLD, _PYTHON_NEW, "Python")

        # The merged result must be a non-empty string, not a raw append
        assert isinstance(merged, str)
        assert merged.strip(), "merged content must not be empty"
        # Both fact sets must be represented: creation year and release year
        assert "1991" in merged or "Guido" in merged, (
            "merged content should preserve original creation facts; got: " + merged[:300]
        )
        assert "2023" in merged or "3.12" in merged, (
            "merged content should include new release facts; got: " + merged[:300]
        )

    def test_merge_with_rlm_fallback_on_bad_model(self):
        """_merge_with_rlm falls back to simple append when RLM fails."""
        import unittest.mock as mock

        from cogniverse_agents.inference.rlm_inference import RLMInference
        from cogniverse_agents.wiki.wiki_manager import _CONTENT_SEPARATOR
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        wm = self._make_wiki_manager()

        # Patch RLMInference to raise so the fallback path is exercised
        bad_config = LLMEndpointConfig(model="ollama/bad-model-xyz", api_base=_OLLAMA_API_BASE)
        broken_rlm = RLMInference(llm_config=bad_config, timeout_seconds=10)

        with mock.patch(
            "cogniverse_agents.wiki.wiki_manager.RLMInference",
            return_value=broken_rlm,
        ):
            merged = wm._merge_with_rlm(_PYTHON_OLD, _PYTHON_NEW, "Python")

        # Fallback path: simple append with separator
        assert _CONTENT_SEPARATOR in merged, (
            "fallback must produce separator-joined content; got: " + merged[:200]
        )
        assert _PYTHON_OLD in merged
        assert _PYTHON_NEW in merged
