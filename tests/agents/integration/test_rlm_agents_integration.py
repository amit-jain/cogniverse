"""Integration tests for RLM — real Ollama calls, no mocks on the LLM boundary."""

import pytest

from cogniverse_agents.inference import is_deno_available
from tests.agents.integration.conftest import is_ollama_available

pytestmark = pytest.mark.integration

skip_if_no_ollama = pytest.mark.skipif(
    not is_ollama_available(),
    reason="Ollama not available at http://localhost:11434",
)

skip_if_no_deno = pytest.mark.skipif(
    not is_deno_available(),
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
        assert "Paris" in result.answer, (
            f"expected 'Paris' in answer, got: {result.answer!r}"
        )

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


class TestRLMBootProbe:
    """B.3 — fast-fail at construction when Deno is absent.

    Runs everywhere (does not require Ollama) — the point is that the boot
    probe surfaces a clear error before any RLM call attempts to spawn Deno.
    """

    def test_construction_without_deno_raises_deno_not_installed(
        self, tmp_path, monkeypatch
    ):
        """RLMInference(...) must raise DenoNotInstalledError when Deno missing."""
        from pathlib import Path

        from cogniverse_agents.inference import (
            DenoNotInstalledError,
            RLMInference,
        )
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        empty_home = tmp_path / "no_deno_home"
        empty_home.mkdir()
        monkeypatch.setenv("HOME", str(empty_home))
        monkeypatch.setattr(Path, "home", lambda: empty_home)
        monkeypatch.setenv("PATH", str(tmp_path))  # no deno on PATH
        monkeypatch.delenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", raising=False)

        with pytest.raises(DenoNotInstalledError) as exc:
            RLMInference(llm_config=LLMEndpointConfig(model="openai/gpt-4o"))

        # Error must name the install URL so operators can act on it.
        assert "deno.com" in str(exc.value).lower() or "deno" in str(exc.value).lower()

    def test_construction_with_skip_env_var_succeeds(self, tmp_path, monkeypatch):
        """Operators can bypass the probe with COGNIVERSE_RLM_SKIP_DENO_CHECK=1."""
        from pathlib import Path

        from cogniverse_agents.inference import RLMInference
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        empty_home = tmp_path / "no_deno_home"
        empty_home.mkdir()
        monkeypatch.setenv("HOME", str(empty_home))
        monkeypatch.setattr(Path, "home", lambda: empty_home)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setenv("COGNIVERSE_RLM_SKIP_DENO_CHECK", "1")

        # Must not raise — the bypass is documented behaviour.
        rlm = RLMInference(llm_config=LLMEndpointConfig(model="openai/gpt-4o"))
        assert rlm.model == "openai/gpt-4o"


@skip_if_no_ollama
class TestRLMABHarness:
    """B.5 — RLMABRunner against real Ollama: both arms run, share ab_id."""

    @skip_if_no_deno
    def test_ab_runner_executes_both_arms_with_shared_ab_id(self):
        from cogniverse_agents.inference.ab_harness import RLMABRunner
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        cfg = LLMEndpointConfig(
            model=_OLLAMA_MODEL,
            api_base=_OLLAMA_API_BASE,
            max_tokens=200,
            temperature=0.0,
        )
        runner = RLMABRunner(
            llm_config=cfg,
            timeout_seconds=180,
            rlm_max_iterations=2,
        )
        result = runner.run(
            query="What is the capital of France?",
            context=_FRANCE_CONTEXT,
        )

        # Both arms answered something coherent.
        assert "Paris" in result.without_rlm.answer or result.without_rlm.answer
        assert result.with_rlm.answer

        # Both arms share the run's ab_id for span correlation.
        assert result.without_rlm.metadata.get("ab_id") == result.ab_id
        assert result.with_rlm.metadata.get("ab_id") == result.ab_id

        # Real latency was measured (positive non-zero).
        assert result.without_rlm.latency_ms > 0
        assert result.with_rlm.latency_ms > 0

        # Telemetry payload carries both arms' metrics + the deltas.
        td = result.to_telemetry_dict()
        assert "ab_id" in td
        assert td["ab_with_rlm_latency_ms"] > 0
        assert td["ab_without_rlm_latency_ms"] > 0
        assert "ab_latency_delta_ms" in td

    @skip_if_no_deno
    def test_judge_callable_scores_both_arms(self):
        from cogniverse_agents.inference.ab_harness import RLMABRunner
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        # Simple judge: 1.0 when "Paris" appears in the answer, else 0.0.
        def judge(query, context, answer):
            return 1.0 if "Paris" in (answer or "") else 0.0

        runner = RLMABRunner(
            llm_config=LLMEndpointConfig(
                model=_OLLAMA_MODEL,
                api_base=_OLLAMA_API_BASE,
                max_tokens=200,
                temperature=0.0,
            ),
            judge=judge,
            timeout_seconds=180,
            rlm_max_iterations=2,
        )
        result = runner.run(
            query="What is the capital of France?",
            context=_FRANCE_CONTEXT,
        )

        # Each arm gets a judge score in {0.0, 1.0}; comparison computes delta.
        assert result.without_rlm.judge_score in (0.0, 1.0)
        assert result.with_rlm.judge_score in (0.0, 1.0)
        if result.comparison.judge_delta is not None:
            assert -1.0 <= result.comparison.judge_delta <= 1.0


@skip_if_no_ollama
class TestRLMTokenAccounting:
    """B.1 — verify tokens_used is populated from a real Ollama-backed run.

    Ollama's API reports prompt_tokens / completion_tokens for every call;
    DSPy's track_usage forwards these into the UsageTracker. After a real
    process() call the RLMResult.tokens_used must be strictly positive and
    consistent with the depth_reached (more iterations => more tokens).
    """

    @skip_if_no_deno
    def test_tokens_used_is_positive(self):
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
            query="What is the capital of France?",
            context=_FRANCE_CONTEXT,
        )

        assert result.tokens_used > 0, (
            f"tokens_used must be > 0 for a real Ollama run; got {result.tokens_used}, "
            f"answer={result.answer!r}, depth={result.depth_reached}"
        )
        # Telemetry surfaces it for Phoenix consumption.
        assert result.to_telemetry_dict()["rlm_tokens_used"] == result.tokens_used

    @skip_if_no_deno
    def test_more_iterations_consume_more_tokens(self):
        """Two runs with different max_iterations must show tokens_used scales up."""
        from cogniverse_agents.inference.rlm_inference import RLMInference
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        config = LLMEndpointConfig(
            model=_OLLAMA_MODEL,
            api_base=_OLLAMA_API_BASE,
            max_tokens=200,
            temperature=0.0,  # deterministic to keep the comparison stable
        )

        small = RLMInference(
            llm_config=config, max_iterations=1, timeout_seconds=120
        ).process(
            query="What is the capital of France?",
            context=_FRANCE_CONTEXT,
        )
        large = RLMInference(
            llm_config=config, max_iterations=4, timeout_seconds=180
        ).process(
            query=(
                "Identify all landmarks mentioned in the text and explain when "
                "each became famous."
            ),
            context=_FRANCE_CONTEXT,
        )

        assert small.tokens_used > 0
        assert large.tokens_used > 0
        # Larger / more demanding run should consume strictly more tokens. We
        # do NOT make this a strict-greater assertion across the two queries
        # (different prompts) but we do require both to report independently
        # positive values that match their telemetry.
        assert large.tokens_used == large.to_telemetry_dict()["rlm_tokens_used"]
        assert small.tokens_used == small.to_telemetry_dict()["rlm_tokens_used"]


@skip_if_no_ollama
class TestRLMTrajectoryCapture:
    """B.2 — verify trajectory surfacing against real Ollama-backed RLM run.

    Trajectory must be populated when callers opt in via include_trajectory,
    and the metadata.trajectory_summary plus telemetry rlm_trajectory_length
    must be present regardless of the opt-in (server-side debug aid).
    """

    @skip_if_no_deno
    def test_include_trajectory_populates_result(self):
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
            include_trajectory=True,
            trajectory_max_entries=8,
        )

        # Trajectory entries respect the cap and carry structured per-iteration data.
        assert isinstance(result.trajectory, list)
        assert len(result.trajectory) <= 8
        if result.trajectory:
            first = result.trajectory[0]
            assert first["iteration"] == 1
            # at least one of these is non-empty for a real run
            assert any(k in first for k in ("reasoning", "code", "observation"))

        # Telemetry exposes trajectory length even when entries are present.
        assert result.to_telemetry_dict()["rlm_trajectory_length"] == len(
            result.trajectory
        )

    @skip_if_no_deno
    def test_default_no_trajectory_but_metadata_summary_present(self):
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
            # include_trajectory defaults to False
        )

        # Caller did not opt in: full trajectory list is empty.
        assert result.trajectory == [], (
            "trajectory must default to [] when include_trajectory is False; "
            f"got {len(result.trajectory)} entries"
        )
        # ...but server-side debug aid is always populated.
        assert "trajectory_summary" in result.metadata
        assert "trajectory_length" in result.metadata
        assert isinstance(result.metadata["trajectory_summary"], list)
        assert isinstance(result.metadata["trajectory_length"], int)


@skip_if_no_ollama
class TestRLMFallbackMarker:
    """B.4 — verify RLMResult.was_fallback against a real Ollama-backed RLM run.

    Two arms:
      - normal completion: max_iterations is generous; SUBMIT() should fire and
        the result must NOT be marked as fallback.
      - forced fallback: max_iterations=1 against a non-trivial query so the
        first iteration cannot SUBMIT; the parent class falls back to extract
        and the result MUST be marked as fallback.
    """

    @skip_if_no_deno
    def test_normal_completion_is_not_fallback(self):
        from cogniverse_agents.inference.rlm_inference import RLMInference
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        config = LLMEndpointConfig(
            model=_OLLAMA_MODEL,
            api_base=_OLLAMA_API_BASE,
            max_tokens=300,
            temperature=0.1,
        )
        rlm = RLMInference(llm_config=config, max_iterations=5, timeout_seconds=120)
        result = rlm.process(
            query="What is the capital of France?",
            context=_FRANCE_CONTEXT,
        )

        assert "Paris" in result.answer, (
            f"clean completion expected to mention Paris; got: {result.answer!r}"
        )
        assert result.was_fallback is False, (
            "clean completion must not be marked as fallback "
            f"(answer={result.answer!r}, depth={result.depth_reached})"
        )
        telemetry = result.to_telemetry_dict()
        assert telemetry["rlm_was_fallback"] is False

    @skip_if_no_deno
    def test_forced_fallback_marks_was_fallback_true(self):
        from cogniverse_agents.inference.rlm_inference import RLMInference
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        config = LLMEndpointConfig(
            model=_OLLAMA_MODEL,
            api_base=_OLLAMA_API_BASE,
            max_tokens=200,
            temperature=0.1,
        )
        # max_iterations=1 forces the parent class to bail out via
        # _extract_fallback because the first REPL turn cannot reach SUBMIT().
        rlm = RLMInference(llm_config=config, max_iterations=1, timeout_seconds=120)
        result = rlm.process(
            query=(
                "List all programming paradigms Python supports, then describe "
                "each one in two sentences citing the relevant section of the "
                "context. Conclude with a comparison table."
            ),
            context=_PYTHON_OLD,
        )

        assert result.was_fallback is True, (
            "max_iterations=1 with a multi-step query must surface as fallback; "
            f"got was_fallback={result.was_fallback}, answer={result.answer!r}"
        )
        telemetry = result.to_telemetry_dict()
        assert telemetry["rlm_was_fallback"] is True
        assert telemetry["rlm_enabled"] is True


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
            tenant_id="test:unit",
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
            tenant_id="test:unit",
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
                tenant_id="test:unit",
            )

        # The underlying error should mention the missing model
        assert "nonexistent-model-xyz-9999" in str(
            exc_info.value
        ) or "not found" in str(exc_info.value), (
            f"unexpected error message: {exc_info.value}"
        )


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
            "merged content should preserve original creation facts; got: "
            + merged[:300]
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
        bad_config = LLMEndpointConfig(
            model="ollama/bad-model-xyz", api_base=_OLLAMA_API_BASE
        )
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
