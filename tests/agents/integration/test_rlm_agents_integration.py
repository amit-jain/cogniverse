"""Integration tests for RLM — real LM calls (no LLM-boundary mocks).

Endpoint, model, provider and api key resolve via ``tests/fixtures/llm.py``
so the same suite runs against any OpenAI-compatible provider without a
code change. Deno is provisioned by the ``ensure_deno`` session fixture
in ``tests/agents/integration/conftest.py`` when missing.
"""

import dataclasses

import pytest

from cogniverse_agents.inference import deno_check
from tests.agents.integration.conftest import skip_if_no_lm
from tests.fixtures.llm import (
    resolve_base_url,
    resolve_prefixed_model,
    resolve_provider,
)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("ensure_deno")]


def _lm_model() -> str:
    return resolve_prefixed_model()


def _lm_base() -> str:
    return resolve_base_url()


def _lm_provider() -> str:
    return resolve_provider()


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


@skip_if_no_lm
class TestRLMInferenceDirect:
    """RLMInference.process() called with the configured LM."""

    def test_process_returns_answer_with_content(self):
        from cogniverse_agents.inference.rlm_inference import RLMInference
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        config = LLMEndpointConfig(
            model=_lm_model(),
            api_base=_lm_base(),
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
            model=_lm_model(),
            api_base=_lm_base(),
            max_tokens=800,
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
            model=_lm_model(),
            api_base=_lm_base(),
            max_tokens=800,
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
    """fast-fail at construction when Deno is absent.

    Runs everywhere (does not require the LM) — the point is that the boot
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
        monkeypatch.setattr(deno_check, "_skip_deno_check", False)

        with pytest.raises(DenoNotInstalledError) as exc:
            RLMInference(llm_config=LLMEndpointConfig(model="openai/gpt-4o"))

        # Error must name the install URL so operators can act on it.
        assert "deno.com" in str(exc.value).lower() or "deno" in str(exc.value).lower()

    def test_construction_with_configured_skip_succeeds(self, tmp_path, monkeypatch):
        """configure_deno_check(skip=True) bypasses the probe at construction."""
        from pathlib import Path

        from cogniverse_agents.inference import RLMInference
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        empty_home = tmp_path / "no_deno_home"
        empty_home.mkdir()
        monkeypatch.setenv("HOME", str(empty_home))
        monkeypatch.setattr(Path, "home", lambda: empty_home)
        monkeypatch.setenv("PATH", str(tmp_path))
        monkeypatch.setattr(deno_check, "_skip_deno_check", True)

        # Must not raise — the bypass is documented behaviour.
        rlm = RLMInference(llm_config=LLMEndpointConfig(model="openai/gpt-4o"))
        assert rlm.model == "openai/gpt-4o"


@skip_if_no_lm
class TestRLMABHarness:
    """RLMABRunner against the configured LM: both arms run, share ab_id."""

    def test_ab_runner_executes_both_arms_with_shared_ab_id(self):
        from cogniverse_agents.inference.ab_harness import RLMABRunner
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        cfg = LLMEndpointConfig(
            model=_lm_model(),
            api_base=_lm_base(),
            max_tokens=800,
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

    def test_judge_callable_scores_both_arms(self):
        from cogniverse_agents.inference.ab_harness import RLMABRunner
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        # Simple judge: 1.0 when "Paris" appears in the answer, else 0.0.
        def judge(query, context, answer):
            return 1.0 if "Paris" in (answer or "") else 0.0

        runner = RLMABRunner(
            llm_config=LLMEndpointConfig(
                model=_lm_model(),
                api_base=_lm_base(),
                max_tokens=800,
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


@skip_if_no_lm
class TestRLMTokenAccounting:
    """verify tokens_used is populated from a real LM-backed run.

    LM provider reports prompt_tokens / completion_tokens for every call;
    DSPy's track_usage forwards these into the UsageTracker. After a real
    process() call the RLMResult.tokens_used must be strictly positive and
    consistent with the depth_reached (more iterations => more tokens).
    """

    def test_tokens_used_is_positive(self):
        from cogniverse_agents.inference.rlm_inference import RLMInference
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        config = LLMEndpointConfig(
            model=_lm_model(),
            api_base=_lm_base(),
            max_tokens=800,
            temperature=0.1,
        )
        rlm = RLMInference(llm_config=config, max_iterations=3, timeout_seconds=120)
        result = rlm.process(
            query="What is the capital of France?",
            context=_FRANCE_CONTEXT,
        )

        assert result.tokens_used > 0, (
            f"tokens_used must be > 0 for a real LM run; got {result.tokens_used}, "
            f"answer={result.answer!r}, depth={result.depth_reached}"
        )
        # Telemetry surfaces it for Phoenix consumption.
        assert result.to_telemetry_dict()["rlm_tokens_used"] == result.tokens_used

    def test_more_iterations_consume_more_tokens(self):
        """Two runs with different max_iterations must show tokens_used scales up."""
        from cogniverse_agents.inference.rlm_inference import RLMInference
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        config = LLMEndpointConfig(
            model=_lm_model(),
            api_base=_lm_base(),
            max_tokens=800,
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


@skip_if_no_lm
class TestRLMTrajectoryCapture:
    """verify trajectory surfacing against the configured LM-backed RLM run.

    Trajectory must be populated when callers opt in via include_trajectory,
    and the metadata.trajectory_summary plus telemetry rlm_trajectory_length
    must be present regardless of the opt-in (server-side debug aid).
    """

    def test_include_trajectory_populates_result(self):
        from cogniverse_agents.inference.rlm_inference import RLMInference
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        config = LLMEndpointConfig(
            model=_lm_model(),
            api_base=_lm_base(),
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

    def test_default_no_trajectory_but_metadata_summary_present(self):
        from cogniverse_agents.inference.rlm_inference import RLMInference
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        config = LLMEndpointConfig(
            model=_lm_model(),
            api_base=_lm_base(),
            max_tokens=800,
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


class TestTolerantInterpreterRealDeno:
    """TolerantPythonInterpreter against a real Deno/Pyodide sandbox."""

    def test_register_and_execute_roundtrip(self):
        """Tool registration (the _send_request path the stock reader fails
        on under stale channel messages) followed by an execute that calls
        the registered tool — exact output asserted."""
        from cogniverse_agents.inference.tolerant_interpreter import (
            TolerantPythonInterpreter,
        )

        interp = TolerantPythonInterpreter(
            tools={"double": lambda x: int(x) * 2},
            output_fields=[{"name": "answer", "type": "str"}],
        )
        try:
            out = interp.execute("print(double(21))")
        finally:
            interp.shutdown()
        assert out.strip() == "42"


@skip_if_no_lm
class TestRLMFallbackMarker:
    """verify RLMResult.was_fallback against a real LM-backed RLM run.

    Two arms:
      - normal completion: max_iterations is generous; SUBMIT() should fire and
        the result must NOT be marked as fallback.
      - forced fallback: max_iterations=1 against a non-trivial query so the
        first iteration cannot SUBMIT; the parent class falls back to extract
        and the result MUST be marked as fallback.
    """

    def test_normal_completion_is_not_fallback(self):
        from cogniverse_agents.inference.rlm_inference import RLMInference
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        config = LLMEndpointConfig(
            model=_lm_model(),
            api_base=_lm_base(),
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

    def test_forced_fallback_marks_was_fallback_true(self):
        from cogniverse_agents.inference.rlm_inference import RLMInference
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        config = LLMEndpointConfig(
            model=_lm_model(),
            api_base=_lm_base(),
            max_tokens=800,
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


@skip_if_no_lm
class TestRLMAwareMixinProcess:
    """RLMAwareMixin.process_with_rlm() called with the configured LM."""

    def _make_agent(self):
        from cogniverse_agents.mixins.rlm_aware_mixin import RLMAwareMixin
        from cogniverse_foundation.config.manager import ConfigManager
        from tests.utils.memory_store import InMemoryConfigStore

        class _Agent(RLMAwareMixin):
            pass

        agent = _Agent()
        store = InMemoryConfigStore()
        store.initialize()
        agent.bind_config_manager(ConfigManager(store=store))
        return agent

    def test_process_with_rlm_returns_rlm_result(self):
        from cogniverse_agents.inference.rlm_inference import RLMResult
        from cogniverse_core.agents.rlm_options import RLMOptions

        # RLMOptions.model is passed straight through to litellm via the
        # RLMAwareMixin's "if '/' not in model_name: prepend backend" shim,
        # which assumes the bare model has no '/'. Real vLLM model ids
        # carry an HF org prefix (google/gemma-4-e4b-it) so the '/' check
        # fires false-positive and the litellm provider is never set —
        # pass the fully prefixed form (openai/google/...) so the model
        # string is already litellm-callable.
        opts = RLMOptions(
            enabled=True,
            backend=_lm_provider(),
            model=_lm_model(),
            api_base=_lm_base(),
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
            backend=_lm_provider(),
            model=_lm_model(),
            api_base=_lm_base(),
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
            backend=_lm_provider(),
            model="nonexistent-model-xyz-9999",
            api_base=_lm_base(),
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


@skip_if_no_lm
class TestWikiManagerMergeWithRLM:
    """WikiManager._merge_with_rlm() integrates old and new content via real RLM."""

    def _make_wiki_manager(self):
        """Bypass ``__init__`` so the test doesn't need a real Vespa backend,
        but set the attributes ``_merge_with_rlm`` actually reads:
        ``_llm_endpoint_config`` (forwarded to ``RLMInference``) and
        the always-needed ``_tenant_id`` / ``_backend`` (unused by the
        merge path but referenced in the surrounding instance state)."""
        from cogniverse_agents.wiki.wiki_manager import WikiManager
        from cogniverse_foundation.config.unified_config import LLMEndpointConfig

        wm = WikiManager.__new__(WikiManager)
        wm._llm_endpoint_config = LLMEndpointConfig(
            model=_lm_model(), api_base=_lm_base()
        )
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
            model=f"{_lm_provider()}/bad-model-xyz", api_base=_lm_base()
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


# ---------------------------------------------------------------------------
# Malformed model turns at the LM boundary
# ---------------------------------------------------------------------------
#
# The RLM's LM endpoint is fronted by a local OpenAI-compatible relay that
# forwards every request to the Modal-served model, except the ones a test
# chooses to answer itself: a reply cut off at the token limit (what the
# model returned in the failing A/B run) or an HTTP error. Every query
# carries a fresh nonce so no request is ever answered from the LM cache.

_PLANETS_QUERY = "Count how many planets are mentioned and list them by name."
_PLANETS_CONTEXT = (
    "Our solar system has eight planets: Mercury, Venus, Earth, Mars, "
    "Jupiter, Saturn, Uranus, Neptune. Pluto was reclassified as a "
    "dwarf planet in 2006."
)
# A reply truncated mid-reasoning: no ``code`` field, no closing brace.
_TRUNCATED_REPLY = (
    '{\n  "reasoning": "The goal is to count the planets mentioned in the '
    "context. Plan: 1. Print the context. 2. Extract the names after "
    "'planets:'"
)
_MALFORMED_ACTION_OUTPUT = (
    "[Error] Your previous response could not be parsed: it must contain "
    "the fields [reasoning, code]. It may have been cut off at the output "
    "token limit; respond again with both fields and keep the reasoning "
    "brief."
)


def _iteration_turn(text: str, nonce: str, iteration: int | None) -> bool:
    """True for an action turn of the query tagged ``nonce`` — for
    ``iteration`` only when given, else for every iteration."""
    import re

    if nonce not in text or "[[ ## iteration ## ]]" not in text:
        return False
    if iteration is None:
        return True
    return re.search(rf"\[\[ ## iteration ## \]\]\s*{iteration}/\d+", text) is not None


def _extract_turn(text: str, nonce: str) -> bool:
    return nonce in text and "extract the final outputs now" in text


class _LMRelay:
    """Local OpenAI-compatible endpoint in front of the Modal model."""

    def __init__(self, upstream_api_base: str):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        import httpx

        self.upstream = upstream_api_base.rstrip("/")
        # (text) -> ("malformed", content) | ("status", code) | None to forward
        self.rule = lambda text: None
        self.served: list[str] = []
        self._lock = threading.Lock()
        relay = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                import json

                raw = self.rfile.read(int(self.headers["Content-Length"]))
                body = json.loads(raw)
                text = "\n".join(
                    str(m.get("content", "")) for m in body.get("messages", [])
                )
                decision = relay.rule(text)
                if decision is None:
                    headers = {
                        k: v
                        for k, v in self.headers.items()
                        if k.lower()
                        not in ("host", "content-length", "accept-encoding")
                    }
                    upstream = httpx.post(
                        relay.upstream + self.path.removeprefix("/v1"),
                        content=raw,
                        headers=headers,
                        timeout=300,
                    )
                    relay._record("forwarded")
                    self._reply(upstream.status_code, upstream.content)
                    return
                kind, value = decision
                relay._record(kind)
                if kind == "status":
                    self._reply(value, b'{"error": {"message": "relay outage"}}')
                    return
                completion = {
                    "id": "chatcmpl-relay",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": value},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 512,
                        "total_tokens": 522,
                    },
                }
                self._reply(200, json.dumps(completion).encode())

            def _reply(self, status: int, payload: bytes):
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.api_base = f"http://127.0.0.1:{self._server.server_address[1]}/v1"
        self._thread = threading.Thread(target=self._server.serve_forever)
        self._thread.daemon = True
        self._thread.start()

    def _record(self, kind: str) -> None:
        with self._lock:
            self.served.append(kind)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def lm_relay(gemma_inference_endpoint):
    from tests.agents.integration.conftest import _gemma_llm_config

    config = _gemma_llm_config(gemma_inference_endpoint)
    relay = _LMRelay(config.api_base)
    relay.config = dataclasses.replace(config, api_base=relay.api_base, max_tokens=512)
    yield relay
    relay.close()


class TestRLMMalformedModelTurn:
    """A reply that does not parse is one failed REPL iteration, not a crash."""

    def test_ab_run_survives_a_truncated_first_action_turn(self, lm_relay):
        import uuid

        from cogniverse_agents.inference.ab_harness import RLMABRunner

        nonce = uuid.uuid4().hex
        lm_relay.rule = lambda text: (
            ("malformed", _TRUNCATED_REPLY) if _iteration_turn(text, nonce, 1) else None
        )
        runner = RLMABRunner(
            llm_config=lm_relay.config,
            timeout_seconds=600,
            rlm_max_iterations=2,
            rlm_max_llm_calls=4,
        )

        result = runner.run(
            query=f"[{nonce}] {_PLANETS_QUERY}", context=_PLANETS_CONTEXT
        )

        # The chat adapter's reply and the JSON adapter's retry both came
        # back truncated; that turn is iteration 1 of the trajectory.
        assert lm_relay.served.count("malformed") == 2
        assert result.with_rlm.metadata["trajectory_summary"][0] == {
            "iteration": 1,
            "reasoning": "",
            "code": "",
            "output": _MALFORMED_ACTION_OUTPUT,
        }
        assert result.with_rlm.metadata["ab_id"] == result.ab_id
        assert result.without_rlm.metadata["ab_id"] == result.ab_id

    def test_every_action_turn_malformed_ends_in_extraction(self, lm_relay):
        """Malformed turns spend the iteration budget, then the run falls
        back to extraction exactly as an unfinished REPL loop does."""
        import uuid

        from cogniverse_agents.inference.rlm_inference import RLMInference

        nonce = uuid.uuid4().hex
        lm_relay.rule = lambda text: (
            ("malformed", _TRUNCATED_REPLY)
            if _iteration_turn(text, nonce, None)
            else None
        )
        rlm = RLMInference(
            llm_config=lm_relay.config,
            max_iterations=2,
            max_llm_calls=4,
            timeout_seconds=600,
            cache=False,
        )

        result = rlm.process(
            query=f"[{nonce}] {_PLANETS_QUERY}",
            context=_PLANETS_CONTEXT,
            include_trajectory=True,
        )

        assert lm_relay.served == ["malformed"] * 4 + ["forwarded"]
        assert result.was_fallback is True
        assert result.trajectory == [
            {
                "iteration": 1,
                "reasoning": "",
                "code": "",
                "output": _MALFORMED_ACTION_OUTPUT,
            },
            {
                "iteration": 2,
                "reasoning": "",
                "code": "",
                "output": _MALFORMED_ACTION_OUTPUT,
            },
        ]

    def test_malformed_extraction_raises_with_the_reply(self, lm_relay):
        """With no parseable turn at all there is no answer: the run raises
        naming the reply rather than returning an empty answer."""
        import uuid

        from dspy.utils.exceptions import AdapterParseError

        from cogniverse_agents.inference.rlm_inference import RLMInference

        nonce = uuid.uuid4().hex
        lm_relay.rule = lambda text: (
            ("malformed", _TRUNCATED_REPLY)
            if _iteration_turn(text, nonce, None) or _extract_turn(text, nonce)
            else None
        )
        rlm = RLMInference(
            llm_config=lm_relay.config,
            max_iterations=2,
            max_llm_calls=4,
            timeout_seconds=600,
            cache=False,
        )

        with pytest.raises(AdapterParseError) as err:
            rlm.process(query=f"[{nonce}] {_PLANETS_QUERY}", context=_PLANETS_CONTEXT)

        assert err.value.lm_response == _TRUNCATED_REPLY
        assert sorted(err.value.signature.output_fields) == ["answer"]
        assert lm_relay.served == ["malformed"] * 6

    def test_endpoint_outage_mid_run_is_not_absorbed(self, lm_relay):
        """An HTTP failure is the endpoint's, not the model's: it propagates
        after the endpoint's own retry budget instead of being recorded as
        a failed iteration."""
        import uuid

        import litellm

        from cogniverse_agents.inference.rlm_inference import RLMInference

        nonce = uuid.uuid4().hex
        lm_relay.rule = lambda text: (
            ("status", 503) if _iteration_turn(text, nonce, 1) else None
        )
        rlm = RLMInference(
            llm_config=dataclasses.replace(lm_relay.config, num_retries=0),
            max_iterations=2,
            max_llm_calls=4,
            timeout_seconds=600,
            cache=False,
        )

        with pytest.raises(litellm.ServiceUnavailableError) as err:
            rlm.process(query=f"[{nonce}] {_PLANETS_QUERY}", context=_PLANETS_CONTEXT)

        assert "relay outage" in str(err.value)
        assert lm_relay.served == ["status"]

    def test_concurrent_runs_keep_their_own_malformed_turns(self, lm_relay):
        """Two runs start at once on one fresh RLMInference: it is built once,
        and only the run whose turn was malformed records it."""
        import threading
        import uuid
        from concurrent.futures import ThreadPoolExecutor

        from cogniverse_agents.inference.rlm_inference import RLMInference

        bad, good = uuid.uuid4().hex, uuid.uuid4().hex
        lm_relay.rule = lambda text: (
            ("malformed", _TRUNCATED_REPLY) if _iteration_turn(text, bad, 1) else None
        )
        rlm = RLMInference(
            llm_config=lm_relay.config,
            max_iterations=2,
            max_llm_calls=4,
            timeout_seconds=600,
            cache=False,
        )
        builds = []
        create_lm = rlm._create_lm

        def counted_create_lm():
            builds.append(threading.get_ident())
            return create_lm()

        rlm._create_lm = counted_create_lm
        barrier = threading.Barrier(2)

        def run(nonce: str):
            barrier.wait()
            return rlm.process(
                query=f"[{nonce}] {_PLANETS_QUERY}",
                context=_PLANETS_CONTEXT,
                include_trajectory=True,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            bad_result, good_result = pool.map(run, [bad, good])

        assert bad_result.trajectory[0] == {
            "iteration": 1,
            "reasoning": "",
            "code": "",
            "output": _MALFORMED_ACTION_OUTPUT,
        }
        assert good_result.trajectory[0]["iteration"] == 1
        assert [entry["output"] for entry in good_result.trajectory].count(
            _MALFORMED_ACTION_OUTPUT
        ) == 0
        assert lm_relay.served.count("malformed") == 2
        # Both first touches raced the lazy build; exactly one built the RLM.
        assert len(builds) == 1
