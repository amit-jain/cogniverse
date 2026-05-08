"""Unit tests for RLMABRunner (B.5)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cogniverse_agents.inference.ab_harness import (
    ABArmResult,
    RLMABRunner,
)
from cogniverse_foundation.config.unified_config import LLMEndpointConfig


def _make_runner(judge=None) -> RLMABRunner:
    return RLMABRunner(
        llm_config=LLMEndpointConfig(model="openai/gpt-4o-mini"),
        judge=judge,
    )


class TestArmResultDataclass:
    def test_to_telemetry_dict_includes_ab_id_and_deltas(self):
        runner = _make_runner()

        without_arm = ABArmResult(
            arm="without_rlm",
            answer="A",
            latency_ms=100.0,
            tokens_used=200,
            was_fallback=False,
            judge_score=0.6,
        )
        with_arm = ABArmResult(
            arm="with_rlm",
            answer="B",
            latency_ms=500.0,
            tokens_used=1500,
            was_fallback=False,
            judge_score=0.85,
        )

        # Patch internals so .run constructs an ABResult with these arms.
        with (
            patch.object(runner, "_run_without_rlm", return_value=without_arm),
            patch.object(runner, "_run_with_rlm", return_value=with_arm),
        ):
            result = runner.run(query="q", context="ctx")

        td = result.to_telemetry_dict()
        # ab_id appears in the telemetry dict and matches the result's ab_id.
        assert td["ab_id"] == result.ab_id
        # Latency / token deltas reflect with_rlm - without_rlm.
        assert td["ab_latency_delta_ms"] == 400.0
        assert td["ab_tokens_delta"] == 1300
        # Judge delta: 0.85 - 0.6 = 0.25 (allow float tolerance).
        assert td["ab_judge_delta"] == pytest.approx(0.25)
        # Both arms present.
        assert td["ab_with_rlm_judge"] == 0.85
        assert td["ab_without_rlm_judge"] == 0.6


class TestSharedAbId:
    def test_both_arms_share_same_ab_id(self):
        runner = _make_runner()

        without = ABArmResult(
            arm="without_rlm",
            answer="A",
            latency_ms=10.0,
            tokens_used=10,
            was_fallback=False,
        )
        with_rlm = ABArmResult(
            arm="with_rlm",
            answer="B",
            latency_ms=50.0,
            tokens_used=80,
            was_fallback=False,
        )

        with (
            patch.object(runner, "_run_without_rlm", return_value=without),
            patch.object(runner, "_run_with_rlm", return_value=with_rlm),
        ):
            result = runner.run(query="q", context="ctx")

        # ab_id stamped in both arms' metadata for span correlation.
        assert result.without_rlm.metadata["ab_id"] == result.ab_id
        assert result.with_rlm.metadata["ab_id"] == result.ab_id


class TestJudgeOptional:
    def test_no_judge_no_judge_delta(self):
        runner = _make_runner(judge=None)
        without = ABArmResult(
            arm="without_rlm",
            answer="A",
            latency_ms=10.0,
            tokens_used=10,
            was_fallback=False,
            judge_score=None,
        )
        with_rlm = ABArmResult(
            arm="with_rlm",
            answer="B",
            latency_ms=20.0,
            tokens_used=30,
            was_fallback=False,
            judge_score=None,
        )
        with (
            patch.object(runner, "_run_without_rlm", return_value=without),
            patch.object(runner, "_run_with_rlm", return_value=with_rlm),
        ):
            result = runner.run(query="q", context="ctx")

        assert result.comparison.judge_delta is None

    def test_judge_called_on_both_arms(self):
        calls = []

        def judge(query: str, context: str, answer: str) -> float:
            calls.append((query, answer))
            return 0.5 if answer == "without_answer" else 0.9

        runner = _make_runner(judge=judge)

        # The runner builds arms by calling _run_without_rlm / _run_with_rlm.
        # In real run those methods invoke judge; for this unit test we
        # replace them with stubs that record calls and apply judge themselves
        # to mirror real behaviour.
        def fake_without(query, context, system_prompt):
            ans = "without_answer"
            return ABArmResult(
                arm="without_rlm",
                answer=ans,
                latency_ms=1.0,
                tokens_used=1,
                was_fallback=False,
                judge_score=judge(query, context, ans),
            )

        def fake_with(query, context, system_prompt):
            ans = "with_answer"
            return ABArmResult(
                arm="with_rlm",
                answer=ans,
                latency_ms=2.0,
                tokens_used=10,
                was_fallback=False,
                judge_score=judge(query, context, ans),
            )

        with (
            patch.object(runner, "_run_without_rlm", side_effect=fake_without),
            patch.object(runner, "_run_with_rlm", side_effect=fake_with),
        ):
            result = runner.run(query="q", context="ctx")

        assert {c[1] for c in calls} == {"without_answer", "with_answer"}
        assert result.comparison.judge_delta == pytest.approx(0.4)


class TestFallbackPropagation:
    def test_rlm_fallback_surfaces_in_comparison(self):
        runner = _make_runner()
        without = ABArmResult(
            arm="without_rlm",
            answer="A",
            latency_ms=10.0,
            tokens_used=10,
            was_fallback=False,
        )
        with_rlm = ABArmResult(
            arm="with_rlm",
            answer="B-fallback",
            latency_ms=200.0,
            tokens_used=400,
            was_fallback=True,
        )
        with (
            patch.object(runner, "_run_without_rlm", return_value=without),
            patch.object(runner, "_run_with_rlm", return_value=with_rlm),
        ):
            result = runner.run(query="q", context="ctx")

        assert result.comparison.rlm_was_fallback is True
        assert result.with_rlm.was_fallback is True
        assert result.without_rlm.was_fallback is False
