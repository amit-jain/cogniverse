"""Unit tests for RLMABRunner."""

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


class TestRlmArmEventRouting:
    """The RLM arm must route its events to the queue's own task_id."""

    def test_rlm_inference_gets_queue_task_id(self, monkeypatch):
        from types import SimpleNamespace

        from cogniverse_agents.inference import ab_harness
        from cogniverse_agents.inference.ab_harness import RLMABRunner

        captured = {}

        class _FakeRLM:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def process(self, **kwargs):
                return SimpleNamespace(
                    answer="a",
                    latency_ms=1.0,
                    tokens_used=0,
                    was_fallback=False,
                    metadata={},
                )

        monkeypatch.setattr(ab_harness, "RLMInference", _FakeRLM)

        runner = object.__new__(RLMABRunner)
        # Distinct sentinels: the RLM arm must consume the ROUTED config (which
        # carries the tenant's semantic-router x-authz-* headers), not the base
        # llm_config. Same-object fixtures couldn't tell the two apart.
        base_config = object()
        routed_config = object()
        runner._llm_config = base_config
        runner._routed_llm_config = routed_config
        runner._rlm_max_iterations = 10
        runner._rlm_max_llm_calls = 30
        runner._timeout_seconds = 300
        runner._event_queue = SimpleNamespace(task_id="task-xyz")
        runner._tenant_id = "acme:acme"
        runner._judge = None

        runner._run_with_rlm("q", "c", None)

        assert captured["task_id"] == "task-xyz"
        assert captured["tenant_id"] == "acme:acme"
        # The routed endpoint, not the base — a regression to self._llm_config
        # would drop the tenant's tier-routing headers on the RLM arm.
        assert captured["llm_config"] is routed_config
        assert captured["llm_config"] is not base_config


class TestABRunnerSemanticRouting:
    """Both arms route through the semantic router when a config_manager + tenant is set."""

    def _patch_enabled(self, monkeypatch):
        from unittest.mock import MagicMock

        from cogniverse_foundation.config.unified_config import SemanticRouterConfig

        router = SemanticRouterConfig(
            enabled=True,
            semantic_router_url="http://semantic-router:8080/v1",
            tenant_tiers={"acme:prod": "pro"},
            default_tier="free",
        )
        cfg = MagicMock()
        cfg.get_semantic_router.return_value = router
        monkeypatch.setattr(
            "cogniverse_foundation.config.utils.get_config", lambda **kw: cfg
        )

    def test_both_arms_share_one_routed_endpoint(self, monkeypatch):
        from unittest.mock import MagicMock

        self._patch_enabled(monkeypatch)
        runner = RLMABRunner(
            llm_config=LLMEndpointConfig(
                model="openai/gpt-4o", api_base="http://direct"
            ),
            tenant_id="acme:prod",
            config_manager=MagicMock(),
        )
        # Resolved once; the identical routed endpoint drives both arms so the
        # router returns the same model for each.
        routed = runner._routed_llm_config
        assert routed.api_base == "http://semantic-router:8080/v1"
        assert routed.model == "openai/auto"
        assert routed.extra_headers == {
            "x-authz-user-id": "acme:prod",
            "x-authz-user-groups": "pro",
        }

    def test_no_config_manager_keeps_direct_endpoint(self):
        runner = RLMABRunner(
            llm_config=LLMEndpointConfig(
                model="openai/gpt-4o", api_base="http://direct"
            ),
            tenant_id="acme:prod",
        )
        assert runner._routed_llm_config.api_base == "http://direct"
        assert runner._routed_llm_config.extra_headers is None
