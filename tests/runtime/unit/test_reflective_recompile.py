"""All-failure agents recompile via dspy.GEPA instead of skipping.

A triggered agent whose human annotations are ALL failures builds an empty
positives trainset, so BootstrapFewShot has nothing to imitate. Rather than
skip, ``_optimize_agent`` routes the failing rows through a reflective GEPA
compile: the reflection LM reads the failing rollouts plus a feedback metric
that rewards NOT reproducing the recorded failing output, proposes improved
instructions, and the candidate STILL goes through the same
``promote_if_better`` gate before it can serve. The flag defaults ON for the
three servable agents.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from cogniverse_agents.routing.config import (
    AutomationRulesConfig,
    OptimizationTriggersConfig,
)
from cogniverse_runtime import optimization_cli
from cogniverse_runtime.optimization_cli import (
    _optimize_agent,
    _reflective_metric,
    _reflective_settings_from_config,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


def _rules(**triggers) -> AutomationRulesConfig:
    return AutomationRulesConfig(
        optimization_triggers=OptimizationTriggersConfig(**triggers)
    )


def _patch_rules(monkeypatch, rules: AutomationRulesConfig) -> None:
    monkeypatch.setattr(
        "cogniverse_runtime.quality_monitor_cli._load_automation_rules",
        lambda tenant_id, config_manager=None: rules,
    )


class TestReflectiveConfig:
    def test_defaults_are_on_min_ten_budget_sixty(self):
        triggers = OptimizationTriggersConfig()
        assert triggers.enable_reflective_recompile is True
        assert triggers.min_reflective_failures == 10
        assert triggers.reflective_max_metric_calls == 60

    def test_accessor_reads_declared_defaults_through_real_config(self):
        from cogniverse_foundation.config.manager import ConfigManager
        from tests.utils.memory_store import InMemoryConfigStore

        store = InMemoryConfigStore()
        store.initialize()
        cm = ConfigManager(store=store)

        enable, min_failures, max_calls = _reflective_settings_from_config(
            "acme:acme", config_manager=cm
        )
        assert enable is True
        assert min_failures == 10
        assert max_calls == 60


class TestReflectiveMetric:
    def test_five_arg_shape_and_summary_exact_score_feedback(self):
        from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

        metric = _reflective_metric("summary")
        assert list(inspect.signature(metric).parameters) == [
            "gold",
            "pred",
            "trace",
            "pred_name",
            "pred_trace",
        ]

        gold = SimpleNamespace(_bad_output="alpha beta")
        reproduced = metric(
            gold, SimpleNamespace(summary="alpha beta"), None, None, None
        )
        assert isinstance(reproduced, ScoreWithFeedback)
        assert reproduced.score == 0.0
        assert reproduced.feedback == (
            "The recorded failing summary was 'alpha beta'. Produce a distinct, "
            "accurate summary that does not reproduce it."
        )

        diverged = metric(
            gold, SimpleNamespace(summary="entirely other wording"), None, None, None
        )
        assert diverged.score == 1.0

    def test_search_metric_scores_enum_validity(self):
        from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

        metric = _reflective_metric("search")
        gold = SimpleNamespace(_bad_output="", query="find cats")

        valid = metric(
            gold,
            SimpleNamespace(
                primary_intent="search",
                complexity_level="simple",
                needs_video_search="true",
            ),
            None,
            None,
            None,
        )
        assert isinstance(valid, ScoreWithFeedback)
        assert valid.score == 1.0
        assert "find cats" in valid.feedback

        invalid = metric(
            gold,
            SimpleNamespace(
                primary_intent="", complexity_level="", needs_video_search=""
            ),
            None,
            None,
            None,
        )
        assert invalid.score == 0.0


def _install_reflective_fakes(monkeypatch, *, agent_name, promote=True):
    """Wire the reflective branch's boundaries to recording fakes.

    Returns ``(fake_gepa, captured, gate_calls)`` so a test can assert the
    trainset GEPA received, the metric/budget/reflection_lm handed to
    ``_build_gepa``, and the ``promote_if_better`` call the serving path made.
    """
    captured: dict = {}
    gate_calls: list = []

    class _FakeGepa:
        def __init__(self):
            self.trainset = None

        def compile(self, module, trainset=None):
            self.trainset = trainset
            predictor = SimpleNamespace(
                signature=SimpleNamespace(
                    instructions="Reflective: avoid the failing output."
                ),
                demos=[],
            )
            return SimpleNamespace(named_predictors=lambda: [("predict", predictor)])

    fake_gepa = _FakeGepa()

    def _fake_build_gepa(metric, reflection_lm, max_metric_calls):
        captured["metric"] = metric
        captured["reflection_lm"] = reflection_lm
        captured["max_metric_calls"] = max_metric_calls
        return fake_gepa

    monkeypatch.setattr(optimization_cli, "_build_gepa", _fake_build_gepa)
    # Baseline scoring would otherwise hit the real LM; pin a winning pair.
    monkeypatch.setattr(
        optimization_cli, "_holdout_scores", lambda *a, **k: (0.30, 0.80)
    )

    class _FakeOptimizer:
        def initialize_language_model(self, endpoint, teacher_endpoint_config=None):
            self.lm = MagicMock(name="student_lm")

        def create_query_analysis_signature(self):
            return SimpleNamespace(kind="search_sig")

        def create_summary_generation_signature(self):
            return SimpleNamespace(kind="summary_sig")

        def create_detailed_report_signature(self):
            return SimpleNamespace(kind="report_sig")

    monkeypatch.setattr(
        "cogniverse_agents.optimizer.dspy_agent_optimizer.DSPyAgentPromptOptimizer",
        _FakeOptimizer,
    )
    monkeypatch.setattr(
        "dspy.ChainOfThought",
        lambda sig: SimpleNamespace(named_predictors=lambda: []),
    )

    class _FakeArtifactManager:
        def __init__(self, telemetry_provider, tenant_id):
            pass

        async def load_prompts(self, agent_type):
            return None

        async def promote_if_better(
            self,
            agent_type,
            candidate_prompts,
            candidate_demos,
            baseline_score,
            candidate_score,
            *,
            min_improvement=0.0,
            serve_versioned=False,
            optimizer="unknown",
            train_examples=None,
            **kwargs,
        ):
            gate_calls.append(
                {
                    "agent_type": agent_type,
                    "candidate_prompts": candidate_prompts,
                    "baseline_score": baseline_score,
                    "candidate_score": candidate_score,
                    "min_improvement": min_improvement,
                    "serve_versioned": serve_versioned,
                    "train_examples": train_examples,
                }
            )
            extras = {"served_version": 5} if promote else {}
            return SimpleNamespace(promoted=promote, extra_metrics=extras)

    monkeypatch.setattr(
        "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
        _FakeArtifactManager,
    )
    return fake_gepa, captured, gate_calls


class TestReflectiveBranch:
    @pytest.mark.asyncio
    async def test_all_failure_summary_compiles_and_serves(self, monkeypatch):
        from dspy.teleprompt.gepa.gepa import ScoreWithFeedback

        _patch_rules(
            monkeypatch,
            _rules(
                enable_reflective_recompile=True,
                min_reflective_failures=5,
                reflective_max_metric_calls=42,
            ),
        )
        fake_gepa, captured, gate_calls = _install_reflective_fakes(
            monkeypatch, agent_name="summary"
        )

        low = pd.DataFrame(
            [
                {
                    "query": f"q{i}",
                    "output": json.dumps({"summary": f"bad {i}"}),
                    "score": 0.2,
                }
                for i in range(12)
            ]
        )

        result = await _optimize_agent(
            agent_name="summary",
            low_scoring_df=low,
            high_scoring_df=pd.DataFrame(),
            llm_endpoint=None,
            config_manager=MagicMock(),
            telemetry_provider=MagicMock(),
            tenant_id="acme:acme",
        )

        assert result["status"] == "success"
        assert result["reflective"] is True
        # 12 failing rows split deterministically -> 9 reflect-train, 3 held-out.
        assert result["training_examples"] == 9
        assert result["holdout_examples"] == 0
        assert result["negative_probes"] == 3

        # GEPA saw exactly the 9 reflect-train rows, each carrying the recorded
        # failing output for the feedback metric — NOT the 3 held-out rows.
        assert fake_gepa.trainset is not None
        assert len(fake_gepa.trainset) == 9
        assert [ex._bad_output for ex in fake_gepa.trainset] == [
            f"bad {i}" for i in range(9)
        ]
        assert fake_gepa.trainset[0].inputs().toDict() == {
            "content": json.dumps({"summary": "bad 0"}),
            "summary_type": "comprehensive",
            "target_audience": "general",
        }
        assert captured["reflection_lm"] is not None
        assert captured["max_metric_calls"] == 42

        # The metric handed to GEPA returns the exact ScoreWithFeedback contract.
        swf = captured["metric"](
            SimpleNamespace(_bad_output="alpha beta"),
            SimpleNamespace(summary="alpha beta"),
            None,
            None,
            None,
        )
        assert isinstance(swf, ScoreWithFeedback)
        assert swf.score == 0.0

        # Serving still routes through the promote_if_better gate, versioned.
        assert len(gate_calls) == 1
        assert gate_calls[0]["agent_type"] == "summarizer_agent"
        assert gate_calls[0]["candidate_prompts"] == {
            "summarizer": "Reflective: avoid the failing output."
        }
        assert gate_calls[0]["serve_versioned"] is True
        assert gate_calls[0]["baseline_score"] == 0.30
        assert gate_calls[0]["candidate_score"] == 0.80
        assert gate_calls[0]["train_examples"] == 9
        assert result["served"] == {
            "served_agent": "summarizer_agent",
            "version": 5,
            "active": True,
            "promoted": True,
            "baseline_score": 0.30,
            "candidate_score": 0.80,
        }

    @pytest.mark.asyncio
    async def test_all_failure_search_populates_gepa_trainset(self, monkeypatch):
        _patch_rules(
            monkeypatch,
            _rules(enable_reflective_recompile=True, min_reflective_failures=5),
        )
        fake_gepa, captured, gate_calls = _install_reflective_fakes(
            monkeypatch, agent_name="search"
        )

        low = pd.DataFrame(
            [{"query": f"query {i}", "output": "{}", "score": 0.2} for i in range(12)]
        )

        result = await _optimize_agent(
            agent_name="search",
            low_scoring_df=low,
            high_scoring_df=pd.DataFrame(),
            llm_endpoint=None,
            config_manager=MagicMock(),
            telemetry_provider=MagicMock(),
            tenant_id="acme:acme",
        )

        assert result["status"] == "success"
        assert result["reflective"] is True
        assert len(fake_gepa.trainset) == 9
        assert fake_gepa.trainset[0].inputs().toDict() == {"query": "query 0"}
        # search rows carry no free-text failing output — the metric scores
        # enum-validity, so _bad_output is the empty string.
        assert all(ex._bad_output == "" for ex in fake_gepa.trainset)
        assert gate_calls[0]["agent_type"] == "search_agent"
        assert result["served"]["promoted"] is True

    @pytest.mark.asyncio
    async def test_flag_off_keeps_no_positive_examples_skip(self, monkeypatch):
        _patch_rules(monkeypatch, _rules(enable_reflective_recompile=False))

        low = pd.DataFrame(
            [{"query": f"q{i}", "output": "{}", "score": 0.2} for i in range(12)]
        )

        result = await _optimize_agent(
            agent_name="search",
            low_scoring_df=low,
            high_scoring_df=pd.DataFrame(),
            llm_endpoint=None,
            config_manager=MagicMock(),
            telemetry_provider=MagicMock(),
            tenant_id="acme:acme",
        )

        assert result == {
            "status": "skipped",
            "reason": "no_positive_examples",
            "negative_examples": 12,
        }

    @pytest.mark.asyncio
    async def test_below_min_failures_skips_with_distinct_reason(self, monkeypatch):
        _patch_rules(
            monkeypatch,
            _rules(enable_reflective_recompile=True, min_reflective_failures=10),
        )

        low = pd.DataFrame(
            [{"query": f"q{i}", "output": "{}", "score": 0.2} for i in range(6)]
        )

        result = await _optimize_agent(
            agent_name="search",
            low_scoring_df=low,
            high_scoring_df=pd.DataFrame(),
            llm_endpoint=None,
            config_manager=MagicMock(),
            telemetry_provider=MagicMock(),
            tenant_id="acme:acme",
        )

        assert result == {
            "status": "skipped",
            "reason": "insufficient_failures_to_reflect",
            "negative_examples": 6,
        }
