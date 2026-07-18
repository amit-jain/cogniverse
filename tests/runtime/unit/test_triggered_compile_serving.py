"""Triggered-mode compiles must WIN a held-out eval before serving.

The compile used to promote every result straight to the ACTIVE artefact —
a recompile that produced worse prompts went live un-gated. Serving now goes
through ``ArtifactManager.promote_if_better``: the candidate is scored against
the currently-active baseline on held-out positives (token-F1 to the labeled
output) plus known-bad probes (reward for NOT reproducing a failing output),
and only a win by at least ``optimization_improvement_threshold`` flips
active. Rejections land in the experiments ledger, not in production.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from cogniverse_runtime.optimization_cli import (
    _SERVE_TARGET,
    _holdout_scores,
    _optimize_agent,
    _serve_compiled_prompts,
    _split_train_holdout,
    _token_f1,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _GateRecordingManager:
    """Records the promote_if_better call the serving path must make."""

    def __init__(self, promote=True):
        self.promote = promote
        self.gate_calls = []

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
        self.gate_calls.append(
            {
                "agent_type": agent_type,
                "candidate_prompts": candidate_prompts,
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "min_improvement": min_improvement,
                "serve_versioned": serve_versioned,
            }
        )
        extras = {"served_version": 7} if self.promote else {}
        return SimpleNamespace(promoted=self.promote, extra_metrics=extras)


def _compiled(instructions="Optimized: rank by intent."):
    predictor = SimpleNamespace(
        signature=SimpleNamespace(instructions=instructions), demos=[]
    )
    return SimpleNamespace(named_predictors=lambda: [("predict", predictor)])


def test_serve_target_maps_compile_names_to_dispatch_agents():
    assert _SERVE_TARGET == {
        "search": ("search_agent", "search_optimizer"),
        "summary": ("summarizer_agent", "summarizer"),
        "report": ("detailed_report_agent", "report_generator"),
    }


class TestEvalPrimitives:
    def test_token_f1_exact_values(self):
        assert _token_f1("rank results by intent", "rank results by intent") == 1.0
        assert _token_f1("alpha beta", "gamma delta") == 0.0
        assert _token_f1("quick brown fox", "brown fox jumps") == pytest.approx(2 / 3)
        assert _token_f1("", "anything") == 0.0
        assert _token_f1("x", "") == 0.0

    def test_split_train_holdout_deterministic(self):
        assert _split_train_holdout(list(range(8))) == ([0, 1, 2, 3, 4, 5], [6, 7])
        assert _split_train_holdout([1, 2, 3]) == ([1, 2], [3])
        assert _split_train_holdout([1, 2]) == ([1], [2])
        assert _split_train_holdout([1]) == ([1], [])
        assert _split_train_holdout([]) == ([], [])

    def test_search_validity_exact(self):
        from cogniverse_runtime.optimization_cli import _search_validity

        good = SimpleNamespace(
            primary_intent="temporal_search",
            complexity_level="Simple",
            needs_video_search="true",
        )
        assert _search_validity(good) == 1.0
        partial = SimpleNamespace(
            primary_intent="find videos about cats",
            complexity_level="moderate",
            needs_video_search="yes",
        )
        assert _search_validity(partial) == pytest.approx(1 / 3)
        assert _search_validity(SimpleNamespace()) == 0.0

    def test_holdout_scores_exact_for_summary(self):
        import dspy

        def _example(label):
            return dspy.Example(
                content="c",
                summary_type="comprehensive",
                target_audience="general",
                summary=label,
                key_points="[]",
                confidence=0.9,
            ).with_inputs("content", "summary_type", "target_audience")

        class _StubModule:
            def __init__(self, outputs):
                self.outputs = list(outputs)
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(summary=self.outputs.pop(0))

        holdout = [_example("alpha beta"), _example("gamma delta")]
        negatives = [
            (
                {
                    "content": "{}",
                    "summary_type": "comprehensive",
                    "target_audience": "general",
                },
                "bad output text",
            )
        ]
        # baseline: exact on ex1, miss on ex2, echoes the known-bad output.
        baseline = _StubModule(["alpha beta", "wrong stuff here", "bad output text"])
        # candidate: exact on both, diverges from the known-bad output.
        candidate = _StubModule(["alpha beta", "gamma delta", "good different text"])

        b, c = _holdout_scores(baseline, candidate, holdout, negatives, "summary")

        assert b == pytest.approx((1.0 + 0.0 + 0.0) / 3)
        assert c == pytest.approx((1.0 + 1.0 + 2 / 3) / 3)
        # Both modules saw the same three input sets, holdout first, with the
        # signature's optional input blank-filled.
        assert baseline.calls == candidate.calls
        assert len(baseline.calls) == 3
        assert all(call["visual_insights"] == "" for call in baseline.calls)
        assert baseline.calls[2]["content"] == "{}"

    def test_holdout_scores_search_uses_validity(self):
        class _StubModule:
            def __init__(self, preds):
                self.preds = list(preds)
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return self.preds.pop(0)

        valid = SimpleNamespace(
            primary_intent="search",
            complexity_level="simple",
            needs_video_search="true",
        )
        invalid = SimpleNamespace(
            primary_intent="",
            complexity_level="",
            needs_video_search="",
        )
        negatives = [({"query": "previously failing query"}, "")]
        baseline = _StubModule([invalid])
        candidate = _StubModule([valid])

        b, c = _holdout_scores(baseline, candidate, [], negatives, "search")

        assert b == 0.0
        assert c == 1.0
        assert baseline.calls == [{"query": "previously failing query", "context": ""}]


class TestGatedServing:
    @pytest.mark.asyncio
    async def test_winning_scores_serve_through_the_gate(self):
        am = _GateRecordingManager(promote=True)

        result = await _serve_compiled_prompts(
            am,
            "search",
            _compiled(),
            baseline_score=0.40,
            candidate_score=0.60,
            min_improvement=0.05,
        )

        assert len(am.gate_calls) == 1
        call = am.gate_calls[0]
        assert call["agent_type"] == "search_agent"
        assert call["candidate_prompts"] == {
            "search_optimizer": "Optimized: rank by intent."
        }
        assert call["baseline_score"] == 0.40
        assert call["candidate_score"] == 0.60
        assert call["min_improvement"] == 0.05
        assert call["serve_versioned"] is True
        assert result == {
            "served_agent": "search_agent",
            "version": 7,
            "active": True,
            "promoted": True,
            "baseline_score": 0.40,
            "candidate_score": 0.60,
        }

    @pytest.mark.asyncio
    async def test_losing_scores_do_not_serve(self):
        am = _GateRecordingManager(promote=False)

        result = await _serve_compiled_prompts(
            am,
            "search",
            _compiled(),
            baseline_score=0.60,
            candidate_score=0.40,
            min_improvement=0.05,
        )

        assert len(am.gate_calls) == 1
        assert result == {
            "served_agent": "search_agent",
            "version": None,
            "active": False,
            "promoted": False,
            "baseline_score": 0.60,
            "candidate_score": 0.40,
        }

    @pytest.mark.asyncio
    async def test_no_eval_material_does_not_promote(self):
        am = _GateRecordingManager(promote=True)

        result = await _serve_compiled_prompts(am, "search", _compiled())

        assert am.gate_calls == []
        assert result == {
            "served_agent": "search_agent",
            "version": None,
            "active": False,
            "promoted": False,
            "reason": "no_eval_material",
        }

    @pytest.mark.asyncio
    async def test_summary_maps_to_summarizer_predictor(self):
        am = _GateRecordingManager(promote=True)

        await _serve_compiled_prompts(
            am,
            "summary",
            _compiled("Be concise."),
            baseline_score=0.3,
            candidate_score=0.9,
        )

        assert am.gate_calls[0]["agent_type"] == "summarizer_agent"
        assert am.gate_calls[0]["candidate_prompts"] == {"summarizer": "Be concise."}

    @pytest.mark.asyncio
    async def test_no_instructions_serves_nothing(self):
        am = _GateRecordingManager()
        compiled = SimpleNamespace(named_predictors=lambda: [])

        result = await _serve_compiled_prompts(
            am, "report", compiled, baseline_score=0.1, candidate_score=0.9
        )

        assert result is None
        assert am.gate_calls == []


class TestNoPositiveExamples:
    @pytest.mark.asyncio
    async def test_skip_reports_the_discarded_negative_count(self):
        """An all-failure agent must say WHY it was skipped — the operator
        needs to know negatives existed and were unusable as a trainset,
        not a generic no-data shrug."""
        low = pd.DataFrame(
            [{"query": f"q{i}", "output": "{}", "score": 0.3} for i in range(3)]
        )

        result = await _optimize_agent(
            agent_name="search",
            low_scoring_df=low,
            high_scoring_df=pd.DataFrame(),
            llm_endpoint=None,
            config_manager=None,
            telemetry_provider=None,
            tenant_id="acme:acme",
        )

        assert result == {
            "status": "skipped",
            "reason": "no_positive_examples",
            "negative_examples": 3,
        }


class TestMinImprovementKnob:
    def test_reads_the_declared_default_from_config(self):
        """The acceptance gate reads the tenant's
        ``optimization_improvement_threshold`` through the real config path;
        with no tenant override the declared default (0.05) applies."""
        from cogniverse_foundation.config.manager import ConfigManager
        from cogniverse_runtime.optimization_cli import _min_improvement_from_config
        from tests.utils.memory_store import InMemoryConfigStore

        store = InMemoryConfigStore()
        store.initialize()
        cm = ConfigManager(store=store)

        assert _min_improvement_from_config("acme:acme", config_manager=cm) == 0.05
