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


def _compiled(instructions="Optimized: rank by intent.", agent_name="search"):
    """A served module whose predictor carries the given instructions."""
    from cogniverse_runtime.optimization_cli import _served_module

    module = _served_module(agent_name)
    for _, predictor in module.named_predictors():
        predictor.signature = predictor.signature.with_instructions(instructions)
    return module


def _expected_state(module):
    import json

    return json.dumps(json.loads(json.dumps(module.dump_state())), sort_keys=True)


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

    def test_probe_score_is_token_f1_against_the_served_output_field(self):
        from cogniverse_runtime.optimization_cli import _probe_score

        pred = SimpleNamespace(enhanced_query="red kite footage")
        assert _probe_score(pred, "red kite footage", "search") == 1.0
        assert _probe_score(pred, "kite footage clips", "search") == pytest.approx(
            2 / 3
        )
        assert _probe_score(SimpleNamespace(enhanced_query=""), "", "search") == 0.0
        assert _probe_score(pred, "", "search") == 1.0

    def test_holdout_scores_exact_for_summary(self):
        import dspy

        def _example(label):
            return dspy.Example(
                content="c",
                query="q",
                summary_type="comprehensive",
                keyframes=[],
                summary=label,
                key_points="[]",
                confidence_score="0.9",
            ).with_inputs("content", "query", "summary_type", "keyframes")

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
                    "query": "q",
                    "summary_type": "comprehensive",
                    "keyframes": [],
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
        # Both modules saw the same three input sets, holdout first, each one
        # exactly the served signature's input fields.
        assert baseline.calls == candidate.calls
        assert [sorted(call) for call in baseline.calls] == [
            ["content", "keyframes", "query", "summary_type"]
        ] * 3
        assert baseline.calls[2]["content"] == "{}"

    def test_holdout_scores_search_uses_the_served_enhanced_query(self):
        class _StubModule:
            def __init__(self, preds):
                self.preds = list(preds)
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return self.preds.pop(0)

        negatives = [
            (
                {"query": "previously failing query", "modality": "video", "top_k": 10},
                "stale rewritten query",
            )
        ]
        baseline = _StubModule(
            [SimpleNamespace(enhanced_query="stale rewritten query")]
        )
        candidate = _StubModule([SimpleNamespace(enhanced_query="red kite footage")])

        b, c = _holdout_scores(baseline, candidate, [], negatives, "search")

        assert b == 0.0
        assert c == 1.0
        assert baseline.calls == [
            {"query": "previously failing query", "modality": "video", "top_k": 10}
        ]


class TestGatedServing:
    @pytest.mark.asyncio
    async def test_winning_scores_serve_through_the_gate(self):
        am = _GateRecordingManager(promote=True)
        compiled = _compiled()

        result = await _serve_compiled_prompts(
            am,
            "search",
            compiled,
            baseline_score=0.40,
            candidate_score=0.60,
            min_improvement=0.05,
        )

        assert len(am.gate_calls) == 1
        call = am.gate_calls[0]
        assert call["agent_type"] == "search_agent"
        assert call["candidate_prompts"] == {
            "__dspy_module__": _expected_state(compiled)
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

        compiled = _compiled("Be concise.", agent_name="summary")
        await _serve_compiled_prompts(
            am,
            "summary",
            compiled,
            baseline_score=0.3,
            candidate_score=0.9,
        )

        assert am.gate_calls[0]["agent_type"] == "summarizer_agent"
        assert am.gate_calls[0]["candidate_prompts"] == {
            "__dspy_module__": _expected_state(compiled)
        }
        assert '"Be concise."' in _expected_state(compiled)

    @pytest.mark.asyncio
    async def test_no_module_state_serves_nothing(self):
        am = _GateRecordingManager()
        compiled = SimpleNamespace(dump_state=lambda: {})

        result = await _serve_compiled_prompts(
            am, "report", compiled, baseline_score=0.1, candidate_score=0.9
        )

        assert result is None
        assert am.gate_calls == []


class TestNoPositiveExamples:
    @pytest.mark.asyncio
    async def test_too_few_failures_to_reflect_reports_negative_count(
        self, monkeypatch
    ):
        """An all-failure agent must say WHY it was skipped — the operator
        needs the discarded negative count and that there was not enough signal
        to reflect, not a generic no-data shrug. With reflective recompile on by
        default, 3 failures fall below the reflect threshold."""
        from cogniverse_agents.routing.config import (
            AutomationRulesConfig,
            OptimizationTriggersConfig,
        )

        monkeypatch.setattr(
            "cogniverse_runtime.quality_monitor_cli._load_automation_rules",
            lambda tenant_id, config_manager=None: AutomationRulesConfig(
                optimization_triggers=OptimizationTriggersConfig(
                    min_reflective_failures=10
                )
            ),
        )
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
            "reason": "insufficient_failures_to_reflect",
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


@pytest.mark.parametrize("agent_name", ["search", "summary", "report"])
def test_triggered_compile_uses_the_served_signature(agent_name):
    from cogniverse_agents.detailed_report_agent import ReportGenerationSignature
    from cogniverse_agents.search_agent import SearchOptimizationSignature
    from cogniverse_agents.summarizer_agent import SummaryGenerationSignature
    from cogniverse_runtime.optimization_cli import (
        _SERVE_TARGET,
        _served_module,
        _signature_for_agent,
    )

    expected = {
        "search": SearchOptimizationSignature,
        "summary": SummaryGenerationSignature,
        "report": ReportGenerationSignature,
    }[agent_name]
    assert _signature_for_agent(agent_name) is expected
    module = _served_module(agent_name)
    _, predictor_attr = _SERVE_TARGET[agent_name]
    predictor = getattr(module, predictor_attr)
    inner = getattr(predictor, "predict", predictor)
    assert set(expected.input_fields) <= set(inner.signature.input_fields)
    assert set(expected.output_fields) <= set(inner.signature.output_fields)


@pytest.mark.asyncio
async def test_triggered_publishes_the_complete_compiled_module_state():
    import json

    import dspy
    from dspy.teleprompt import BootstrapFewShot

    from cogniverse_agents.summarizer_agent import SummarizationModule

    example = dspy.Example(
        content="Saturn has rings.",
        query="Describe Saturn",
        summary_type="brief",
        keyframes=[],
        summary="Saturn has rings.",
        key_points="rings",
        confidence_score="1.0",
    ).with_inputs("content", "query", "summary_type", "keyframes")
    compiled = BootstrapFewShot(max_bootstrapped_demos=0, max_labeled_demos=1).compile(
        SummarizationModule(), trainset=[example]
    )
    expected = json.loads(json.dumps(compiled.dump_state()))
    assert expected["summarizer.predict"]["demos"] == [dict(example)]
    manager = _GateRecordingManager()
    result = await _serve_compiled_prompts(
        manager,
        "summary",
        compiled,
        baseline_score=0.0,
        candidate_score=1.0,
    )
    assert result["promoted"] is True
    assert manager.gate_calls[0]["candidate_prompts"] == {
        "__dspy_module__": json.dumps(expected, sort_keys=True)
    }


@pytest.mark.asyncio
async def test_compiled_module_overlay_isolated_across_concurrent_requests():
    import asyncio
    import json

    from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
    from cogniverse_agents.summarizer_agent import SummarizationModule
    from cogniverse_core.agents.base import _dispatched_prompt_overlay

    module = SummarizationModule()
    stock = json.loads(json.dumps(module.dump_state()))
    ready = asyncio.Event()
    arrivals = 0

    async def request(label):
        nonlocal arrivals
        state = json.loads(json.dumps(stock))
        state["summarizer.predict"]["signature"]["instructions"] = f"Answer {label}."
        state["summarizer.predict"]["demos"] = [{"content": label, "summary": label}]
        agent = MemoryAwareMixin()
        agent.set_dispatched_artefact(
            {"prompts": {"__dspy_module__": json.dumps(state)}}
        )
        arrivals += 1
        if arrivals == 2:
            ready.set()
        await ready.wait()
        with _dispatched_prompt_overlay(agent, module) as consumed:
            assert consumed.dump_state() == state
        return state["summarizer.predict"]["demos"]

    assert await asyncio.gather(request("Saturn"), request("Jupiter")) == [
        [{"content": "Saturn", "summary": "Saturn"}],
        [{"content": "Jupiter", "summary": "Jupiter"}],
    ]
    assert module.dump_state() == stock


def test_corrupt_compiled_module_overlay_raises():
    import json

    from cogniverse_agents.memory_aware_mixin import MemoryAwareMixin
    from cogniverse_agents.summarizer_agent import SummarizationModule
    from cogniverse_core.agents.base import _dispatched_prompt_overlay

    agent = MemoryAwareMixin()
    agent.set_dispatched_artefact({"prompts": {"__dspy_module__": "{"}})
    with pytest.raises(json.JSONDecodeError):
        with _dispatched_prompt_overlay(agent, SummarizationModule()):
            pytest.fail("Corrupt compiled state was accepted")
