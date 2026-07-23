"""Unit tests for the live Inspect scorers in core/inspect_scorers.py.

Focus on precision@k / recall@k, which are now wired into the production
``get_configured_scorers`` and score against the sample's ground-truth target.
"""

from types import SimpleNamespace

import pytest

from cogniverse_evaluation.core.inspect_scorers import (
    get_configured_scorers,
    precision_scorer,
    recall_scorer,
)
from cogniverse_evaluation.core.solver_output import pack_solver_output


def _state_with(results: list) -> SimpleNamespace:
    """Build a minimal TaskState-shaped object carrying packed solver output."""
    packed = pack_solver_output(
        query="q", search_results={"cfg": {"success": True, "results": results}}
    )
    message = SimpleNamespace(content=packed)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(output=SimpleNamespace(choices=[choice]))


@pytest.mark.unit
@pytest.mark.ci_fast
class TestPrecisionRecallScorers:
    @pytest.mark.asyncio
    async def test_precision_at_k(self):
        # retrieved {a,b,c}, relevant {a,b} -> precision = 2/3
        state = _state_with([{"video_id": "a"}, {"video_id": "b"}, {"video_id": "c"}])
        score = await precision_scorer()(state, ["a", "b"])
        assert score.value == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_recall_at_k_full(self):
        # retrieved {a,b,c}, relevant {a,b} -> recall = 2/2 = 1.0
        state = _state_with([{"video_id": "a"}, {"video_id": "b"}, {"video_id": "c"}])
        score = await recall_scorer()(state, ["a", "b"])
        assert score.value == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_recall_at_k_partial(self):
        # retrieved {a}, relevant {a,b,c} -> recall = 1/3
        state = _state_with([{"video_id": "a"}])
        score = await recall_scorer()(state, ["a", "b", "c"])
        assert score.value == pytest.approx(1 / 3)

    @pytest.mark.asyncio
    async def test_vacuous_when_no_ground_truth(self):
        # No target -> precision/recall are undefined; score vacuously 1.0.
        state = _state_with([{"video_id": "a"}])
        assert (await precision_scorer()(state, [])).value == pytest.approx(1.0)
        assert (await recall_scorer()(state, [])).value == pytest.approx(1.0)

    def test_default_config_includes_precision_recall(self):
        # Default set: relevance, diversity, result_count, precision, recall.
        assert len(get_configured_scorers({})) == 5

    def test_precision_recall_can_be_disabled(self):
        assert len(get_configured_scorers({"use_precision_recall": False})) == 3

    def test_visual_scorers_wired_when_enabled(self):
        # Visual-judge / quality scorers are opt-in and were previously defined
        # but never added to the production scorer set.
        base = len(get_configured_scorers({}))
        assert len(get_configured_scorers({"enable_llm_evaluators": True})) == base + 1
        assert (
            len(
                get_configured_scorers(
                    {"enable_llm_evaluators": True, "enable_quality_evaluators": True}
                )
            )
            == base + 2
        )
        # experiment_tracker nests the flags under "evaluation" — still wired.
        nested = {
            "evaluation": {
                "enable_llm_evaluators": True,
                "enable_quality_evaluators": True,
            }
        }
        assert len(get_configured_scorers(nested)) == base + 2


@pytest.mark.unit
@pytest.mark.ci_fast
class TestDefaultScorerClosures:
    """The relevance/diversity/result_count scorers only executed inside the
    non-CI e2e tier; drive their score closures directly so a scoring
    regression is caught in the fast lane."""

    @pytest.mark.asyncio
    async def test_relevance_scorer_scores_keyword_overlap(self):
        from cogniverse_evaluation.core.inspect_scorers import relevance_scorer

        results = [{"content": "a red kite flying over the green field"}]
        packed = pack_solver_output(
            query="red kite",
            search_results={"cfg": {"success": True, "results": results}},
        )
        state = SimpleNamespace(
            output=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=packed))]
            )
        )
        score = await relevance_scorer()(state, ["v1"])
        assert 0.0 < score.value <= 1.0
        assert "cfg" in score.metadata["individual_scores"]

    @pytest.mark.asyncio
    async def test_result_count_scorer_scales_with_count(self):
        from cogniverse_evaluation.core.inspect_scorers import result_count_scorer

        five = [{"video_id": f"v{i}"} for i in range(5)]
        packed = pack_solver_output(
            query="q", search_results={"cfg": {"success": True, "results": five}}
        )
        state = SimpleNamespace(
            output=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=packed))]
            )
        )
        score = await result_count_scorer()(state, ["v1"])
        # 5 results / 10 = 0.5
        assert score.value == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_diversity_scorer_runs_without_error(self):
        from cogniverse_evaluation.core.inspect_scorers import diversity_scorer

        results = [
            {"content": "alpha beta"},
            {"content": "gamma delta"},
        ]
        packed = pack_solver_output(
            query="q", search_results={"cfg": {"success": True, "results": results}}
        )
        state = SimpleNamespace(
            output=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=packed))]
            )
        )
        score = await diversity_scorer()(state, ["v1"])
        assert 0.0 <= score.value <= 1.0
