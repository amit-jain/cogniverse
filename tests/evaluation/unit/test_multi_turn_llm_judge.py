"""Unit tests for LLMJudgeCore score extraction."""

import pytest

from cogniverse_evaluation.evaluators.llm_judge import LLMJudgeCore

pytestmark = pytest.mark.unit


class TestScoreExtraction:
    """_extract_score_from_response must distinguish a real score (including
    a real 0.5) from an unscored/failed reply (None)."""

    def _judge(self) -> LLMJudgeCore:
        return LLMJudgeCore(model_name="x", base_url="http://unused")

    def test_parses_x_out_of_ten(self):
        score, _ = self._judge()._extract_score_from_response("Score: 8/10. Good.")
        assert score == 0.8

    def test_real_half_score_is_not_none(self):
        score, _ = self._judge()._extract_score_from_response("rating: 0.5")
        assert score == 0.5

    def test_transport_failure_string_yields_none(self):
        score, _ = self._judge()._extract_score_from_response(
            "Evaluation failed: connection refused"
        )
        assert score is None

    def test_reply_without_score_yields_none(self):
        score, _ = self._judge()._extract_score_from_response(
            "The results look reasonable overall."
        )
        assert score is None


class TestScoreClamping:
    """LM replies like "12/10" or "-3/10" must clamp into [0, 1] — an
    out-of-range score flows into the quality monitor's persisted means and
    its 0.8/0.5 example-classification gates."""

    def _judge(self) -> LLMJudgeCore:
        return LLMJudgeCore(model_name="x", base_url="http://unused")

    def test_over_ten_clamps_to_one(self):
        score, _ = self._judge()._extract_score_from_response("Score: 12/10")
        assert score == 1.0

    def test_hundred_out_of_ten_clamps_to_one(self):
        score, _ = self._judge()._extract_score_from_response("the value is 100/10")
        assert score == 1.0

    def test_ten_point_five_clamps_to_one(self):
        score, _ = self._judge()._extract_score_from_response("Score: 10.5/10")
        assert score == 1.0

    def test_plain_score_unchanged(self):
        score, _ = self._judge()._extract_score_from_response("Score: 7/10")
        assert score == 0.7


class TestScoreRegexEdges:
    def _judge(self) -> LLMJudgeCore:
        return LLMJudgeCore(model_name="x", base_url="http://unused")

    def test_out_of_100_is_not_treated_as_out_of_10(self):
        assert self._judge()._extract_score_from_response("Score: 85/100")[0] == 0.85
        assert self._judge()._extract_score_from_response("3/100")[0] == 0.03

    def test_negative_score_clamps_to_zero_not_positive(self):
        # -3/10 must be 0.0, not 0.3 (sign dropped by the old regex).
        assert self._judge()._extract_score_from_response("Score: -3/10")[0] == 0.0

    def test_plain_fractions_unchanged(self):
        assert self._judge()._extract_score_from_response("Score: 5/10")[0] == 0.5
        assert self._judge()._extract_score_from_response("rating: 0.5")[0] == 0.5
        assert (
            self._judge()._extract_score_from_response("the value is 100/10")[0] == 1.0
        )
