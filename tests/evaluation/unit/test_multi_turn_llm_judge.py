"""Unit tests for LLMJudgeCore score extraction."""

from cogniverse_evaluation.evaluators.llm_judge import LLMJudgeCore


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
