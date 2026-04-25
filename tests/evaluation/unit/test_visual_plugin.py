"""
Unit tests for visual evaluator plugin.
"""

from unittest.mock import Mock, patch

import pytest

from cogniverse_evaluation.plugins.visual_evaluator import (
    VisualEvaluatorPlugin,
    get_visual_scorers,
)


class TestVisualEvaluatorPlugin:
    """Test visual evaluator plugin functionality."""

    @pytest.mark.unit
    def test_get_visual_scorers_no_config(self):
        """Test that no scorers returned when not configured."""
        config = {"enable_llm_evaluators": False, "enable_quality_evaluators": False}
        scorers = get_visual_scorers(config)
        assert len(scorers) == 0

    @pytest.mark.unit
    def test_get_visual_scorers_with_llm(self):
        """Test visual judge scorer creation."""
        config = {
            "enable_llm_evaluators": True,
            "evaluator_name": "test_judge",
            "enable_quality_evaluators": False,
        }
        scorers = get_visual_scorers(config)
        assert len(scorers) == 1

    @pytest.mark.unit
    def test_get_visual_scorers_with_quality(self):
        """Test quality scorer creation."""
        config = {"enable_llm_evaluators": False, "enable_quality_evaluators": True}
        scorers = get_visual_scorers(config)
        assert len(scorers) == 1

    @pytest.mark.unit
    def test_get_visual_scorers_with_both(self):
        """Test both scorers creation."""
        config = {
            "enable_llm_evaluators": True,
            "enable_quality_evaluators": True,
            "evaluator_name": "test_judge",
        }
        scorers = get_visual_scorers(config)
        assert len(scorers) == 2

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_visual_judge_scorer_no_config(self):
        """Test visual judge scorer when evaluator not configured."""
        scorer = VisualEvaluatorPlugin.create_visual_judge_scorer("missing_judge")

        # Mock state
        state = Mock()
        state.input = {"query": "test query"}
        state.outputs = {}

        with patch("cogniverse_core.config.utils.get_config") as mock_config:
            mock_config.return_value = {"evaluators": {}}

            score = await scorer(state, None)
            assert score.value == 0.0
            assert "not configured" in score.explanation

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_visual_judge_scorer_with_results(self):
        """Test visual judge scorer with search results."""
        scorer = VisualEvaluatorPlugin.create_visual_judge_scorer("test_judge")

        # Mock state
        state = Mock()
        state.input = {"query": "test query"}
        state.outputs = {
            "profile1_strategy1": {
                "success": True,
                "results": [
                    {"video_id": "video1", "score": 0.9},
                    {"video_id": "video2", "score": 0.8},
                ],
            }
        }

        with (
            patch("cogniverse_core.config.utils.get_config") as mock_config,
            patch(
                "cogniverse_evaluation.evaluators.configurable_visual_judge.ConfigurableVisualJudge"
            ) as mock_judge_class,
        ):
            mock_config.return_value = {
                "evaluators": {
                    "test_judge": {"provider": "ollama", "model": "test_model"}
                }
            }

            mock_judge = Mock()
            mock_eval_result = Mock()
            mock_eval_result.score = 0.85
            mock_judge.evaluate.return_value = mock_eval_result
            mock_judge_class.return_value = mock_judge

            score = await scorer(state, None)
            assert score.value == 0.85
            assert "visual_evaluator" in score.metadata["plugin"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_quality_scorer(self):
        """Test quality evaluator scorer."""
        scorer = VisualEvaluatorPlugin.create_quality_scorer()

        # Mock state
        state = Mock()
        state.input = {"query": "test query"}
        state.outputs = {
            "profile1_strategy1": {"success": True, "results": [{"video_id": "video1"}]}
        }

        with patch(
            "cogniverse_evaluation.evaluators.sync_reference_free.create_sync_evaluators"
        ) as mock_create:
            mock_evaluator = Mock()
            mock_eval_result = Mock()
            mock_eval_result.score = 0.75
            mock_evaluator.evaluate.return_value = mock_eval_result
            mock_evaluator.__class__.__name__ = "TestEvaluator"
            mock_create.return_value = [mock_evaluator]

            score = await scorer(state, None)
            assert score.value == 0.75
            assert "visual_evaluator" in score.metadata["plugin"]


class TestConfigurableVisualJudgeGetVideoPath:
    """Regression tests for ConfigurableVisualJudge._get_video_path extension lookup."""

    @staticmethod
    def _bare_judge(cache_root):
        from cogniverse_core.common.media import MediaConfig, MediaLocator
        from cogniverse_evaluation.evaluators.configurable_visual_judge import (
            ConfigurableVisualJudge,
        )

        judge = ConfigurableVisualJudge.__new__(ConfigurableVisualJudge)
        judge.locator = MediaLocator(
            tenant_id="test", config=MediaConfig(), cache_root=cache_root
        )
        return judge

    @pytest.mark.unit
    def test_video_id_alone_no_longer_resolves(self, tmp_path, monkeypatch):
        """Legacy probe removed: source_url is now required."""
        sample_dir = tmp_path / "data" / "testset" / "evaluation" / "sample_videos"
        sample_dir.mkdir(parents=True)
        (sample_dir / "v_-HpCLXdtcas.mp4").write_bytes(b"")
        monkeypatch.chdir(tmp_path)

        judge = self._bare_judge(tmp_path / "cache")
        assert judge._get_video_path({"video_id": "v_-HpCLXdtcas"}) is None

    @pytest.mark.unit
    @pytest.mark.parametrize("ext", ["mp4", "mkv", "avi", "mov"])
    def test_finds_video_via_source_url_for_each_extension(self, tmp_path, ext):
        clip = tmp_path / f"v.{ext}"
        clip.write_bytes(b"video")
        judge = self._bare_judge(tmp_path / "cache")

        result = judge._get_video_path({"source_url": f"file://{clip}"})

        assert result == str(clip)

    @pytest.mark.unit
    def test_finds_video_via_source_url(self, tmp_path):
        clip = tmp_path / "v.mp4"
        clip.write_bytes(b"video")

        judge = self._bare_judge(tmp_path / "cache")
        result = judge._get_video_path({"source_url": f"file://{clip}"})

        assert result == str(clip)

    @pytest.mark.unit
    def test_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        judge = self._bare_judge(tmp_path / "cache")
        assert judge._get_video_path({"video_id": "does_not_exist"}) is None
