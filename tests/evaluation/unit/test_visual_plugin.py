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

        with patch("cogniverse_foundation.config.utils.get_config") as mock_config:
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
            patch("cogniverse_foundation.config.utils.get_config") as mock_config,
            patch(
                "cogniverse_evaluation.evaluators.configurable_visual_judge.ConfigurableVisualJudge"
            ) as mock_judge_class,
        ):
            mock_config.return_value = {
                "evaluators": {
                    "test_judge": {"provider": "openai", "model": "test_model"}
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


class TestVisualJudgeTenantAndModelOverride:
    """The experiment's tenant and --llm-model must reach the visual judge —
    they were dropped in favor of the system tenant and the config model."""

    @pytest.mark.unit
    def test_tenant_and_model_overrides_applied(self):
        from unittest.mock import patch

        from cogniverse_core.common.media import MediaLocator
        from cogniverse_evaluation.evaluators.configurable_visual_judge import (
            ConfigurableVisualJudge,
        )

        captured = {}

        def fake_get_config(tenant_id, config_manager=None):
            captured["tenant_id"] = tenant_id
            return {
                "evaluators": {
                    "visual_judge": {
                        "provider": "openai",
                        "model": "config-model",
                        "base_url": "http://config",
                    }
                }
            }

        locator = MediaLocator.__new__(MediaLocator)
        with patch(
            "cogniverse_evaluation.evaluators.configurable_visual_judge.get_config",
            side_effect=fake_get_config,
        ):
            judge = ConfigurableVisualJudge(
                locator=locator,
                evaluator_name="visual_judge",
                tenant_id="acme:acme",
                model="override-model",
                base_url="http://override",
            )

        # The experiment tenant was used for config resolution, and the
        # explicit model/base_url won over the config values.
        assert captured["tenant_id"] == "acme:acme"
        assert judge.model == "override-model"
        assert judge.base_url == "http://override"

    @pytest.mark.unit
    def test_falls_back_to_config_when_no_override(self):
        from unittest.mock import patch

        from cogniverse_core.common.media import MediaLocator
        from cogniverse_evaluation.evaluators.configurable_visual_judge import (
            ConfigurableVisualJudge,
        )

        cfg = {
            "evaluators": {
                "visual_judge": {"model": "config-model", "base_url": "http://config"}
            }
        }
        locator = MediaLocator.__new__(MediaLocator)
        with patch(
            "cogniverse_evaluation.evaluators.configurable_visual_judge.get_config",
            return_value=cfg,
        ):
            judge = ConfigurableVisualJudge(locator=locator)

        assert judge.model == "config-model"
        assert judge.base_url == "http://config"


class TestVisualJudgeScorerFailureContract:
    """Judge failures are excluded from the mean and surfaced in metadata; a
    judge outage (every attempted call failing) refuses to produce a score
    instead of reporting a uniform 0.0 quality collapse."""

    def _state(self, outputs):
        state = Mock()
        state.input = {"query": "test query"}
        state.outputs = outputs
        return state

    def _config_patch(self):
        return patch(
            "cogniverse_foundation.config.utils.get_config",
            return_value={
                "evaluators": {
                    "test_judge": {
                        "provider": "openai",
                        "model": "m",
                        "base_url": "http://config",
                    }
                }
            },
        )

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_failures_excluded_from_mean(self):
        scorer = VisualEvaluatorPlugin.create_visual_judge_scorer("test_judge")
        state = self._state(
            {
                "p1_s1": {"success": True, "results": [{"video_id": "v1"}]},
                "p2_s1": {"success": True, "results": [{"video_id": "v2"}]},
            }
        )

        good = Mock()
        good.score = 0.8
        good.label = "excellent_match"

        judge = Mock()
        judge.evaluate.side_effect = [good, RuntimeError("Vision API error: 503")]

        with (
            self._config_patch(),
            patch(
                "cogniverse_evaluation.evaluators.configurable_visual_judge.ConfigurableVisualJudge",
                return_value=judge,
            ),
        ):
            score = await scorer(state, None)

        assert score.value == 0.8
        assert score.metadata["individual_scores"] == {"p1_s1": 0.8}
        assert "503" in score.metadata["failed_evaluations"]["p2_s1"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_all_attempted_failing_raises(self):
        scorer = VisualEvaluatorPlugin.create_visual_judge_scorer("test_judge")
        state = self._state(
            {
                "p1_s1": {"success": True, "results": [{"video_id": "v1"}]},
                "empty": {"success": True, "results": []},
            }
        )

        judge = Mock()
        judge.evaluate.side_effect = RuntimeError("connection refused")

        with (
            self._config_patch(),
            patch(
                "cogniverse_evaluation.evaluators.configurable_visual_judge.ConfigurableVisualJudge",
                return_value=judge,
            ),
        ):
            with pytest.raises(RuntimeError, match="every attempted configuration"):
                await scorer(state, None)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_evaluation_failed_label_is_failure_not_zero(self):
        scorer = VisualEvaluatorPlugin.create_visual_judge_scorer("test_judge")
        state = self._state(
            {
                "p1_s1": {"success": True, "results": [{"video_id": "v1"}]},
                "p2_s1": {"success": True, "results": [{"video_id": "v2"}]},
            }
        )

        failed_result = Mock()
        failed_result.score = 0.0
        failed_result.label = "evaluation_failed"
        failed_result.explanation = "Visual evaluation failed: timeout"

        good = Mock()
        good.score = 0.6
        good.label = "good_match"

        judge = Mock()
        judge.evaluate.side_effect = [failed_result, good]

        with (
            self._config_patch(),
            patch(
                "cogniverse_evaluation.evaluators.configurable_visual_judge.ConfigurableVisualJudge",
                return_value=judge,
            ),
        ):
            score = await scorer(state, None)

        # The failed judgment is not averaged in as a 0.0 quality signal.
        assert score.value == 0.6
        assert score.metadata["individual_scores"] == {"p2_s1": 0.6}
        assert "timeout" in score.metadata["failed_evaluations"]["p1_s1"]


class TestVisualJudgeRequestTimeout:
    """The vision call is bounded — a hung or dead endpoint fails within the
    configured timeout instead of hanging the experiment run."""

    def _judge(self, base_url, timeout_s):
        from cogniverse_core.common.media import MediaLocator
        from cogniverse_evaluation.evaluators.configurable_visual_judge import (
            ConfigurableVisualJudge,
        )

        with patch(
            "cogniverse_evaluation.evaluators.configurable_visual_judge.get_config",
            return_value={
                "evaluators": {
                    "visual_judge": {
                        "provider": "openai",
                        "model": "m",
                        "base_url": base_url,
                        "request_timeout_s": timeout_s,
                    }
                }
            },
        ):
            return ConfigurableVisualJudge(locator=MediaLocator.__new__(MediaLocator))

    @pytest.mark.unit
    def test_hung_endpoint_times_out(self, tmp_path):
        import socket
        import threading
        import time

        import requests as requests_lib

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        server.settimeout(0.2)
        stop = threading.Event()
        conns = []

        def _accept():
            # Accept connections but never respond.
            while not stop.is_set():
                try:
                    conn, _ = server.accept()
                    conns.append(conn)
                except socket.timeout:
                    continue

        thread = threading.Thread(target=_accept, daemon=True)
        thread.start()

        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"\xff\xd8\xff\xdbfakejpeg")

        judge = self._judge(f"http://127.0.0.1:{port}", timeout_s=1.0)
        assert judge.request_timeout_s == 1.0

        started = time.monotonic()
        try:
            with pytest.raises(requests_lib.exceptions.RequestException):
                judge._score_frames("query", [str(frame)])
            assert time.monotonic() - started < 10
        finally:
            stop.set()
            thread.join(timeout=2)
            for conn in conns:
                conn.close()
            server.close()

    @pytest.mark.unit
    def test_dead_endpoint_raises_promptly(self, tmp_path):
        import requests as requests_lib

        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"\xff\xd8\xff\xdbfakejpeg")

        judge = self._judge("http://127.0.0.1:29071", timeout_s=1.0)
        with pytest.raises(requests_lib.exceptions.RequestException):
            judge._score_frames("query", [str(frame)])
