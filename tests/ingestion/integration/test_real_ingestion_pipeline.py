"""
REAL Integration tests for video ingestion pipeline.

These tests use ACTUAL:
- Video files (no mocks)
- Processors (real keyframe extraction, transcription, embeddings)
- Vespa backend (real document creation)

NO MOCKS - this tests the real end-to-end pipeline.
"""

import shutil
from pathlib import Path

import pytest

# Mark all tests as integration and requiring models
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_models,
    pytest.mark.slow,
]


@pytest.mark.integration
@pytest.mark.requires_cv2
class TestRealIngestionPipeline:
    """Real integration tests with actual video processing."""

    @pytest.fixture
    def test_video_path(self):
        """Get path to smallest test video."""
        # Use the smallest available test video
        video_path = Path("data/testset/evaluation/sample_videos/v_-6dz6tBH77I.mp4")
        return video_path

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "ingestion_output"
        output_dir.mkdir()
        yield output_dir
        # Cleanup
        if output_dir.exists():
            shutil.rmtree(output_dir)

    def test_real_strategy_resolution_frame_based(self):
        """The shipped ColPali frame profile resolves to frame processing, one
        document per frame, and its schema's ranking strategies."""
        from cogniverse_runtime.ingestion.strategy import StrategyConfig

        config = StrategyConfig(Path(__file__).resolve().parents[3] / "configs")
        strategy = config.get_strategy("video_colpali_smol500_mv_frame")

        assert (
            strategy.processing_type,
            strategy.segmentation,
            strategy.storage_mode,
        ) == ("frame_based", "frames", "multi_doc")
        assert strategy.schema_name == "video_colpali_smol500_mv_frame"
        assert strategy.default_ranking == "default"
        assert sorted(strategy.ranking_strategies) == [
            "binary_binary",
            "bm25_no_description",
            "bm25_only",
            "default",
            "float_binary",
            "float_float",
            "hybrid_binary_bm25",
            "hybrid_binary_bm25_no_description",
            "hybrid_bm25_binary",
            "hybrid_bm25_binary_no_description",
            "hybrid_bm25_float",
            "hybrid_bm25_float_no_description",
            "hybrid_float_bm25",
            "hybrid_float_bm25_no_description",
            "phased",
        ]
        assert config.get_strategy("video_colpali_smol500_mv_frame") is strategy
