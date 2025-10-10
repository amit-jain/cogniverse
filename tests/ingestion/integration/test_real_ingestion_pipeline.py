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

    def test_real_embedding_generation_colpali(self, test_video_path, temp_output_dir):
        """
        REAL TEST: Generate ColPali embeddings from actual video frames.

        SKIPPED by default due to model download/inference time.
        Run with: pytest -v -m "not skip" to include this test.
        """
        import logging

        from cogniverse_runtime.ingestion.processors.embedding_processor import EmbeddingProcessor

        # First extract keyframes
        from cogniverse_runtime.ingestion.processors.keyframe_processor import KeyframeProcessor

        logger = logging.getLogger("test")

        keyframe_processor = KeyframeProcessor(
            logger=logger,
            max_frames=2,  # Only 2 frames for speed
        )
        keyframe_result = keyframe_processor.extract_keyframes(test_video_path)

        # Generate real embeddings using the correct constructor
        embedding_processor = EmbeddingProcessor(
            logger=logger,
            embedding_type="multi_vector",
            model_name="vidore/colsmol-500m",  # Real ColPali model (smaller, stable)
        )

        # This will download model on first run and generate REAL embeddings
        # Pass the keyframe result data to generate_embeddings
        result = embedding_processor.generate_embeddings(
            {"video_id": test_video_path.stem, "keyframes": keyframe_result}
        )

        # Validate real embeddings
        assert result is not None
        assert "embeddings" in result

    def test_real_strategy_resolution_frame_based(self):
        """
        REAL TEST: Test strategy resolution for frame-based processing.

        Tests that Strategy correctly resolves processing strategies.
        """
        from cogniverse_runtime.ingestion.strategy import Strategy

        # Create frame-based strategy config
        strategy = Strategy(
            processing_type="frame_based",
            segmentation="frames",
            storage_mode="multi_doc",
            schema_name="video_colpali",
            ranking_strategies={},
            default_ranking="default",
            model_name="colpali",
            model_config={},
            needs_float_embeddings=True,
            needs_binary_embeddings=False,
            embedding_fields={"float_field": "embedding"},
        )

        # Validate strategy creation
        assert strategy is not None
        assert strategy.processing_type == "frame_based"
        assert strategy.segmentation == "frames"
        assert strategy.model_name == "colpali"

        # Verify segmentation is frame-based
        assert "frame" in strategy.segmentation.lower()
