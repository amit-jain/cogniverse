"""
Integration tests for video ingestion with different backends and models.

Tests actual document ingestion with various video processing profiles.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from tests.utils.markers import (skip_heavy_models_in_ci, skip_if_ci,
                                 skip_if_low_memory, skip_if_no_vespa)

# Import components for integration testing
try:
    from src.app.ingestion.pipeline import VideoIngestionPipeline
    from src.app.ingestion.pipeline_builder import PipelineBuilder
    from src.processing.unified_video_pipeline import PipelineConfig
except ImportError:
    # Handle missing imports gracefully
    VideoIngestionPipeline = None
    PipelineBuilder = None
    PipelineConfig = None


@pytest.mark.integration
@pytest.mark.ci_safe
class TestMockBackendIngestion:
    """Test ingestion pipeline with mock backend (CI-safe)."""

    @pytest.fixture
    def temp_video_dir(self):
        """Create temporary directory with mock video files."""
        temp_dir = tempfile.mkdtemp()
        video_dir = Path(temp_dir) / "videos"
        video_dir.mkdir()

        # Create mock video files
        for i in range(3):
            video_file = video_dir / f"test_video_{i}.mp4"
            video_file.write_bytes(b"mock video content")

        yield video_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_pipeline_config(self, temp_video_dir):
        """Create mock pipeline configuration."""
        return {
            "video_dir": temp_video_dir,
            "backend": "mock",
            "max_frames_per_video": 2,
            "transcribe_audio": False,
            "generate_descriptions": False,
        }

    def test_basic_pipeline_creation(self, mock_pipeline_config):
        """Test basic pipeline creation with mock backend."""
        if not PipelineBuilder:
            pytest.skip("Pipeline components not available")

        builder = PipelineBuilder()
        pipeline = builder.build_pipeline(mock_pipeline_config)

        assert pipeline is not None
        assert hasattr(pipeline, "process_video")

    @pytest.mark.requires_cv2
    def test_keyframe_extraction_integration(self, temp_video_dir):
        """Test keyframe extraction in isolation."""
        import logging

        from src.app.ingestion.processors.keyframe_processor import \
            KeyframeProcessor

        logger = logging.getLogger("test")
        processor = KeyframeProcessor(logger, max_frames=5)

        # Create a mock video file
        video_file = temp_video_dir / "test_keyframe.mp4"
        video_file.write_bytes(b"mock video for keyframe test")

        with patch("cv2.VideoCapture") as mock_cap:
            # Mock successful video opening
            mock_cap_instance = Mock()
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.return_value = 30.0  # Mock FPS
            mock_cap_instance.read.return_value = (False, None)  # No frames
            mock_cap.return_value = mock_cap_instance

            with patch("src.common.utils.output_manager.get_output_manager"):
                result = processor.extract_keyframes(video_file)

                assert "video_id" in result
                assert "keyframes" in result
                assert result["video_id"] == "test_keyframe"

    @pytest.mark.requires_ffmpeg
    def test_chunk_extraction_integration(self, temp_video_dir):
        """Test chunk extraction in isolation."""
        import logging

        from src.app.ingestion.processors.chunk_processor import ChunkProcessor

        logger = logging.getLogger("test")
        processor = ChunkProcessor(logger, chunk_duration=30.0)

        video_file = temp_video_dir / "test_chunk.mp4"
        video_file.write_bytes(b"mock video for chunk test")

        with patch("subprocess.run") as mock_subprocess:
            # Mock ffprobe for duration
            mock_subprocess.return_value.stdout = "60.0\n"

            with patch("src.common.utils.output_manager.get_output_manager"):
                result = processor.extract_chunks(video_file)

                assert "video_id" in result
                assert "chunks" in result
                assert result["video_id"] == "test_chunk"


@pytest.mark.integration
@pytest.mark.requires_vespa
@skip_if_no_vespa
class TestVespaBackendIngestion:
    """Test ingestion with actual Vespa backend."""

    @pytest.fixture
    def vespa_test_videos(self):
        """Get test videos for Vespa integration."""
        test_dir = Path("data/testset/evaluation/sample_videos")
        if test_dir.exists():
            return list(test_dir.glob("*.mp4"))[:2]  # Limit to 2 videos
        else:
            pytest.skip("Test videos not available")

    def test_vespa_connection(self):
        """Test basic Vespa connectivity."""
        import requests

        try:
            response = requests.get(
                "http://localhost:8080/ApplicationStatus", timeout=5
            )
            assert response.status_code == 200
        except requests.RequestException:
            pytest.fail("Vespa not accessible at localhost:8080")

    @pytest.mark.slow
    def test_lightweight_vespa_ingestion(self, vespa_test_videos):
        """Test lightweight ingestion to Vespa (no heavy models)."""
        if not vespa_test_videos:
            pytest.skip("No test videos available")

        # Test with basic frame extraction only
        from src.processing.unified_video_pipeline import (
            PipelineConfig, VideoIngestionPipeline)

        config = PipelineConfig.from_config()
        config.video_dir = vespa_test_videos[0].parent
        config.search_backend = "vespa"
        config.transcribe_audio = False
        config.generate_descriptions = False
        config.max_frames_per_video = 1

        pipeline = VideoIngestionPipeline(config)

        # Process just one video
        result = pipeline.process_video(vespa_test_videos[0])

        assert result is not None
        assert "video_id" in result

    @pytest.mark.local_only
    @pytest.mark.requires_colpali
    @skip_heavy_models_in_ci
    def test_colpali_vespa_ingestion(self, vespa_test_videos):
        """Test ColPali model ingestion to Vespa (local only)."""
        if not vespa_test_videos:
            pytest.skip("No test videos available")

        from src.processing.unified_video_pipeline import (
            PipelineConfig, VideoIngestionPipeline)

        config = PipelineConfig.from_config()
        config.video_dir = vespa_test_videos[0].parent
        config.search_backend = "vespa"
        config.active_video_profile = "video_colpali_smol500_mv_frame"
        config.max_frames_per_video = 2

        pipeline = VideoIngestionPipeline(config)
        result = pipeline.process_video(vespa_test_videos[0])

        assert result is not None
        assert "embeddings" in result.get("results", {})

    @pytest.mark.local_only
    @pytest.mark.requires_videoprism
    @skip_heavy_models_in_ci
    @skip_if_low_memory
    def test_videoprism_vespa_ingestion(self, vespa_test_videos):
        """Test VideoPrism model ingestion to Vespa (local only)."""
        if not vespa_test_videos:
            pytest.skip("No test videos available")

        from src.processing.unified_video_pipeline import (
            PipelineConfig, VideoIngestionPipeline)

        config = PipelineConfig.from_config()
        config.video_dir = vespa_test_videos[0].parent
        config.search_backend = "vespa"
        config.active_video_profile = "video_videoprism_base_mv_chunk_30s"
        config.max_frames_per_video = 1

        pipeline = VideoIngestionPipeline(config)
        result = pipeline.process_video(vespa_test_videos[0])

        assert result is not None
        assert "embeddings" in result.get("results", {})

    @pytest.mark.local_only
    @pytest.mark.requires_colqwen
    @skip_heavy_models_in_ci
    def test_colqwen_vespa_ingestion(self, vespa_test_videos):
        """Test ColQwen model ingestion to Vespa (local only)."""
        if not vespa_test_videos:
            pytest.skip("No test videos available")

        from src.processing.unified_video_pipeline import (
            PipelineConfig, VideoIngestionPipeline)

        config = PipelineConfig.from_config()
        config.video_dir = vespa_test_videos[0].parent
        config.search_backend = "vespa"
        config.active_video_profile = "video_colqwen_omni_mv_chunk_30s"
        config.max_frames_per_video = 1

        pipeline = VideoIngestionPipeline(config)
        result = pipeline.process_video(vespa_test_videos[0])

        assert result is not None
        assert "embeddings" in result.get("results", {})


@pytest.mark.integration
@pytest.mark.local_only
@skip_if_ci
class TestComprehensiveIngestion:
    """Comprehensive ingestion tests (local development only)."""

    @pytest.fixture
    def all_test_videos(self):
        """Get all available test videos."""
        test_dir = Path("data/testset/evaluation/sample_videos")
        if test_dir.exists():
            return list(test_dir.glob("*.mp4"))
        else:
            pytest.skip("Test videos not available")

    @pytest.mark.slow
    @pytest.mark.requires_vespa
    def test_multi_profile_ingestion(self, all_test_videos):
        """Test ingestion with multiple profiles."""
        if not all_test_videos:
            pytest.skip("No test videos available")

        profiles_to_test = [
            "video_colpali_smol500_mv_frame",
            "video_videoprism_base_mv_chunk_30s",
            "video_colqwen_omni_mv_chunk_30s",
        ]

        from src.processing.unified_video_pipeline import (
            PipelineConfig, VideoIngestionPipeline)

        results = {}
        for profile in profiles_to_test:
            try:
                config = PipelineConfig.from_config()
                config.video_dir = all_test_videos[0].parent
                config.search_backend = "vespa"
                config.active_video_profile = profile
                config.max_frames_per_video = 1

                pipeline = VideoIngestionPipeline(config)
                result = pipeline.process_video(all_test_videos[0])
                results[profile] = result

            except Exception as e:
                # Log but don't fail - some models might not be available
                print(f"Profile {profile} failed: {e}")

        # At least one profile should succeed
        assert len(results) > 0

    @pytest.mark.benchmark
    def test_ingestion_performance(self, all_test_videos):
        """Benchmark ingestion performance."""
        if not all_test_videos:
            pytest.skip("No test videos available")

        import time

        from src.processing.unified_video_pipeline import (
            PipelineConfig, VideoIngestionPipeline)

        config = PipelineConfig.from_config()
        config.video_dir = all_test_videos[0].parent
        config.search_backend = "vespa"
        config.active_video_profile = "video_colpali_smol500_mv_frame"
        config.max_frames_per_video = 5

        pipeline = VideoIngestionPipeline(config)

        start_time = time.time()
        result = pipeline.process_video(all_test_videos[0])
        end_time = time.time()

        processing_time = end_time - start_time
        print(f"Processing time: {processing_time:.2f} seconds")

        assert result is not None
        assert processing_time < 300  # Should complete within 5 minutes
