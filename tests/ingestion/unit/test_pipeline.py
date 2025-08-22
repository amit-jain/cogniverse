#!/usr/bin/env python3
"""
Unit tests for VideoIngestionPipeline.

Tests the video processing pipeline functionality with proper mocking.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.app.ingestion.pipeline import (
    PipelineConfig,
    PipelineStep,
    VideoIngestionPipeline,
)


@pytest.mark.unit
class TestPipelineConfig:
    """Test suite for PipelineConfig class."""

    def test_pipeline_config_defaults(self):
        """Test PipelineConfig initialization with defaults."""
        config = PipelineConfig()

        assert config.extract_keyframes is True
        assert config.transcribe_audio is True
        assert config.generate_descriptions is True
        assert config.generate_embeddings is True
        assert config.keyframe_threshold == 0.999
        assert config.max_frames_per_video == 3000
        assert config.vlm_batch_size == 500
        assert config.video_dir == Path("data/videos")
        assert config.output_dir is None
        assert config.search_backend == "byaldi"

    def test_pipeline_config_custom_values(self):
        """Test PipelineConfig with custom values."""
        config = PipelineConfig(
            extract_keyframes=False,
            transcribe_audio=False,
            generate_descriptions=False,
            generate_embeddings=False,
            keyframe_threshold=0.95,
            max_frames_per_video=1000,
            vlm_batch_size=100,
            video_dir=Path("/custom/path"),
            search_backend="vespa",
        )

        assert config.extract_keyframes is False
        assert config.transcribe_audio is False
        assert config.generate_descriptions is False
        assert config.generate_embeddings is False
        assert config.keyframe_threshold == 0.95
        assert config.max_frames_per_video == 1000
        assert config.vlm_batch_size == 100
        assert config.video_dir == Path("/custom/path")
        assert config.search_backend == "vespa"

    @patch("src.common.utils.output_manager.get_output_manager")
    @patch("src.app.ingestion.pipeline.get_config")
    def test_from_config_method(self, mock_get_config, mock_get_output_manager):
        """Test PipelineConfig.from_config method."""
        # Mock config data
        config_data = {
            "pipeline_config": {
                "extract_keyframes": False,
                "transcribe_audio": True,
                "generate_descriptions": False,
                "generate_embeddings": True,
                "keyframe_threshold": 0.95,
                "max_frames_per_video": 2000,
                "vlm_batch_size": 200,
            },
            "search_backend": "vespa",
        }
        mock_get_config.return_value = config_data

        # Mock output manager
        mock_output_manager = Mock()
        mock_output_manager.get_processing_dir.return_value = Path("/test/output")
        mock_get_output_manager.return_value = mock_output_manager

        config = PipelineConfig.from_config()

        assert config.extract_keyframes is False
        assert config.transcribe_audio is True
        assert config.generate_descriptions is False
        assert config.generate_embeddings is True
        assert config.keyframe_threshold == 0.95
        assert config.max_frames_per_video == 2000
        assert config.vlm_batch_size == 200
        assert config.search_backend == "vespa"
        assert config.output_dir == Path("/test/output")

    @patch("src.common.utils.output_manager.get_output_manager")
    @patch("src.app.ingestion.pipeline.get_config")
    def test_from_profile_method(self, mock_get_config, mock_get_output_manager):
        """Test PipelineConfig.from_profile method."""
        # Mock config data with profiles
        config_data = {
            "video_processing_profiles": {
                "test_profile": {
                    "pipeline_config": {
                        "extract_keyframes": True,
                        "transcribe_audio": False,
                        "generate_descriptions": True,
                        "generate_embeddings": False,
                        "keyframe_threshold": 0.99,
                        "max_frames_per_video": 1500,
                        "vlm_batch_size": 300,
                    }
                }
            },
            "search_backend": "byaldi",
        }
        mock_get_config.return_value = config_data

        # Mock output manager
        mock_output_manager = Mock()
        mock_output_manager.get_processing_dir.return_value = Path("/profile/output")
        mock_get_output_manager.return_value = mock_output_manager

        config = PipelineConfig.from_profile("test_profile")

        assert config.extract_keyframes is True
        assert config.transcribe_audio is False
        assert config.generate_descriptions is True
        assert config.generate_embeddings is False
        assert config.keyframe_threshold == 0.99
        assert config.max_frames_per_video == 1500
        assert config.vlm_batch_size == 300
        assert config.search_backend == "byaldi"
        assert config.output_dir == Path("/profile/output")

    @patch("src.app.ingestion.pipeline.get_config")
    def test_from_profile_method_profile_not_found(self, mock_get_config):
        """Test PipelineConfig.from_profile with non-existent profile."""
        config_data = {"video_processing_profiles": {}, "search_backend": "byaldi"}
        mock_get_config.return_value = config_data

        with pytest.raises(
            ValueError, match="Profile 'missing_profile' not found in config"
        ):
            PipelineConfig.from_profile("missing_profile")


@pytest.mark.unit
class TestVideoIngestionPipelineUtilityMethods:
    """Test suite for VideoIngestionPipeline utility methods."""

    @pytest.fixture
    def pipeline_config(self):
        """Create a test pipeline config."""
        return PipelineConfig(
            extract_keyframes=True,
            transcribe_audio=True,
            generate_descriptions=True,
            generate_embeddings=True,
            max_frames_per_video=100,
            search_backend="vespa",
        )

    @pytest.fixture
    def mock_pipeline(self, pipeline_config):
        """Create a mock pipeline for testing utility methods."""
        with patch.multiple(
            VideoIngestionPipeline,
            __init__=Mock(return_value=None),  # Skip actual initialization
            process_video_async=Mock(return_value={"video_id": "test", "status": "completed"}),
            generate_embeddings=Mock(return_value={"embeddings": "mock"}),
            _check_cache_async=Mock(return_value={}),
            _save_to_cache_async=Mock(return_value=None),
            _get_cached_data=Mock(return_value={}),
            _process_segmentation=Mock(return_value={}),
        ):
            pipeline = VideoIngestionPipeline.__new__(VideoIngestionPipeline)
            pipeline.config = pipeline_config
            pipeline.logger = Mock()
            pipeline.schema_name = "test_schema"
            # Create mock strategy_set with segmentation
            mock_strategy_set = Mock()
            mock_segmentation = Mock()
            mock_segmentation.chunk_duration = 30.0
            mock_strategy_set.segmentation = mock_segmentation
            pipeline.strategy_set = mock_strategy_set

            # Add app_config for _get_chunk_duration method
            pipeline.app_config = Mock()
            pipeline.app_config.get.return_value = {}
            # Create a proper mock strategy object
            mock_strategy = Mock()
            mock_strategy.processing_type = "frame"
            mock_strategy.storage_mode = "vector"
            mock_strategy.schema_name = "test_schema"
            pipeline.strategy = mock_strategy
            pipeline.profile_output_dir = Path("/test/output")  # Add output directory
            return pipeline

    @patch("cv2.VideoCapture")
    def test_get_video_duration_success(self, mock_video_capture, mock_pipeline):
        """Test successful video duration extraction."""
        # Mock cv2.VideoCapture
        mock_cap = Mock()

        # Mock the get method to return fps=25, frame_count=3012.5 for 120.5 seconds
        def mock_get(prop):
            import cv2

            if prop == cv2.CAP_PROP_FPS:
                return 25.0
            elif prop == cv2.CAP_PROP_FRAME_COUNT:
                return 3012.5
            return 0

        mock_cap.get.side_effect = mock_get
        mock_video_capture.return_value = mock_cap

        video_path = Path("/test/video.mp4")
        duration = mock_pipeline._get_video_duration(video_path)

        assert duration == 120.5
        mock_video_capture.assert_called_once_with(str(video_path))
        mock_cap.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_get_video_duration_error(self, mock_video_capture, mock_pipeline):
        """Test video duration extraction with cv2 error."""
        mock_video_capture.side_effect = Exception("OpenCV error")

        video_path = Path("/test/video.mp4")
        duration = mock_pipeline._get_video_duration(video_path)

        assert duration == 0.0

    def test_get_video_files(self, mock_pipeline, tmp_path):
        """Test get_video_files method."""
        # Create test video files
        video_dir = tmp_path / "videos"
        video_dir.mkdir()

        (video_dir / "video1.mp4").touch()
        (video_dir / "video2.avi").touch()
        (video_dir / "video3.mov").touch()
        (video_dir / "not_video.txt").touch()
        (video_dir / "video4.mkv").touch()

        video_files = mock_pipeline.get_video_files(video_dir)

        # Should find 4 video files (excluding .txt)
        assert len(video_files) == 4
        video_names = [f.name for f in video_files]
        assert "video1.mp4" in video_names
        assert "video2.avi" in video_names
        assert "video3.mov" in video_names
        assert "video4.mkv" in video_names
        assert "not_video.txt" not in video_names

    def test_get_video_files_empty_directory(self, mock_pipeline, tmp_path):
        """Test get_video_files with empty directory."""
        video_dir = tmp_path / "empty"
        video_dir.mkdir()

        video_files = mock_pipeline.get_video_files(video_dir)

        assert len(video_files) == 0

    def test_extract_base_video_data(self, mock_pipeline):
        """Test _extract_base_video_data method."""
        # Test input with various result types
        results = {
            "video_id": "test_video",
            "video_path": "/test/video.mp4",
            "duration": 120.5,
            "keyframes": {"frame_count": 100},
            "audio": {"transcript": "test transcript"},
            "descriptions": {"desc_count": 50},
            "other_data": "should_be_ignored",
        }

        base_data = mock_pipeline._extract_base_video_data(results)

        assert base_data["video_id"] == "test_video"
        assert base_data["video_path"] == "/test/video.mp4"
        assert base_data["duration"] == 120.5
        assert "other_data" not in base_data

    def test_add_strategy_metadata(self, mock_pipeline):
        """Test _add_strategy_metadata method."""
        video_data = {"video_id": "test_video"}

        result = mock_pipeline._add_strategy_metadata(video_data)

        assert result["video_id"] == "test_video"
        assert result["processing_type"] == "frame"
        assert result["storage_mode"] == "vector"
        assert result["schema_name"] == "test_schema"

    def test_prepare_base_results(self, mock_pipeline):
        """Test _prepare_base_results method."""
        video_path = Path("/test/video.mp4")

        with patch.object(mock_pipeline, "_get_video_duration", return_value=120.5):
            results = mock_pipeline._prepare_base_results(video_path)

            assert results["video_id"] == "video"
            assert results["video_path"] == str(video_path)
            assert results["duration"] == 120.5

    def test_get_chunk_duration_with_strategy(self, mock_pipeline):
        """Test _get_chunk_duration method with strategy."""
        # Mock strategy set with chunk strategy
        mock_chunk_strategy = Mock()
        mock_chunk_strategy.chunk_duration = 30.0
        mock_pipeline.strategy_set.segmentation = mock_chunk_strategy

        duration = mock_pipeline._get_chunk_duration()

        assert duration == 30.0

    def test_get_chunk_duration_no_strategy(self, mock_pipeline):
        """Test _get_chunk_duration method with no strategy."""
        # No strategy set
        mock_pipeline.strategy_set = None

        duration = mock_pipeline._get_chunk_duration()

        assert duration == 30.0  # Default value

    def test_convert_embedding_result_dict(self, mock_pipeline):
        """Test _convert_embedding_result method with proper result object."""
        # Create a mock result object with all required attributes
        mock_result = Mock()
        mock_result.video_id = "test_video"
        mock_result.total_documents = 10
        mock_result.documents_processed = 8
        mock_result.documents_fed = 8
        mock_result.processing_time = 1.5
        mock_result.errors = []
        mock_result.metadata = {"test": "data"}

        converted = mock_pipeline._convert_embedding_result(mock_result)

        assert converted["video_id"] == "test_video"
        assert converted["total_documents"] == 10
        assert converted["documents_processed"] == 8
        assert converted["backend"] == "vespa"

    def test_convert_embedding_result_string(self, mock_pipeline):
        """Test _convert_embedding_result method with minimal result object."""
        mock_result = Mock()
        mock_result.video_id = "test"
        mock_result.total_documents = 0
        mock_result.documents_processed = 0
        mock_result.documents_fed = 0
        mock_result.processing_time = 0.0
        mock_result.errors = ["error message"]
        mock_result.metadata = {}

        converted = mock_pipeline._convert_embedding_result(mock_result)
        assert converted["errors"] == ["error message"]

    def test_convert_embedding_result_other_type(self, mock_pipeline):
        """Test _convert_embedding_result method with successful result."""
        mock_result = Mock()
        mock_result.video_id = "success_video"
        mock_result.total_documents = 5
        mock_result.documents_processed = 5
        mock_result.documents_fed = 5
        mock_result.processing_time = 2.0
        mock_result.errors = []
        mock_result.metadata = {"status": "success"}

        converted = mock_pipeline._convert_embedding_result(mock_result)
        assert converted["video_id"] == "success_video"
        assert converted["metadata"]["status"] == "success"

    def test_process_frame_data(self, mock_pipeline):
        """Test _process_frame_data method."""
        video_data = {"video_id": "test_video"}
        # Test input data with correct structure
        results = {
            "results": {
                "keyframes": {
                    "keyframes": [
                        {"frame_id": "frame_1", "timestamp": 1.0},
                        {"frame_id": "frame_2", "timestamp": 2.0},
                    ]
                },
                "descriptions": {
                    "descriptions": {
                        "frame_1": "Description 1",
                        "frame_2": "Description 2",
                    }
                },
            }
        }

        result_data = mock_pipeline._process_frame_data(video_data, results)

        # The method returns video_data with keyframes and descriptions added
        assert result_data["video_id"] == "test_video"
        assert "keyframes" in result_data
        assert "descriptions" in result_data

    def test_process_chunk_data(self, mock_pipeline):
        """Test _process_chunk_data method."""
        video_data = {"video_id": "test_video"}
        results = {
            "results": {
                "chunks": {
                    "chunks": [
                        {"chunk_id": "chunk_1", "start_time": 0.0, "end_time": 30.0},
                        {"chunk_id": "chunk_2", "start_time": 30.0, "end_time": 60.0},
                    ]
                },
                "audio": {"transcript": "Full transcript"},
            }
        }

        result_data = mock_pipeline._process_chunk_data(video_data, results)

        # The method returns video_data with chunks added
        assert result_data["video_id"] == "test_video"
        assert "chunks" in result_data

    def test_process_single_vector_data(self, mock_pipeline):
        """Test _process_single_vector_data method."""
        video_data = {"video_id": "test_video"}
        results = {
            "results": {
                "single_vector_processing": {
                    "segments": [
                        {"start_time": 0.0, "end_time": 60.0, "text": "Test segment"}
                    ],
                    "metadata": {"test": "data"},
                    "full_transcript": "Test transcript",
                    "document_structure": {"sections": ["intro"]},
                }
            }
        }

        result_data = mock_pipeline._process_single_vector_data(video_data, results)

        # The method returns video_data with single vector processing data added
        assert result_data["video_id"] == "test_video"
        assert "segments" in result_data
        assert "processing_metadata" in result_data
        assert "full_transcript" in result_data
        assert "document_structure" in result_data

    @patch("asyncio.run")
    def test_process_video_sync_wrapper(self, mock_asyncio_run, mock_pipeline):
        """Test process_video method (sync wrapper)."""
        mock_result = {"video_id": "test", "status": "completed"}
        mock_asyncio_run.return_value = mock_result

        video_path = Path("/test/video.mp4")
        result = mock_pipeline.process_video(video_path)

        assert result == mock_result
        mock_asyncio_run.assert_called_once()

    def test_prepare_video_data_frame_strategy(self, mock_pipeline):
        """Test _prepare_video_data with frame strategy."""
        # Structure data as expected by the actual implementation
        results = {
            "video_id": "test_video",
            "video_path": "/test/path.mp4",
            "duration": 120.0,
            "results": {
                "keyframes": {"keyframes": [{"frame_id": "frame_1"}]},
                "descriptions": {"descriptions": {"frame_1": "Test description"}},
            },
        }

        # Set processing_type to something other than video_chunks to trigger frame processing
        mock_pipeline.strategy.processing_type = "frames"

        video_data = mock_pipeline._prepare_video_data(results)

        assert video_data["video_id"] == "test_video"
        assert "keyframes" in video_data  # _process_frame_data adds keyframes directly
        assert (
            "descriptions" in video_data
        )  # _process_frame_data adds descriptions directly

    def test_prepare_video_data_chunk_strategy(self, mock_pipeline):
        """Test _prepare_video_data with chunk strategy."""
        results = {
            "video_id": "test_video",
            "video_path": "/test/path.mp4",
            "duration": 120.0,
            "results": {
                "chunks": {"chunks": [{"chunk_id": "chunk_1"}]},
                "audio": {"transcript": "Test transcript"},
            },
        }

        video_data = mock_pipeline._prepare_video_data(results)

        assert video_data["video_id"] == "test_video"
        assert "chunks" in video_data  # _process_chunk_data adds chunks directly

    def test_prepare_video_data_single_vector_strategy(self, mock_pipeline):
        """Test _prepare_video_data with single_vector strategy."""
        results = {
            "video_id": "test_video",
            "video_path": "/test/path.mp4",
            "duration": 120.0,
            "results": {
                "single_vector_processing": {
                    "segments": [
                        {"start_time": 0.0, "end_time": 60.0, "text": "Test segment"}
                    ],
                    "metadata": {"test": "data"},
                    "full_transcript": "Test transcript",
                    "document_structure": {"sections": ["intro"]},
                }
            },
        }

        video_data = mock_pipeline._prepare_video_data(results)

        assert video_data["video_id"] == "test_video"
        assert (
            "segments" in video_data
        )  # _process_single_vector_data adds segments directly


@pytest.mark.unit
class TestPipelineStep:
    """Test PipelineStep enum."""

    def test_pipeline_step_values(self):
        """Test PipelineStep enum values."""
        assert PipelineStep.EXTRACT_KEYFRAMES.value == "extract_keyframes"
        assert PipelineStep.EXTRACT_CHUNKS.value == "extract_chunks"
        assert PipelineStep.TRANSCRIBE_AUDIO.value == "transcribe_audio"
        assert PipelineStep.GENERATE_DESCRIPTIONS.value == "generate_descriptions"
        assert PipelineStep.GENERATE_EMBEDDINGS.value == "generate_embeddings"
