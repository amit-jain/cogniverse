#!/usr/bin/env python3
"""
Unit tests for VideoChunkExtractor.
"""

import json
import subprocess
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pytest
from cogniverse_runtime.ingestion.processors.video_chunk_extractor import (
    VideoChunkExtractor,
)


@pytest.mark.unit
class TestVideoChunkExtractor:
    """Test suite for VideoChunkExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a basic VideoChunkExtractor instance."""
        return VideoChunkExtractor(
            chunk_duration=30.0, chunk_overlap=0.0, max_chunks=None, cache_chunks=True
        )

    @pytest.fixture
    def custom_extractor(self):
        """Create a custom VideoChunkExtractor instance."""
        return VideoChunkExtractor(
            chunk_duration=15.0, chunk_overlap=2.0, max_chunks=5, cache_chunks=False
        )

    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        return Mock()

    @pytest.fixture
    def video_path(self, tmp_path):
        """Create a mock video path."""
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()
        return video_file

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create a mock output directory."""
        return tmp_path / "output"

    def test_initialization_defaults(self):
        """Test VideoChunkExtractor initialization with defaults."""
        extractor = VideoChunkExtractor()

        assert extractor.chunk_duration == 30.0
        assert extractor.chunk_overlap == 0.0
        assert extractor.max_chunks is None
        assert extractor.cache_chunks is True
        assert hasattr(extractor, "logger")

    def test_initialization_custom_values(self):
        """Test VideoChunkExtractor initialization with custom values."""
        extractor = VideoChunkExtractor(
            chunk_duration=45.0, chunk_overlap=5.0, max_chunks=10, cache_chunks=False
        )

        assert extractor.chunk_duration == 45.0
        assert extractor.chunk_overlap == 5.0
        assert extractor.max_chunks == 10
        assert extractor.cache_chunks is False

    @patch("subprocess.run")
    def test_get_video_duration_success(self, mock_run, extractor, video_path):
        """Test successful video duration extraction."""
        # Mock ffprobe output
        mock_result = Mock()
        mock_result.stdout = "120.500\n"
        mock_run.return_value = mock_result

        duration = extractor._get_video_duration(video_path)

        assert duration == 120.5
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "ffprobe" in args
        assert str(video_path) in args

    @patch("subprocess.run")
    def test_get_video_duration_error(
        self, mock_run, extractor, video_path, mock_logger
    ):
        """Test video duration extraction error handling."""
        extractor.logger = mock_logger
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffprobe")

        duration = extractor._get_video_duration(video_path)

        assert duration == 0.0
        mock_logger.error.assert_called_once()

    @patch("subprocess.run")
    def test_extract_chunk_ffmpeg_success(
        self, mock_run, extractor, video_path, tmp_path
    ):
        """Test successful chunk extraction with ffmpeg."""
        output_path = tmp_path / "chunk.mp4"
        mock_run.return_value = Mock()  # Successful call

        result = extractor._extract_chunk_ffmpeg(video_path, output_path, 10.0, 30.0)

        assert result is True
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "ffmpeg" in args
        assert str(video_path) in args
        assert str(output_path) in args
        assert "10.0" in args
        assert "30.0" in args

    @patch("subprocess.run")
    def test_extract_chunk_ffmpeg_error(
        self, mock_run, extractor, video_path, tmp_path, mock_logger
    ):
        """Test ffmpeg chunk extraction error handling."""
        extractor.logger = mock_logger
        output_path = tmp_path / "chunk.mp4"
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "ffmpeg", stderr=b"Error message"
        )

        result = extractor._extract_chunk_ffmpeg(video_path, output_path, 10.0, 30.0)

        assert result is False
        mock_logger.error.assert_called_once()

    @patch("cv2.VideoCapture")
    @patch("cv2.imwrite")
    def test_extract_preview_frame_success(
        self, mock_imwrite, mock_cap_class, extractor, video_path, tmp_path
    ):
        """Test successful preview frame extraction."""
        output_path = tmp_path / "preview.jpg"

        # Mock cv2.VideoCapture
        mock_cap = Mock()
        mock_cap_class.return_value = mock_cap
        mock_cap.get.return_value = 100  # 100 total frames
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_imwrite.return_value = True

        result = extractor._extract_preview_frame(video_path, output_path)

        assert result is True
        mock_cap.set.assert_called_with(1, 50)  # cv2.CAP_PROP_POS_FRAMES = 1
        mock_cap.read.assert_called_once()
        mock_imwrite.assert_called_once()
        mock_cap.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_extract_preview_frame_no_frames(
        self, mock_cap_class, extractor, video_path, tmp_path
    ):
        """Test preview frame extraction with no frames."""
        output_path = tmp_path / "preview.jpg"

        mock_cap = Mock()
        mock_cap_class.return_value = mock_cap
        mock_cap.get.return_value = 0  # 0 total frames

        result = extractor._extract_preview_frame(video_path, output_path)

        assert result is False
        mock_cap.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_extract_preview_frame_error(
        self, mock_cap_class, extractor, video_path, tmp_path, mock_logger
    ):
        """Test preview frame extraction error handling."""
        extractor.logger = mock_logger
        output_path = tmp_path / "preview.jpg"
        mock_cap_class.side_effect = Exception("CV2 error")

        result = extractor._extract_preview_frame(video_path, output_path)

        assert result is False
        mock_logger.error.assert_called_once()

    def test_load_cached_chunks_not_exist(self, extractor, video_path, output_dir):
        """Test loading cached chunks when metadata doesn't exist."""
        result = extractor.load_cached_chunks(video_path, output_dir)

        assert result is None

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    @patch("pathlib.Path.exists")
    def test_load_cached_chunks_success(
        self,
        mock_exists,
        mock_json_load,
        mock_file,
        extractor,
        video_path,
        output_dir,
        mock_logger,
    ):
        """Test successful loading of cached chunks."""
        extractor.logger = mock_logger

        # Mock metadata file exists
        mock_exists.return_value = True  # All paths exist

        # Mock metadata content
        metadata = {
            "video_id": "test_video",
            "chunks": [
                {"chunk_path": "/path/to/chunk1.mp4"},
                {"chunk_path": "/path/to/chunk2.mp4"},
            ],
        }
        mock_json_load.return_value = metadata

        result = extractor.load_cached_chunks(video_path, output_dir)

        assert result == metadata
        mock_logger.info.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    @patch("pathlib.Path.exists")
    def test_load_cached_chunks_missing_files(
        self,
        mock_exists,
        mock_json_load,
        mock_file,
        extractor,
        video_path,
        output_dir,
        mock_logger,
    ):
        """Test loading cached chunks when some chunk files are missing."""
        extractor.logger = mock_logger

        # Mock metadata file exists but chunk files don't
        # First call (metadata file) returns True, subsequent calls (chunk files) return False
        mock_exists.side_effect = [True, False, False]

        metadata = {
            "chunks": [
                {"chunk_path": "/path/to/chunk1.mp4"},
                {"chunk_path": "/path/to/chunk2.mp4"},
            ]
        }
        mock_json_load.return_value = metadata

        result = extractor.load_cached_chunks(video_path, output_dir)

        assert result is None
        mock_logger.warning.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    @patch("json.load")
    @patch("pathlib.Path.exists")
    def test_load_cached_chunks_json_error(
        self,
        mock_exists,
        mock_json_load,
        mock_file,
        extractor,
        video_path,
        output_dir,
        mock_logger,
    ):
        """Test loading cached chunks with JSON error."""
        extractor.logger = mock_logger
        mock_exists.return_value = True
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        result = extractor.load_cached_chunks(video_path, output_dir)

        assert result is None
        mock_logger.error.assert_called_once()

    @patch.object(VideoChunkExtractor, "_get_video_duration")
    def test_extract_chunks_invalid_duration(
        self, mock_duration, extractor, video_path, output_dir, mock_logger
    ):
        """Test extract_chunks with invalid video duration."""
        extractor.logger = mock_logger
        mock_duration.return_value = 0.0

        result = extractor.extract_chunks(video_path, output_dir)

        assert result["chunks"] == []
        assert result["error"] == "Invalid duration"
        mock_logger.error.assert_called_once()

    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("time.time")
    @patch.object(VideoChunkExtractor, "_extract_preview_frame")
    @patch.object(VideoChunkExtractor, "_extract_chunk_ffmpeg")
    @patch.object(VideoChunkExtractor, "_get_video_duration")
    def test_extract_chunks_no_overlap_success(
        self,
        mock_duration,
        mock_ffmpeg,
        mock_preview,
        mock_time,
        mock_json_dump,
        mock_file,
        mock_mkdir,
        extractor,
        video_path,
        output_dir,
        mock_logger,
    ):
        """Test successful chunk extraction without overlap."""
        extractor.logger = mock_logger
        mock_duration.return_value = 75.0  # 75 second video
        mock_ffmpeg.return_value = True
        mock_preview.return_value = True
        mock_time.return_value = 1234567890.0

        result = extractor.extract_chunks(video_path, output_dir)

        # Should create 3 chunks (0-30, 30-60, 60-75)
        assert len(result["chunks"]) == 3
        assert result["total_duration"] == 75.0
        assert result["num_chunks"] == 3
        assert mock_ffmpeg.call_count == 3

    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("time.time")
    @patch.object(VideoChunkExtractor, "_extract_preview_frame")
    @patch.object(VideoChunkExtractor, "_extract_chunk_ffmpeg")
    @patch.object(VideoChunkExtractor, "_get_video_duration")
    def test_extract_chunks_with_overlap(
        self,
        mock_duration,
        mock_ffmpeg,
        mock_preview,
        mock_time,
        mock_json_dump,
        mock_file,
        mock_mkdir,
        custom_extractor,
        video_path,
        output_dir,
        mock_logger,
    ):
        """Test chunk extraction with overlap."""
        custom_extractor.logger = mock_logger
        mock_duration.return_value = 50.0  # 50 second video
        mock_ffmpeg.return_value = True
        mock_preview.return_value = True
        mock_time.return_value = 1234567890.0

        # chunk_duration=15.0, chunk_overlap=2.0, so stride=13.0
        # Chunks at: 0, 13, 26, 39 (last one truncated)
        result = custom_extractor.extract_chunks(video_path, output_dir)

        assert len(result["chunks"]) == 4
        assert result["chunks"][0]["start_time"] == 0.0
        assert result["chunks"][1]["start_time"] == 13.0
        assert result["chunks"][2]["start_time"] == 26.0
        assert result["chunks"][3]["start_time"] == 39.0

    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("time.time")
    @patch.object(VideoChunkExtractor, "_extract_preview_frame")
    @patch.object(VideoChunkExtractor, "_extract_chunk_ffmpeg")
    @patch.object(VideoChunkExtractor, "_get_video_duration")
    def test_extract_chunks_max_chunks_limit(
        self,
        mock_duration,
        mock_ffmpeg,
        mock_preview,
        mock_time,
        mock_json_dump,
        mock_file,
        mock_mkdir,
        custom_extractor,
        video_path,
        output_dir,
        mock_logger,
    ):
        """Test chunk extraction with max_chunks limit."""
        custom_extractor.logger = mock_logger
        custom_extractor.max_chunks = 2  # Limit to 2 chunks
        mock_duration.return_value = 100.0  # Long video
        mock_ffmpeg.return_value = True
        mock_preview.return_value = True
        mock_time.return_value = 1234567890.0

        result = custom_extractor.extract_chunks(video_path, output_dir)

        # Should only create 2 chunks despite long video
        assert len(result["chunks"]) == 2
        assert mock_ffmpeg.call_count == 2

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("time.time")
    @patch.object(VideoChunkExtractor, "_extract_preview_frame")
    @patch.object(VideoChunkExtractor, "_extract_chunk_ffmpeg")
    @patch.object(VideoChunkExtractor, "_get_video_duration")
    def test_extract_chunks_with_caching(
        self,
        mock_duration,
        mock_ffmpeg,
        mock_preview,
        mock_time,
        mock_json_dump,
        mock_file,
        mock_mkdir,
        mock_exists,
        extractor,
        video_path,
        output_dir,
        mock_logger,
    ):
        """Test chunk extraction with caching enabled."""
        extractor.logger = mock_logger
        mock_duration.return_value = 45.0
        mock_ffmpeg.return_value = True
        mock_preview.return_value = True
        mock_time.return_value = 1234567890.0

        # Mock that first chunk exists (cached), second doesn't
        # Just return True for most cases to make test simpler
        mock_exists.return_value = True

        result = extractor.extract_chunks(video_path, output_dir)

        # All chunks are cached in this simple mock
        assert len(result["chunks"]) == 2
        assert mock_ffmpeg.call_count == 0  # No chunks extracted (all cached)

    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("time.time")
    @patch.object(VideoChunkExtractor, "_extract_preview_frame")
    @patch.object(VideoChunkExtractor, "_extract_chunk_ffmpeg")
    @patch.object(VideoChunkExtractor, "_get_video_duration")
    def test_extract_chunks_ffmpeg_failure(
        self,
        mock_duration,
        mock_ffmpeg,
        mock_preview,
        mock_time,
        mock_json_dump,
        mock_file,
        mock_mkdir,
        extractor,
        video_path,
        output_dir,
        mock_logger,
    ):
        """Test chunk extraction when ffmpeg fails."""
        extractor.logger = mock_logger
        mock_duration.return_value = 45.0
        mock_ffmpeg.return_value = False  # Ffmpeg fails
        mock_preview.return_value = True
        mock_time.return_value = 1234567890.0

        result = extractor.extract_chunks(video_path, output_dir)

        # Should continue processing despite failures
        assert len(result["chunks"]) == 0  # No successful chunks
        mock_logger.error.assert_called()  # Should log errors

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("time.time")
    @patch.object(VideoChunkExtractor, "_extract_preview_frame")
    @patch.object(VideoChunkExtractor, "_extract_chunk_ffmpeg")
    @patch.object(VideoChunkExtractor, "_get_video_duration")
    def test_extract_chunks_metadata_structure(
        self,
        mock_duration,
        mock_ffmpeg,
        mock_preview,
        mock_time,
        mock_json_dump,
        mock_file,
        mock_mkdir,
        mock_exists,
        extractor,
        video_path,
        output_dir,
        mock_logger,
    ):
        """Test that chunk metadata has correct structure."""
        extractor.logger = mock_logger
        mock_duration.return_value = 35.0
        mock_ffmpeg.return_value = True
        mock_preview.return_value = True
        mock_time.return_value = 1234567890.0
        mock_exists.return_value = True  # Preview exists

        result = extractor.extract_chunks(video_path, output_dir)

        # Check top-level metadata structure
        assert "video_id" in result
        assert "video_path" in result
        assert "total_duration" in result
        assert "chunk_duration" in result
        assert "chunk_overlap" in result
        assert "num_chunks" in result
        assert "chunks" in result
        assert "extracted_at" in result

        # Check chunk metadata structure
        chunk = result["chunks"][0]
        assert "chunk_id" in chunk
        assert "chunk_path" in chunk
        assert "start_time" in chunk
        assert "end_time" in chunk
        assert "duration" in chunk
        assert "filename" in chunk
        assert "preview_path" in chunk  # Should exist since mock_exists returns True
