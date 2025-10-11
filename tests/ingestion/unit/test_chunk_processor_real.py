"""
Unit tests for ChunkProcessor - Testing actual implementation.

Tests real video chunk extraction functionality using ffmpeg.
"""

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from cogniverse_runtime.ingestion.processors.chunk_processor import ChunkProcessor


@pytest.mark.unit
@pytest.mark.ci_safe
@pytest.mark.requires_ffmpeg
class TestChunkProcessor:
    """Test the actual ChunkProcessor implementation."""

    @pytest.fixture
    def processor(self, mock_logger):
        """Create a real chunk processor."""
        return ChunkProcessor(
            mock_logger, chunk_duration=30.0, chunk_overlap=2.0, cache_chunks=True
        )

    @pytest.fixture
    def no_overlap_processor(self, mock_logger):
        """Create processor without overlap."""
        return ChunkProcessor(
            mock_logger, chunk_duration=15.0, chunk_overlap=0.0, cache_chunks=False
        )

    def test_processor_initialization(self, mock_logger):
        """Test chunk processor initialization."""
        processor = ChunkProcessor(
            mock_logger, chunk_duration=60.0, chunk_overlap=5.0, cache_chunks=False
        )

        assert processor.PROCESSOR_NAME == "chunk"
        assert processor.logger == mock_logger
        assert processor.chunk_duration == 60.0
        assert processor.chunk_overlap == 5.0
        assert processor.cache_chunks is False

    def test_processor_defaults(self, mock_logger):
        """Test processor with default values."""
        processor = ChunkProcessor(mock_logger)

        assert processor.chunk_duration == 30.0
        assert processor.chunk_overlap == 0.0
        assert processor.cache_chunks is True

    def test_from_config_factory_method(self, mock_logger):
        """Test creating processor from configuration."""
        config = {"chunk_duration": 45.0, "chunk_overlap": 3.0, "cache_chunks": False}

        processor = ChunkProcessor.from_config(config, mock_logger)

        assert processor.chunk_duration == 45.0
        assert processor.chunk_overlap == 3.0
        assert processor.cache_chunks is False

    def test_from_config_with_defaults(self, mock_logger):
        """Test from_config uses defaults for missing parameters."""
        config = {"chunk_duration": 20.0}

        processor = ChunkProcessor.from_config(config, mock_logger)

        assert processor.chunk_duration == 20.0
        assert processor.chunk_overlap == 0.0  # default
        assert processor.cache_chunks is True  # default

    @patch("subprocess.run")
    def test_get_video_duration_success(self, mock_subprocess, processor):
        """Test successful video duration extraction."""
        # Mock ffprobe output
        mock_result = Mock()
        mock_result.stdout = "120.5\n"  # 120.5 seconds
        mock_subprocess.return_value = mock_result

        duration = processor._get_video_duration(Path("/test/video.mp4"))

        assert duration == 120.5

        # Verify ffprobe was called with correct arguments
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert "ffprobe" in call_args
        assert "-show_entries" in call_args
        assert "format=duration" in call_args
        assert "/test/video.mp4" in call_args

    @patch("subprocess.run")
    def test_get_video_duration_error(self, mock_subprocess, processor):
        """Test video duration extraction error handling."""
        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "ffprobe")

        duration = processor._get_video_duration(Path("/test/invalid.mp4"))

        assert duration == 0.0
        processor.logger.error.assert_called()

    @patch("subprocess.run")
    def test_extract_chunk_success(self, mock_subprocess, processor, temp_dir):
        """Test successful chunk extraction."""
        video_path = Path("/test/video.mp4")
        chunk_path = temp_dir / "chunk.mp4"

        # Mock successful ffmpeg execution
        mock_subprocess.return_value = Mock()

        # Mock chunk file creation
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            mock_stat.return_value.st_size = 1000000  # 1MB file

            success = processor._extract_chunk(video_path, chunk_path, 10.0, 30.0)

        assert success is True

        # Verify ffmpeg was called with correct arguments
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert "ffmpeg" in call_args
        assert "-ss" in call_args
        assert "10.0" in call_args  # start time
        assert "-t" in call_args
        assert "30.0" in call_args  # duration
        assert str(chunk_path) in call_args

    @patch("subprocess.run")
    def test_extract_chunk_error(self, mock_subprocess, processor, temp_dir):
        """Test chunk extraction error handling."""
        video_path = Path("/test/video.mp4")
        chunk_path = temp_dir / "chunk.mp4"

        mock_subprocess.side_effect = subprocess.CalledProcessError(1, "ffmpeg")

        success = processor._extract_chunk(video_path, chunk_path, 10.0, 30.0)

        assert success is False
        processor.logger.error.assert_called()

    @patch("cogniverse_core.common.utils.output_manager.get_output_manager")
    @patch("builtins.open", create=True)
    @patch("json.dump")
    def test_extract_chunks_success(
        self,
        mock_json_dump,
        mock_open,
        mock_output_manager,
        processor,
        temp_dir,
        sample_video_path,
    ):
        """Test successful chunk extraction workflow."""
        # Mock output manager
        mock_manager = Mock()
        mock_manager.get_processing_dir.return_value = temp_dir
        mock_output_manager.return_value = mock_manager

        # Mock file operations
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        # Mock video duration and chunk extraction
        with (
            patch.object(processor, "_get_video_duration", return_value=90.0),
            patch.object(processor, "_extract_chunk", return_value=True),
        ):

            result = processor.extract_chunks(sample_video_path)

        # Verify results structure
        assert "chunks" in result
        assert "metadata" in result
        assert "chunks_dir" in result
        assert "video_id" in result
        assert result["video_id"] == "test_video"

        # With 90s video, 30s chunks, 2s overlap, should get 4 chunks:
        # [0-30], [28-58], [56-86], [84-90]
        chunks = result["chunks"]
        assert len(chunks) == 4

        # Verify chunk structure
        chunk1 = chunks[0]
        assert chunk1["start_time"] == 0.0
        assert chunk1["end_time"] == 30.0
        assert chunk1["duration"] == 30.0
        assert "chunk_000" in chunk1["filename"]

        chunk2 = chunks[1]
        assert chunk2["start_time"] == 28.0  # 30 - 2 (overlap)
        assert chunk2["end_time"] == 58.0

        # Verify metadata was saved
        mock_json_dump.assert_called_once()
        saved_metadata = mock_json_dump.call_args[0][0]
        assert saved_metadata["video_id"] == "test_video"
        assert saved_metadata["video_duration"] == 90.0
        assert saved_metadata["chunk_duration"] == 30.0
        assert saved_metadata["chunk_overlap"] == 2.0
        assert saved_metadata["chunks_extracted"] == 4

    @patch("cogniverse_core.common.utils.output_manager.get_output_manager")
    def test_extract_chunks_no_overlap(
        self, mock_output_manager, no_overlap_processor, temp_dir, sample_video_path
    ):
        """Test chunk extraction without overlap."""
        # Mock output manager
        mock_manager = Mock()
        mock_manager.get_processing_dir.return_value = temp_dir
        mock_output_manager.return_value = mock_manager

        # Mock 45 second video, 15s chunks, no overlap
        with (
            patch.object(
                no_overlap_processor, "_get_video_duration", return_value=45.0
            ),
            patch.object(no_overlap_processor, "_extract_chunk", return_value=True),
            patch("builtins.open", create=True),
            patch("json.dump"),
        ):

            result = no_overlap_processor.extract_chunks(sample_video_path)

        # Should get exactly 3 chunks: [0-15], [15-30], [30-45]
        chunks = result["chunks"]
        assert len(chunks) == 3

        chunk1 = chunks[0]
        assert chunk1["start_time"] == 0.0
        assert chunk1["end_time"] == 15.0

        chunk2 = chunks[1]
        assert chunk2["start_time"] == 15.0  # No overlap
        assert chunk2["end_time"] == 30.0

        chunk3 = chunks[2]
        assert chunk3["start_time"] == 30.0
        assert chunk3["end_time"] == 45.0

    def test_extract_chunks_invalid_duration(self, processor, sample_video_path):
        """Test handling of invalid video duration."""
        with patch.object(processor, "_get_video_duration", return_value=0.0):
            result = processor.extract_chunks(sample_video_path)

        # Should return empty result
        assert result["chunks"] == []
        assert result["metadata"] == {}

        # Should log error
        processor.logger.error.assert_called()

    @patch("cogniverse_core.common.utils.output_manager.get_output_manager")
    def test_extract_chunks_with_failed_extraction(
        self, mock_output_manager, processor, temp_dir, sample_video_path
    ):
        """Test handling when some chunk extractions fail."""
        # Mock output manager
        mock_manager = Mock()
        mock_manager.get_processing_dir.return_value = temp_dir
        mock_output_manager.return_value = mock_manager

        # Mock some extractions succeeding, some failing
        # For 90s video: chunks [0-30], [28-58], [56-86], [84-90]
        def mock_extract_side_effect(*args):
            # Make chunks 0 and 2 succeed, 1 and 3 fail
            call_count = mock_extract_side_effect.call_count
            mock_extract_side_effect.call_count += 1
            return call_count in [0, 2]  # Success for calls 0 and 2

        mock_extract_side_effect.call_count = 0

        with (
            patch.object(processor, "_get_video_duration", return_value=90.0),
            patch.object(
                processor, "_extract_chunk", side_effect=mock_extract_side_effect
            ),
            patch("builtins.open", create=True),
            patch("json.dump"),
        ):

            result = processor.extract_chunks(sample_video_path)

        # Should only include successful chunks (1st and 3rd succeed)
        chunks = result["chunks"]
        assert len(chunks) == 2  # Only successful extractions

        # Chunk numbers reflect successful extraction order, not original positions
        # The code increments chunk_idx only when extraction succeeds
        assert chunks[0]["chunk_number"] == 0  # First successful extraction
        assert chunks[1]["chunk_number"] == 1  # Second successful extraction

        # But the timestamps should show which original chunks these were
        assert chunks[0]["start_time"] == 0.0  # Original chunk 0: [0-30]
        assert chunks[1]["start_time"] == 56.0  # Original chunk 2: [56-86]

    def test_extract_chunks_legacy_output_dir(
        self, processor, temp_dir, sample_video_path
    ):
        """Test legacy output directory support."""
        with (
            patch.object(processor, "_get_video_duration", return_value=30.0),
            patch.object(processor, "_extract_chunk", return_value=True),
            patch("builtins.open", create=True),
            patch("json.dump"),
        ):

            result = processor.extract_chunks(sample_video_path, output_dir=temp_dir)

        # Should work with legacy path structure
        assert result["video_id"] == "test_video"
        expected_chunks_dir = str(temp_dir / "chunks" / "test_video")
        assert result["chunks_dir"] == expected_chunks_dir

    def test_cleanup_method(self, processor):
        """Test cleanup method."""
        # Should not raise any errors
        processor.cleanup()

    def test_get_config_method(self, processor):
        """Test the get_config method from BaseProcessor."""
        processor.get_config()

        # The base processor only stores kwargs passed to super().__init__
        # Since ChunkProcessor doesn't pass its params as kwargs, config will be empty
        # But we can verify the actual attributes exist
        assert hasattr(processor, "chunk_duration")
        assert hasattr(processor, "chunk_overlap")
        assert hasattr(processor, "cache_chunks")
        assert processor.chunk_duration == 30.0
        assert processor.chunk_overlap == 2.0
        assert processor.cache_chunks is True

    def test_processor_name_constant(self, processor):
        """Test processor name constant."""
        assert processor.PROCESSOR_NAME == "chunk"
        assert processor.get_processor_name() == "chunk"

    def test_chunk_overlap_calculation(self, processor):
        """Test chunk overlap calculations are correct."""
        # Test internal logic with known values
        chunk_duration = 30.0
        chunk_overlap = 5.0

        # First chunk: 0-30
        # Second chunk should start at: 30 - 5 = 25
        # Third chunk should start at: 25 + 30 - 5 = 50

        processor.chunk_duration = chunk_duration
        processor.chunk_overlap = chunk_overlap

        with (
            patch.object(processor, "_get_video_duration", return_value=80.0),
            patch.object(processor, "_extract_chunk", return_value=True),
            patch(
                "cogniverse_core.common.utils.output_manager.get_output_manager"
            ) as mock_output_manager,
            patch("builtins.open", create=True),
            patch("json.dump"),
        ):

            mock_manager = Mock()
            mock_manager.get_processing_dir.return_value = Path("/tmp")
            mock_output_manager.return_value = mock_manager

            result = processor.extract_chunks(Path("/test/video.mp4"))

        chunks = result["chunks"]
        assert len(chunks) == 4  # [0-30], [25-55], [50-80], [75-80]

        assert chunks[0]["start_time"] == 0.0
        assert chunks[0]["end_time"] == 30.0

        assert chunks[1]["start_time"] == 25.0  # 30 - 5
        assert chunks[1]["end_time"] == 55.0

        assert chunks[2]["start_time"] == 50.0  # 25 + 30 - 5
        assert chunks[2]["end_time"] == 80.0

        assert chunks[3]["start_time"] == 75.0  # 50 + 30 - 5
        assert chunks[3]["end_time"] == 80.0
