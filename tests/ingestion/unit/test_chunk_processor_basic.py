"""
Basic unit tests for ChunkProcessor to improve coverage.

Tests basic initialization and configuration without external dependencies.
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from src.app.ingestion.processors.chunk_processor import ChunkProcessor


class TestChunkProcessorBasic:
    """Basic tests for ChunkProcessor."""

    @pytest.fixture
    def mock_logger(self):
        return Mock()

    def test_processor_initialization_defaults(self, mock_logger):
        """Test processor initialization with default values."""
        processor = ChunkProcessor(mock_logger)

        assert processor.PROCESSOR_NAME == "chunk"
        assert processor.logger == mock_logger
        assert processor.chunk_duration == 30.0
        assert processor.chunk_overlap == 0.0
        assert processor.cache_chunks is True

    def test_processor_initialization_custom(self, mock_logger):
        """Test processor initialization with custom values."""
        processor = ChunkProcessor(
            mock_logger, chunk_duration=15.0, chunk_overlap=2.0, cache_chunks=False
        )

        assert processor.chunk_duration == 15.0
        assert processor.chunk_overlap == 2.0
        assert processor.cache_chunks is False

    def test_from_config_factory_method(self, mock_logger):
        """Test creating processor from configuration."""
        config = {"chunk_duration": 45.0, "chunk_overlap": 5.0, "cache_chunks": False}

        processor = ChunkProcessor.from_config(config, mock_logger)

        assert processor.chunk_duration == 45.0
        assert processor.chunk_overlap == 5.0
        assert processor.cache_chunks is False

    def test_from_config_with_defaults(self, mock_logger):
        """Test from_config uses defaults for missing parameters."""
        config = {"chunk_duration": 20.0}

        processor = ChunkProcessor.from_config(config, mock_logger)

        assert processor.chunk_duration == 20.0
        assert processor.chunk_overlap == 0.0  # default
        assert processor.cache_chunks is True  # default

    @patch("subprocess.run")
    def test_get_video_duration_success(self, mock_run, mock_logger):
        """Test successful video duration retrieval."""
        mock_run.return_value.stdout = "120.5\n"
        processor = ChunkProcessor(mock_logger)

        duration = processor._get_video_duration(Path("/test/video.mp4"))

        assert duration == 120.5
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_get_video_duration_error(self, mock_run, mock_logger):
        """Test video duration retrieval error handling."""
        mock_run.side_effect = Exception("ffprobe error")
        processor = ChunkProcessor(mock_logger)

        duration = processor._get_video_duration(Path("/test/video.mp4"))

        assert duration == 0.0
        mock_logger.error.assert_called()

    def test_cleanup_method(self, mock_logger):
        """Test cleanup method."""
        processor = ChunkProcessor(mock_logger)

        # Should not raise any exceptions
        processor.cleanup()

    def test_processor_name_constant(self, mock_logger):
        """Test processor name constant."""
        processor = ChunkProcessor(mock_logger)
        assert processor.PROCESSOR_NAME == "chunk"
        assert processor.get_processor_name() == "chunk"
