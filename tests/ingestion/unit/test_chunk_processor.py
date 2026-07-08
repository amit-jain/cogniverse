#!/usr/bin/env python3
"""
Tests for ChunkProcessor
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from cogniverse_runtime.ingestion.processors.chunk_processor import ChunkProcessor


@pytest.fixture
def mock_logger():
    """Create mock logger."""
    return MagicMock()


@pytest.fixture
def processor(mock_logger):
    """Create processor instance with test settings."""
    return ChunkProcessor(
        logger=mock_logger,
        chunk_duration=30.0,
        chunk_overlap=5.0,
        cache_chunks=False,
    )


@pytest.fixture
def sample_video_path(tmp_path):
    """Create sample video path."""
    video = tmp_path / "test_video.mp4"
    video.touch()
    return video


class TestChunkProcessor:
    """Test ChunkProcessor functionality."""

    def test_processor_initialization(self, mock_logger):
        """Test processor initialization with custom values."""
        processor = ChunkProcessor(
            logger=mock_logger,
            chunk_duration=60.0,
            chunk_overlap=5.0,
            cache_chunks=False,
        )

        assert processor.chunk_duration == 60.0
        assert processor.chunk_overlap == 5.0
        assert processor.cache_chunks is False

    def test_processor_initialization_defaults(self, mock_logger):
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

    @patch("cogniverse_runtime.ingestion.processors.chunk_processor.subprocess.run")
    def test_get_video_duration_success(
        self, mock_subprocess, processor, sample_video_path
    ):
        """Test successful video duration extraction."""
        mock_subprocess.return_value = Mock(stdout="120.5\n", returncode=0)

        duration = processor._get_video_duration(sample_video_path)

        assert duration == 120.5
        mock_subprocess.assert_called_once()

    @patch("cogniverse_runtime.ingestion.processors.chunk_processor.subprocess.run")
    def test_get_video_duration_error(
        self, mock_subprocess, processor, sample_video_path
    ):
        """Test video duration extraction error handling."""
        mock_subprocess.side_effect = Exception("ffprobe error")

        duration = processor._get_video_duration(sample_video_path)

        assert duration == 0.0
        processor.logger.error.assert_called()

    @patch("cogniverse_runtime.ingestion.processors.chunk_processor.subprocess.run")
    def test_extract_chunk_success(
        self, mock_subprocess, processor, sample_video_path, tmp_path
    ):
        """Test successful chunk extraction."""
        chunk_path = tmp_path / "chunk.mp4"
        mock_subprocess.return_value = Mock(returncode=0)

        # Create the file that would be created by ffmpeg
        chunk_path.write_text("dummy content")

        result = processor._extract_chunk(sample_video_path, chunk_path, 10.0, 30.0)

        assert result is True
        mock_subprocess.assert_called_once()

    @patch("cogniverse_runtime.ingestion.processors.chunk_processor.subprocess.run")
    def test_extract_chunk_failure(
        self, mock_subprocess, processor, sample_video_path, tmp_path
    ):
        """Test chunk extraction failure."""
        chunk_path = tmp_path / "chunk.mp4"
        mock_subprocess.side_effect = Exception("ffmpeg error")

        result = processor._extract_chunk(sample_video_path, chunk_path, 10.0, 30.0)

        assert result is False
        processor.logger.error.assert_called()

    @patch.object(ChunkProcessor, "_extract_chunk")
    @patch.object(ChunkProcessor, "_get_video_duration")
    def test_extract_chunks_success(
        self, mock_duration, mock_extract, processor, sample_video_path, tmp_path
    ):
        """Test successful chunk extraction process."""
        mock_duration.return_value = 90.0
        mock_extract.return_value = True

        result = processor.extract_chunks(sample_video_path, output_dir=tmp_path)

        assert "chunks" in result
        assert "metadata" in result
        assert "chunks_dir" in result
        assert "video_id" in result
        assert (
            len(result["chunks"]) == 4
        )  # 90s with 30s chunks and 5s overlap: 0-30, 25-55, 50-80, 75-90
        assert result["metadata"]["video_duration"] == 90.0

    @patch.object(ChunkProcessor, "_get_video_duration")
    def test_extract_chunks_invalid_duration(
        self, mock_duration, processor, sample_video_path
    ):
        """Test chunk extraction with invalid video duration."""
        mock_duration.return_value = 0.0

        result = processor.extract_chunks(sample_video_path)

        assert result["chunks"] == []
        assert result["metadata"] == {}
        processor.logger.error.assert_called()

    def test_process_method(self, processor, sample_video_path):
        """Test process method delegates to extract_chunks."""
        with patch.object(processor, "extract_chunks") as mock_extract:
            mock_extract.return_value = {"chunks": []}

            result = processor.process(sample_video_path, output_dir="/test")

            assert result == {"chunks": []}
            mock_extract.assert_called_once_with(sample_video_path, "/test")

    def test_cleanup_method(self, processor):
        """Test cleanup method exists and is callable."""
        # Should not raise any exceptions
        processor.cleanup()


class TestChunkProcessorIntegration:
    """Integration tests for ChunkProcessor."""

    @patch("cogniverse_runtime.ingestion.processors.chunk_processor.subprocess.run")
    def test_full_extraction_workflow(
        self, mock_subprocess, mock_logger, sample_video_path, tmp_path
    ):
        """Test complete chunk extraction workflow."""
        # Mock ffprobe for duration
        mock_subprocess.return_value = Mock(stdout="120.0\n", returncode=0)

        processor = ChunkProcessor(
            logger=mock_logger,
            chunk_duration=30.0,
            chunk_overlap=10.0,
            cache_chunks=True,
        )

        # First call is for duration, subsequent are for extraction
        with patch.object(processor, "_extract_chunk", return_value=True):
            result = processor.extract_chunks(sample_video_path, output_dir=tmp_path)

        # Verify result structure
        assert "chunks" in result
        assert "metadata" in result
        assert result["metadata"]["video_duration"] == 120.0
        assert result["metadata"]["chunk_duration"] == 30.0
        assert result["metadata"]["chunk_overlap"] == 10.0

        # Check chunks were created correctly
        # With 30s chunks and 10s overlap: 0-30, 20-50, 40-70, 60-90, 80-110, 100-120
        expected_chunks = 6
        assert len(result["chunks"]) == expected_chunks

        # Verify metadata file was saved
        metadata_file = tmp_path / "metadata" / f"{sample_video_path.stem}_chunks.json"
        assert metadata_file.exists()

        with open(metadata_file) as f:
            saved_metadata = json.load(f)
        assert saved_metadata["video_duration"] == 120.0

    def test_processor_with_output_manager(
        self, mock_logger, sample_video_path, tmp_path
    ):
        """Test processor uses OutputManager when no output_dir specified."""
        processor = ChunkProcessor(logger=mock_logger)

        with patch(
            "cogniverse_core.common.utils.output_manager.get_output_manager"
        ) as mock_get_om:
            mock_om = MagicMock()
            mock_get_om.return_value = mock_om
            mock_om.get_processing_dir.return_value = tmp_path / "output"

            with patch.object(processor, "_get_video_duration", return_value=0.0):
                processor.extract_chunks(sample_video_path)

            # Verify OutputManager was used
            mock_get_om.assert_called_once()
            assert mock_om.get_processing_dir.call_count == 2  # chunks and metadata


class TestChunkCaching:
    """cache_chunks controls whether extract_chunks reuses on-disk chunks or
    always re-extracts. The flag was previously stored and never consulted."""

    def _make_processor(self, mock_logger, cache_chunks):
        return ChunkProcessor(
            logger=mock_logger,
            chunk_duration=30.0,
            chunk_overlap=0.0,
            cache_chunks=cache_chunks,
        )

    def _run(self, processor, video_path, output_dir):
        """Run extract_chunks with ffprobe/ffmpeg faked; count extraction
        calls and materialize real chunk files so cache validation can see
        them. Returns (result, extract_call_count)."""
        calls = {"n": 0}

        def fake_extract(video, chunk_path, start, dur):
            calls["n"] += 1
            chunk_path.parent.mkdir(parents=True, exist_ok=True)
            chunk_path.write_bytes(b"CHUNK")
            return True

        with (
            patch.object(processor, "_get_video_duration", return_value=60.0),
            patch.object(processor, "_extract_chunk", side_effect=fake_extract),
        ):
            result = processor.extract_chunks(video_path, output_dir=output_dir)
        return result, calls["n"]

    def test_cache_enabled_reuses_chunks_on_second_call(
        self, mock_logger, sample_video_path, tmp_path
    ):
        proc = self._make_processor(mock_logger, cache_chunks=True)
        out = tmp_path / "out"

        first, first_calls = self._run(proc, sample_video_path, out)
        assert first_calls == 2  # 60s / 30s
        assert len(first["chunks"]) == 2

        second, second_calls = self._run(proc, sample_video_path, out)
        assert second_calls == 0  # fully reused
        assert second["chunks"] == first["chunks"]

    def test_cache_disabled_reextracts_every_call(
        self, mock_logger, sample_video_path, tmp_path
    ):
        proc = self._make_processor(mock_logger, cache_chunks=False)
        out = tmp_path / "out"

        _, first_calls = self._run(proc, sample_video_path, out)
        _, second_calls = self._run(proc, sample_video_path, out)
        assert first_calls == 2
        assert second_calls == 2  # no reuse

    def test_cache_invalidated_when_a_chunk_file_missing(
        self, mock_logger, sample_video_path, tmp_path
    ):
        proc = self._make_processor(mock_logger, cache_chunks=True)
        out = tmp_path / "out"

        first, _ = self._run(proc, sample_video_path, out)
        # Delete one chunk file — the cached set is now incomplete.
        from pathlib import Path

        Path(first["chunks"][0]["path"]).unlink()

        _, recount = self._run(proc, sample_video_path, out)
        assert recount == 2  # re-extracted because cache was invalid
