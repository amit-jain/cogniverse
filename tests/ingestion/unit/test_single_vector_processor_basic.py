"""
Basic unit tests for SingleVectorVideoProcessor to improve coverage.

Tests basic initialization and configuration without external dependencies.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from cogniverse_runtime.ingestion.processors.single_vector_processor import (
    SingleVectorVideoProcessor,
    VideoSegment,
)


@pytest.mark.unit
@pytest.mark.ci_safe
class TestSingleVectorVideoProcessor:
    """Basic tests for SingleVectorVideoProcessor."""

    @pytest.fixture
    def mock_logger(self):
        return Mock()

    def test_processor_initialization_defaults(self, mock_logger):
        """Test processor initialization with default values."""
        processor = SingleVectorVideoProcessor(mock_logger)

        assert processor.PROCESSOR_NAME == "single_vector"
        assert processor.logger == mock_logger
        assert processor.strategy == "chunks"
        assert processor.segment_duration == 6.0
        assert processor.segment_overlap == 1.0
        assert processor.sampling_fps == 2.0
        assert processor.max_frames_per_segment == 12
        assert processor.min_segment_duration == 2.0
        assert processor.store_as_single_doc is True

    def test_processor_initialization_custom(self, mock_logger):
        """Test processor initialization with custom values."""
        processor = SingleVectorVideoProcessor(
            mock_logger,
            strategy="windows",
            segment_duration=30.0,
            segment_overlap=5.0,
            sampling_fps=1.0,
            max_frames_per_segment=30,
            min_segment_duration=5.0,
            store_as_single_doc=False,
        )

        assert processor.strategy == "windows"
        assert processor.segment_duration == 30.0
        assert processor.segment_overlap == 5.0
        assert processor.sampling_fps == 1.0
        assert processor.max_frames_per_segment == 30
        assert processor.min_segment_duration == 5.0
        assert processor.store_as_single_doc is False

    def test_processor_initialization_global_strategy(self, mock_logger):
        """Test processor initialization with global strategy."""
        processor = SingleVectorVideoProcessor(
            mock_logger,
            strategy="global",
            segment_overlap=5.0,  # Should be ignored for global
        )

        assert processor.strategy == "global"
        assert processor.segment_overlap == 0  # Should be 0 for global strategy

    def test_get_video_info(self, mock_logger):
        """Test _get_video_info method."""
        processor = SingleVectorVideoProcessor(mock_logger)

        with patch("cv2.VideoCapture") as mock_cv2:
            mock_cap = Mock()
            mock_cap.get.side_effect = [
                25.0,
                1000,
                1920,
                1080,
            ]  # fps, frames, width, height
            mock_cv2.return_value = mock_cap

            info = processor._get_video_info(Path("/test/video.mp4"))

            assert info["fps"] == 25.0
            assert info["total_frames"] == 1000
            assert info["width"] == 1920
            assert info["height"] == 1080
            assert info["duration"] == 40.0  # 1000 frames / 25 fps
            mock_cap.release.assert_called_once()

    def test_calculate_segment_boundaries_global(self, mock_logger):
        """Test segment boundary calculation for global strategy."""
        processor = SingleVectorVideoProcessor(mock_logger, strategy="global")

        boundaries = processor._calculate_segment_boundaries(120.0)

        assert len(boundaries) == 1
        assert boundaries[0] == (0.0, 120.0)

    def test_calculate_segment_boundaries_chunks(self, mock_logger):
        """Test segment boundary calculation for chunks strategy."""
        processor = SingleVectorVideoProcessor(
            mock_logger,
            strategy="chunks",
            segment_duration=10.0,
            segment_overlap=2.0,
            min_segment_duration=2.0,
        )

        boundaries = processor._calculate_segment_boundaries(30.0)

        # Should have segments: [0-10], [8-18], [16-30] (last extended)
        assert len(boundaries) == 3
        assert boundaries[0] == (0.0, 10.0)
        assert boundaries[1] == (8.0, 18.0)  # 10 - 2 overlap
        assert boundaries[2] == (16.0, 30.0)  # Extended to avoid tiny segment

    def test_calculate_segment_boundaries_no_overlap(self, mock_logger):
        """Test segment boundary calculation without overlap."""
        processor = SingleVectorVideoProcessor(
            mock_logger,
            strategy="chunks",
            segment_duration=10.0,
            segment_overlap=0.0,
            min_segment_duration=2.0,
        )

        boundaries = processor._calculate_segment_boundaries(25.0)

        # Should have segments: [0-10], [10-25] (second extended since 25-20=5 >= min_duration)
        assert len(boundaries) == 2
        assert boundaries[0] == (0.0, 10.0)
        assert boundaries[1] == (10.0, 25.0)  # Extended to avoid tiny segment

    def test_calculate_segment_boundaries_extend_last(self, mock_logger):
        """Test that small remaining duration extends last segment."""
        processor = SingleVectorVideoProcessor(
            mock_logger,
            strategy="chunks",
            segment_duration=10.0,
            segment_overlap=0.0,
            min_segment_duration=3.0,
        )

        boundaries = processor._calculate_segment_boundaries(21.0)

        # Should have one segment [0-21] since 21-10=11 > 10, but then 21-20=1 < 3
        # So the algorithm extends the first segment to cover everything
        assert len(boundaries) == 1
        assert boundaries[0] == (0.0, 21.0)  # Extended to cover entire duration

    def test_align_transcript_segments(self, mock_logger):
        """Test transcript segment alignment."""
        processor = SingleVectorVideoProcessor(mock_logger)

        transcript_segments = [
            {"start": 5.0, "end": 15.0, "text": "First segment"},
            {"start": 12.0, "end": 25.0, "text": "Second segment"},
            {"start": 30.0, "end": 40.0, "text": "Third segment"},
        ]

        # Align with segment [10.0, 20.0]
        aligned = processor._align_transcript_segments(transcript_segments, 10.0, 20.0)

        assert len(aligned) == 2  # First two segments overlap
        assert aligned[0]["text"] == "First segment"
        assert aligned[0]["relative_start"] == 0.0  # max(0, 5-10)
        assert aligned[0]["relative_end"] == 5.0  # min(10, 15-10)

        assert aligned[1]["text"] == "Second segment"
        assert aligned[1]["relative_start"] == 2.0  # max(0, 12-10)
        assert aligned[1]["relative_end"] == 10.0  # min(10, 25-10)

    def test_combine_transcripts_global(self, mock_logger):
        """Test transcript combination for global strategy."""
        processor = SingleVectorVideoProcessor(mock_logger, strategy="global")

        segment = VideoSegment(
            segment_id=0,
            start_time=0.0,
            end_time=60.0,
            frames=[],
            frame_timestamps=[],
            transcript_segments=[],
            transcript_text="Full video transcript",
        )

        combined = processor._combine_transcripts([segment])
        assert combined == "Full video transcript"

    def test_combine_transcripts_overlapping(self, mock_logger):
        """Test transcript combination with overlapping segments."""
        processor = SingleVectorVideoProcessor(mock_logger, strategy="chunks")

        segments = [
            VideoSegment(
                segment_id=0,
                start_time=0.0,
                end_time=10.0,
                frames=[],
                frame_timestamps=[],
                transcript_segments=[
                    {"start": 5.0, "text": "First part"},
                    {"start": 8.0, "text": "Second part"},
                ],
                transcript_text="",
            ),
            VideoSegment(
                segment_id=1,
                start_time=8.0,
                end_time=18.0,
                frames=[],
                frame_timestamps=[],
                transcript_segments=[
                    {"start": 8.0, "text": "Second part"},  # Duplicate
                    {"start": 12.0, "text": "Third part"},
                ],
                transcript_text="",
            ),
        ]

        combined = processor._combine_transcripts(segments)
        assert combined == "First part Second part Third part"  # No duplicates

    def test_get_document_structure_single(self, mock_logger):
        """Test document structure for single document storage."""
        processor = SingleVectorVideoProcessor(mock_logger, store_as_single_doc=True)

        structure = processor._get_document_structure()

        assert structure["type"] == "single_document"
        assert "All segments stored in one document" in structure["description"]

    def test_get_document_structure_multiple(self, mock_logger):
        """Test document structure for multiple document storage."""
        processor = SingleVectorVideoProcessor(mock_logger, store_as_single_doc=False)

        structure = processor._get_document_structure()

        assert structure["type"] == "multiple_documents"
        assert "Each segment stored as separate document" in structure["description"]

    def test_prepare_for_embedding_generation(self, mock_logger):
        """Test preparation for embedding generation."""
        processor = SingleVectorVideoProcessor(mock_logger)

        frames = [np.random.rand(100, 100, 3) for _ in range(3)]
        segment = VideoSegment(
            segment_id=0,
            start_time=0.0,
            end_time=10.0,
            frames=frames,
            frame_timestamps=[0.0, 5.0, 10.0],
            transcript_segments=[],
            transcript_text="Test transcript",
        )

        prepared = processor.prepare_for_embedding_generation([segment], "videoprism")

        assert len(prepared) == 1
        assert prepared[0]["segment_id"] == 0
        assert len(prepared[0]["frames"]) == 3
        assert prepared[0]["frame_timestamps"] == [0.0, 5.0, 10.0]
        assert prepared[0]["metadata"]["start_time"] == 0.0
        assert prepared[0]["metadata"]["end_time"] == 10.0
        assert prepared[0]["metadata"]["transcript"] == "Test transcript"
        assert prepared[0]["metadata"]["model_type"] == "videoprism"

    def test_processor_name_constant(self, mock_logger):
        """Test processor name constant."""
        processor = SingleVectorVideoProcessor(mock_logger)
        assert processor.PROCESSOR_NAME == "single_vector"
        assert processor.get_processor_name() == "single_vector"


@pytest.mark.unit
@pytest.mark.ci_safe
class TestVideoSegment:
    """Tests for VideoSegment dataclass."""

    def test_video_segment_creation(self):
        """Test VideoSegment creation."""
        frames = [np.random.rand(100, 100, 3) for _ in range(2)]
        segment = VideoSegment(
            segment_id=1,
            start_time=10.0,
            end_time=20.0,
            frames=frames,
            frame_timestamps=[10.0, 15.0],
            transcript_segments=[{"start": 12.0, "text": "test"}],
            transcript_text="test transcript",
            metadata={"test": "value"},
        )

        assert segment.segment_id == 1
        assert segment.start_time == 10.0
        assert segment.end_time == 20.0
        assert len(segment.frames) == 2
        assert segment.frame_timestamps == [10.0, 15.0]
        assert segment.transcript_text == "test transcript"
        assert segment.metadata["test"] == "value"

    def test_video_segment_to_dict(self):
        """Test VideoSegment to_dict method."""
        frames = [np.random.rand(50, 50, 3) for _ in range(3)]
        segment = VideoSegment(
            segment_id=2,
            start_time=5.0,
            end_time=15.0,
            frames=frames,
            frame_timestamps=[5.0, 10.0, 15.0],
            transcript_segments=[],
            transcript_text="segment text",
            metadata={"key": "value"},
        )

        segment_dict = segment.to_dict()

        assert segment_dict["segment_id"] == 2
        assert segment_dict["start_time"] == 5.0
        assert segment_dict["end_time"] == 15.0
        assert segment_dict["frame_count"] == 3
        assert segment_dict["transcript_text"] == "segment text"
        assert segment_dict["metadata"]["key"] == "value"
        # Frames should not be included in dict representation
        assert "frames" not in segment_dict


@pytest.mark.unit
@pytest.mark.ci_safe
class TestSegmentationDoesNotDecodeFrames:
    """Segmentation must not decode + retain raw frames (tens of GB on a long
    video); the embedding stage re-decodes each time range on demand."""

    def test_process_video_keeps_no_decoded_frames(self, tmp_path):
        import logging

        import cv2

        mp4 = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(
            str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64)
        )
        try:
            for _ in range(30):  # 3s @ 10fps
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        finally:
            writer.release()

        proc = SingleVectorVideoProcessor(
            logging.getLogger("test"),
            strategy="chunks",
            segment_duration=1.0,
            segment_overlap=0.0,
            sampling_fps=2.0,
        )
        segments = proc.process_video(video_path=mp4)["segments"]

        assert segments, "expected at least one segment"
        # No decoded frames retained anywhere.
        assert all(len(seg.frames) == 0 for seg in segments)
        # But the planned sample count is still recorded (boundary math).
        assert any(seg.metadata["frame_count"] > 0 for seg in segments)
        for seg in segments:
            assert len(seg.frame_timestamps) == seg.metadata["frame_count"]
            for ts in seg.frame_timestamps:
                assert seg.start_time - 0.2 <= ts <= seg.end_time + 0.2


@pytest.mark.unit
@pytest.mark.ci_safe
class TestSegmentationCache:
    """When a pipeline artifact cache is wired in, re-segmenting the same
    video with the same strategy params and transcript serves the cached
    result (no container probe) and a transcript change misses."""

    def _make_video(self, tmp_path):
        import cv2

        mp4 = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(
            str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64)
        )
        try:
            for _ in range(30):
                writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
        finally:
            writer.release()
        return mp4

    class _RecordingCache:
        def __init__(self, hit=None):
            self.hit = hit
            self.get_calls = []
            self.set_calls = []

        async def get_segmentation(self, video_path, **kwargs):
            self.get_calls.append({"video_path": video_path, **kwargs})
            return self.hit

        async def set_segmentation(self, video_path, result, **kwargs):
            self.set_calls.append(
                {"video_path": video_path, "result": result, **kwargs}
            )
            return True

    def _processor(self, cache):
        import logging

        return SingleVectorVideoProcessor(
            logging.getLogger("test"),
            strategy="chunks",
            segment_duration=1.0,
            segment_overlap=0.0,
            sampling_fps=2.0,
            cache=cache,
        )

    def test_miss_computes_and_stores_full_fidelity_segments(self, tmp_path):
        mp4 = self._make_video(tmp_path)
        cache = self._RecordingCache()
        proc = self._processor(cache)

        result = proc.process_video(video_path=mp4)

        assert len(cache.get_calls) == 1
        assert len(cache.set_calls) == 1
        call = cache.set_calls[0]
        assert call["video_path"] == str(mp4)
        assert call["strategy"] == "chunks"
        assert call["segment_duration"] == 1.0
        assert call["segment_overlap"] == 0.0
        assert call["sampling_fps"] == 2.0
        assert call["max_frames"] == 12
        assert call["transcript_fingerprint"] == "none"
        payload_segments = call["result"]["segments"]
        assert len(payload_segments) == len(result["segments"])
        for d, seg in zip(payload_segments, result["segments"]):
            assert d["segment_id"] == seg.segment_id
            assert d["frame_timestamps"] == seg.frame_timestamps
            assert d["transcript_segments"] == seg.transcript_segments
            assert d["transcript_text"] == seg.transcript_text

    def test_hit_serves_cached_segments_without_probing_the_video(self, tmp_path):
        mp4 = self._make_video(tmp_path)
        first = self._RecordingCache()
        result = self._processor(first).process_video(
            video_path=mp4, metadata={"run": "one"}
        )
        cached_payload = first.set_calls[0]["result"]

        cache = self._RecordingCache(hit=cached_payload)
        proc = self._processor(cache)
        proc._get_video_info = Mock(
            side_effect=AssertionError("cache hit must not probe the video")
        )

        out = proc.process_video(video_path=mp4, metadata={"run": "two"})

        assert cache.set_calls == []
        assert len(out["segments"]) == len(result["segments"])
        for got, want in zip(out["segments"], result["segments"]):
            assert isinstance(got, VideoSegment)
            assert got.segment_id == want.segment_id
            assert got.start_time == want.start_time
            assert got.end_time == want.end_time
            assert got.frame_timestamps == want.frame_timestamps
            assert got.transcript_text == want.transcript_text
            assert got.frames == []
        assert out["metadata"]["run"] == "two"
        assert out["metadata"]["num_segments"] == len(result["segments"])
        # Serving from cache must not poison the cached payload for later hits.
        assert "run" not in cached_payload["metadata"]

    def test_transcript_changes_the_fingerprint(self, tmp_path):
        mp4 = self._make_video(tmp_path)
        cache = self._RecordingCache()
        proc = self._processor(cache)

        transcript_a = {"segments": [{"start": 0.0, "end": 1.0, "text": "a"}]}
        transcript_b = {"segments": [{"start": 0.0, "end": 1.0, "text": "b"}]}
        proc.process_video(video_path=mp4, transcript_data=transcript_a)
        proc.process_video(video_path=mp4, transcript_data=transcript_b)

        fp_a = cache.get_calls[0]["transcript_fingerprint"]
        fp_b = cache.get_calls[1]["transcript_fingerprint"]
        assert fp_a != fp_b
        assert fp_a != "none"
        assert fp_b != "none"
