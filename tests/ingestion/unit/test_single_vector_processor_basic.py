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
        assert processor.min_segment_duration == 2.0

    def test_processor_initialization_custom(self, mock_logger):
        """Test processor initialization with custom values."""
        processor = SingleVectorVideoProcessor(
            mock_logger,
            strategy="windows",
            segment_duration=30.0,
            segment_overlap=5.0,
            min_segment_duration=5.0,
        )

        assert processor.strategy == "windows"
        assert processor.segment_duration == 30.0
        assert processor.segment_overlap == 5.0
        assert processor.min_segment_duration == 5.0

    @pytest.mark.parametrize(
        "param", ["sampling_fps", "max_frames_per_segment", "store_as_single_doc"]
    )
    def test_a_profile_naming_a_frame_or_storage_knob_is_refused(self, param):
        """The embedding server samples the frames and the schema decides the
        storage, so a profile still naming either knob fails to build."""
        from cogniverse_runtime.ingestion.strategies import (
            SingleVectorSegmentationStrategy,
        )

        with pytest.raises(TypeError) as caught:
            SingleVectorSegmentationStrategy(strategy="chunks", **{param: 1})
        assert str(caught.value) == (
            "SingleVectorSegmentationStrategy.__init__() got an unexpected "
            f"keyword argument '{param}'"
        )

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
        """Overlapping chunks run to the end; the last one is a full chunk
        whenever it is at least min_segment_duration long."""
        processor = SingleVectorVideoProcessor(
            mock_logger,
            strategy="chunks",
            segment_duration=10.0,
            segment_overlap=2.0,
            min_segment_duration=2.0,
        )

        assert processor._calculate_segment_boundaries(30.0) == [
            (0.0, 10.0),
            (8.0, 18.0),
            (16.0, 26.0),
            (24.0, 30.0),
        ]

    def test_calculate_segment_boundaries_no_overlap(self, mock_logger):
        processor = SingleVectorVideoProcessor(
            mock_logger,
            strategy="chunks",
            segment_duration=10.0,
            segment_overlap=0.0,
            min_segment_duration=2.0,
        )

        assert processor._calculate_segment_boundaries(25.0) == [
            (0.0, 10.0),
            (10.0, 20.0),
            (20.0, 25.0),
        ]

    def test_calculate_segment_boundaries_extend_last(self, mock_logger):
        """A remainder shorter than min_segment_duration joins the chunk
        before it instead of standing alone."""
        processor = SingleVectorVideoProcessor(
            mock_logger,
            strategy="chunks",
            segment_duration=10.0,
            segment_overlap=0.0,
            min_segment_duration=3.0,
        )

        assert processor._calculate_segment_boundaries(21.0) == [
            (0.0, 10.0),
            (10.0, 21.0),
        ]

    @pytest.mark.parametrize(
        ("duration", "boundaries"),
        [
            (1.5, [(0.0, 1.5)]),
            (6.0, [(0.0, 6.0)]),
            (10.0, [(0.0, 6.0), (5.0, 10.0)]),
            (
                26.5,
                [(0.0, 6.0), (5.0, 11.0), (10.0, 16.0), (15.0, 21.0), (20.0, 26.5)],
            ),
            (
                30.0,
                [
                    (0.0, 6.0),
                    (5.0, 11.0),
                    (10.0, 16.0),
                    (15.0, 21.0),
                    (20.0, 26.0),
                    (25.0, 30.0),
                ],
            ),
            (
                31.0,
                [
                    (0.0, 6.0),
                    (5.0, 11.0),
                    (10.0, 16.0),
                    (15.0, 21.0),
                    (20.0, 26.0),
                    (25.0, 31.0),
                ],
            ),
        ],
    )
    def test_the_shipped_chunk_profile_cuts_six_second_chunks(
        self, mock_logger, duration, boundaries
    ):
        """6 s chunks overlapping by 1 s, as video_embeddinggemma2_sv_chunk_6s
        ships them: a chunk ending exactly at the end of the video is kept,
        never merged into the one before."""
        processor = SingleVectorVideoProcessor(
            mock_logger, strategy="chunks", segment_duration=6.0, segment_overlap=1.0
        )

        assert processor._calculate_segment_boundaries(duration) == boundaries

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
                transcript_segments=[
                    {"start": 8.0, "text": "Second part"},  # Duplicate
                    {"start": 12.0, "text": "Third part"},
                ],
                transcript_text="",
            ),
        ]

        combined = processor._combine_transcripts(segments)
        assert combined == "First part Second part Third part"  # No duplicates

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
        segment = VideoSegment(
            segment_id=1,
            start_time=10.0,
            end_time=20.0,
            transcript_segments=[{"start": 12.0, "text": "test"}],
            transcript_text="test transcript",
            metadata={"test": "value"},
        )

        assert segment.segment_id == 1
        assert segment.start_time == 10.0
        assert segment.end_time == 20.0
        assert segment.transcript_segments == [{"start": 12.0, "text": "test"}]
        assert segment.transcript_text == "test transcript"
        assert segment.metadata["test"] == "value"

    def test_video_segment_to_dict(self):
        """Test VideoSegment to_dict method."""
        segment = VideoSegment(
            segment_id=2,
            start_time=5.0,
            end_time=15.0,
            transcript_segments=[{"start": 6.0, "text": "segment text"}],
            transcript_text="segment text",
            metadata={"key": "value"},
        )

        assert segment.to_dict() == {
            "segment_id": 2,
            "start_time": 5.0,
            "end_time": 15.0,
            "transcript_text": "segment text",
            "metadata": {"key": "value"},
        }


@pytest.mark.unit
@pytest.mark.ci_safe
class TestSegmentationOutput:
    """Segmentation reads only the video's header; the embedding stage reads
    each segment's time range from the video itself."""

    def test_process_video_returns_segments_and_their_metadata(self, tmp_path):
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

        proc = SingleVectorVideoProcessor(logging.getLogger("test"), strategy="global")
        result = proc.process_video(
            video_path=mp4,
            transcript_data={"segments": [{"start": 0.5, "end": 1.5, "text": " hi "}]},
        )
        metadata = dict(result["metadata"])
        metadata.pop("processed_at")

        assert sorted(result) == ["full_transcript", "metadata", "segments"]
        assert [seg.to_dict() for seg in result["segments"]] == [
            {
                "segment_id": 0,
                "start_time": 0.0,
                "end_time": 3.0,
                "transcript_text": "hi",
                "metadata": {"duration": 3.0, "has_transcript": True},
            }
        ]
        assert result["full_transcript"] == "hi"
        assert metadata == {
            "video_id": "clip",
            "duration": 3.0,
            "fps": 10.0,
            "total_frames": 30,
            "num_segments": 1,
            "strategy": "global",
            "segment_duration": 6.0,
            "segment_overlap": 0,
        }


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
            cache=cache,
        )

    def test_miss_computes_and_stores_full_fidelity_segments(self, tmp_path):
        mp4 = self._make_video(tmp_path)
        cache = self._RecordingCache()
        proc = self._processor(cache)

        result = proc.process_video(video_path=mp4)

        assert len(cache.get_calls) == 1
        assert len(cache.set_calls) == 1
        call = dict(cache.set_calls[0])
        payload = call.pop("result")
        assert call == {
            "video_path": str(mp4),
            "strategy": "chunks",
            "segment_duration": 1.0,
            "segment_overlap": 0.0,
            "transcript_fingerprint": "none",
        }
        assert cache.get_calls == [{k: v for k, v in call.items()}]
        assert sorted(payload) == ["full_transcript", "metadata", "segments"]
        assert payload["segments"] == [
            {
                "segment_id": seg.segment_id,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "transcript_segments": seg.transcript_segments,
                "transcript_text": seg.transcript_text,
                "metadata": seg.metadata,
            }
            for seg in result["segments"]
        ]

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
            assert got.transcript_segments == want.transcript_segments
            assert got.transcript_text == want.transcript_text
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


def _clip(tmp_path, frames: int, fps: float = 10.0) -> Path:
    import cv2

    mp4 = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    try:
        for _ in range(frames):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    finally:
        writer.release()
    return mp4


@pytest.mark.unit
@pytest.mark.ci_safe
class TestSegmentationParameters:
    """Parameters that would loop forever or cut nothing are refused."""

    def test_a_non_positive_segment_duration_is_refused(self):
        with pytest.raises(ValueError) as zero:
            SingleVectorVideoProcessor(Mock(), segment_duration=0.0)
        with pytest.raises(ValueError) as negative:
            SingleVectorVideoProcessor(Mock(), segment_duration=-6.0)

        assert str(zero.value) == "segment_duration must be positive, got 0.0"
        assert str(negative.value) == "segment_duration must be positive, got -6.0"

    def test_an_overlap_that_would_not_advance_is_refused(self):
        """An overlap at least as long as the segment never moves the next
        segment's start, so the cut would never end."""
        with pytest.raises(ValueError) as equal:
            SingleVectorVideoProcessor(
                Mock(), segment_duration=6.0, segment_overlap=6.0
            )
        with pytest.raises(ValueError) as longer:
            SingleVectorVideoProcessor(
                Mock(), segment_duration=6.0, segment_overlap=9.0
            )
        with pytest.raises(ValueError) as negative:
            SingleVectorVideoProcessor(
                Mock(), segment_duration=6.0, segment_overlap=-1.0
            )

        assert str(equal.value) == (
            "segment_overlap must be at least 0 and less than segment_duration "
            "(6.0), got 6.0"
        )
        assert str(longer.value) == (
            "segment_overlap must be at least 0 and less than segment_duration "
            "(6.0), got 9.0"
        )
        assert str(negative.value) == (
            "segment_overlap must be at least 0 and less than segment_duration "
            "(6.0), got -1.0"
        )

    def test_global_segmentation_ignores_the_overlap(self):
        processor = SingleVectorVideoProcessor(
            Mock(), strategy="global", segment_duration=6.0, segment_overlap=9.0
        )

        assert processor.segment_overlap == 0
        assert processor._calculate_segment_boundaries(95.0) == [(0.0, 95.0)]

    def test_a_negative_minimum_segment_is_refused(self):
        with pytest.raises(ValueError) as caught:
            SingleVectorVideoProcessor(Mock(), min_segment_duration=-1.0)

        assert (
            str(caught.value) == "min_segment_duration must not be negative, got -1.0"
        )

    def test_a_video_with_no_readable_duration_is_refused(self, tmp_path):
        """A header that reports no frames would otherwise cut zero segments
        and ingest nothing, reported as a video with no content."""
        empty = tmp_path / "empty.mp4"
        empty.write_bytes(b"not a video")
        processor = SingleVectorVideoProcessor(Mock())

        with pytest.raises(ValueError) as caught:
            processor.process_video(video_path=empty)

        assert str(caught.value) == (
            "empty.mp4 has no readable duration (fps 0.0, 0 frames)"
        )


@pytest.mark.unit
@pytest.mark.ci_safe
class TestSegmentationContracts:
    def test_windows_cut_thirty_second_windows_overlapping_by_five(self):
        processor = SingleVectorVideoProcessor(
            Mock(), strategy="windows", segment_duration=30.0, segment_overlap=5.0
        )

        assert processor._calculate_segment_boundaries(95.0) == [
            (0.0, 30.0),
            (25.0, 55.0),
            (50.0, 80.0),
            (75.0, 95.0),
        ]
        assert processor._calculate_segment_boundaries(81.0) == [
            (0.0, 30.0),
            (25.0, 55.0),
            (50.0, 80.0),
            (75.0, 81.0),
        ]
        assert processor._calculate_segment_boundaries(76.0) == [
            (0.0, 30.0),
            (25.0, 55.0),
            (50.0, 76.0),
        ]

    def test_transcript_alignment_excludes_segments_touching_only_an_edge(self):
        processor = SingleVectorVideoProcessor(Mock())
        transcript = [
            {"start": 0.0, "end": 10.0, "text": "ends at the start"},
            {"start": 20.0, "end": 25.0, "text": "starts at the end"},
            {"start": 19.5, "end": 20.5, "text": "straddles the end"},
        ]

        aligned = processor._align_transcript_segments(transcript, 10.0, 20.0)

        assert aligned == [
            {
                "start": 19.5,
                "end": 20.5,
                "text": "straddles the end",
                "relative_start": 9.5,
                "relative_end": 10.0,
            }
        ]
        assert transcript[2] == {
            "start": 19.5,
            "end": 20.5,
            "text": "straddles the end",
        }

    def test_overlapping_chunks_join_their_transcript_once_in_time_order(
        self, tmp_path
    ):
        mp4 = _clip(tmp_path, frames=120)
        processor = SingleVectorVideoProcessor(
            Mock(), strategy="chunks", segment_duration=6.0, segment_overlap=1.0
        )
        transcript = {
            "segments": [
                {"start": 8.0, "end": 9.0, "text": " third "},
                {"start": 0.5, "end": 1.5, "text": "first"},
                {"start": 5.2, "end": 5.8, "text": "second"},
                {"start": 9.5, "end": 9.9, "text": "   "},
            ]
        }

        result = processor.process_video(video_path=mp4, transcript_data=transcript)

        assert [
            (seg.start_time, seg.end_time, seg.transcript_text)
            for seg in result["segments"]
        ] == [
            (0.0, 6.0, "first second"),
            (5.0, 11.0, "second third"),
            (10.0, 12.0, ""),
        ]
        assert result["full_transcript"] == "first second third"
        assert [seg.metadata for seg in result["segments"]] == [
            {"duration": 6.0, "has_transcript": True},
            {"duration": 6.0, "has_transcript": True},
            {"duration": 2.0, "has_transcript": False},
        ]

    def test_to_dict_of_a_segment_without_metadata(self):
        segment = VideoSegment(
            segment_id=0, start_time=0.0, end_time=6.0, transcript_segments=[]
        )

        assert segment.to_dict() == {
            "segment_id": 0,
            "start_time": 0.0,
            "end_time": 6.0,
            "transcript_text": "",
            "metadata": {},
        }

    def test_a_cached_segment_round_trips_and_owns_its_data(self):
        from cogniverse_runtime.ingestion.processors.single_vector_processor import (
            _segment_cache_payload,
            _segment_from_cache,
        )

        segment = VideoSegment(
            segment_id=3,
            start_time=15.0,
            end_time=21.0,
            transcript_segments=[{"start": 16.0, "text": "kick"}],
            transcript_text="kick",
            metadata={"duration": 6.0, "has_transcript": True},
        )

        payload = _segment_cache_payload(segment)
        segment.transcript_segments[0]["text"] = "changed"
        segment.metadata["duration"] = 0.0
        restored = _segment_from_cache(payload)
        restored.transcript_segments.append({"start": 20.0, "text": "late"})

        assert payload == {
            "segment_id": 3,
            "start_time": 15.0,
            "end_time": 21.0,
            "transcript_segments": [{"start": 16.0, "text": "kick"}],
            "transcript_text": "kick",
            "metadata": {"duration": 6.0, "has_transcript": True},
        }
        assert restored == VideoSegment(
            segment_id=3,
            start_time=15.0,
            end_time=21.0,
            transcript_segments=[
                {"start": 16.0, "text": "kick"},
                {"start": 20.0, "text": "late"},
            ],
            transcript_text="kick",
            metadata={"duration": 6.0, "has_transcript": True},
        )

    def test_the_transcript_fingerprint_ignores_key_order(self):
        from cogniverse_runtime.ingestion.processors.single_vector_processor import (
            _transcript_fingerprint,
        )

        one = {"segments": [{"start": 0.0, "text": "a"}], "language": "en"}
        same = {"language": "en", "segments": [{"text": "a", "start": 0.0}]}
        other = {"language": "en", "segments": [{"text": "b", "start": 0.0}]}

        assert _transcript_fingerprint(None) == "none"
        assert _transcript_fingerprint({}) == "none"
        assert _transcript_fingerprint(one) == _transcript_fingerprint(same)
        assert _transcript_fingerprint(one) != _transcript_fingerprint(other)
        assert _transcript_fingerprint(one) == "47e8a14c060e7dac"


@pytest.mark.unit
@pytest.mark.ci_safe
class TestSegmentationStrategyDrivesTheProcessor:
    """The strategy's processor config builds the processor the pipeline
    runs, and its segment() hands back serializable segments."""

    def _strategy(self):
        from cogniverse_runtime.ingestion.strategies import (
            SingleVectorSegmentationStrategy,
        )

        return SingleVectorSegmentationStrategy(
            strategy="chunks", segment_duration=6.0, segment_overlap=1.0
        )

    def test_the_required_processor_config_builds_the_processor(self):
        config = self._strategy().get_required_processors()
        processor = SingleVectorVideoProcessor.from_config(
            config["single_vector"], Mock()
        )

        assert config == {
            "single_vector": {
                "strategy": "chunks",
                "segment_duration": 6.0,
                "segment_overlap": 1.0,
            }
        }
        assert (
            processor.strategy,
            processor.segment_duration,
            processor.segment_overlap,
            processor.min_segment_duration,
        ) == ("chunks", 6.0, 1.0, 2.0)

    def test_segment_refuses_a_context_without_the_processor(self, tmp_path):
        import asyncio
        from types import SimpleNamespace

        strategy = self._strategy()
        with pytest.raises(ValueError) as no_manager:
            asyncio.run(strategy.segment(tmp_path / "v.mp4", SimpleNamespace()))
        empty = SimpleNamespace(
            processor_manager=SimpleNamespace(get_processor=lambda name: None)
        )
        with pytest.raises(ValueError) as no_processor:
            asyncio.run(strategy.segment(tmp_path / "v.mp4", empty))

        assert str(no_manager.value) == "Pipeline context missing processor_manager"
        assert str(no_processor.value) == (
            "No single-vector processor available in processor_manager"
        )

    def test_segment_returns_the_processor_output_with_segments_as_dicts(
        self, tmp_path
    ):
        import asyncio
        from types import SimpleNamespace

        mp4 = _clip(tmp_path, frames=100)
        strategy = self._strategy()
        processor = SingleVectorVideoProcessor.from_config(
            strategy.get_required_processors()["single_vector"], Mock()
        )
        context = SimpleNamespace(
            processor_manager=SimpleNamespace(
                get_processor=lambda name: {"single_vector": processor}.get(name)
            ),
            cache=None,
        )

        out = asyncio.run(
            strategy.segment(
                mp4,
                context,
                transcript_data={
                    "segments": [{"start": 4.0, "end": 7.0, "text": "goal"}]
                },
            )
        )
        result = out["single_vector_processing"]

        assert sorted(out) == ["single_vector_processing"]
        assert result["segments"] == [
            {
                "segment_id": 0,
                "start_time": 0.0,
                "end_time": 6.0,
                "transcript_text": "goal",
                "metadata": {"duration": 6.0, "has_transcript": True},
            },
            {
                "segment_id": 1,
                "start_time": 5.0,
                "end_time": 10.0,
                "transcript_text": "goal",
                "metadata": {"duration": 5.0, "has_transcript": True},
            },
        ]
        assert result["full_transcript"] == "goal"
        assert result["metadata"]["num_segments"] == 2

    def test_process_forwards_the_transcript_and_metadata(self, tmp_path):
        mp4 = _clip(tmp_path, frames=30)
        processor = SingleVectorVideoProcessor(Mock(), strategy="global")
        transcript = {"segments": [{"start": 0.0, "end": 1.0, "text": "hello"}]}

        result = processor.process(
            mp4, transcript_data=transcript, metadata={"tenant": "acme:acme"}
        )

        assert result["full_transcript"] == "hello"
        assert result["metadata"]["tenant"] == "acme:acme"
        assert result["segments"][0].transcript_segments == [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello",
                "relative_start": 0.0,
                "relative_end": 1.0,
            }
        ]


@pytest.mark.unit
@pytest.mark.ci_safe
class TestVideoHeaderAndCachedCopies:
    def test_an_unreadable_file_reports_an_empty_header(self, tmp_path):
        junk = tmp_path / "junk.mp4"
        junk.write_bytes(b"\x00" * 64)

        assert SingleVectorVideoProcessor(Mock())._get_video_info(junk) == {
            "fps": 0.0,
            "total_frames": 0,
            "width": 0,
            "height": 0,
            "duration": 0,
        }

    def test_a_cache_hit_hands_out_segments_it_does_not_share(self, tmp_path):
        mp4 = _clip(tmp_path, frames=30)
        transcript = {"segments": [{"start": 0.0, "end": 1.0, "text": "kickoff"}]}
        recording = TestSegmentationCache._RecordingCache()
        processor = SingleVectorVideoProcessor(
            Mock(), strategy="global", cache=recording
        )
        processor.process_video(video_path=mp4, transcript_data=transcript)
        payload = recording.set_calls[0]["result"]
        cached = SingleVectorVideoProcessor(
            Mock(),
            strategy="global",
            cache=TestSegmentationCache._RecordingCache(payload),
        )

        first = cached.process_video(video_path=mp4, transcript_data=transcript)
        first["segments"][0].transcript_segments[0]["text"] = "changed"
        first["segments"][0].metadata["duration"] = 0.0
        second = cached.process_video(video_path=mp4, transcript_data=transcript)

        assert second["segments"][0].transcript_segments == [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "kickoff",
                "relative_start": 0.0,
                "relative_end": 1.0,
            }
        ]
        assert second["segments"][0].metadata == {
            "duration": 3.0,
            "has_transcript": True,
        }
        assert payload["segments"][0]["transcript_segments"][0]["text"] == "kickoff"
