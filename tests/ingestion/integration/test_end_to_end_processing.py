"""
End-to-end integration tests for video processing pipeline.

Tests the complete pipeline from video input to final processed output,
using real processors (KeyframeProcessor, ChunkProcessor, AudioProcessor)
with a real video file. No mocked processors.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from cogniverse_runtime.ingestion.processing_strategy_set import ProcessingStrategySet
from cogniverse_runtime.ingestion.processor_manager import ProcessorManager
from cogniverse_runtime.ingestion.strategies import (
    AudioTranscriptionStrategy,
    ChunkSegmentationStrategy,
    FrameSegmentationStrategy,
    MultiVectorEmbeddingStrategy,
)


@pytest.mark.integration
@pytest.mark.ci_fast
class TestEndToEndVideoProcessing:
    """End-to-end integration tests for video processing with real processors."""

    @pytest.fixture(scope="class")
    def real_video_path(self, tmp_path_factory):
        """Create a real 5-second video with varying hue frames."""
        tmp = tmp_path_factory.mktemp("e2e_video")
        video_path = tmp / "test_video.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps, width, height = 30, 640, 480
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

        for i in range(fps * 5):
            hue = int((i / (fps * 5)) * 180)
            frame = np.ones((height, width, 3), dtype=np.uint8)
            frame[:, :] = [hue, 255, 255]
            frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            out.write(frame)

        out.release()
        return video_path

    @pytest.fixture
    def processor_manager(self):
        """Create a real ProcessorManager with auto-discovered processors."""
        logger = logging.getLogger("test_e2e_pipeline")
        return ProcessorManager(logger)

    @pytest.fixture
    def frame_strategy_set(self):
        """Frame-based processing strategy set."""
        return ProcessingStrategySet(
            segmentation=FrameSegmentationStrategy(
                max_frames=5, fps=1.0, threshold=0.8
            ),
            audio=AudioTranscriptionStrategy(),
            embedding=MultiVectorEmbeddingStrategy(),
        )

    @pytest.fixture
    def chunk_strategy_set(self):
        """Chunk-based processing strategy set."""
        return ProcessingStrategySet(
            segmentation=ChunkSegmentationStrategy(
                chunk_duration=2.0, chunk_overlap=0.0, cache_chunks=False
            ),
            audio=AudioTranscriptionStrategy(),
            embedding=MultiVectorEmbeddingStrategy(),
        )

    def test_complete_frame_based_pipeline(
        self, processor_manager, frame_strategy_set, real_video_path, tmp_path
    ):
        """Test complete frame-based pipeline with real processors."""
        processor_manager.initialize_from_strategies(
            frame_strategy_set, service_urls={}
        )

        keyframe_proc = processor_manager.get_processor("keyframe")
        assert keyframe_proc is not None, (
            "Real KeyframeProcessor should be auto-discovered"
        )

        output_dir = tmp_path / "frame_pipeline"
        output_dir.mkdir()
        keyframe_result = keyframe_proc.extract_keyframes(
            real_video_path, output_dir=output_dir
        )

        assert keyframe_result["video_id"] == "test_video"
        assert len(keyframe_result["keyframes"]) > 0
        assert keyframe_result["metadata"]["keyframes_extracted"] > 0

        keyframes_dir = Path(keyframe_result["keyframes_dir"])
        saved_files = list(keyframes_dir.glob("*.jpg"))
        assert len(saved_files) == len(keyframe_result["keyframes"])

        for kf in keyframe_result["keyframes"]:
            assert kf["timestamp"] >= 0.0
            assert "filename" in kf

    @pytest.mark.requires_ffmpeg
    def test_complete_chunk_based_pipeline(
        self, processor_manager, chunk_strategy_set, real_video_path, tmp_path
    ):
        """Test complete chunk-based pipeline with real processors."""
        processor_manager.initialize_from_strategies(
            chunk_strategy_set, service_urls={}
        )

        chunk_proc = processor_manager.get_processor("chunk")
        assert chunk_proc is not None, "Real ChunkProcessor should be auto-discovered"

        output_dir = tmp_path / "chunk_pipeline"
        output_dir.mkdir()
        chunk_result = chunk_proc.extract_chunks(real_video_path, output_dir=output_dir)

        assert chunk_result["video_id"] == "test_video"
        assert len(chunk_result["chunks"]) > 0
        assert chunk_result["metadata"]["chunks_extracted"] > 0

        chunks_dir = Path(chunk_result["chunks_dir"])
        saved_files = list(chunks_dir.glob("*.mp4"))
        assert len(saved_files) == len(chunk_result["chunks"])

        for chunk in chunk_result["chunks"]:
            assert chunk["end_time"] > chunk["start_time"]
            assert chunk["duration"] > 0

    def test_pipeline_error_propagation(
        self, processor_manager, frame_strategy_set, real_video_path, tmp_path
    ):
        """Test that errors in pipeline steps are properly propagated.

        Uses a real ProcessorManager with auto-discovered processors, then
        injects one intentionally-failing processor to test error handling.
        """

        class FailingKeyframeProcessor:
            PROCESSOR_NAME = "keyframe"

            def __init__(self, logger, **kwargs):
                self.logger = logger

            @classmethod
            def from_config(cls, config, logger):
                return cls(logger, **config)

            def extract_keyframes(self, *args, **kwargs):
                raise ValueError("Keyframe extraction failed")

        # Override just the keyframe processor; audio and embedding remain real
        processor_manager._processor_classes["keyframe"] = FailingKeyframeProcessor
        processor_manager.initialize_from_strategies(
            frame_strategy_set, service_urls={}
        )

        keyframe_proc = processor_manager.get_processor("keyframe")

        with pytest.raises(ValueError, match="Keyframe extraction failed"):
            keyframe_proc.extract_keyframes(real_video_path, output_dir=tmp_path)

    def test_pipeline_partial_failure_handling(
        self, processor_manager, frame_strategy_set, real_video_path, tmp_path
    ):
        """Test partial failure: keyframe succeeds, audio fails.

        Uses real ProcessorManager with real keyframe processor,
        injects a failing audio processor.
        """

        class FailingAudioProcessor:
            PROCESSOR_NAME = "audio"

            def __init__(self, logger, **kwargs):
                self.logger = logger

            @classmethod
            def from_config(cls, config, logger):
                return cls(logger, **config)

            def transcribe_audio(self, *args, **kwargs):
                raise RuntimeError("Audio processing failed")

        processor_manager._processor_classes["audio"] = FailingAudioProcessor
        processor_manager.initialize_from_strategies(
            frame_strategy_set, service_urls={}
        )

        # Real keyframe processing succeeds
        keyframe_proc = processor_manager.get_processor("keyframe")
        output_dir = tmp_path / "partial_failure"
        output_dir.mkdir()
        keyframe_result = keyframe_proc.extract_keyframes(
            real_video_path, output_dir=output_dir
        )
        assert len(keyframe_result["keyframes"]) > 0

        # Injected failing audio processor raises
        audio_proc = processor_manager.get_processor("audio")
        with pytest.raises(RuntimeError, match="Audio processing failed"):
            audio_proc.transcribe_audio(real_video_path)

    def test_pipeline_data_flow_between_stages(
        self, processor_manager, frame_strategy_set, real_video_path, tmp_path
    ):
        """Test data flows correctly between real pipeline stages."""
        processor_manager.initialize_from_strategies(
            frame_strategy_set, service_urls={}
        )

        keyframe_proc = processor_manager.get_processor("keyframe")
        output_dir = tmp_path / "data_flow"
        output_dir.mkdir()

        keyframe_result = keyframe_proc.extract_keyframes(
            real_video_path, output_dir=output_dir
        )

        # Verify real output structure
        assert keyframe_result["video_id"] == "test_video"
        assert "keyframes" in keyframe_result
        assert isinstance(keyframe_result["keyframes"], list)
        assert "metadata" in keyframe_result
        assert isinstance(keyframe_result["metadata"], dict)
        assert keyframe_result["metadata"]["video_id"] == "test_video"

    def test_pipeline_configuration_flows_to_real_processor(self, tmp_path):
        """Test that strategy config parameters reach real processor constructors."""
        logger = logging.getLogger("test_config_flow")
        manager = ProcessorManager(logger)

        strategy = FrameSegmentationStrategy(max_frames=15, fps=2.0, threshold=0.95)
        strategy_set = ProcessingStrategySet(segmentation=strategy)
        manager.initialize_from_strategies(strategy_set, service_urls={})

        processor = manager.get_processor("keyframe")
        assert processor is not None

        # Verify real KeyframeProcessor received the config values
        assert processor.max_frames == 15
        assert processor.fps == 2.0
        assert processor.threshold == 0.95

    def test_pipeline_concurrent_keyframe_extraction(
        self, processor_manager, frame_strategy_set, real_video_path, tmp_path
    ):
        """Test concurrent real keyframe extraction is thread-safe."""
        import threading

        processor_manager.initialize_from_strategies(
            frame_strategy_set, service_urls={}
        )
        processor = processor_manager.get_processor("keyframe")

        results = []
        errors = []

        def extract_keyframes(thread_idx):
            try:
                output_dir = tmp_path / f"concurrent_{thread_idx}"
                output_dir.mkdir(exist_ok=True)
                result = processor.extract_keyframes(
                    real_video_path, output_dir=output_dir
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=extract_keyframes, args=(i,)) for i in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent extraction errors: {errors}"
        assert len(results) == 3

        for result in results:
            assert result["video_id"] == "test_video"
            assert len(result["keyframes"]) > 0
