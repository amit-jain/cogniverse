"""
End-to-end integration tests for video processing pipeline.

Tests the complete pipeline from video input to final processed output,
including keyframe extraction, audio transcription, and embedding generation.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.app.ingestion.processing_strategy_set import ProcessingStrategySet
from src.app.ingestion.processor_manager import ProcessorManager
from src.app.ingestion.strategies import (
    AudioTranscriptionStrategy,
    ChunkSegmentationStrategy,
    FrameSegmentationStrategy,
    MultiVectorEmbeddingStrategy,
)


@pytest.mark.integration
class TestEndToEndVideoProcessing:
    """End-to-end integration tests for video processing."""

    @pytest.fixture
    def mock_pipeline_config(self, temp_dir):
        """Create a mock pipeline configuration."""
        return {
            "video_dir": temp_dir,
            "output_dir": temp_dir / "output",
            "cache_dir": temp_dir / "cache",
            "max_concurrent_videos": 1,
            "search_backend": "vespa",
            "profiles": ["test_profile"],
        }

    @pytest.fixture
    def frame_based_strategy_set(self):
        """Create frame-based processing strategy set."""
        frame_strategy = FrameSegmentationStrategy(max_frames=5, fps=1.0, threshold=0.8)
        audio_strategy = AudioTranscriptionStrategy()
        embedding_strategy = MultiVectorEmbeddingStrategy()
        return ProcessingStrategySet(
            segmentation=frame_strategy,
            audio=audio_strategy,
            embedding=embedding_strategy,
        )

    @pytest.fixture
    def chunk_based_strategy_set(self):
        """Create chunk-based processing strategy set."""
        chunk_strategy = ChunkSegmentationStrategy(
            chunk_duration=10.0, chunk_overlap=1.0, cache_chunks=False
        )
        audio_strategy = AudioTranscriptionStrategy()
        embedding_strategy = MultiVectorEmbeddingStrategy()
        return ProcessingStrategySet(
            segmentation=chunk_strategy,
            audio=audio_strategy,
            embedding=embedding_strategy,
        )

    @patch("src.app.ingestion.processors.keyframe_processor.KeyframeProcessor")
    @patch("src.app.ingestion.processors.audio_processor.AudioProcessor")
    @patch("src.app.ingestion.processors.embedding_processor.EmbeddingProcessor")
    def test_complete_frame_based_pipeline(
        self,
        mock_embedding_class,
        mock_audio_class,
        mock_keyframe_class,
        mock_logger,
        frame_based_strategy_set,
        temp_dir,
        sample_video_path,
    ):
        """Test complete frame-based video processing pipeline."""

        # Mock keyframe processor
        mock_keyframe = Mock()
        mock_keyframe.PROCESSOR_NAME = "keyframe"
        mock_keyframe_result = {
            "video_id": "test_video",
            "total_keyframes": 3,
            "fps": 1.0,
            "keyframes": [
                {"frame_index": 0, "timestamp": 0.0, "filename": "frame_0.jpg"},
                {"frame_index": 30, "timestamp": 1.0, "filename": "frame_1.jpg"},
                {"frame_index": 60, "timestamp": 2.0, "filename": "frame_2.jpg"},
            ],
        }
        mock_keyframe.extract_keyframes.return_value = mock_keyframe_result
        mock_keyframe_class.return_value = mock_keyframe
        mock_keyframe_class.from_config = Mock(return_value=mock_keyframe)

        # Mock audio processor
        mock_audio = Mock()
        mock_audio.PROCESSOR_NAME = "audio"
        mock_audio_result = {
            "video_id": "test_video",
            "text": "This is test audio transcription.",
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "This is test audio"},
                {"start": 2.5, "end": 5.0, "text": "transcription."},
            ],
            "language": "en",
        }
        mock_audio.transcribe_audio.return_value = mock_audio_result
        mock_audio_class.return_value = mock_audio
        mock_audio_class.from_config = Mock(return_value=mock_audio)

        # Mock embedding processor
        mock_embedding = Mock()
        mock_embedding.PROCESSOR_NAME = "embedding"
        mock_embedding_result = {
            "video_id": "test_video",
            "embeddings": {
                "frame_embeddings": [
                    {"frame_index": 0, "embedding": np.random.rand(128).tolist()},
                    {"frame_index": 1, "embedding": np.random.rand(128).tolist()},
                    {"frame_index": 2, "embedding": np.random.rand(128).tolist()},
                ],
                "audio_embedding": np.random.rand(128).tolist(),
            },
        }
        mock_embedding.generate_embeddings.return_value = mock_embedding_result
        mock_embedding_class.return_value = mock_embedding
        mock_embedding_class.from_config = Mock(return_value=mock_embedding)

        # Create processor manager with mocked processors
        with patch(
            "src.app.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []

            manager = ProcessorManager(mock_logger)
            manager._processor_classes["keyframe"] = mock_keyframe_class
            manager._processor_classes["audio"] = mock_audio_class
            manager._processor_classes["embedding"] = mock_embedding_class

            # Initialize from strategies
            manager.initialize_from_strategies(frame_based_strategy_set)

            # Execute pipeline steps
            keyframe_processor = manager.get_processor("keyframe")
            audio_processor = manager.get_processor("audio")
            embedding_processor = manager.get_processor("embedding")

            # Process video through complete pipeline
            keyframe_result = keyframe_processor.extract_keyframes(
                sample_video_path, temp_dir
            )
            audio_result = audio_processor.transcribe_audio(sample_video_path)
            embedding_result = embedding_processor.generate_embeddings(
                {"keyframes": keyframe_result, "transcript": audio_result}
            )

            # Verify pipeline results
            assert keyframe_result["total_keyframes"] == 3
            assert audio_result["language"] == "en"
            assert "frame_embeddings" in embedding_result["embeddings"]
            assert len(embedding_result["embeddings"]["frame_embeddings"]) == 3

            # Verify processors were called with correct data
            mock_keyframe.extract_keyframes.assert_called_once_with(
                sample_video_path, temp_dir
            )
            mock_audio.transcribe_audio.assert_called_once_with(sample_video_path)
            mock_embedding.generate_embeddings.assert_called_once()

    @patch("src.app.ingestion.processors.chunk_processor.ChunkProcessor")
    @patch("src.app.ingestion.processors.audio_processor.AudioProcessor")
    @patch("src.app.ingestion.processors.embedding_processor.EmbeddingProcessor")
    def test_complete_chunk_based_pipeline(
        self,
        mock_embedding_class,
        mock_audio_class,
        mock_chunk_class,
        mock_logger,
        chunk_based_strategy_set,
        temp_dir,
        sample_video_path,
    ):
        """Test complete chunk-based video processing pipeline."""

        # Mock chunk processor
        mock_chunk = Mock()
        mock_chunk.PROCESSOR_NAME = "chunk"
        mock_chunk_result = {
            "video_id": "test_video",
            "total_chunks": 2,
            "chunks": [
                {"start_time": 0.0, "end_time": 10.0, "filename": "chunk_0.mp4"},
                {"start_time": 9.0, "end_time": 19.0, "filename": "chunk_1.mp4"},
            ],
        }
        mock_chunk.extract_chunks.return_value = mock_chunk_result
        mock_chunk_class.return_value = mock_chunk
        mock_chunk_class.from_config = Mock(return_value=mock_chunk)

        # Mock audio processor
        mock_audio = Mock()
        mock_audio.PROCESSOR_NAME = "audio"
        mock_audio_result = {
            "video_id": "test_video",
            "text": "Chunk-based audio transcription.",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "Chunk-based audio transcription."}
            ],
            "language": "en",
        }
        mock_audio.transcribe_audio.return_value = mock_audio_result
        mock_audio_class.return_value = mock_audio
        mock_audio_class.from_config = Mock(return_value=mock_audio)

        # Mock embedding processor
        mock_embedding = Mock()
        mock_embedding.PROCESSOR_NAME = "embedding"
        mock_embedding_result = {
            "video_id": "test_video",
            "embeddings": {
                "chunk_embeddings": [
                    {"chunk_index": 0, "embedding": np.random.rand(128).tolist()},
                    {"chunk_index": 1, "embedding": np.random.rand(128).tolist()},
                ],
            },
        }
        mock_embedding.generate_embeddings.return_value = mock_embedding_result
        mock_embedding_class.return_value = mock_embedding
        mock_embedding_class.from_config = Mock(return_value=mock_embedding)

        # Create processor manager
        with patch(
            "src.app.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []

            manager = ProcessorManager(mock_logger)
            manager._processor_classes["chunk"] = mock_chunk_class
            manager._processor_classes["audio"] = mock_audio_class
            manager._processor_classes["embedding"] = mock_embedding_class

            # Initialize from strategies
            manager.initialize_from_strategies(chunk_based_strategy_set)

            # Execute chunk-based pipeline
            chunk_processor = manager.get_processor("chunk")
            audio_processor = manager.get_processor("audio")
            embedding_processor = manager.get_processor("embedding")

            # Process video
            chunk_result = chunk_processor.extract_chunks(
                sample_video_path, video_duration=20.0
            )
            audio_result = audio_processor.transcribe_audio(sample_video_path)
            embedding_result = embedding_processor.generate_embeddings(
                {"chunks": chunk_result, "transcript": audio_result}
            )

            # Verify results
            assert chunk_result["total_chunks"] == 2
            assert audio_result["text"] == "Chunk-based audio transcription."
            assert "chunk_embeddings" in embedding_result["embeddings"]

            # Verify processing calls
            mock_chunk.extract_chunks.assert_called_once()
            mock_audio.transcribe_audio.assert_called_once_with(sample_video_path)
            mock_embedding.generate_embeddings.assert_called_once()

    def test_pipeline_error_propagation(
        self, mock_logger, frame_based_strategy_set, temp_dir, sample_video_path
    ):
        """Test that errors in pipeline steps are properly propagated."""

        # Create processor that fails
        class FailingKeyframeProcessor:
            PROCESSOR_NAME = "keyframe"

            def __init__(self, logger, **kwargs):
                self.logger = logger

            @classmethod
            def from_config(cls, config, logger):
                return cls(logger, **config)

            def extract_keyframes(self, *args, **kwargs):
                raise ValueError("Keyframe extraction failed")

        # Add working processors for audio and embedding (test focuses on keyframe failure)
        class WorkingAudioProcessor:
            PROCESSOR_NAME = "audio"

            def __init__(self, logger, **kwargs):
                self.logger = logger

            @classmethod
            def from_config(cls, config, logger):
                return cls(logger, **config)

            def transcribe_audio(self, *args, **kwargs):
                return {"text": "test audio", "language": "en"}

        class WorkingEmbeddingProcessor:
            PROCESSOR_NAME = "embedding"

            def __init__(self, logger, **kwargs):
                self.logger = logger

            @classmethod
            def from_config(cls, config, logger):
                return cls(logger, **config)

            def generate_embeddings(self, *args, **kwargs):
                return {"embeddings": {"frame_embeddings": []}}

        with patch(
            "src.app.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []

            manager = ProcessorManager(mock_logger)
            manager._processor_classes["keyframe"] = FailingKeyframeProcessor
            manager._processor_classes["audio"] = WorkingAudioProcessor
            manager._processor_classes["embedding"] = WorkingEmbeddingProcessor
            manager.initialize_from_strategies(frame_based_strategy_set)

            keyframe_processor = manager.get_processor("keyframe")

            # Pipeline should propagate the error
            with pytest.raises(ValueError, match="Keyframe extraction failed"):
                keyframe_processor.extract_keyframes(sample_video_path, temp_dir)

    def test_pipeline_partial_failure_handling(
        self, mock_logger, frame_based_strategy_set, temp_dir, sample_video_path
    ):
        """Test handling of partial failures in pipeline (some processors succeed, others fail)."""

        # Mock successful keyframe processor
        class SuccessfulKeyframeProcessor:
            PROCESSOR_NAME = "keyframe"

            def __init__(self, logger, **kwargs):
                self.logger = logger

            @classmethod
            def from_config(cls, config, logger):
                return cls(logger, **config)

            def extract_keyframes(self, *args, **kwargs):
                return {
                    "video_id": "test_video",
                    "total_keyframes": 1,
                    "keyframes": [{"frame_index": 0, "timestamp": 0.0}],
                }

        # Mock failing audio processor
        class FailingAudioProcessor:
            PROCESSOR_NAME = "audio"

            def __init__(self, logger, **kwargs):
                self.logger = logger

            @classmethod
            def from_config(cls, config, logger):
                return cls(logger, **config)

            def transcribe_audio(self, *args, **kwargs):
                raise Exception("Audio processing failed")

        # Mock working embedding processor (test focuses on audio failure)
        class WorkingEmbeddingProcessor:
            PROCESSOR_NAME = "embedding"

            def __init__(self, logger, **kwargs):
                self.logger = logger

            @classmethod
            def from_config(cls, config, logger):
                return cls(logger, **config)

            def generate_embeddings(self, *args, **kwargs):
                return {"embeddings": {"frame_embeddings": []}}

        with patch(
            "src.app.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []

            manager = ProcessorManager(mock_logger)
            manager._processor_classes["keyframe"] = SuccessfulKeyframeProcessor
            manager._processor_classes["audio"] = FailingAudioProcessor
            manager._processor_classes["embedding"] = WorkingEmbeddingProcessor

            manager.initialize_from_strategies(frame_based_strategy_set)

            # Keyframe processing should succeed
            keyframe_processor = manager.get_processor("keyframe")
            keyframe_result = keyframe_processor.extract_keyframes(
                sample_video_path, temp_dir
            )
            assert keyframe_result["total_keyframes"] == 1

            # Audio processing should fail
            audio_processor = manager.get_processor("audio")
            with pytest.raises(Exception, match="Audio processing failed"):
                audio_processor.transcribe_audio(sample_video_path)

    @patch("src.app.ingestion.processors.keyframe_processor.KeyframeProcessor")
    @patch("src.app.ingestion.processors.audio_processor.AudioProcessor")
    @patch("src.app.ingestion.processors.embedding_processor.EmbeddingProcessor")
    def test_pipeline_data_flow_validation(
        self,
        mock_embedding_class,
        mock_audio_class,
        mock_keyframe_class,
        mock_logger,
        frame_based_strategy_set,
        temp_dir,
        sample_video_path,
    ):
        """Test that data flows correctly between pipeline stages."""

        # Mock processors with specific return formats
        mock_keyframe = Mock()
        mock_keyframe_result = {
            "video_id": "test_video",
            "total_keyframes": 2,
            "keyframes": [
                {"frame_index": 0, "timestamp": 0.0, "filename": "frame_0.jpg"},
                {"frame_index": 1, "timestamp": 1.0, "filename": "frame_1.jpg"},
            ],
        }
        mock_keyframe.extract_keyframes.return_value = mock_keyframe_result
        mock_keyframe_class.return_value = mock_keyframe
        mock_keyframe_class.from_config = Mock(return_value=mock_keyframe)

        mock_audio = Mock()
        mock_audio_result = {
            "video_id": "test_video",
            "text": "Test audio",
            "segments": [{"start": 0.0, "end": 2.0, "text": "Test audio"}],
        }
        mock_audio.transcribe_audio.return_value = mock_audio_result
        mock_audio_class.return_value = mock_audio
        mock_audio_class.from_config = Mock(return_value=mock_audio)

        # Mock embedding processor
        mock_embedding = Mock()
        mock_embedding_result = {
            "video_id": "test_video",
            "embeddings": {"frame_embeddings": [
                {"frame_index": 0, "embedding": [0.1, 0.2, 0.3]},
                {"frame_index": 1, "embedding": [0.4, 0.5, 0.6]},
            ]},
        }
        mock_embedding.generate_embeddings.return_value = mock_embedding_result
        mock_embedding_class.return_value = mock_embedding
        mock_embedding_class.from_config = Mock(return_value=mock_embedding)

        with patch(
            "src.app.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []

            manager = ProcessorManager(mock_logger)
            manager._processor_classes["keyframe"] = mock_keyframe_class
            manager._processor_classes["audio"] = mock_audio_class
            manager._processor_classes["embedding"] = mock_embedding_class

            manager.initialize_from_strategies(frame_based_strategy_set)

            # Execute pipeline and verify data flow
            keyframe_processor = manager.get_processor("keyframe")
            audio_processor = manager.get_processor("audio")
            embedding_processor = manager.get_processor("embedding")

            # Step 1: Extract keyframes
            keyframes = keyframe_processor.extract_keyframes(
                sample_video_path, temp_dir
            )

            # Step 2: Process audio
            transcript = audio_processor.transcribe_audio(sample_video_path)

            # Step 3: Generate embeddings
            embeddings = embedding_processor.generate_embeddings({
                "keyframes": keyframes,
                "transcript": transcript
            })

            # Verify data consistency
            assert keyframes["video_id"] == transcript["video_id"] == embeddings["video_id"]
            assert keyframes["video_id"] == "test_video"

            # Verify data format compliance
            assert "total_keyframes" in keyframes
            assert "keyframes" in keyframes
            assert isinstance(keyframes["keyframes"], list)

            assert "text" in transcript
            assert "segments" in transcript
            assert isinstance(transcript["segments"], list)

    def test_pipeline_configuration_consistency(self, mock_logger, temp_dir):
        """Test that pipeline configurations are applied consistently across processors."""

        # Create strategy with specific configuration
        frame_strategy = FrameSegmentationStrategy(
            max_frames=15, fps=2.0, threshold=0.95
        )
        strategy_set = ProcessingStrategySet(segmentation=frame_strategy)

        with patch(
            "src.app.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []

            # Mock processor that stores configuration
            class ConfigurableProcessor:
                PROCESSOR_NAME = "keyframe"

                def __init__(self, logger, **kwargs):
                    self.logger = logger
                    self.config = kwargs

                @classmethod
                def from_config(cls, config, logger):
                    return cls(logger, **config)

                def get_config(self):
                    return self.config

            manager = ProcessorManager(mock_logger)
            manager._processor_classes["keyframe"] = ConfigurableProcessor
            manager.initialize_from_strategies(strategy_set)

            # Verify processor got correct configuration
            processor = manager.get_processor("keyframe")
            config = processor.get_config()

            assert config["max_frames"] == 15
            assert config["fps"] == 2.0
            assert config["threshold"] == 0.95

    def test_pipeline_concurrent_processing_safety(
        self, mock_logger, frame_based_strategy_set, temp_dir, sample_video_path
    ):
        """Test pipeline safety under concurrent processing scenarios."""
        import threading
        import time

        # Mock thread-safe processor
        class ThreadSafeProcessor:
            PROCESSOR_NAME = "keyframe"

            def __init__(self, logger, **kwargs):
                self.logger = logger
                self.call_count = 0
                self._lock = threading.Lock()

            @classmethod
            def from_config(cls, config, logger):
                return cls(logger, **config)

            def extract_keyframes(self, *args, **kwargs):
                with self._lock:
                    self.call_count += 1
                    time.sleep(0.01)  # Simulate processing time
                    return {
                        "video_id": f"video_{self.call_count}",
                        "total_keyframes": 1,
                    }

        # Mock working audio processor
        class WorkingAudioProcessor:
            PROCESSOR_NAME = "audio"

            def __init__(self, logger, **kwargs):
                self.logger = logger

            @classmethod
            def from_config(cls, config, logger):
                return cls(logger, **config)

            def transcribe_audio(self, *args, **kwargs):
                return {"text": "test audio", "language": "en"}

        # Mock working embedding processor  
        class WorkingEmbeddingProcessor:
            PROCESSOR_NAME = "embedding"

            def __init__(self, logger, **kwargs):
                self.logger = logger

            @classmethod
            def from_config(cls, config, logger):
                return cls(logger, **config)

            def generate_embeddings(self, *args, **kwargs):
                return {"embeddings": {"frame_embeddings": []}}

        with patch(
            "src.app.ingestion.processor_manager.pkgutil.iter_modules"
        ) as mock_iter:
            mock_iter.return_value = []

            manager = ProcessorManager(mock_logger)
            manager._processor_classes["keyframe"] = ThreadSafeProcessor
            manager._processor_classes["audio"] = WorkingAudioProcessor
            manager._processor_classes["embedding"] = WorkingEmbeddingProcessor
            manager.initialize_from_strategies(frame_based_strategy_set)

            processor = manager.get_processor("keyframe")
            results = []

            def process_video():
                result = processor.extract_keyframes(sample_video_path, temp_dir)
                results.append(result)

            # Run multiple threads concurrently
            threads = [threading.Thread(target=process_video) for _ in range(3)]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            # All threads should complete successfully
            assert len(results) == 3

            # Results should have unique video IDs (indicating separate processing)
            video_ids = [result["video_id"] for result in results]
            assert len(set(video_ids)) == 3
