"""
REAL Integration tests for video ingestion pipeline.

These tests use ACTUAL:
- Video files (no mocks)
- Processors (real keyframe extraction, transcription, embeddings)
- Vespa backend (real document creation)

NO MOCKS - this tests the real end-to-end pipeline.
"""

import pytest
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mark all tests as integration and requiring models
pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_models,
    pytest.mark.slow,
]


@pytest.mark.integration
@pytest.mark.requires_cv2
class TestRealIngestionPipeline:
    """Real integration tests with actual video processing."""

    @pytest.fixture
    def test_video_path(self):
        """Get path to smallest test video."""
        # Use the smallest available test video
        video_path = Path("data/testset/evaluation/sample_videos/v_-6dz6tBH77I.mp4")
        if not video_path.exists():
            pytest.skip(f"Test video not found: {video_path}")
        return video_path

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "ingestion_output"
        output_dir.mkdir()
        yield output_dir
        # Cleanup
        if output_dir.exists():
            shutil.rmtree(output_dir)

    def test_real_keyframe_extraction(self, test_video_path, temp_output_dir):
        """
        REAL TEST: Extract keyframes from actual video using OpenCV.

        No mocks - tests real KeyframeProcessor with real video.
        """
        from src.app.ingestion.processors.keyframe_processor import KeyframeProcessor
        import logging

        # Create real keyframe processor with minimal config
        config = {
            "method": "uniform",  # Simple uniform sampling
            "max_frames": 3,  # Only 3 frames for speed
            "output_dir": str(temp_output_dir),
        }

        # Create logger
        logger = logging.getLogger("test_keyframe")

        processor = KeyframeProcessor(logger)
        processor.configure(config)

        # Extract keyframes from REAL video
        result = processor.extract_keyframes(test_video_path)

        # Validate real output
        assert result is not None
        assert "keyframes" in result
        assert len(result["keyframes"]) <= 3  # Should respect max_frames
        assert "total_keyframes" in result
        assert "fps" in result

        # Verify actual image files were created
        for keyframe in result["keyframes"]:
            frame_path = Path(keyframe["filename"])
            assert frame_path.exists(), f"Frame file should exist: {frame_path}"
            assert frame_path.stat().st_size > 0, "Frame file should not be empty"

    @pytest.mark.requires_whisper
    def test_real_audio_transcription(self, test_video_path):
        """
        REAL TEST: Transcribe audio from actual video using Whisper.

        No mocks - tests real AudioProcessor with real video.
        """
        from src.app.ingestion.processors.audio_processor import AudioProcessor

        # Create real audio processor with minimal config
        config = {
            "model": "base",  # Smallest Whisper model
            "language": "en",
        }

        processor = AudioProcessor(config)

        # Transcribe REAL audio
        result = processor.transcribe_audio(test_video_path)

        # Validate real output
        assert result is not None
        assert "text" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 0  # Should have some transcription
        assert "language" in result

    @pytest.mark.skip(reason="Embedding generation too slow for regular CI")
    def test_real_embedding_generation_colpali(self, test_video_path, temp_output_dir):
        """
        REAL TEST: Generate ColPali embeddings from actual video frames.

        SKIPPED by default due to model download/inference time.
        Run with: pytest -v -m "not skip" to include this test.
        """
        from src.app.ingestion.processors.embedding_processor import EmbeddingProcessor

        # First extract keyframes
        from src.app.ingestion.processors.keyframe_processor import KeyframeProcessor

        keyframe_config = {
            "method": "uniform",
            "max_frames": 2,  # Only 2 frames for speed
            "output_dir": str(temp_output_dir),
        }

        keyframe_processor = KeyframeProcessor(keyframe_config)
        keyframe_result = keyframe_processor.extract_keyframes(test_video_path)

        # Generate real embeddings
        embedding_config = {
            "model_type": "colpali",
            "model_name": "vidore/colpali-v1.2-hf",  # Real ColPali model
        }

        embedding_processor = EmbeddingProcessor(embedding_config)

        # This will download model on first run and generate REAL embeddings
        result = embedding_processor.generate_embeddings(
            video_id=test_video_path.stem, keyframe_result=keyframe_result
        )

        # Validate real embeddings
        assert result is not None
        assert "embeddings" in result
        assert len(result["embeddings"]) > 0

    def test_real_strategy_resolution_frame_based(self):
        """
        REAL TEST: Test strategy resolution for frame-based processing.

        Tests that StrategyConfig correctly resolves processing strategies.
        """
        from src.app.processing.strategy import StrategyConfig, ProcessingType

        # Create frame-based strategy config
        strategy_config = StrategyConfig(
            processing_type=ProcessingType.FRAME_BASED,
            embedding_model="colpali",
            chunk_duration_seconds=None,  # Not used for frame-based
            segment_duration_seconds=None,
        )

        # Resolve strategies
        strategy_set = strategy_config.resolve_strategies()

        # Validate strategy resolution
        assert strategy_set is not None
        assert strategy_set.segmentation_strategy is not None
        assert strategy_set.transcription_strategy is not None
        assert strategy_set.description_strategy is not None

        # Verify segmentation is frame-based
        assert "frame" in strategy_set.segmentation_strategy.method.lower()

    def test_real_strategy_resolution_chunk_based(self):
        """
        REAL TEST: Test strategy resolution for chunk-based processing.
        """
        from src.app.processing.strategy import StrategyConfig, ProcessingType

        # Create chunk-based strategy config
        strategy_config = StrategyConfig(
            processing_type=ProcessingType.CHUNK_BASED,
            embedding_model="videoprism",
            chunk_duration_seconds=30.0,
            segment_duration_seconds=None,
        )

        # Resolve strategies
        strategy_set = strategy_config.resolve_strategies()

        # Validate strategy resolution
        assert strategy_set is not None
        assert strategy_set.segmentation_strategy is not None

        # Verify segmentation is chunk-based
        assert (
            "chunk" in strategy_set.segmentation_strategy.method.lower()
            or "window" in strategy_set.segmentation_strategy.method.lower()
        )
        assert strategy_set.segmentation_strategy.chunk_duration == 30.0

    @pytest.mark.skip(reason="Full pipeline too slow - use for manual testing only")
    @pytest.mark.requires_vespa
    @pytest.mark.asyncio
    async def test_real_end_to_end_ingestion_to_vespa(
        self, test_video_path, temp_output_dir
    ):
        """
        REAL E2E TEST: Complete ingestion pipeline to Vespa.

        This test:
        1. Extracts keyframes from real video
        2. Transcribes audio with Whisper
        3. Generates embeddings (mocked for speed)
        4. Creates Vespa documents
        5. Feeds to real Vespa instance
        6. Queries and verifies results

        SKIPPED by default - too slow for CI.
        """
        from src.app.ingestion.pipeline import IngestionPipeline
        from src.backends.vespa.client import VespaSearchClient

        # Create minimal pipeline config
        pipeline_config = {
            "keyframe": {"method": "uniform", "max_frames": 3},
            "audio": {"model": "base"},
            "embedding": {
                "model_type": "mock"
            },  # Mock embeddings for speed in full pipeline test
            "output_dir": str(temp_output_dir),
        }

        # Initialize pipeline
        pipeline = IngestionPipeline(pipeline_config)

        # Run REAL ingestion
        result = await pipeline.process_video_async(test_video_path)

        # Validate result
        assert result is not None
        assert "video_id" in result
        assert "documents" in result
        assert len(result["documents"]) > 0

        # Verify Vespa integration
        vespa_client = VespaSearchClient()

        # Feed documents to Vespa
        for doc in result["documents"]:
            response = vespa_client.feed_document(doc)
            assert (
                response.status_code == 200
            ), f"Vespa feed failed: {response.text}"

        # Query Vespa to verify
        search_results = vespa_client.search(
            query="test", schema="video_frames", hits=10
        )

        assert search_results is not None
        assert len(search_results) > 0

    def test_document_builder_formats(self):
        """
        REAL TEST: Test document builders create correct formats.
        """
        from src.app.ingestion.document_builder import DocumentBuilder
        from src.common.document import UniversalDocument

        # Mock processed data (structure is real, values can be mock)
        processed_data = {
            "video_id": "test_video",
            "keyframes": [
                {"frame_index": 0, "timestamp": 0.0, "filename": "/tmp/frame_0.jpg"}
            ],
            "transcription": {
                "text": "Test transcription",
                "segments": [{"start": 0.0, "end": 1.0, "text": "Test"}],
            },
            "embeddings": [{"frame_index": 0, "embedding": [0.1] * 128}],
        }

        # Test multi-document format
        builder = DocumentBuilder(storage_format="multi_doc")
        documents = builder.build_documents(processed_data)

        assert isinstance(documents, list)
        assert len(documents) > 0
        assert all(isinstance(doc, UniversalDocument) for doc in documents)

        # Validate document structure
        for doc in documents:
            assert doc.doc_id is not None
            assert doc.media_type in ["video_frame", "video_audio", "video_chunk"]
            assert doc.embeddings is not None or doc.text is not None

        # Test single-document format
        builder = DocumentBuilder(storage_format="single_doc")
        documents = builder.build_documents(processed_data)

        assert isinstance(documents, list)
        assert len(documents) == 1  # Should be single document
        assert isinstance(documents[0], UniversalDocument)


@pytest.mark.unit
class TestIngestionConfigValidation:
    """Unit tests for ingestion configuration validation."""

    def test_strategy_config_validation(self):
        """Test that StrategyConfig validates inputs correctly."""
        from src.app.processing.strategy import StrategyConfig, ProcessingType

        # Valid config
        config = StrategyConfig(
            processing_type=ProcessingType.FRAME_BASED, embedding_model="colpali"
        )
        assert config.processing_type == ProcessingType.FRAME_BASED

        # Test validation of chunk duration for chunk-based processing
        with pytest.raises(ValueError):
            StrategyConfig(
                processing_type=ProcessingType.CHUNK_BASED,
                embedding_model="videoprism",
                chunk_duration_seconds=None,  # Should raise - required for chunk-based
            )

    def test_profile_loading(self):
        """Test that profiles can be loaded correctly."""
        from src.app.ingestion.config import ProfileLoader

        # Test loading existing profile
        loader = ProfileLoader()

        # This should not raise
        profiles = loader.list_profiles()
        assert isinstance(profiles, list)
        # At least one profile should exist
        assert len(profiles) > 0
