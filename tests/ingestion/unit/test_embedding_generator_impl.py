"""
Comprehensive unit tests for EmbeddingGeneratorImpl to improve coverage.

Tests the unified embedding generation logic with proper mocking.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from cogniverse_runtime.ingestion.processors.embedding_generator.embedding_generator import (
    EmbeddingResult,
)
from cogniverse_runtime.ingestion.processors.embedding_generator.embedding_generator_impl import (
    EmbeddingGeneratorImpl,
)


@pytest.mark.unit
@pytest.mark.ci_safe
class TestEmbeddingGeneratorImpl:
    """Tests for EmbeddingGeneratorImpl."""

    @pytest.fixture
    def mock_logger(self):
        return Mock()

    @pytest.fixture
    def mock_backend_client(self):
        client = Mock()
        client.feed_document.return_value = True
        return client

    @pytest.fixture
    def basic_config(self):
        return {
            "schema_name": "test_schema",
            "embedding_model": "test_model",
            "storage_mode": "multi_doc",
            "embedding_type": "video_chunks",
            "model_loader": "colqwen",
        }

    @pytest.fixture
    def frame_based_config(self):
        return {
            "schema_name": "test_schema",
            "embedding_model": "test_model",
            "storage_mode": "multi_doc",
            "embedding_type": "frame_based",
            "model_loader": "colpali",
        }

    @patch("cogniverse_core.common.models.get_or_load_model")
    def test_initialization_with_model_load(
        self, mock_get_model, basic_config, mock_logger, mock_backend_client
    ):
        """Test EmbeddingGeneratorImpl initialization with model loading."""
        mock_model = Mock()
        mock_processor = Mock()
        mock_get_model.return_value = (mock_model, mock_processor)

        generator = EmbeddingGeneratorImpl(
            basic_config, mock_logger, mock_backend_client
        )

        assert generator.profile_config == basic_config
        assert generator.model_name == "test_model"
        assert generator.backend_client == mock_backend_client
        assert generator.schema_name == "test_schema"
        assert generator.storage_mode == "multi_doc"
        assert generator.model == mock_model
        assert generator.processor == mock_processor

        mock_get_model.assert_called_once_with("test_model", basic_config, mock_logger)

    @patch("cogniverse_core.common.models.get_or_load_model")
    def test_initialization_frame_based_no_model_load(
        self, mock_get_model, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test EmbeddingGeneratorImpl initialization without model loading for frame-based."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        assert generator.model is None
        assert generator.processor is None
        assert generator.videoprism_loader is None

        # Model should not be loaded for frame_based
        mock_get_model.assert_not_called()

    @patch("cogniverse_core.common.models.get_or_load_model")
    def test_initialization_videoprism_model(
        self, mock_get_model, mock_logger, mock_backend_client
    ):
        """Test EmbeddingGeneratorImpl initialization with VideoPrism model."""
        config = {
            "schema_name": "test_schema",
            "embedding_model": "videoprism_large",
            "embedding_type": "direct_video_segment",
            "model_loader": "videoprism",
        }

        mock_loader = Mock()
        mock_get_model.return_value = (mock_loader, None)

        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)

        assert generator.videoprism_loader == mock_loader
        assert generator.model is None

    @patch("cogniverse_core.common.models.get_or_load_model")
    def test_load_model_error_handling(
        self, mock_get_model, basic_config, mock_logger, mock_backend_client
    ):
        """Test _load_model error handling."""
        mock_get_model.side_effect = Exception("Model loading failed")

        with pytest.raises(Exception, match="Model loading failed"):
            EmbeddingGeneratorImpl(basic_config, mock_logger, mock_backend_client)

        mock_logger.error.assert_called_with(
            "Failed to load model: Model loading failed"
        )

    def test_should_load_model_logic(self, mock_logger, mock_backend_client):
        """Test _should_load_model logic for different embedding types."""
        # Frame-based should NOT load model
        config = {"embedding_type": "frame_based", "embedding_model": "vidore/colsmol-500m", "model_loader": "colpali"}
        with patch(
            "cogniverse_core.common.models.get_or_load_model",
            return_value=(Mock(), Mock()),
        ):
            generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)
            assert generator._should_load_model() is False

        # Other valid types should load model
        test_cases = [
            ("video_chunks", "colqwen"),
            ("direct_video_segment", "videoprism"),
            ("single_vector", "videoprism"),
            ("document_colbert", "colbert"),
        ]
        for embedding_type, model_loader in test_cases:
            config = {"embedding_type": embedding_type, "embedding_model": "test_model", "model_loader": model_loader}
            with patch(
                "cogniverse_core.common.models.get_or_load_model",
                return_value=(Mock(), Mock()),
            ):
                generator = EmbeddingGeneratorImpl(
                    config, mock_logger, mock_backend_client
                )
                assert (
                    generator._should_load_model() is True
                ), f"Should load model for {embedding_type}"

    def test_extract_segments_basic(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _extract_segments with basic segments key."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        video_data = {
            "segments": [
                {"start_time": 0.0, "end_time": 10.0},
                {"start_time": 10.0, "end_time": 20.0},
            ]
        }

        segments = generator._extract_segments(video_data)

        assert len(segments) == 2
        assert segments[0]["start_time"] == 0.0
        assert segments[1]["end_time"] == 20.0

    def test_extract_segments_keyframes(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _extract_segments with keyframes key."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        video_data = {
            "keyframes": [
                {"timestamp": 0.0, "frame_data": "frame1"},
                {"timestamp": 5.0, "frame_data": "frame2"},
            ]
        }

        segments = generator._extract_segments(video_data)

        assert len(segments) == 2
        assert segments[0]["timestamp"] == 0.0
        assert segments[1]["frame_data"] == "frame2"

    def test_extract_segments_nested_structure(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _extract_segments with nested dict structure."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        video_data = {
            "frames": {
                "keyframes": [
                    {"timestamp": 0.0, "frame_data": "frame1"},
                    {"timestamp": 5.0, "frame_data": "frame2"},
                ]
            }
        }

        segments = generator._extract_segments(video_data)

        assert len(segments) == 2
        assert segments[0]["timestamp"] == 0.0

    def test_extract_segments_single_vector_processing(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _extract_segments with single_vector_processing structure."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        video_data = {
            "single_vector_processing": {
                "segments": [
                    {"start_time": 0.0, "end_time": 30.0},
                    {"start_time": 30.0, "end_time": 60.0},
                ]
            }
        }

        segments = generator._extract_segments(video_data)

        assert len(segments) == 2
        assert segments[0]["start_time"] == 0.0
        assert segments[1]["end_time"] == 60.0

    def test_extract_segments_no_segments_found(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _extract_segments when no segments found."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        video_data = {"video_id": "test", "other_data": "value"}

        segments = generator._extract_segments(video_data)

        assert segments == []

    def test_generate_embeddings_no_segments(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test generate_embeddings when no segments found."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        video_data = {"video_id": "test_video"}

        result = generator.generate_embeddings(video_data, Path("/tmp"))

        assert result.video_id == "test_video"
        assert result.total_documents == 0
        assert result.documents_processed == 0
        assert result.documents_fed == 0
        assert "No segments found in video_data" in result.errors
        assert result.processing_time >= 0

    def test_generate_embeddings_multi_doc_mode(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test generate_embeddings in multi-document mode."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        video_data = {
            "video_id": "test_video",
            "segments": [
                {"start_time": 0.0, "end_time": 10.0},
                {"start_time": 10.0, "end_time": 20.0},
            ],
            "storage_mode": "multi_doc",
        }

        expected_result = EmbeddingResult(
            video_id="test_video",
            total_documents=2,
            documents_processed=2,
            documents_fed=2,
            processing_time=0,
            errors=[],
            metadata={"num_segments": 2},
        )

        with patch.object(
            generator, "_process_multi_documents", return_value=expected_result
        ) as mock_multi:
            result = generator.generate_embeddings(video_data, Path("/tmp"))

            mock_multi.assert_called_once()
            assert result.video_id == "test_video"
            assert result.total_documents == 2
            assert result.processing_time > 0  # Should be set by generate_embeddings

    def test_generate_embeddings_single_doc_mode(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test generate_embeddings in single-document mode."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        video_data = {
            "video_id": "test_video",
            "segments": [
                {"start_time": 0.0, "end_time": 10.0},
                {"start_time": 10.0, "end_time": 20.0},
            ],
            "storage_mode": "single_doc",
        }

        expected_result = EmbeddingResult(
            video_id="test_video",
            total_documents=1,
            documents_processed=1,
            documents_fed=1,
            processing_time=0,
            errors=[],
            metadata={"num_segments": 2},
        )

        with patch.object(
            generator, "_process_single_document", return_value=expected_result
        ) as mock_single:
            result = generator.generate_embeddings(video_data, Path("/tmp"))

            mock_single.assert_called_once()
            assert result.video_id == "test_video"
            assert result.total_documents == 1

    def test_process_multi_documents_success(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _process_multi_documents successful processing."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        video_data = {
            "video_id": "test_video",
            "video_path": "/path/to/video.mp4",
            "transcript": {"full_text": "Test transcript"},
            "descriptions": {"0": "First segment", "1": "Second segment"},
        }

        segments = [
            {"start_time": 0.0, "end_time": 10.0},
            {"start_time": 10.0, "end_time": 20.0},
        ]

        # Mock dependencies
        mock_embeddings = np.random.rand(128)
        mock_doc = {"id": "test_doc"}

        with (
            patch.object(
                generator, "_extract_transcript_text", return_value="Test transcript"
            ),
            patch.object(
                generator, "_generate_segment_embeddings", return_value=mock_embeddings
            ),
            patch.object(generator, "_create_segment_document", return_value=mock_doc),
            patch.object(generator, "_feed_document", return_value=True),
        ):

            result = generator._process_multi_documents(video_data, segments)

            assert result.video_id == "test_video"
            assert result.total_documents == 2
            assert result.documents_processed == 2
            assert result.documents_fed == 2
            assert len(result.errors) == 0

    def test_process_multi_documents_segment_error(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _process_multi_documents with segment processing error."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        video_data = {"video_id": "test_video", "video_path": "/path/to/video.mp4"}

        segments = [
            {"start_time": 0.0, "end_time": 10.0},
            {"start_time": 10.0, "end_time": 20.0},
        ]

        with (
            patch.object(generator, "_extract_transcript_text", return_value=""),
            patch.object(
                generator,
                "_generate_segment_embeddings",
                side_effect=Exception("Embedding error"),
            ),
        ):

            result = generator._process_multi_documents(video_data, segments)

            assert result.video_id == "test_video"
            assert result.documents_processed == 0
            assert "Segment 0: Embedding error" in result.errors
            assert "Segment 1: Embedding error" in result.errors

    def test_process_multi_documents_no_embeddings(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _process_multi_documents when no embeddings generated."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        video_data = {"video_id": "test_video", "video_path": "/path/to/video.mp4"}

        segments = [{"start_time": 0.0, "end_time": 10.0}]

        with (
            patch.object(generator, "_extract_transcript_text", return_value=""),
            patch.object(generator, "_generate_segment_embeddings", return_value=None),
        ):

            result = generator._process_multi_documents(video_data, segments)

            assert result.documents_processed == 0
            assert result.documents_fed == 0

    def test_process_single_document_success(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _process_single_document successful processing."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        video_data = {"video_id": "test_video", "video_path": "/path/to/video.mp4"}

        segments = [
            {"start_time": 0.0, "end_time": 10.0},
            {"start_time": 10.0, "end_time": 20.0},
        ]

        # Mock embeddings with different shapes to test stacking
        mock_embeddings1 = np.random.rand(1, 128)
        mock_embeddings2 = np.random.rand(1, 128)
        mock_doc = {"id": "test_doc"}

        with (
            patch.object(
                generator,
                "_generate_segment_embeddings",
                side_effect=[mock_embeddings1, mock_embeddings2],
            ),
            patch.object(generator, "_create_combined_document", return_value=mock_doc),
            patch.object(generator, "_feed_document", return_value=True),
        ):

            result = generator._process_single_document(video_data, segments)

            assert result.video_id == "test_video"
            assert result.total_documents == 1
            assert result.documents_processed == 1
            assert result.documents_fed == 1
            assert len(result.errors) == 0

    def test_process_single_document_no_embeddings(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _process_single_document when no embeddings generated."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        video_data = {"video_id": "test_video", "video_path": "/path/to/video.mp4"}

        segments = [{"start_time": 0.0, "end_time": 10.0}]

        with patch.object(generator, "_generate_segment_embeddings", return_value=None):

            result = generator._process_single_document(video_data, segments)

            assert result.video_id == "test_video"
            assert result.total_documents == 1
            assert result.documents_processed == 0
            assert result.documents_fed == 0
            assert "No embeddings generated" in result.errors

    def test_process_single_document_single_embedding(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _process_single_document with only one embedding (no stacking)."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        video_data = {"video_id": "test_video", "video_path": "/path/to/video.mp4"}

        segments = [{"start_time": 0.0, "end_time": 10.0}]

        mock_embeddings = np.random.rand(1, 128)
        mock_doc = {"id": "test_doc"}

        with (
            patch.object(
                generator, "_generate_segment_embeddings", return_value=mock_embeddings
            ),
            patch.object(
                generator, "_create_combined_document", return_value=mock_doc
            ) as mock_create,
            patch.object(generator, "_feed_document", return_value=True),
        ):

            result = generator._process_single_document(video_data, segments)

            assert result.documents_processed == 1
            # Verify that combined embeddings is the single embedding (no stacking)
            mock_create.assert_called_once()
            call_args = mock_create.call_args[1]
            np.testing.assert_array_equal(call_args["embeddings"], mock_embeddings)

    def test_process_single_document_segment_error(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _process_single_document with segment processing error."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        video_data = {"video_id": "test_video", "video_path": "/path/to/video.mp4"}

        segments = [
            {"start_time": 0.0, "end_time": 10.0},
            {"start_time": 10.0, "end_time": 20.0},
        ]

        # First segment succeeds, second fails
        mock_embeddings = np.random.rand(1, 128)
        mock_doc = {"id": "test_doc"}

        with (
            patch.object(
                generator,
                "_generate_segment_embeddings",
                side_effect=[mock_embeddings, Exception("Segment error")],
            ),
            patch.object(generator, "_create_combined_document", return_value=mock_doc),
            patch.object(generator, "_feed_document", return_value=True),
        ):

            result = generator._process_single_document(video_data, segments)

            assert (
                result.documents_processed == 1
            )  # Still creates document with successful embeddings
            assert result.documents_fed == 1
            assert "Segment 1: Segment error" in result.errors

    def test_generate_segment_embeddings_chunk_path(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _generate_segment_embeddings with chunk_path."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        segment = {"chunk_path": "/path/to/chunk.mp4"}
        mock_embeddings = np.random.rand(128)

        with patch.object(
            generator, "_generate_chunk_embeddings", return_value=mock_embeddings
        ) as mock_chunk:
            result = generator._generate_segment_embeddings(
                segment, Path("/video.mp4"), {}
            )

            mock_chunk.assert_called_once_with(Path("/path/to/chunk.mp4"))
            np.testing.assert_array_equal(result, mock_embeddings)

    def test_generate_segment_embeddings_video_path(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _generate_segment_embeddings with video file path."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        segment = {"path": "/path/to/video.mp4"}
        mock_embeddings = np.random.rand(128)

        with patch.object(
            generator, "_generate_chunk_embeddings", return_value=mock_embeddings
        ) as mock_chunk:
            result = generator._generate_segment_embeddings(
                segment, Path("/video.mp4"), {}
            )

            mock_chunk.assert_called_once_with(Path("/path/to/video.mp4"))
            np.testing.assert_array_equal(result, mock_embeddings)

    def test_generate_segment_embeddings_image_path(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _generate_segment_embeddings with image file path."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        segment = {"path": "/path/to/frame.jpg"}
        mock_embeddings = np.random.rand(128)

        with patch.object(
            generator, "_generate_frame_embeddings", return_value=mock_embeddings
        ) as mock_frame:
            result = generator._generate_segment_embeddings(
                segment, Path("/video.mp4"), {}
            )

            mock_frame.assert_called_once_with(Path("/path/to/frame.jpg"))
            np.testing.assert_array_equal(result, mock_embeddings)

    def test_generate_segment_embeddings_frame_path(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _generate_segment_embeddings with frame_path."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        segment = {"frame_path": "/path/to/frame.png"}
        mock_embeddings = np.random.rand(128)

        with patch.object(
            generator, "_generate_frame_embeddings", return_value=mock_embeddings
        ) as mock_frame:
            result = generator._generate_segment_embeddings(
                segment, Path("/video.mp4"), {}
            )

            mock_frame.assert_called_once_with(Path("/path/to/frame.png"))
            np.testing.assert_array_equal(result, mock_embeddings)

    def test_generate_segment_embeddings_time_segment(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _generate_segment_embeddings with time-based segment."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        segment = {"start_time": 10.0, "end_time": 20.0}
        mock_embeddings = np.random.rand(128)

        with patch.object(
            generator, "_generate_time_segment_embeddings", return_value=mock_embeddings
        ) as mock_time:
            result = generator._generate_segment_embeddings(
                segment, Path("/video.mp4"), {}
            )

            mock_time.assert_called_once_with(Path("/video.mp4"), 10.0, 20.0)
            np.testing.assert_array_equal(result, mock_embeddings)

    def test_generate_segment_embeddings_unknown_type(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _generate_segment_embeddings with unknown segment type."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        segment = {"unknown_field": "unknown_value"}

        result = generator._generate_segment_embeddings(segment, Path("/video.mp4"), {})

        assert result is None
        mock_logger.warning.assert_called_with(
            "Unknown segment type: dict_keys(['unknown_field'])"
        )

    @patch("torch.no_grad")
    @patch("PIL.Image.open")
    def test_generate_frame_embeddings_success(
        self,
        mock_image_open,
        mock_no_grad,
        frame_based_config,
        mock_logger,
        mock_backend_client,
    ):
        """Test _generate_frame_embeddings successful processing."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )
        generator.model = Mock()
        generator.processor = Mock()
        generator.model.device = "cpu"

        # Mock image and processing
        mock_image = Mock()
        mock_image_open.return_value.convert.return_value = mock_image

        # Mock processor chain - use real dict for batch inputs
        mock_batch_inputs = {"input_ids": Mock(), "attention_mask": Mock()}
        mock_batch_inputs_wrapper = Mock()
        mock_batch_inputs_wrapper.to.return_value = mock_batch_inputs
        generator.processor.process_images.return_value = mock_batch_inputs_wrapper

        # Mock model output
        mock_embeddings = Mock()
        mock_embeddings.cpu.return_value.to.return_value.numpy.return_value = (
            np.random.rand(1, 128)
        )
        generator.model.return_value = mock_embeddings

        # Mock torch.no_grad as proper context manager
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=None)
        context_manager.__exit__ = Mock(return_value=None)
        mock_no_grad.return_value = context_manager

        result = generator._generate_frame_embeddings(Path("/path/to/frame.jpg"))

        assert result is not None
        assert result.shape == (1, 128)
        mock_image_open.assert_called_once_with(Path("/path/to/frame.jpg"))
        generator.processor.process_images.assert_called_once_with([mock_image])

    @patch("PIL.Image.open")
    def test_generate_frame_embeddings_no_model(
        self, mock_image_open, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _generate_frame_embeddings when model not loaded."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )

        with patch.object(generator, "_load_model") as mock_load:
            # Model still None after load attempt
            result = generator._generate_frame_embeddings(Path("/path/to/frame.jpg"))

            assert result is None
            mock_load.assert_called_once()
            mock_logger.error.assert_called_with("Model or processor not loaded")

    @patch("PIL.Image.open")
    def test_generate_frame_embeddings_error(
        self, mock_image_open, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _generate_frame_embeddings error handling."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )
        generator.model = Mock()
        generator.processor = Mock()

        mock_image_open.side_effect = Exception("Image load error")

        result = generator._generate_frame_embeddings(Path("/path/to/frame.jpg"))

        assert result is None
        mock_logger.error.assert_called_with(
            "Error generating frame embeddings: Image load error"
        )

    @patch("cogniverse_core.common.models.get_or_load_model")
    @patch("torch.no_grad")
    @patch("cv2.cvtColor")
    @patch("PIL.Image.fromarray")
    @patch("cv2.VideoCapture")
    def test_generate_chunk_embeddings_colqwen(
        self,
        mock_video_capture,
        mock_from_array,
        mock_cvt_color,
        mock_no_grad,
        mock_get_model,
        frame_based_config,
        mock_logger,
        mock_backend_client,
    ):
        """Test _generate_chunk_embeddings with ColQwen model."""
        mock_get_model.return_value = (Mock(), Mock())
        config = {**frame_based_config, "fps": 1.0, "embedding_type": "video_chunks", "model_loader": "colqwen"}
        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)
        generator.model_name = "colqwen_test"
        generator.model = Mock()
        generator.processor = Mock()
        generator.model.device = "cpu"

        # Mock video capture
        mock_cap = Mock()
        mock_cap.get.side_effect = [25.0, 100]  # fps, total_frames
        mock_cap.read.return_value = (True, np.random.rand(100, 100, 3))
        mock_video_capture.return_value = mock_cap

        # Mock image processing
        mock_pil_image = Mock()
        mock_from_array.return_value = mock_pil_image
        mock_cvt_color.return_value = np.random.rand(100, 100, 3)

        # Mock batch processing - use real dict for batch inputs
        mock_batch_inputs = {"input_ids": Mock(), "attention_mask": Mock()}
        mock_batch_inputs_wrapper = Mock()
        mock_batch_inputs_wrapper.to.return_value = mock_batch_inputs
        generator.processor.process_images.return_value = mock_batch_inputs_wrapper

        # Mock model output
        mock_embeddings = Mock()
        mock_embeddings.cpu.return_value.numpy.return_value = np.random.rand(10, 128)
        generator.model.return_value = mock_embeddings

        # Mock torch.no_grad as proper context manager
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=None)
        context_manager.__exit__ = Mock(return_value=None)
        mock_no_grad.return_value = context_manager

        result = generator._generate_chunk_embeddings(Path("/path/to/chunk.mp4"))

        assert result is not None
        assert result.shape == (128,)  # Averaged across frames
        mock_cap.release.assert_called_once()

    @patch("subprocess.run")
    def test_generate_chunk_embeddings_videoprism(
        self, mock_subprocess, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _generate_chunk_embeddings with VideoPrism model."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )
        generator.videoprism_loader = Mock()

        # Mock ffprobe output
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "30.5"
        mock_subprocess.return_value = mock_result

        # Mock videoprism processing
        mock_embeddings = np.random.rand(768)
        generator.videoprism_loader.process_video_segment.return_value = {
            "embeddings_np": mock_embeddings
        }

        result = generator._generate_chunk_embeddings(Path("/path/to/chunk.mp4"))

        np.testing.assert_array_equal(result, mock_embeddings)
        generator.videoprism_loader.process_video_segment.assert_called_once_with(
            Path("/path/to/chunk.mp4"), 0, 30.5
        )

    @patch("cogniverse_core.common.models.get_or_load_model")
    @patch("cv2.VideoCapture")
    def test_generate_chunk_embeddings_no_frames(
        self,
        mock_video_capture,
        mock_get_model,
        frame_based_config,
        mock_logger,
        mock_backend_client,
    ):
        """Test _generate_chunk_embeddings when no frames extracted."""
        mock_get_model.return_value = (Mock(), Mock())
        config = {**frame_based_config, "embedding_type": "video_chunks", "model_loader": "colqwen"}
        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)
        generator.model_name = "colqwen_test"
        generator.model = Mock()
        generator.processor = Mock()

        # Mock video capture with no frames
        mock_cap = Mock()
        mock_cap.get.side_effect = [25.0, 100]
        mock_cap.read.return_value = (False, None)  # No frames
        mock_video_capture.return_value = mock_cap

        result = generator._generate_chunk_embeddings(Path("/path/to/chunk.mp4"))

        assert result is None
        mock_logger.error.assert_called_with("No frames extracted from chunk")

    @patch("cogniverse_core.common.models.get_or_load_model")
    def test_generate_chunk_embeddings_error(
        self, mock_get_model, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _generate_chunk_embeddings error handling."""
        mock_get_model.return_value = (Mock(), Mock())
        config = {**frame_based_config, "embedding_type": "video_chunks", "model_loader": "colqwen"}
        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)
        generator.model_name = "colqwen_test"

        with patch("cv2.VideoCapture", side_effect=Exception("Video error")):
            result = generator._generate_chunk_embeddings(Path("/path/to/chunk.mp4"))

            assert result is None
            mock_logger.error.assert_called_with(
                "Error generating chunk embeddings: Video error"
            )

    def test_generate_time_segment_embeddings_videoprism(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _generate_time_segment_embeddings with VideoPrism."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )
        generator.videoprism_loader = Mock()

        mock_embeddings = np.random.rand(768)
        generator.videoprism_loader.process_video_segment.return_value = {
            "embeddings_np": mock_embeddings
        }

        result = generator._generate_time_segment_embeddings(
            Path("/video.mp4"), 10.0, 20.0
        )

        np.testing.assert_array_equal(result, mock_embeddings)
        generator.videoprism_loader.process_video_segment.assert_called_once_with(
            Path("/video.mp4"), 10.0, 20.0
        )

    def test_generate_time_segment_embeddings_videoprism_no_result(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _generate_time_segment_embeddings with VideoPrism returning no result."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )
        generator.videoprism_loader = Mock()
        generator.videoprism_loader.process_video_segment.return_value = None

        result = generator._generate_time_segment_embeddings(
            Path("/video.mp4"), 10.0, 20.0
        )

        assert result is None

    @patch("torch.no_grad")
    @patch("cv2.cvtColor")
    @patch("PIL.Image.fromarray")
    @patch("cv2.VideoCapture")
    def test_generate_time_segment_embeddings_other_models(
        self,
        mock_video_capture,
        mock_from_array,
        mock_cvt_color,
        mock_no_grad,
        frame_based_config,
        mock_logger,
        mock_backend_client,
    ):
        """Test _generate_time_segment_embeddings with other models."""
        config = {**frame_based_config, "fps": 2.0}
        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)
        generator.model = Mock()
        generator.processor = Mock()
        generator.model.device = "cpu"

        # Mock video capture
        mock_cap = Mock()
        mock_cap.get.return_value = 30.0  # fps
        mock_cap.read.return_value = (True, np.random.rand(100, 100, 3))
        mock_video_capture.return_value = mock_cap

        # Mock image processing
        mock_pil_image = Mock()
        mock_from_array.return_value = mock_pil_image
        mock_cvt_color.return_value = np.random.rand(100, 100, 3)

        # Mock batch processing - use real dict for batch inputs
        mock_batch_inputs = {"input_ids": Mock(), "attention_mask": Mock()}
        mock_batch_inputs_wrapper = Mock()
        mock_batch_inputs_wrapper.to.return_value = mock_batch_inputs
        generator.processor.process_images.return_value = mock_batch_inputs_wrapper

        # Mock model output
        mock_embeddings = Mock()
        mock_embeddings.cpu.return_value.numpy.return_value = np.random.rand(5, 128)
        generator.model.return_value = mock_embeddings

        # Mock torch.no_grad as proper context manager
        context_manager = Mock()
        context_manager.__enter__ = Mock(return_value=None)
        context_manager.__exit__ = Mock(return_value=None)
        mock_no_grad.return_value = context_manager

        result = generator._generate_time_segment_embeddings(
            Path("/video.mp4"), 10.0, 20.0
        )

        assert result is not None
        assert result.shape == (128,)  # Averaged across frames
        mock_cap.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_generate_time_segment_embeddings_no_frames(
        self, mock_video_capture, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _generate_time_segment_embeddings when no frames extracted."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )
        generator.model = Mock()
        generator.processor = Mock()

        # Mock video capture with no frames
        mock_cap = Mock()
        mock_cap.get.return_value = 30.0
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap

        result = generator._generate_time_segment_embeddings(
            Path("/video.mp4"), 10.0, 20.0
        )

        assert result is None

    def test_extract_segments_document_files(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _extract_segments with document_files key."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )
        doc_files = [
            {"document_id": "doc1", "extracted_text": "Hello", "path": "/a.txt"},
            {"document_id": "doc2", "extracted_text": "World", "path": "/b.md"},
        ]
        segments = generator._extract_segments({"document_files": doc_files})
        assert len(segments) == 2
        assert segments[0]["document_id"] == "doc1"
        assert segments[1]["extracted_text"] == "World"

    def test_extract_segments_audio_files(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test _extract_segments with audio_files key."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )
        audio_files = [
            {"audio_id": "clip1", "path": "/a.mp3"},
            {"audio_id": "clip2", "path": "/b.wav"},
        ]
        segments = generator._extract_segments({"audio_files": audio_files})
        assert len(segments) == 2
        assert segments[0]["audio_id"] == "clip1"

    def test_generate_embeddings_dispatches_document_content(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test generate_embeddings dispatches to _process_document_segments."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )
        doc_files = [{"document_id": "d1", "extracted_text": "x", "path": "/a.txt"}]
        video_data = {"video_id": "test", "document_files": doc_files}

        expected = EmbeddingResult(
            video_id="test", total_documents=1, documents_processed=1,
            documents_fed=1, processing_time=0, errors=[], metadata={},
        )
        with patch.object(
            generator, "_process_document_segments", return_value=expected
        ) as mock_doc:
            result = generator.generate_embeddings(video_data, Path("/tmp"))
            mock_doc.assert_called_once_with(video_data, doc_files)
            assert result.video_id == "test"

    def test_generate_embeddings_dispatches_audio_content(
        self, frame_based_config, mock_logger, mock_backend_client
    ):
        """Test generate_embeddings dispatches to _process_audio_segments."""
        generator = EmbeddingGeneratorImpl(
            frame_based_config, mock_logger, mock_backend_client
        )
        audio_files = [{"audio_id": "a1", "path": "/a.mp3"}]
        video_data = {"video_id": "test", "audio_files": audio_files}

        expected = EmbeddingResult(
            video_id="test", total_documents=1, documents_processed=1,
            documents_fed=1, processing_time=0, errors=[], metadata={},
        )
        with patch.object(
            generator, "_process_audio_segments", return_value=expected
        ) as mock_audio:
            result = generator.generate_embeddings(video_data, Path("/tmp"))
            mock_audio.assert_called_once_with(video_data, audio_files)
            assert result.video_id == "test"

    @patch("cogniverse_core.common.models.get_or_load_model")
    def test_load_model_colbert(self, mock_get_model, mock_logger, mock_backend_client):
        """Test _load_model loads ColBERT model when model name contains 'colbert'."""
        config = {
            "schema_name": "document_text",
            "embedding_model": "lightonai/GTE-ModernColBERT-v1",
            "embedding_type": "document_colbert",
            "model_loader": "colbert",
        }
        mock_colbert = Mock()
        mock_get_model.return_value = (mock_colbert, None)

        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)

        assert generator.colbert_model == mock_colbert
        assert generator.model is None
        assert generator.videoprism_loader is None
        mock_get_model.assert_called_once_with(
            "lightonai/GTE-ModernColBERT-v1", config, mock_logger
        )

    @patch("cogniverse_core.common.models.get_or_load_model")
    def test_load_model_audio_dual(self, mock_get_model, mock_logger, mock_backend_client):
        """Test _load_model loads ColBERT for semantic when embedding_type is audio_dual."""
        config = {
            "schema_name": "audio_content",
            "embedding_model": "laion/clap-htsat-unfused",
            "embedding_type": "audio_dual",
            "model_loader": "colbert",
            "semantic_model": "lightonai/GTE-ModernColBERT-v1",
        }
        mock_colbert = Mock()
        mock_get_model.return_value = (mock_colbert, None)

        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)

        assert generator.colbert_model == mock_colbert
        assert generator.model is None
        mock_get_model.assert_called_once_with(
            "lightonai/GTE-ModernColBERT-v1", config, mock_logger
        )

    def test_process_document_segments_calls_colbert(
        self, mock_logger, mock_backend_client
    ):
        """Test _process_document_segments uses ColBERT model to encode text."""
        config = {
            "schema_name": "document_text",
            "embedding_model": "lightonai/GTE-ModernColBERT-v1",
            "embedding_type": "frame_based",
            "model_loader": "colpali",
        }
        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)

        mock_colbert = Mock()
        mock_colbert.encode.return_value = [np.random.rand(10, 128)]
        generator.colbert_model = mock_colbert
        mock_backend_client.ingest_documents.return_value = {"success_count": 1}

        doc_files = [
            {
                "document_id": "doc1",
                "filename": "readme.txt",
                "path": "/tmp/readme.txt",
                "document_type": "txt",
                "extracted_text": "This is test document content.",
            }
        ]
        video_data = {"video_id": "content_dir", "document_files": doc_files}

        result = generator._process_document_segments(video_data, doc_files)

        assert result.documents_processed == 1
        assert result.documents_fed == 1
        mock_colbert.encode.assert_called_once_with(
            ["This is test document content."], is_query=False
        )
        mock_backend_client.ingest_documents.assert_called_once()

    def test_process_document_segments_empty_text_raises(
        self, mock_logger, mock_backend_client
    ):
        """Test _process_document_segments raises on empty text."""
        config = {"embedding_type": "frame_based", "embedding_model": "test_model", "model_loader": "colpali"}
        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)
        generator.colbert_model = Mock()

        doc_files = [
            {"document_id": "d1", "filename": "empty.txt", "path": "/tmp/e.txt",
             "extracted_text": "   "}
        ]
        result = generator._process_document_segments(
            {"video_id": "test", "document_files": doc_files}, doc_files
        )
        assert result.documents_processed == 0
        assert len(result.errors) == 1
        assert "no extracted text" in result.errors[0]


@pytest.mark.unit
@pytest.mark.ci_safe
class TestModelLoaderFactoryModelLoader:
    """Test ModelLoaderFactory dispatches on model_loader from config."""

    def test_colbert_loader(self):
        from cogniverse_core.common.models.model_loaders import (
            ColBERTModelLoader,
            ModelLoaderFactory,
        )

        loader = ModelLoaderFactory.create_loader(
            "lightonai/GTE-ModernColBERT-v1",
            {"model_loader": "colbert"},
            None,
        )
        assert isinstance(loader, ColBERTModelLoader)

    def test_colpali_loader(self):
        from cogniverse_core.common.models.model_loaders import (
            ColPaliModelLoader,
            ModelLoaderFactory,
        )

        loader = ModelLoaderFactory.create_loader(
            "vidore/colsmol-500m",
            {"model_loader": "colpali"},
            None,
        )
        assert isinstance(loader, ColPaliModelLoader)

    def test_colqwen_loader(self):
        from cogniverse_core.common.models.model_loaders import (
            ColQwenModelLoader,
            ModelLoaderFactory,
        )

        loader = ModelLoaderFactory.create_loader(
            "vidore/colqwen-omni-v0.1",
            {"model_loader": "colqwen"},
            None,
        )
        assert isinstance(loader, ColQwenModelLoader)

    def test_videoprism_loader(self):
        from cogniverse_core.common.models.model_loaders import (
            ModelLoaderFactory,
            VideoPrismModelLoader,
        )

        loader = ModelLoaderFactory.create_loader(
            "videoprism_public_v1_base_hf",
            {"model_loader": "videoprism"},
            None,
        )
        assert isinstance(loader, VideoPrismModelLoader)

    def test_missing_model_loader_raises(self):
        from cogniverse_core.common.models.model_loaders import ModelLoaderFactory

        with pytest.raises(ValueError, match="must contain 'model_loader'"):
            ModelLoaderFactory.create_loader("some-model", {}, None)

    def test_unknown_model_loader_raises(self):
        from cogniverse_core.common.models.model_loaders import ModelLoaderFactory

        with pytest.raises(ValueError, match="Unknown model_loader='bogus'"):
            ModelLoaderFactory.create_loader(
                "some-model", {"model_loader": "bogus"}, None
            )
