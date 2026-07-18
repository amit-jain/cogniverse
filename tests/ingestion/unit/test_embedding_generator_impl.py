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
            "embedding_type": "multi_vector",
            "model_loader": "videoprism",
        }

    @pytest.fixture
    def frame_based_config(self):
        return {
            "schema_name": "test_schema",
            "embedding_model": "test_model",
            "storage_mode": "multi_doc",
            "embedding_type": "multi_vector",
            "model_loader": "colpali",
        }

    @patch("cogniverse_core.common.models.get_or_load_model")
    def test_initialization_with_model_load(
        self, mock_get_model, basic_config, mock_logger, mock_backend_client
    ):
        """Test EmbeddingGeneratorImpl initialization loads model for videoprism."""
        mock_loader = Mock()
        mock_get_model.return_value = (mock_loader, None)

        generator = EmbeddingGeneratorImpl(
            basic_config, mock_logger, mock_backend_client
        )

        assert generator.profile_config == basic_config
        assert generator.model_name == "test_model"
        assert generator.backend_client == mock_backend_client
        assert generator.schema_name == "test_schema"
        assert generator.storage_mode == "multi_doc"
        assert generator.videoprism_loader == mock_loader

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
            "embedding_type": "multi_vector",
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
        """Test _should_load_model dispatches on model_loader."""
        # ColPali/ColQwen should NOT load model at init (deferred to first use)
        for loader in ("colpali", "colqwen"):
            config = {
                "embedding_type": "multi_vector",
                "embedding_model": "test_model",
                "model_loader": loader,
            }
            with patch(
                "cogniverse_core.common.models.get_or_load_model",
                return_value=(Mock(), Mock()),
            ):
                generator = EmbeddingGeneratorImpl(
                    config, mock_logger, mock_backend_client
                )
                assert generator._should_load_model() is False, (
                    f"model_loader={loader} should defer model loading"
                )

        # ColBERT and VideoPrism should load model at init
        for loader in ("colbert", "videoprism"):
            config = {
                "embedding_type": "multi_vector",
                "embedding_model": "test_model",
                "model_loader": loader,
            }
            with patch(
                "cogniverse_core.common.models.get_or_load_model",
                return_value=(Mock(), Mock()),
            ):
                generator = EmbeddingGeneratorImpl(
                    config, mock_logger, mock_backend_client
                )
                assert generator._should_load_model() is True, (
                    f"model_loader={loader} should load model at init"
                )

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
            patch.object(
                generator,
                "_feed_documents",
                side_effect=lambda docs, errors=None: len(docs),
            ),
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

        # Mock image and processing. Image.open is used as a context manager
        # now (``with Image.open(...) as image:``) so the mock must support
        # __enter__/__exit__; the converted image is what process_images sees.
        mock_image = Mock()
        mock_image_open.return_value.__enter__ = Mock(
            return_value=mock_image_open.return_value
        )
        mock_image_open.return_value.__exit__ = Mock(return_value=None)
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

    def _make_remote_generator(self, config, mock_logger, mock_backend_client):
        """Build a generator whose model/processor is a real
        RemoteInferenceClient so ``_generate_frame_embeddings`` takes the remote
        branch (the only branch that applies document-side token pooling)."""
        from cogniverse_core.common.models.model_loaders import RemoteInferenceClient

        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)
        client = RemoteInferenceClient("http://remote.invalid", logger=mock_logger)
        generator.model = client
        generator.processor = client
        return generator, client

    def test_generate_frame_embeddings_pools_when_factor_set(
        self, frame_based_config, mock_logger, mock_backend_client, tmp_path
    ):
        """Frame remote path pools the multi-vector when token_pool_factor set.

        The pooler clusters by cosine similarity (HierarchicalTokenPooler), so
        30 DISTINCT token rows with pool_factor=3 collapse to ceil(30/3)=10
        clusters -> shape (10, 320). Identical rows would collapse to 1, which
        is why the stub returns distinct (seeded) vectors.
        """
        from PIL import Image as PILImage

        config = {**frame_based_config, "token_pool_factor": 3}
        generator, client = self._make_remote_generator(
            config, mock_logger, mock_backend_client
        )

        rng = np.random.default_rng(0)
        tokens = rng.standard_normal((1, 30, 320)).astype(np.float32)

        frame_path = tmp_path / "frame.png"
        PILImage.new("RGB", (8, 8), (10, 20, 30)).save(frame_path)

        with patch.object(
            client, "process_images", return_value={"embeddings": tokens}
        ) as mock_proc:
            result = generator._generate_frame_embeddings(frame_path)

        mock_proc.assert_called_once()
        assert result is not None
        assert result.shape == (10, 320)
        assert result.dtype == np.float32

    def test_generate_frame_embeddings_no_pooling_without_factor(
        self, frame_based_config, mock_logger, mock_backend_client, tmp_path
    ):
        """Control: with no token_pool_factor, the remote path is unpooled."""
        from PIL import Image as PILImage

        generator, client = self._make_remote_generator(
            frame_based_config, mock_logger, mock_backend_client
        )

        rng = np.random.default_rng(0)
        tokens = rng.standard_normal((1, 30, 320)).astype(np.float32)

        frame_path = tmp_path / "frame.png"
        PILImage.new("RGB", (8, 8), (10, 20, 30)).save(frame_path)

        with patch.object(
            client, "process_images", return_value={"embeddings": tokens}
        ):
            result = generator._generate_frame_embeddings(frame_path)

        assert result is not None
        assert result.shape == (30, 320)

    @patch("PIL.Image.fromarray")
    @patch("cv2.cvtColor")
    @patch("cv2.VideoCapture")
    def test_generate_chunk_embeddings_pools_when_factor_set(
        self,
        mock_video_capture,
        mock_cvt_color,
        mock_from_array,
        frame_based_config,
        mock_logger,
        mock_backend_client,
    ):
        """Chunk remote path pools the per-chunk multi-vector when the factor
        is set.

        The remote client returns [N_frames, T, D]; the method mean-pools over
        the frame dim to a 2D (T, D) multi-vector, which is then token-pooled.
        30 DISTINCT token rows + pool_factor=3 -> ceil(30/3)=10 clusters.
        """
        config = {
            **frame_based_config,
            "fps": 1.0,
            "embedding_type": "multi_vector",
            "model_loader": "colqwen",
            "token_pool_factor": 3,
        }
        generator, client = self._make_remote_generator(
            config, mock_logger, mock_backend_client
        )
        generator.model_name = "colqwen_test"

        mock_cap = Mock()
        mock_cap.get.side_effect = [25.0, 100]  # fps, total_frames
        mock_cap.read.return_value = (True, np.zeros((8, 8, 3), dtype=np.uint8))
        mock_video_capture.return_value = mock_cap
        mock_cvt_color.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
        mock_from_array.return_value = Mock()

        # Single extracted frame -> [1, 30, 320]; mean(axis=0) -> (30, 320).
        rng = np.random.default_rng(0)
        tokens = rng.standard_normal((1, 30, 320)).astype(np.float32)

        with patch.object(
            client, "process_images", return_value={"embeddings": tokens}
        ):
            result = generator._generate_chunk_embeddings(Path("/path/to/chunk.mp4"))

        assert result is not None
        assert result.shape == (10, 320)
        assert result.dtype == np.float32

    @patch("cogniverse_core.common.models.get_or_load_model")
    @patch("subprocess.run")
    def test_generate_chunk_embeddings_videoprism_single_vector_not_pooled(
        self,
        mock_subprocess,
        mock_get_or_load_model,
        frame_based_config,
        mock_logger,
        mock_backend_client,
    ):
        """A VideoPrism-style single-vector chunk return is never pooled, even
        when token_pool_factor is set — pooling a single vector is corrupting.

        The shape guard (ndim == 2 and shape[0] > 1) protects it.
        """
        mock_get_or_load_model.return_value = (Mock(), None)
        config = {
            **frame_based_config,
            "embedding_type": "multi_vector",
            "model_loader": "videoprism",
            "token_pool_factor": 3,
        }
        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)
        generator.videoprism_loader = Mock()

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "30.0"
        mock_subprocess.return_value = mock_result

        rng = np.random.default_rng(1)
        single_vector = rng.standard_normal((768,)).astype(np.float32)
        generator.videoprism_loader.process_video_segment.return_value = {
            "embeddings_np": single_vector
        }

        result = generator._generate_chunk_embeddings(Path("/path/to/chunk.mp4"))

        assert result is not None
        assert result.shape == (768,)
        np.testing.assert_array_equal(result, single_vector)

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
        config = {
            **frame_based_config,
            "fps": 1.0,
            "embedding_type": "multi_vector",
            "model_loader": "colqwen",
        }
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

    @patch("cogniverse_core.common.models.get_or_load_model")
    @patch("subprocess.run")
    def test_generate_chunk_embeddings_videoprism(
        self,
        mock_subprocess,
        mock_get_or_load_model,
        mock_logger,
        mock_backend_client,
    ):
        """Test _generate_chunk_embeddings with VideoPrism model."""
        # Constructor calls ``get_or_load_model`` for model_loader=videoprism;
        # without this mock the real loader fires ``import jax`` (the
        # JAX/flax stack now lives only in the deploy/videoprism sidecar
        # image, not in base deps) and the test fails to even instantiate.
        mock_get_or_load_model.return_value = (Mock(), None)
        videoprism_config = {
            "schema_name": "test_schema",
            "embedding_model": "test_model",
            "storage_mode": "multi_doc",
            "embedding_type": "multi_vector",
            "model_loader": "videoprism",
        }
        generator = EmbeddingGeneratorImpl(
            videoprism_config, mock_logger, mock_backend_client
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
        config = {
            **frame_based_config,
            "embedding_type": "multi_vector",
            "model_loader": "colqwen",
        }
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
        config = {
            **frame_based_config,
            "embedding_type": "multi_vector",
            "model_loader": "colqwen",
        }
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

        # The capture is cached for reuse across the video's segments —
        # reopening per segment paid container parsing every time. A second
        # segment must reuse the same capture, and release happens when the
        # generator finishes the video.
        mock_cap.isOpened.return_value = True
        generator._generate_time_segment_embeddings(Path("/video.mp4"), 20.0, 30.0)
        assert mock_video_capture.call_count == 1, "capture must be reused"
        mock_cap.release.assert_not_called()
        generator._release_video_capture()
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
            video_id="test",
            total_documents=1,
            documents_processed=1,
            documents_fed=1,
            processing_time=0,
            errors=[],
            metadata={},
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
            video_id="test",
            total_documents=1,
            documents_processed=1,
            documents_fed=1,
            processing_time=0,
            errors=[],
            metadata={},
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
            "embedding_model": "lightonai/Reason-ModernColBERT",
            "embedding_type": "multi_vector",
            "model_loader": "colbert",
        }
        mock_colbert = Mock()
        mock_get_model.return_value = (mock_colbert, None)

        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)

        assert generator.colbert_model == mock_colbert
        assert generator.model is None
        assert generator.videoprism_loader is None
        mock_get_model.assert_called_once_with(
            "lightonai/Reason-ModernColBERT", config, mock_logger
        )

    @patch("cogniverse_core.common.models.get_or_load_model")
    def test_load_model_audio_dual(
        self, mock_get_model, mock_logger, mock_backend_client
    ):
        """Test _load_model loads ColBERT for semantic when model_loader is colbert with semantic_model."""
        config = {
            "schema_name": "audio_content",
            "embedding_model": "laion/clap-htsat-unfused",
            "embedding_type": "multi_vector",
            "model_loader": "colbert",
            "semantic_model": "lightonai/Reason-ModernColBERT",
        }
        mock_colbert = Mock()
        mock_get_model.return_value = (mock_colbert, None)

        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)

        assert generator.colbert_model == mock_colbert
        assert generator.model is None
        mock_get_model.assert_called_once_with(
            "lightonai/Reason-ModernColBERT", config, mock_logger
        )

    def test_process_document_segments_calls_colbert(
        self, mock_logger, mock_backend_client
    ):
        """Test _process_document_segments uses ColBERT model to encode text."""
        config = {
            "schema_name": "document_text",
            "embedding_model": "lightonai/Reason-ModernColBERT",
            "embedding_type": "multi_vector",
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
        config = {
            "embedding_type": "multi_vector",
            "embedding_model": "test_model",
            "model_loader": "colpali",
        }
        generator = EmbeddingGeneratorImpl(config, mock_logger, mock_backend_client)
        generator.colbert_model = Mock()

        doc_files = [
            {
                "document_id": "d1",
                "filename": "empty.txt",
                "path": "/tmp/e.txt",
                "extracted_text": "   ",
            }
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
            "lightonai/Reason-ModernColBERT",
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
            "TomoroAI/tomoro-colqwen3-embed-4b",
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


class _StubColbert:
    """Minimal ColBERT stand-in: returns a multi-vector (T, 128) per text."""

    def encode(self, texts, is_query=False):
        return [np.random.rand(6, 128).astype(np.float32) for _ in texts]


class _CapturingBackend:
    """Records the Documents fed so they can be run through the prod mapping."""

    def __init__(self):
        self.docs = []
        self.schema_name = None

    def ingest_documents(self, docs, schema_name):
        self.docs.extend(docs)
        self.schema_name = schema_name
        return {"success_count": len(docs)}


@pytest.mark.integration
class TestCodeSegmentIngestion:
    """Code ingestion (`cogniverse index --type code`, schema code_lateon_mv).

    Guards the CRIT that crashed pipeline construction (code_file unhandled)
    and the cross-layer drop that yielded zero documents: code chunks must
    flow dispatch -> _extract_segments -> _process_code_segments -> Documents
    whose fields map onto the real code schema (validated via the production
    VespaPyClient.process mapping — a field-name mismatch would 400 on Vespa).
    """

    CODE_SCHEMA = "code_lateon_mv"

    def _make_generator(self, backend):
        config = {
            "schema_name": self.CODE_SCHEMA,
            "embedding_model": "lightonai/LateOn-Code-edge",
            "model_loader": "colpali",  # lazy — skip model load at init
            "storage_mode": "multi_doc",
        }
        gen = EmbeddingGeneratorImpl(config, Mock(), backend)
        gen.colbert_model = _StubColbert()
        return gen

    def test_extract_segments_returns_code_files(self):
        gen = self._make_generator(_CapturingBackend())
        segs = [{"document_id": "m_foo_1", "extracted_text": "def foo(): pass"}]
        assert gen._extract_segments({"code_files": segs}) == segs

    def test_dispatch_maps_code_chunks_onto_code_schema(self, tmp_path):
        backend = _CapturingBackend()
        gen = self._make_generator(backend)
        segments = [
            {
                "document_id": "mymodule_foo_1",
                "file_index": 0,
                "path": "/x/mymodule.py",
                "filename": "mymodule.py",
                "document_type": "py",
                "extracted_text": "def foo():\n    return 1",
                "text_length": 22,
                "chunk_type": "function",
                "chunk_name": "foo",
                "signature": "def foo()",
                "line_start": 1,
                "line_end": 2,
                "language": "python",
            },
            {
                "document_id": "mymodule_Bar_5",
                "file_index": 0,
                "path": "/x/mymodule.py",
                "filename": "mymodule.py",
                "document_type": "py",
                "extracted_text": "class Bar:\n    pass",
                "text_length": 18,
                "chunk_type": "class",
                "chunk_name": "Bar",
                "signature": "class Bar",
                "line_start": 5,
                "line_end": 6,
                "language": "python",
            },
        ]
        video_data = {
            "video_id": "mymodule",
            "video_path": "/x/mymodule.py",
            "code_files": segments,
        }

        result = gen.generate_embeddings(video_data, tmp_path)

        # Dispatch reached _process_code_segments and fed every chunk.
        assert result.documents_processed == 2
        assert result.documents_fed == 2
        assert result.errors == []
        assert backend.schema_name == self.CODE_SCHEMA
        assert len(backend.docs) == 2

        # Validate each Document through the production field mapping against
        # the REAL code schema (no Vespa server needed; mismatch -> 400 in prod).
        from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
        from cogniverse_vespa.ingestion_client import VespaPyClient

        schemas_dir = Path(__file__).resolve().parents[3] / "configs" / "schemas"
        client = VespaPyClient(
            {
                "schema_name": self.CODE_SCHEMA,
                "url": "http://localhost",
                "port": 8080,
                "schema_loader": FilesystemSchemaLoader(schemas_dir),
            }
        )
        by_name = {}
        for doc in backend.docs:
            fields = client.process(doc)["fields"]
            by_name[fields["chunk_name"]] = fields

        foo = by_name["foo"]
        assert foo["code_id"] == "mymodule_foo_1"
        assert foo["file_path"] == "/x/mymodule.py"
        assert foo["chunk_type"] == "function"
        assert foo["language"] == "python"
        assert foo["signature"] == "def foo()"
        assert foo["source_code"] == "def foo():\n    return 1"
        assert foo["line_start"] == 1
        assert foo["line_end"] == 2
        assert "embedding" in foo

        bar = by_name["Bar"]
        assert bar["code_id"] == "mymodule_Bar_5"
        assert bar["chunk_type"] == "class"
        assert bar["source_code"] == "class Bar:\n    pass"


class TestVlmDescriptionMapping:
    """VLM frame descriptions must reach the right keyframe documents.

    Regression: video_data["descriptions"] is the VLM wrapper
    {"video_id", "descriptions": {<frame_ref>: text}, ...}; the old code read
    it at the top level and keyed by the segment ENUMERATION index, so every
    keyframe document got description="". The per-frame text lives one level
    down, keyed by the keyframe's frame_number/frame_id.
    """

    def test_frame_description_map_unwraps_vlm_wrapper(self):
        wrapper = {
            "video_id": "v1",
            "descriptions": {"0": "a sunrise", "30": "a city street"},
            "total_descriptions": 2,
        }
        assert EmbeddingGeneratorImpl._frame_description_map(wrapper) == {
            "0": "a sunrise",
            "30": "a city street",
        }

    def test_frame_description_map_handles_empty_and_flat(self):
        assert EmbeddingGeneratorImpl._frame_description_map({}) == {}
        assert EmbeddingGeneratorImpl._frame_description_map(None) == {}
        # Legacy flat {frame_ref: text} map passes through unchanged.
        assert EmbeddingGeneratorImpl._frame_description_map({"0": "x"}) == {"0": "x"}

    def test_segment_frame_ref_prefers_frame_id_then_number(self):
        assert EmbeddingGeneratorImpl._segment_frame_ref({"frame_id": 7}) == "7"
        assert EmbeddingGeneratorImpl._segment_frame_ref({"frame_number": 30}) == "30"
        assert EmbeddingGeneratorImpl._segment_frame_ref({}) is None

    def test_sparse_frames_align_text_to_correct_keyframe(self):
        """Sparse frame_numbers (0, 30, 90) must align each description to its
        own keyframe; an undescribed frame gets ''. The old idx-on-wrapper
        lookup returned '' for every frame."""
        wrapper = {
            "video_id": "v1",
            "descriptions": {"0": "sunrise", "30": "city street"},
            "total_descriptions": 2,
        }
        frame_map = EmbeddingGeneratorImpl._frame_description_map(wrapper)
        segments = [
            {"frame_number": 0},
            {"frame_number": 30},
            {"frame_number": 90},  # no VLM description for this frame
        ]
        resolved = [
            frame_map.get(EmbeddingGeneratorImpl._segment_frame_ref(s), "")
            for s in segments
        ]
        assert resolved == ["sunrise", "city street", ""]

    @patch("cogniverse_core.common.models.get_or_load_model")
    def test_multi_document_feed_is_batched(self, mock_get_model):
        """Segments are fed in batches (one backend call per ~50), not one
        Vespa round-trip per segment."""
        mock_get_model.return_value = (Mock(), None)
        frame_based_config = {
            "schema_name": "test_schema",
            "embedding_model": "test_model",
            "storage_mode": "multi_doc",
            "embedding_type": "multi_vector",
            "model_loader": "colpali",
        }
        mock_logger = Mock()
        backend = Mock()
        calls: list[list] = []

        def _ingest(docs, schema):
            calls.append(list(docs))
            return {"success_count": len(docs)}

        backend.ingest_documents = Mock(side_effect=_ingest)

        gen = EmbeddingGeneratorImpl(frame_based_config, mock_logger, backend)

        segments = [
            {"start_time": i, "end_time": i + 1, "frame_id": i} for i in range(120)
        ]
        with (
            patch.object(
                gen,
                "_generate_segment_embeddings",
                return_value=np.zeros((4, 128), dtype=np.float32),
            ),
            patch.object(
                gen, "_create_segment_document", side_effect=lambda **kw: Mock()
            ),
        ):
            result = gen._process_multi_documents(
                {
                    "video_id": "v",
                    "video_path": "",
                    "transcript": {},
                    "descriptions": {},
                    "source_url": "",
                },
                segments,
            )

        # 120 segments -> 50 + 50 + 20 = 3 batched feeds, not 120.
        assert backend.ingest_documents.call_count == 3
        assert sum(len(c) for c in calls) == 120
        assert result.documents_processed == 120
        assert result.documents_fed == 120


@pytest.mark.unit
@pytest.mark.ci_safe
class TestRemoteFrameBatching:
    """Consecutive frame segments served by a RemoteInferenceClient must be
    encoded in batched process_images calls, preserving segment order and
    per-segment error containment."""

    @pytest.fixture
    def frame_config(self):
        return {
            "schema_name": "test_schema",
            "embedding_model": "test_model",
            "storage_mode": "multi_doc",
            "embedding_type": "multi_vector",
            "model_loader": "colpali",
        }

    def _generator_with_remote(self, frame_config, calls):
        """Build a generator whose processor is a real RemoteInferenceClient
        with process_images stubbed to record batch sizes."""
        from cogniverse_core.common.models.model_loaders import (
            RemoteInferenceClient,
        )

        generator = EmbeddingGeneratorImpl(frame_config, Mock(), Mock())
        client = RemoteInferenceClient(endpoint_url="http://unused:1")

        def fake_process_images(images, **kwargs):
            calls.append(len(images))
            per_image = [
                np.full((4, 16), fill_value=len(calls) * 100 + i, dtype=np.float32)
                for i in range(len(images))
            ]
            embeddings = (
                per_image[0]
                if len(per_image) == 1
                else np.array(per_image, dtype=object)
            )
            return {"embeddings": embeddings}

        client.process_images = fake_process_images
        generator.model = client
        generator.processor = client
        return generator

    def _frame_segments(self, tmp_path, count):
        from PIL import Image as PILImage

        segments = []
        for i in range(count):
            p = tmp_path / f"frame_{i}.png"
            PILImage.new("RGB", (2, 2), color=(i, 0, 0)).save(p)
            segments.append({"frame_path": str(p), "frame_id": i})
        return segments

    def test_consecutive_frames_share_one_remote_call(self, frame_config, tmp_path):
        calls: list[int] = []
        generator = self._generator_with_remote(frame_config, calls)
        segments = self._frame_segments(tmp_path, 8)

        results = list(generator._iter_segment_embeddings(segments, Path("v.mp4"), {}))

        assert calls == [8], f"expected one batched call, got batches {calls}"
        assert [idx for idx, _, _ in results] == list(range(8))
        for i, (_, seg, emb) in enumerate(results):
            assert seg["frame_id"] == i
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (4, 16)
            assert emb.dtype == np.float32
            # Row i of the single batch carries value 100 + i.
            assert float(emb[0, 0]) == 100.0 + i

    def test_mixed_segments_batch_only_frame_runs(self, frame_config, tmp_path):
        calls: list[int] = []
        generator = self._generator_with_remote(frame_config, calls)
        frames = self._frame_segments(tmp_path, 3)
        time_seg = {"start_time": 0.0, "end_time": 5.0}
        segments = [frames[0], frames[1], time_seg, frames[2]]

        sentinel = np.ones((2, 16), dtype=np.float32)
        generator._generate_segment_embeddings = Mock(return_value=sentinel)

        results = list(generator._iter_segment_embeddings(segments, Path("v.mp4"), {}))

        # Two frame runs → two remote calls of sizes 2 and 1; the time
        # segment goes through the single-segment path in position.
        assert calls == [2, 1]
        assert [idx for idx, _, _ in results] == [0, 1, 2, 3]
        assert results[2][2] is sentinel
        generator._generate_segment_embeddings.assert_called_once_with(
            time_seg, Path("v.mp4"), {}
        )

    def test_failed_batch_falls_back_to_single_frames(self, frame_config, tmp_path):
        calls: list[int] = []
        generator = self._generator_with_remote(frame_config, calls)
        segments = self._frame_segments(tmp_path, 3)

        original = generator.processor.process_images

        def flaky_process_images(images, **kwargs):
            if len(images) > 1:
                calls.append(len(images))
                raise ConnectionError("batch endpoint hiccup")
            return original(images, **kwargs)

        generator.processor.process_images = flaky_process_images

        results = list(generator._iter_segment_embeddings(segments, Path("v.mp4"), {}))

        # One failed batch of 3, then three single-frame retries.
        assert calls == [3, 1, 1, 1]
        assert [idx for idx, _, _ in results] == [0, 1, 2]
        for _, _, emb in results:
            assert isinstance(emb, np.ndarray)
            assert emb.shape == (4, 16)


def _write_short_mp4(path, n_frames: int = 30, size: int = 64) -> None:
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (size, size))
    try:
        for _ in range(n_frames):
            writer.write(np.zeros((size, size, 3), dtype=np.uint8))
    finally:
        writer.release()


@pytest.mark.unit
class TestThreadLocalCapture:
    """The cv2 capture cache must be per-thread so a concurrent video thread
    opening a different file cannot release another thread's live handle."""

    def test_capture_survives_peer_thread_opening_other_video(self, tmp_path):
        import threading

        p1 = tmp_path / "v1.mp4"
        p2 = tmp_path / "v2.mp4"
        _write_short_mp4(p1)
        _write_short_mp4(p2)

        gen = EmbeddingGeneratorImpl(
            {
                "schema_name": "s",
                "embedding_model": "m",
                "model_loader": "colqwen",  # lazy: no model load at init
            },
            Mock(),
        )

        holder = {}
        a_opened = threading.Event()
        b_done = threading.Event()

        def thread_a():
            holder["cap_a"] = gen._get_video_capture(p1)
            a_opened.set()
            assert b_done.wait(timeout=5)
            holder["cap_a2"] = gen._get_video_capture(p1)

        t = threading.Thread(target=thread_a)
        t.start()
        assert a_opened.wait(timeout=5)

        # Main thread opens a DIFFERENT video. On the old shared-attribute code
        # this released thread A's capture; with thread-local storage it does not.
        gen._get_video_capture(p2)
        b_done.set()
        t.join(timeout=5)

        cap_a = holder["cap_a"]
        try:
            assert cap_a.isOpened() is True
            # A's cached handle is intact and re-returned (still its own thread's).
            assert holder["cap_a2"] is cap_a
        finally:
            cap_a.release()
            gen._release_video_capture()


class TestFeedRejectionSurfacing:
    """The backend names rejected docs in failed_documents; dropping them
    reported a partial feed as fully successful."""

    def _impl(self):
        impl = object.__new__(EmbeddingGeneratorImpl)
        impl.schema_name = "video_test"
        impl.logger = Mock()
        client = Mock()
        client.ingest_documents.return_value = {
            "success_count": 2,
            "failed_count": 1,
            "failed_documents": [{"id": "doc-bad", "error": "HTTP 400"}],
            "total_documents": 3,
        }
        impl.backend_client = client
        return impl

    def test_feed_documents_records_rejections_into_errors(self):
        impl = self._impl()
        errors: list = []

        fed = impl._feed_documents([Mock()] * 3, errors=errors)

        assert fed == 2
        assert errors == ["document doc-bad: rejected at feed (HTTP 400)"]

    def test_feed_documents_without_errors_list_still_returns_count(self):
        impl = self._impl()
        assert impl._feed_documents([Mock()] * 3) == 2

    def test_feed_documents_clean_batch_records_nothing(self):
        impl = self._impl()
        impl.backend_client.ingest_documents.return_value = {
            "success_count": 3,
            "failed_count": 0,
            "failed_documents": [],
            "total_documents": 3,
        }
        errors: list = []
        assert impl._feed_documents([Mock()] * 3, errors=errors) == 3
        assert errors == []
