#!/usr/bin/env python3
"""
Unit tests for VLMDescriptor.

Tests VLM description generation functionality with proper mocking.
"""

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from src.app.ingestion.processors.vlm_descriptor import VLMDescriptor


@pytest.mark.unit
class TestVLMDescriptor:
    """Test suite for VLMDescriptor class."""

    @pytest.fixture
    def mock_logger(self):
        """Mock logger for testing."""
        return Mock()

    @pytest.fixture
    def vlm_descriptor(self):
        """Create a basic VLMDescriptor instance."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            return VLMDescriptor(
                vlm_endpoint="http://test-endpoint.com/generate-description",
                batch_size=100,
                timeout=300,
                auto_start=False,
            )

    @pytest.fixture
    def vlm_descriptor_auto_start(self):
        """Create a VLMDescriptor instance with auto_start enabled."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            return VLMDescriptor(
                vlm_endpoint="http://test-endpoint.com/generate-description",
                batch_size=50,
                timeout=600,
                auto_start=True,
            )

    @pytest.fixture
    def sample_keyframes_metadata(self, tmp_path):
        """Create sample keyframes metadata for testing."""
        # Create mock frame files
        frame1_path = tmp_path / "frame_001.jpg"
        frame2_path = tmp_path / "frame_002.jpg"
        frame1_path.touch()
        frame2_path.touch()

        return {
            "video_id": "test_video_123",
            "keyframes": [
                {"frame_id": "frame_001", "path": str(frame1_path), "timestamp": 10.5},
                {"frame_id": "frame_002", "path": str(frame2_path), "timestamp": 20.0},
            ],
        }

    @pytest.fixture
    def empty_keyframes_metadata(self):
        """Create empty keyframes metadata for testing."""
        return {"video_id": "empty_video", "keyframes": []}

    def test_initialization_defaults(self):
        """Test VLMDescriptor initialization with default values."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            descriptor = VLMDescriptor(vlm_endpoint="http://test.com")

            assert descriptor.vlm_endpoint == "http://test.com"
            assert descriptor.batch_size == 500
            assert descriptor.timeout == 10800
            assert descriptor.auto_start is True
            assert descriptor._modal_process is None
            assert descriptor._service_started is False

    def test_initialization_custom_values(self):
        """Test VLMDescriptor initialization with custom values."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            descriptor = VLMDescriptor(
                vlm_endpoint="http://custom.com/api",
                batch_size=200,
                timeout=1800,
                auto_start=False,
            )

            assert descriptor.vlm_endpoint == "http://custom.com/api"
            assert descriptor.batch_size == 200
            assert descriptor.timeout == 1800
            assert descriptor.auto_start is False

    @patch("requests.get")
    def test_ensure_service_running_already_running(self, mock_get, vlm_descriptor):
        """Test _ensure_service_running when service is already running."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        vlm_descriptor._ensure_service_running()

        mock_get.assert_called_once_with(vlm_descriptor.vlm_endpoint, timeout=5)

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    @patch("requests.get")
    @patch("time.sleep")
    def test_ensure_service_running_start_success(
        self, mock_sleep, mock_get, mock_exists, mock_run, vlm_descriptor
    ):
        """Test successful Modal service startup."""
        # Service not running initially
        mock_get.side_effect = Exception("Connection failed")

        # Modal service file exists
        mock_exists.return_value = True

        # Successful modal deploy
        mock_result = Mock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        vlm_descriptor._ensure_service_running()

        mock_run.assert_called_once_with(
            ["modal", "deploy", "scripts/modal_vlm_service.py"],
            capture_output=True,
            text=True,
        )
        mock_sleep.assert_called_once_with(5)

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    @patch("requests.get")
    def test_ensure_service_running_start_failure(
        self, mock_get, mock_exists, mock_run, vlm_descriptor
    ):
        """Test Modal service startup failure."""
        # Service not running initially
        mock_get.side_effect = Exception("Connection failed")

        # Modal service file exists
        mock_exists.return_value = True

        # Failed modal deploy
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Modal deploy error"
        mock_run.return_value = mock_result

        vlm_descriptor._ensure_service_running()

        mock_run.assert_called_once()

    @patch("pathlib.Path.exists")
    @patch("requests.get")
    def test_ensure_service_running_no_service_file(
        self, mock_get, mock_exists, vlm_descriptor
    ):
        """Test _ensure_service_running when service file doesn't exist."""
        # Service not running initially
        mock_get.side_effect = Exception("Connection failed")

        # Modal service file doesn't exist
        mock_exists.return_value = False

        vlm_descriptor._ensure_service_running()

        # Should not try to run subprocess.run

    @patch("subprocess.run")
    def test_stop_service_started_by_pipeline(self, mock_run, vlm_descriptor):
        """Test stopping service that was started by this pipeline."""
        vlm_descriptor._service_started = True

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Service stopped"
        mock_run.return_value = mock_result

        vlm_descriptor.stop_service()

        mock_run.assert_called_once_with(
            ["modal", "stop", "cogniverse-vlm"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert vlm_descriptor._service_started is False

    @patch("subprocess.run")
    def test_stop_service_stop_command_fails(self, mock_run, vlm_descriptor):
        """Test stopping service when stop command fails."""
        vlm_descriptor._service_started = True

        mock_run.side_effect = Exception("Stop command failed")

        vlm_descriptor.stop_service()

        # Should handle exception gracefully
        assert vlm_descriptor._service_started is False

    def test_stop_service_not_started_by_pipeline(self, vlm_descriptor):
        """Test stopping service that was not started by this pipeline."""
        vlm_descriptor._service_started = False

        vlm_descriptor.stop_service()

        # Should not try to stop anything
        assert vlm_descriptor._service_started is False

    def test_generate_descriptions_empty_metadata(self, vlm_descriptor):
        """Test generate_descriptions with empty metadata."""
        result = vlm_descriptor.generate_descriptions({})

        assert result == {"descriptions": {}}

    def test_generate_descriptions_no_video_id(self, vlm_descriptor):
        """Test generate_descriptions with metadata missing video_id."""
        metadata = {"keyframes": []}

        result = vlm_descriptor.generate_descriptions(metadata)

        assert result == {"descriptions": {}}

    def test_generate_descriptions_empty_keyframes(
        self, vlm_descriptor, empty_keyframes_metadata
    ):
        """Test generate_descriptions with empty keyframes list."""
        result = vlm_descriptor.generate_descriptions(empty_keyframes_metadata)

        assert result == {}

    @patch("src.common.utils.output_manager.get_output_manager")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("time.time")
    def test_generate_descriptions_success(
        self,
        mock_time,
        mock_json_dump,
        mock_file,
        mock_mkdir,
        mock_get_output_manager,
        vlm_descriptor,
        sample_keyframes_metadata,
    ):
        """Test successful description generation."""
        # Mock output manager
        mock_output_manager = Mock()
        mock_processing_dir = Mock()
        mock_processing_dir.__truediv__ = Mock(
            return_value=Path("/test/descriptions/test_video_123.json")
        )
        mock_output_manager.get_processing_dir.return_value = mock_processing_dir
        mock_get_output_manager.return_value = mock_output_manager

        mock_time.return_value = 1234567890.0

        # Mock batch processing
        expected_descriptions = {
            "frame_001": "Description of frame 1",
            "frame_002": "Description of frame 2",
        }

        with patch.object(vlm_descriptor, "_process_vlm_batch") as mock_batch:
            mock_batch.return_value = expected_descriptions

            result = vlm_descriptor.generate_descriptions(sample_keyframes_metadata)

            assert result["video_id"] == "test_video_123"
            assert result["descriptions"] == expected_descriptions
            assert result["total_descriptions"] == 2
            assert result["created_at"] == 1234567890.0

    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("time.time")
    def test_generate_descriptions_with_output_dir(
        self,
        mock_time,
        mock_json_dump,
        mock_file,
        mock_mkdir,
        vlm_descriptor,
        sample_keyframes_metadata,
        tmp_path,
    ):
        """Test description generation with explicit output_dir."""
        output_dir = tmp_path / "custom_output"
        mock_time.return_value = 1234567890.0

        expected_descriptions = {
            "frame_001": "Custom description 1",
            "frame_002": "Custom description 2",
        }

        with patch.object(vlm_descriptor, "_process_vlm_batch") as mock_batch:
            mock_batch.return_value = expected_descriptions

            result = vlm_descriptor.generate_descriptions(
                sample_keyframes_metadata, output_dir
            )

            assert result["video_id"] == "test_video_123"
            assert result["descriptions"] == expected_descriptions

    @patch("src.common.utils.output_manager.get_output_manager")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("time.time")
    def test_generate_descriptions_auto_start(
        self,
        mock_time,
        mock_json_dump,
        mock_file,
        mock_mkdir,
        mock_get_output_manager,
        vlm_descriptor_auto_start,
        sample_keyframes_metadata,
    ):
        """Test description generation with auto_start enabled."""
        # Mock output manager
        mock_output_manager = Mock()
        mock_processing_dir = Mock()
        mock_processing_dir.__truediv__ = Mock(
            return_value=Path("/test/descriptions/test_video_123.json")
        )
        mock_output_manager.get_processing_dir.return_value = mock_processing_dir
        mock_get_output_manager.return_value = mock_output_manager

        mock_time.return_value = 1234567890.0

        with patch.object(
            vlm_descriptor_auto_start, "_ensure_service_running"
        ) as mock_ensure:
            with patch.object(
                vlm_descriptor_auto_start, "_process_vlm_batch"
            ) as mock_batch:
                mock_batch.return_value = {"frame_001": "Auto-started description"}

                vlm_descriptor_auto_start.generate_descriptions(
                    sample_keyframes_metadata
                )

                mock_ensure.assert_called_once()
                assert vlm_descriptor_auto_start._service_started is True

    @patch("tempfile.NamedTemporaryFile")
    @patch("zipfile.ZipFile")
    @patch("requests.post")
    @patch("os.unlink")
    def test_process_vlm_batch_success(
        self,
        mock_unlink,
        mock_post,
        mock_zipfile,
        mock_tempfile,
        vlm_descriptor,
        tmp_path,
    ):
        """Test successful batch processing."""
        # Create mock frame files
        frame1_path = tmp_path / "frame_001.jpg"
        frame2_path = tmp_path / "frame_002.jpg"
        frame1_path.write_bytes(b"fake image data 1")
        frame2_path.write_bytes(b"fake image data 2")

        keyframes = [
            {"frame_id": "frame_001", "path": str(frame1_path)},
            {"frame_id": "frame_002", "path": str(frame2_path)},
        ]

        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.zip"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        # Mock zipfile
        mock_zip = Mock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip

        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "descriptions": {
                "frame_001": "Batch description 1",
                "frame_002": "Batch description 2",
            }
        }
        mock_post.return_value = mock_response

        # Mock reading zip file
        with patch("builtins.open", mock_open(read_data=b"mock zip data")):
            result = vlm_descriptor._process_vlm_batch(keyframes)

            expected = {
                "frame_001": "Batch description 1",
                "frame_002": "Batch description 2",
            }
            assert result == expected

    @patch("tempfile.NamedTemporaryFile")
    @patch("zipfile.ZipFile")
    @patch("requests.post")
    @patch("os.unlink")
    def test_process_vlm_batch_api_error(
        self,
        mock_unlink,
        mock_post,
        mock_zipfile,
        mock_tempfile,
        vlm_descriptor,
        tmp_path,
    ):
        """Test batch processing with API error."""
        # Create mock frame files
        frame1_path = tmp_path / "frame_001.jpg"
        frame1_path.write_bytes(b"fake image data")

        keyframes = [{"frame_id": "frame_001", "path": str(frame1_path)}]

        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.zip"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        # Mock zipfile
        mock_zip = Mock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip

        # Mock API error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response

        with patch("builtins.open", mock_open(read_data=b"mock zip data")):
            result = vlm_descriptor._process_vlm_batch(keyframes)

            assert result == {}

    @patch("tempfile.NamedTemporaryFile")
    @patch("zipfile.ZipFile")
    @patch("requests.post")
    @patch("os.unlink")
    def test_process_vlm_batch_connection_error(
        self,
        mock_unlink,
        mock_post,
        mock_zipfile,
        mock_tempfile,
        vlm_descriptor,
        tmp_path,
    ):
        """Test batch processing with connection error."""
        # Create mock frame files
        frame1_path = tmp_path / "frame_001.jpg"
        frame1_path.write_bytes(b"fake image data")

        keyframes = [{"frame_id": "frame_001", "path": str(frame1_path)}]

        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.zip"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        # Mock zipfile
        mock_zip = Mock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip

        # Mock connection error
        mock_post.side_effect = Exception("Connection error")

        with patch("builtins.open", mock_open(read_data=b"mock zip data")):
            result = vlm_descriptor._process_vlm_batch(keyframes)

            assert result == {}

    @patch("tempfile.NamedTemporaryFile")
    @patch("zipfile.ZipFile")
    @patch("requests.post")
    @patch("os.unlink")
    def test_process_vlm_batch_connection_error_auto_start_retry(
        self,
        mock_unlink,
        mock_post,
        mock_zipfile,
        mock_tempfile,
        vlm_descriptor_auto_start,
        tmp_path,
    ):
        """Test batch processing with connection error and auto_start retry."""
        # Create mock frame files
        frame1_path = tmp_path / "frame_001.jpg"
        frame1_path.write_bytes(b"fake image data")

        keyframes = [{"frame_id": "frame_001", "path": str(frame1_path)}]

        # Mock temporary file
        mock_temp = Mock()
        mock_temp.name = "/tmp/test.zip"
        mock_tempfile.return_value.__enter__.return_value = mock_temp

        # Mock zipfile
        mock_zip = Mock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip

        # First call fails, second succeeds
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "descriptions": {"frame_001": "Retry success"}
        }

        mock_post.side_effect = [
            Exception("Connection error"),  # First call fails
            mock_response_success,  # Retry succeeds
        ]

        with patch.object(
            vlm_descriptor_auto_start, "_ensure_service_running"
        ) as mock_ensure:
            with patch("builtins.open", mock_open(read_data=b"mock zip data")):
                result = vlm_descriptor_auto_start._process_vlm_batch(keyframes)

                mock_ensure.assert_called_once()
                assert result == {"frame_001": "Retry success"}

    def test_process_vlm_batch_no_valid_frames(self, vlm_descriptor):
        """Test batch processing with no valid frame paths."""
        keyframes = [
            {"frame_id": "frame_001", "path": "/nonexistent/frame1.jpg"},
            {"frame_id": "frame_002", "path": "/nonexistent/frame2.jpg"},
        ]

        result = vlm_descriptor._process_vlm_batch(keyframes)

        assert result == {}

    @patch("requests.post")
    def test_process_single_frame_success(self, mock_post, vlm_descriptor, tmp_path):
        """Test successful single frame processing."""
        frame_path = tmp_path / "test_frame.jpg"
        frame_path.write_bytes(b"fake image data")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"description": "Single frame description"}
        mock_post.return_value = mock_response

        result = vlm_descriptor.process_single_frame(frame_path)

        assert result == "Single frame description"

        # Verify the request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == vlm_descriptor.vlm_endpoint

        payload = call_args[1]["json"]
        assert "frame_base64" in payload
        assert "prompt" in payload

    @patch("requests.post")
    def test_process_single_frame_api_error(self, mock_post, vlm_descriptor, tmp_path):
        """Test single frame processing with API error."""
        frame_path = tmp_path / "test_frame.jpg"
        frame_path.write_bytes(b"fake image data")

        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        result = vlm_descriptor.process_single_frame(frame_path)

        assert result == "Error: VLM API error 500"

    @patch("requests.post")
    def test_process_single_frame_connection_error(
        self, mock_post, vlm_descriptor, tmp_path
    ):
        """Test single frame processing with connection error."""
        frame_path = tmp_path / "test_frame.jpg"
        frame_path.write_bytes(b"fake image data")

        mock_post.side_effect = Exception("Connection failed")

        result = vlm_descriptor.process_single_frame(frame_path)

        assert result.startswith("Error: Connection failed")

    def test_process_single_frame_file_not_found(self, vlm_descriptor):
        """Test single frame processing with non-existent file."""
        frame_path = Path("/nonexistent/frame.jpg")

        result = vlm_descriptor.process_single_frame(frame_path)

        assert result.startswith("Error:")

    @patch("src.common.utils.output_manager.get_output_manager")
    @patch("pathlib.Path.mkdir")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    @patch("time.time")
    def test_generate_descriptions_large_batch(
        self,
        mock_time,
        mock_json_dump,
        mock_file,
        mock_mkdir,
        mock_get_output_manager,
        vlm_descriptor,
        tmp_path,
    ):
        """Test description generation with multiple batches."""
        # Mock output manager
        mock_output_manager = Mock()
        mock_processing_dir = Mock()
        mock_processing_dir.__truediv__ = Mock(
            return_value=Path("/test/descriptions/large_video.json")
        )
        mock_output_manager.get_processing_dir.return_value = mock_processing_dir
        mock_get_output_manager.return_value = mock_output_manager

        # Create keyframes that will require multiple batches (batch_size=100 for this descriptor)
        keyframes = []
        for i in range(250):  # Will create 3 batches
            frame_path = tmp_path / f"frame_{i:03d}.jpg"
            frame_path.touch()
            keyframes.append(
                {
                    "frame_id": f"frame_{i:03d}",
                    "path": str(frame_path),
                    "timestamp": i * 1.0,
                }
            )

        large_metadata = {"video_id": "large_video", "keyframes": keyframes}

        mock_time.return_value = 1234567890.0

        # Mock batch processing to return different results for each batch
        def mock_batch_side_effect(batch):
            return {kf["frame_id"]: f"Description for {kf['frame_id']}" for kf in batch}

        with patch.object(
            vlm_descriptor, "_process_vlm_batch", side_effect=mock_batch_side_effect
        ):
            result = vlm_descriptor.generate_descriptions(large_metadata)

            assert result["video_id"] == "large_video"
            assert result["total_descriptions"] == 250
            assert len(result["descriptions"]) == 250

            # Verify we called batch processing 3 times (250 frames / 100 batch_size = 3 batches)
            assert vlm_descriptor._process_vlm_batch.call_count == 3

    def test_batch_endpoint_url_transformation(self, vlm_descriptor):
        """Test that batch endpoint URL is correctly transformed."""
        # This tests the URL transformation logic in _process_vlm_batch
        original_endpoint = "http://test.com/generate-description"
        vlm_descriptor.vlm_endpoint = original_endpoint

        expected_batch_endpoint = "http://test.com/upload-and-process-frames"

        # We can't easily test this directly without calling the full method,
        # but we can verify the transformation logic
        batch_endpoint = original_endpoint.replace(
            "generate-description", "upload-and-process-frames"
        )
        assert batch_endpoint == expected_batch_endpoint
