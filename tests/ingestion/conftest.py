"""
Shared fixtures for ingestion tests.

Provides common test utilities, mock objects, and fixtures
for testing the video processing pipeline.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

# Test data constants
TEST_VIDEO_WIDTH = 640
TEST_VIDEO_HEIGHT = 480
TEST_VIDEO_FPS = 30
TEST_VIDEO_DURATION = 5  # seconds


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock(spec=logging.Logger)
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    return logger


@pytest.fixture
def sample_video_path(temp_dir):
    """Create a sample video file for testing."""
    video_path = temp_dir / "test_video.mp4"

    # Create a simple test video using OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(video_path), fourcc, TEST_VIDEO_FPS, (TEST_VIDEO_WIDTH, TEST_VIDEO_HEIGHT)
    )

    # Generate frames with different colors to test keyframe extraction
    total_frames = TEST_VIDEO_FPS * TEST_VIDEO_DURATION
    for i in range(total_frames):
        # Create frames with varying colors
        hue = int((i / total_frames) * 180)  # HSV hue varies from 0-180
        frame = np.ones((TEST_VIDEO_HEIGHT, TEST_VIDEO_WIDTH, 3), dtype=np.uint8)
        frame[:, :] = [hue, 255, 255]  # HSV color
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        out.write(frame)

    out.release()
    return video_path


@pytest.fixture
def sample_audio_transcript():
    """Sample audio transcript for testing."""
    return {
        "text": "This is a test transcript for video processing.",
        "segments": [
            {
                "start": 0.0,
                "end": 2.5,
                "text": "This is a test transcript",
            },
            {
                "start": 2.5,
                "end": 5.0,
                "text": "for video processing.",
            },
        ],
        "language": "en",
    }


@pytest.fixture
def sample_keyframes_data():
    """Sample keyframes metadata for testing."""
    return {
        "video_id": "test_video",
        "total_keyframes": 3,
        "extraction_method": "histogram",
        "keyframes": [
            {
                "frame_index": 0,
                "timestamp": 0.0,
                "filename": "test_video_keyframe_0000.jpg",
                "histogram_difference": 0.0,
            },
            {
                "frame_index": 75,
                "timestamp": 2.5,
                "filename": "test_video_keyframe_0001.jpg",
                "histogram_difference": 0.8,
            },
            {
                "frame_index": 150,
                "timestamp": 5.0,
                "filename": "test_video_keyframe_0002.jpg",
                "histogram_difference": 0.9,
            },
        ],
    }


@pytest.fixture
def sample_embedding_vector():
    """Sample embedding vector for testing."""
    return np.random.rand(128).astype(np.float32)


@pytest.fixture
def sample_binary_embedding():
    """Sample binary embedding for testing."""
    return np.random.randint(0, 256, size=16, dtype=np.uint8)


@pytest.fixture
def mock_processor_config():
    """Sample processor configuration."""
    return {
        "keyframe": {"threshold": 0.999, "max_frames": 100, "fps": None},
        "audio": {"model": "whisper-base", "language": "auto"},
        "chunk": {"chunk_duration": 30.0, "chunk_overlap": 2.0, "cache_chunks": False},
        "embedding": {
            "model_name": "test-embedding-model",
            "batch_size": 32,
            "embedding_dim": 128,
        },
    }


@pytest.fixture
def mock_cache_manager():
    """Mock cache manager for testing."""
    cache = Mock()
    cache.get = Mock(return_value=None)
    cache.set = Mock()
    cache.exists = Mock(return_value=False)
    cache.delete = Mock()
    return cache


@pytest.fixture
def mock_config():
    """Mock global configuration."""
    config = {
        "cache": {"enabled": True, "type": "file", "base_dir": "/tmp/test_cache"},
        "models": {"whisper": {"model": "base"}},
        "processing": {"max_concurrent_videos": 2},
    }
    return config


@pytest.fixture
def mock_cv2_video_capture():
    """Mock OpenCV VideoCapture for testing."""
    with patch("cv2.VideoCapture") as mock_cap:
        mock_instance = Mock()
        mock_instance.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: TEST_VIDEO_FPS,
            cv2.CAP_PROP_FRAME_COUNT: TEST_VIDEO_FPS * TEST_VIDEO_DURATION,
            cv2.CAP_PROP_FRAME_WIDTH: TEST_VIDEO_WIDTH,
            cv2.CAP_PROP_FRAME_HEIGHT: TEST_VIDEO_HEIGHT,
        }.get(prop, 0)

        mock_instance.isOpened.return_value = True
        mock_instance.release.return_value = None

        # Mock frame reading
        frame_count = 0

        def mock_read():
            nonlocal frame_count
            if frame_count < TEST_VIDEO_FPS * TEST_VIDEO_DURATION:
                frame = np.random.randint(
                    0, 255, (TEST_VIDEO_HEIGHT, TEST_VIDEO_WIDTH, 3), dtype=np.uint8
                )
                frame_count += 1
                return True, frame
            return False, None

        mock_instance.read.side_effect = mock_read
        mock_cap.return_value = mock_instance
        yield mock_cap


@pytest.fixture
def mock_whisper():
    """Mock Whisper model for audio transcription testing."""
    with patch("whisper.load_model") as mock_load:
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "This is a test transcript.",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "This is a test transcript."}
            ],
            "language": "en",
        }
        mock_load.return_value = mock_model
        yield mock_model


@pytest.fixture
def mock_torch():
    """Mock PyTorch for embedding model testing."""
    with (
        patch("torch.tensor") as mock_tensor,
        patch("torch.load") as mock_load,
        patch("torch.no_grad"),
    ):

        mock_tensor.return_value = Mock()
        mock_load.return_value = Mock()
        yield


class MockProcessor:
    """Base mock processor for testing."""

    def __init__(self, name: str, logger: logging.Logger, **kwargs):
        self.name = name
        self.logger = logger
        self._config = kwargs
        self.PROCESSOR_NAME = name

    def get_config(self) -> Dict[str, Any]:
        return self._config


@pytest.fixture
def mock_base_processor():
    """Mock base processor for testing."""
    return MockProcessor


# Pytest markers for ingestion tests
pytestmark = [pytest.mark.ingestion]
