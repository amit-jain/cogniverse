"""
Real integration test for video ingestion pipeline.

Tests the full keyframe extraction path using a real sample video.
Skips gracefully if no sample videos are available.
"""

import logging
import shutil
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_SAMPLE_VIDEO_DIR = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "testset"
    / "evaluation"
    / "sample_videos"
)


def _sample_video() -> Path | None:
    """Return path to the smallest available sample video, or None."""
    if not _SAMPLE_VIDEO_DIR.exists():
        return None
    candidates = sorted(_SAMPLE_VIDEO_DIR.glob("*.mp4")) + sorted(
        _SAMPLE_VIDEO_DIR.glob("*.mkv")
    )
    return candidates[0] if candidates else None


_video = _sample_video()

skip_if_no_video = pytest.mark.skipif(
    _video is None,
    reason=f"No sample videos found in {_SAMPLE_VIDEO_DIR}",
)

skip_if_no_cv2 = pytest.mark.skipif(
    not __import__("importlib").util.find_spec("cv2"),
    reason="cv2 not installed",
)


@pytest.fixture(scope="module")
def sample_video_path():
    """Return path to the first available sample video."""
    return _video


@pytest.fixture
def output_dir(tmp_path):
    """Isolated output directory per test."""
    d = tmp_path / "pipeline_output"
    d.mkdir()
    yield d
    if d.exists():
        shutil.rmtree(d)


@pytest.mark.integration
@skip_if_no_video
@skip_if_no_cv2
def test_pipeline_processes_test_video(sample_video_path, output_dir):
    """
    Real video file must yield at least 1 extracted keyframe.

    Runs through KeyframeProcessor with max_frames=1 so the test
    completes quickly without downloading any models.
    """
    from cogniverse_runtime.ingestion.processors.keyframe_processor import (
        KeyframeProcessor,
    )

    logger = logging.getLogger("test_pipeline")

    processor = KeyframeProcessor(
        logger=logger,
        max_frames=1,
        threshold=0.9,
    )

    result = processor.extract_keyframes(sample_video_path, output_dir=output_dir)

    assert result is not None, "extract_keyframes returned None"
    assert "keyframes" in result, (
        f"Result missing 'keyframes' key. Keys: {list(result.keys())}"
    )

    keyframes = result["keyframes"]
    assert len(keyframes) >= 1, (
        f"Expected at least 1 keyframe from {sample_video_path.name}, got {len(keyframes)}"
    )

    # Verify keyframe file actually exists on disk
    first_frame = keyframes[0]
    frame_path_key = "path" if "path" in first_frame else "frame_path"
    frame_path = Path(first_frame[frame_path_key])
    assert frame_path.exists(), f"Keyframe file does not exist at {frame_path}"
