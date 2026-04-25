"""Tests for the shared evaluator media-resolution helpers."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cogniverse_core.common.media import MediaConfig, MediaLocator
from cogniverse_evaluation.evaluators._media_helpers import (
    extract_frames,
    resolve_frame_from_result,
    resolve_video_from_result,
)


@pytest.fixture
def locator(tmp_path):
    return MediaLocator(
        tenant_id="test", config=MediaConfig(), cache_root=tmp_path / "cache"
    )


@pytest.fixture
def fake_video(tmp_path):
    """Create a 5-frame test video using cv2 so extract_frames has real bytes to decode."""
    import cv2
    import numpy as np

    p = tmp_path / "v.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(p), fourcc, 5.0, (32, 32))
    for i in range(5):
        frame = np.full((32, 32, 3), i * 30, dtype=np.uint8)
        out.write(frame)
    out.release()
    return p


class TestResolveVideoFromResult:
    def test_source_url_takes_precedence(self, locator, tmp_path):
        clip = tmp_path / "v.mp4"
        clip.write_bytes(b"video")

        result = resolve_video_from_result(
            {"source_url": f"file://{clip}", "video_id": "v"}, locator
        )
        assert result == clip

    def test_video_path_used_when_no_source_url(self, locator, tmp_path):
        clip = tmp_path / "v.mp4"
        clip.write_bytes(b"video")

        result = resolve_video_from_result({"video_path": str(clip)}, locator)
        assert result == clip

    def test_legacy_probe_when_neither_set(
        self, locator, tmp_path, monkeypatch, caplog
    ):
        sample_dir = tmp_path / "data" / "testset" / "evaluation" / "sample_videos"
        sample_dir.mkdir(parents=True)
        clip = sample_dir / "v_legacy.mp4"
        clip.write_bytes(b"")
        monkeypatch.chdir(tmp_path)

        with caplog.at_level("WARNING"):
            result = resolve_video_from_result({"video_id": "v_legacy"}, locator)

        assert result == Path("data/testset/evaluation/sample_videos/v_legacy.mp4")
        assert any("legacy" in m.lower() for m in caplog.messages)

    def test_returns_none_when_nothing_resolves(self, locator):
        assert (
            resolve_video_from_result({"video_id": "does_not_exist"}, locator) is None
        )

    def test_non_dict_returns_none(self, locator):
        assert resolve_video_from_result("not a dict", locator) is None
        assert resolve_video_from_result(None, locator) is None


class TestResolveFrameFromResult:
    def test_frame_path_takes_precedence(self, locator, tmp_path):
        frame = tmp_path / "f.jpg"
        frame.write_bytes(b"jpeg")

        result = resolve_frame_from_result({"frame_path": str(frame)}, locator)
        assert result == frame

    def test_extracts_from_source_url_when_no_frame_path(self, locator, fake_video):
        result = resolve_frame_from_result(
            {"source_url": f"file://{fake_video}", "frame_id": 2}, locator
        )
        assert result is not None
        assert result.exists()
        assert result.suffix == ".jpg"

    def test_legacy_probe_for_pre_extracted_frame(
        self, locator, tmp_path, monkeypatch, caplog
    ):
        frames_dir = tmp_path / "data" / "frames" / "v_legacy"
        frames_dir.mkdir(parents=True)
        (frames_dir / "frame_3.jpg").write_bytes(b"")
        monkeypatch.chdir(tmp_path)

        with caplog.at_level("WARNING"):
            result = resolve_frame_from_result(
                {"video_id": "v_legacy", "frame_id": 3}, locator
            )

        assert result == Path("data/frames/v_legacy/frame_3.jpg")
        assert any("legacy" in m.lower() for m in caplog.messages)

    def test_returns_none_when_nothing_resolves(self, locator):
        assert resolve_frame_from_result({"video_id": "missing"}, locator) is None


class TestExtractFrames:
    def test_extracts_n_frames(self, fake_video):
        frames = extract_frames(fake_video, num_frames=3)
        assert len(frames) == 3
        for p in frames:
            assert p.exists()
            assert p.suffix == ".jpg"

    def test_specific_frame_index(self, fake_video):
        frames = extract_frames(fake_video, frame_index=2)
        assert len(frames) == 1

    def test_invalid_video_returns_empty(self, tmp_path):
        bogus = tmp_path / "bogus.mp4"
        bogus.write_bytes(b"not a video")
        assert extract_frames(bogus, num_frames=3) == []

    def test_sample_all_caps_at_max_total(self, fake_video):
        frames = extract_frames(fake_video, sample_all=True, max_total_frames=2)
        assert len(frames) <= 2


class TestSourceUrlAcrossAllJudges:
    """Pre-fix code would have failed these — read paths broken in any non-dev env."""

    def test_resolve_video_finds_minio_uri_via_locator(self, monkeypatch, tmp_path):
        clip = tmp_path / "v.mp4"
        clip.write_bytes(b"video")

        fake_locator = MagicMock()
        fake_locator.localize.return_value = clip
        fake_locator.to_canonical_uri.side_effect = lambda raw: (
            raw if "://" in raw else f"file://{Path(raw).resolve()}"
        )

        result = resolve_video_from_result(
            {"source_url": "s3://corpus/v.mp4"}, fake_locator
        )

        assert result == clip
        fake_locator.localize.assert_called_once_with("s3://corpus/v.mp4")

    def test_resolve_frame_extracts_from_uri_when_no_frame_path(
        self, locator, fake_video
    ):
        result = resolve_frame_from_result(
            {"source_url": f"file://{fake_video}", "frame_id": 1}, locator
        )
        assert result is not None
        assert result.suffix == ".jpg"
