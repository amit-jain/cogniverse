"""The sample and evaluation corpus the e2e session ingests, and what ingesting
it is expected to produce."""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pytest

DATA_ROOT = Path(__file__).parent.parent.parent / "data"


def _expected_chunk_count(
    duration_s: float, chunk_duration: float, chunk_overlap: float
) -> int:
    """Mirror ChunkProcessor.extract_chunks: one chunk per loop iteration from 0
    while start < duration, stepping by chunk_duration - chunk_overlap."""
    if duration_s <= 0:
        raise AssertionError(f"video duration must be positive, got {duration_s!r}")
    step = chunk_duration - chunk_overlap
    if step <= 0:
        raise AssertionError(
            f"chunk_duration {chunk_duration!r} must exceed chunk_overlap {chunk_overlap!r}"
        )
    count = 0
    start = 0.0
    while start < duration_s:
        count += 1
        start += step
    return count


def _video_duration_seconds(path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if video_fps <= 0 or total_frames <= 0:
        raise AssertionError(
            f"Could not determine duration for tracked video {path!r}: "
            f"fps={video_fps!r}, frames={total_frames!r}"
        )
    return total_frames / video_fps


def _expected_sample_documents_fed(path: Path, profile: str, media_type: str) -> int:
    if not media_type.startswith("video/"):
        return 1

    config_path = DATA_ROOT.parent / "configs" / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    profile_def = config.get("backend", {}).get("profiles", {}).get(profile, {})
    segmentation = profile_def.get("strategies", {}).get("segmentation", {})
    if segmentation.get("class") == "ChunkSegmentationStrategy":
        # Multi-vector chunk profiles feed one document per chunk
        # (strategy.py: num_patches > 1 -> multi_doc).
        params = segmentation.get("params", {})
        return _expected_chunk_count(
            _video_duration_seconds(path),
            float(params.get("chunk_duration", 30.0)),
            float(params.get("chunk_overlap", 0.0)),
        )
    pipeline_config = profile_def.get("pipeline_config", {}) if profile_def else {}
    target_fps = pipeline_config.get("keyframe_fps")
    if not isinstance(target_fps, (int, float)) or target_fps <= 0:
        target_fps = (
            profile_def.get("strategies", {})
            .get("segmentation", {})
            .get("params", {})
            .get("fps", 0.5)
        )
    if not isinstance(target_fps, (int, float)) or target_fps <= 0:
        raise AssertionError(
            f"Could not determine keyframe fps for profile {profile!r}: {profile_def}"
        )

    import cv2

    cap = cv2.VideoCapture(str(path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if video_fps <= 0 or total_frames <= 0:
        raise AssertionError(
            f"Could not determine frame count for tracked video {path!r}: "
            f"fps={video_fps!r}, frames={total_frames!r}"
        )
    frame_interval = int(video_fps / target_fps) if video_fps > target_fps else 1
    return sum(
        1 for frame_idx in range(total_frames) if frame_idx % frame_interval == 0
    )


EVALUATION_QUERY_ASSET = (
    DATA_ROOT / "testset" / "evaluation" / "sample_videos_retrieval_queries.json"
)


@functools.lru_cache(maxsize=1)
def _evaluation_query_rows() -> tuple[dict[str, object], ...]:
    rows = json.loads(EVALUATION_QUERY_ASSET.read_text())
    if not isinstance(rows, list):
        raise AssertionError(f"{EVALUATION_QUERY_ASSET} did not load a JSON list")
    return tuple(row for row in rows if isinstance(row, dict))


def profile_selection_corpus_videos() -> tuple[Path, ...]:
    """Every video the profile-selection truth asset references, sorted by id.

    Ids normalize through the same production helper the label rule uses, so
    the corpus and the labels cannot disagree about what counts as a video id.
    """
    from cogniverse_runtime.optimization_cli import _profile_selection_expected_videos

    sample_videos_dir = _EVALUATION_CORPUS_DIR / "evaluation" / "sample_videos"
    expected_ids = sorted(
        {
            video_id
            for row in _evaluation_query_rows()
            for video_id in _profile_selection_expected_videos(row)
        }
    )
    if not expected_ids:
        pytest.fail(
            f"Profile-selection truth asset {EVALUATION_QUERY_ASSET} yielded no "
            "expected videos"
        )

    missing_ids: list[str] = []
    duplicate_ids: list[str] = []
    corpus_paths: list[Path] = []
    for video_id in expected_ids:
        matches = sorted(
            path for path in sample_videos_dir.glob(f"{video_id}.*") if path.is_file()
        )
        if len(matches) == 1:
            corpus_paths.append(matches[0])
        elif not matches:
            missing_ids.append(video_id)
        else:
            duplicate_ids.append(video_id)
    if missing_ids or duplicate_ids:
        details = []
        if missing_ids:
            details.append(f"missing ids: {missing_ids!r}")
        if duplicate_ids:
            details.append(f"duplicate ids: {duplicate_ids!r}")
        pytest.fail(
            f"Profile-selection sample video corpus mismatch in {sample_videos_dir}: "
            + "; ".join(details)
        )
    return tuple(corpus_paths)


_SAMPLE_VIDEO_MEDIA_TYPES = {".mp4": "video/mp4", ".mkv": "video/x-matroska"}


def _sample_video_media_type(path: Path) -> str:
    """Upload MIME for a sampled video, by suffix (no system mime database)."""
    try:
        return _SAMPLE_VIDEO_MEDIA_TYPES[path.suffix.lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported sample video suffix {path.suffix!r} for {path.name!r}"
        ) from None


_EVALUATION_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "testset"


_EVALUATION_TEXT_CORPUS_DIR = _EVALUATION_CORPUS_DIR / "evaluation" / "processed"


def _evaluation_text_corpus_paths() -> tuple[Path, ...]:
    # sample_videos_retrieval_queries.json is deliberately NOT ingested. It is the
    # ground truth this tenant is evaluated against -- profile labels derive from its
    # expected_videos and the quality monitor uses it as its golden set. Ingesting it
    # puts a document holding every evaluation query verbatim into the corpus being
    # searched, so it matches any of those queries by construction and outranks the
    # content that should answer them.
    return (
        _EVALUATION_CORPUS_DIR / "dataset_summary.md",
        *_sorted_evaluation_corpus_paths("descriptions"),
        *_sorted_evaluation_corpus_paths("transcripts"),
    )


def _sorted_evaluation_corpus_paths(subdir: str) -> tuple[Path, ...]:
    return tuple(sorted((_EVALUATION_TEXT_CORPUS_DIR / subdir).glob("*.json")))
