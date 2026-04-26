"""Shared media-resolution helpers for visual evaluators.

``source_url`` is the single canonical pointer. Visual evaluators feed result
dicts (originating from search) into these helpers; the helpers resolve the
URI through :class:`MediaLocator` and either return a local video path or
extract a single frame on demand. There are no alternative fields and no
fallback probes — if ``source_url`` cannot be resolved, resolution returns
``None`` so the caller can take the explicit no-frames branch.

Frame extraction (cv2.VideoCapture → JPEGs) lives in :func:`extract_frames`
so all three judges share one implementation.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Optional

from cogniverse_core.common.media import MediaLocator

logger = logging.getLogger(__name__)


def resolve_video_from_result(
    result: dict[str, Any], locator: MediaLocator
) -> Optional[Path]:
    """Return a local Path for the video referenced by ``result["source_url"]``.

    Returns ``None`` when ``source_url`` is missing/empty or cannot be
    localized. ``source_url`` is the only field consulted — there are no
    alternative fields and no probe paths.
    """
    uri = result.get("source_url")
    if not uri:
        return None
    try:
        return locator.localize(uri)
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.warning("Failed to localize source_url %s: %s", uri, exc)
        return None


def resolve_frame_from_result(
    result: dict[str, Any],
    locator: MediaLocator,
) -> Optional[Path]:
    """Return a local Path for a single frame extracted from ``source_url``.

    Frame index comes from ``result.get("frame_id", 0)``. Returns ``None``
    when the video cannot be resolved or cv2 cannot decode the frame.
    """
    video_path = resolve_video_from_result(result, locator)
    if video_path is None:
        return None
    frame_id = int(result.get("frame_id", 0) or 0)
    frames = extract_frames(video_path, num_frames=1, frame_index=frame_id)
    return frames[0] if frames else None


def extract_frames(
    video_path: Path,
    num_frames: int = 30,
    timestamp: float = 0,
    sample_all: bool = False,
    max_total_frames: int = 60,
    frame_index: Optional[int] = None,
) -> list[Path]:
    """Extract frames from a local video file.

    Args:
        video_path: Local path to the video (cv2/ffmpeg need a real path).
        num_frames: Number of frames to extract.
        timestamp: Starting timestamp in seconds.
        sample_all: If True, sample evenly across the whole video up to
            ``max_total_frames``.
        max_total_frames: Cap when ``sample_all`` is True.
        frame_index: When set, return exactly that single frame index. Used
            by ``resolve_frame_from_result`` for on-the-fly extraction of a
            specific frame.

    Returns the list of extracted frame paths (JPEG tempfiles). Empty list on
    decode failure.
    """
    import cv2

    frames: list[Path] = []
    cap: Optional[Any] = None
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("cv2 failed to open %s", video_path)
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 1.0

        if frame_index is not None:
            indices = [max(0, min(frame_index, max(total_frames - 1, 0)))]
        else:
            start_frame = int(timestamp * fps) if timestamp > 0 else 0
            usable = max(0, total_frames - start_frame)
            if usable == 0:
                return []
            count = min(usable, max_total_frames if sample_all else num_frames)
            interval = max(1, usable // max(count, 1))
            indices = [start_frame + (i * interval) for i in range(count)]
            indices = [i for i in indices if i < total_frames]

        for frame_no in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ok, frame = cap.read()
            if not ok:
                continue
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)
                frames.append(Path(tmp.name))
    except Exception as exc:
        logger.error("Frame extraction failed for %s: %s", video_path, exc)
    finally:
        if cap is not None:
            cap.release()

    return frames
