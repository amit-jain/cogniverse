"""Shared media-resolution helpers for visual evaluators.

The three visual judges (`ConfigurableVisualJudge`, `Qwen2VLVisualJudge`,
`VisualRelevanceEvaluator`) historically reimplemented the same path-probing
logic against hardcoded local directories. That broke them in any environment
where the source videos / pre-extracted frames did not happen to live under
``data/testset/...`` — pods, CI, anything not run from the repo root.

These helpers centralize the resolution path through :class:`MediaLocator`:

- ``source_url`` (canonical URI written by ingestion) is the primary source
  of truth.
- For pre-extracted frames a ``frame_path`` field is consulted first; if
  absent, a single frame is extracted on the fly from ``source_url`` at
  ``result.get("frame_id", 0)``.

Frame extraction (cv2.VideoCapture → JPEGs) lives in :func:`extract_frames` so
all three judges share one implementation and none of them assume someone
else pre-decoded.
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
    """Return a local Path for the video referenced by ``result``.

    Resolution order:
      1. ``result["source_url"]`` via :meth:`MediaLocator.localize`.
      2. ``result["video_path"]`` (treat as URI if it has a scheme, else as a
         local path that the locator canonicalizes to ``file://``).

    Returns ``None`` when neither succeeds.
    """
    if not isinstance(result, dict):
        return None

    uri = result.get("source_url")
    if uri:
        try:
            return locator.localize(uri)
        except (FileNotFoundError, ValueError, OSError) as exc:
            logger.warning("Failed to localize source_url %s: %s", uri, exc)

    raw_path = result.get("video_path")
    if raw_path:
        try:
            return locator.localize(locator.to_canonical_uri(str(raw_path)))
        except (FileNotFoundError, ValueError, OSError) as exc:
            logger.warning("Failed to localize video_path %s: %s", raw_path, exc)

    return None


def resolve_frame_from_result(
    result: dict[str, Any],
    locator: MediaLocator,
) -> Optional[Path]:
    """Return a local Path for the frame referenced by ``result``.

    Resolution order:
      1. ``result["frame_path"]`` if it exists locally (or via the locator
         if it is a URI).
      2. Extract a frame on the fly from ``result["source_url"]`` at
         ``result.get("frame_id", 0)``.

    Returns ``None`` when neither succeeds.
    """
    if not isinstance(result, dict):
        return None

    frame_path = result.get("frame_path")
    if frame_path:
        try:
            return locator.localize(locator.to_canonical_uri(str(frame_path)))
        except (FileNotFoundError, ValueError, OSError) as exc:
            logger.warning("Failed to localize frame_path %s: %s", frame_path, exc)

    video_path = resolve_video_from_result(result, locator)
    if video_path is not None:
        frame_id = int(result.get("frame_id", 0) or 0)
        frames = extract_frames(video_path, num_frames=1, frame_index=frame_id)
        if frames:
            return frames[0]

    return None


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
