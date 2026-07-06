"""Answer-time keyframe injection.

Turns retrieved video/image search hits into ``dspy.Image`` inputs so an
answer-generation agent grounds its output in the frames the hits actually
represent, not just their titles and scores. Shared by every agent whose LLM
answers over retrieved visual content (detailed report, summarizer, deep
research) so they derive and fetch keyframes identically — a divergent
derivation would silently make frames unfetchable for one agent but not
another.

The keyframe's object key comes from the one shared contract in
``cogniverse_core.common.media.keyframes`` (the same function the ingestion
write side uses). The bucket and tenant are read from the hit's own
``source_url`` (``s3://{bucket}/{tenant_id}/{uuid}.ext``), so a keyframe is
always looked up in the same bucket its video was uploaded to.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Iterable, Optional

import dspy

from cogniverse_core.common.media import MediaLocator, keyframe_uri

logger = logging.getLogger(__name__)


def _bucket_and_tenant(source_url: str) -> Optional[tuple[str, str]]:
    """Parse ``(bucket, tenant_id)`` from an ``s3://bucket/tenant_id/…`` URL.

    Returns None for any non-s3 URL (local/http videos have no MinIO keyframes)
    or a layout that lacks both a bucket and a tenant segment.
    """
    if not source_url.startswith("s3://"):
        return None
    parts = source_url[len("s3://") :].split("/")
    if len(parts) < 3 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def hit_keyframe_uri(hit: dict[str, Any]) -> Optional[str]:
    """Derive the ``s3://`` keyframe URI for a search hit, or None if the hit
    lacks the fields needed to locate one."""
    bt = _bucket_and_tenant(str(hit.get("source_url") or ""))
    if bt is None:
        return None
    bucket, tenant_id = bt
    video_id = hit.get("video_id")
    segment_id = hit.get("segment_id")
    if not video_id or segment_id is None:
        return None
    try:
        return keyframe_uri(bucket, tenant_id, str(video_id), segment_id)
    except (ValueError, TypeError):
        return None


class KeyframeImageResolver:
    """Collects top-K keyframe images from search hits, with a bounded encode
    cache so repeated answers over the same clips reuse the encoding.

    A keyframe that is missing (not uploaded yet) or fails with a transient
    IO/network error is skipped — a missing frame degrades to a text-only
    answer, it does not break generation. An unexpected error (bad URI scheme,
    a bug) is NOT swallowed: it propagates so real misconfiguration surfaces
    instead of every report silently losing its frames.
    """

    def __init__(self, locator: MediaLocator, cache_size: int = 64):
        self._locator = locator
        self._cache: "OrderedDict[str, dspy.Image]" = OrderedDict()
        self._cache_size = max(1, cache_size)

    def collect(
        self, hits: Iterable[dict[str, Any]], max_images: int
    ) -> list[dspy.Image]:
        images: list[dspy.Image] = []
        if max_images <= 0:
            return images
        for hit in hits:
            if len(images) >= max_images:
                break
            uri = hit_keyframe_uri(hit)
            if uri is None:
                continue
            img = self._image_for(uri)
            if img is not None:
                images.append(img)
        return images

    def _image_for(self, uri: str) -> Optional[dspy.Image]:
        cached = self._cache.get(uri)
        if cached is not None:
            self._cache.move_to_end(uri)
            return cached
        try:
            path = self._locator.localize(uri)
        except FileNotFoundError:
            return None  # keyframe not in object storage yet
        except OSError as e:
            logger.warning("keyframe fetch failed for %s: %r", uri, e)
            return None
        img = dspy.Image.from_file(str(path))
        self._cache[uri] = img
        self._cache.move_to_end(uri)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return img
