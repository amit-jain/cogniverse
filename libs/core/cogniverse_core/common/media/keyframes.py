"""Shared keyframe object-key contract.

The single source of truth for where a video's keyframe lives in object
storage, used by BOTH the ingestion write path (uploading each extracted
keyframe) and the answer-time agent read path (fetching a keyframe for the
answer LLM). Keeping it in one place prevents the two sides from drifting — a
divergent key would silently make every keyframe unfetchable.

``segment_id`` is the keyframe's ordinal during extraction: the same value
Vespa returns as ``segment_id`` on a search hit and the ``NNNN`` in the
extractor's ``{video_id}_keyframe_{NNNN:04d}.jpg`` filename.
"""

from __future__ import annotations


def keyframe_object_key(tenant_id: str, video_id: str, segment_id: int) -> str:
    """Bucket-relative object key for a keyframe.

    e.g. ``keyframe_object_key("acme:acme", "vid123", 7)`` ->
    ``"acme:acme/keyframes/vid123/0007.jpg"``.
    """
    if not tenant_id or not video_id:
        raise ValueError("tenant_id and video_id are required for a keyframe key")
    return f"{tenant_id}/keyframes/{video_id}/{int(segment_id):04d}.jpg"


def keyframe_uri(bucket: str, tenant_id: str, video_id: str, segment_id: int) -> str:
    """Full ``s3://`` URI a ``MediaLocator`` can ``localize``.

    e.g. ``keyframe_uri("media", "acme:acme", "vid123", 7)`` ->
    ``"s3://media/acme:acme/keyframes/vid123/0007.jpg"``.
    """
    if not bucket:
        raise ValueError("bucket is required for a keyframe URI")
    return f"s3://{bucket}/{keyframe_object_key(tenant_id, video_id, segment_id)}"
