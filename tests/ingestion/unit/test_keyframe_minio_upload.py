"""Keyframe write→read key agreement.

The ingestion write side (``minio_client.upload_keyframes``) and the
answer-time read side (``multimodal.hit_keyframe_uri``) must derive the SAME
object key from the same (tenant, video, segment) — otherwise every uploaded
frame is unfetchable. Both go through the one shared ``keyframe_object_key``
contract; this pins that they agree end to end.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from cogniverse_agents.multimodal import hit_keyframe_uri
from cogniverse_runtime.ingestion.processing_strategy_set import ProcessingStrategySet
from cogniverse_runtime.ingestion_worker import minio_client


def test_upload_keyframes_key_matches_agent_read_side(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"vid1_keyframe_{i:04d}.jpg"
        p.write_bytes(b"jpeg-bytes")
        paths.append(str(p))

    fake_client = MagicMock()
    with patch.object(minio_client, "_client", return_value=fake_client):
        uris = minio_client.upload_keyframes(
            tenant_id="acme:acme",
            video_id="vid1",
            keyframe_paths=paths,
            bucket="media",
        )

    # write side: exact keys + exact returned URIs, in segment order
    written_keys = [c.kwargs["Key"] for c in fake_client.put_object.call_args_list]
    assert written_keys == [
        "acme:acme/keyframes/vid1/0000.jpg",
        "acme:acme/keyframes/vid1/0001.jpg",
        "acme:acme/keyframes/vid1/0002.jpg",
    ]
    assert uris == [
        "s3://media/acme:acme/keyframes/vid1/0000.jpg",
        "s3://media/acme:acme/keyframes/vid1/0001.jpg",
        "s3://media/acme:acme/keyframes/vid1/0002.jpg",
    ]
    # jpeg bytes + content type actually went to the store
    assert fake_client.put_object.call_args_list[0].kwargs["Body"] == b"jpeg-bytes"
    assert (
        fake_client.put_object.call_args_list[0].kwargs["ContentType"] == "image/jpeg"
    )

    # read side: a hit carrying that source_url bucket/tenant + video_id +
    # segment_id derives the SAME uri the write side produced
    for segment_id, uri in enumerate(uris):
        hit = {
            "source_url": "s3://media/acme:acme/some-upload.mp4",
            "video_id": "vid1",
            "segment_id": segment_id,
        }
        assert hit_keyframe_uri(hit) == uri


def test_upload_keyframes_uses_default_bucket(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIO_DEFAULT_BUCKET", "corpus")
    p = tmp_path / "v_keyframe_0000.jpg"
    p.write_bytes(b"x")
    fake_client = MagicMock()
    with patch.object(minio_client, "_client", return_value=fake_client):
        uris = minio_client.upload_keyframes(
            tenant_id="t:t", video_id="v", keyframe_paths=[str(p)]
        )
    assert uris == ["s3://corpus/t:t/keyframes/v/0000.jpg"]


def _keyframe_strategy():
    strat = Mock()
    strat.get_required_processors.return_value = ["keyframe"]
    return strat


@pytest.mark.asyncio
async def test_segmentation_uploads_keyframes_on_fresh_extraction(tmp_path):
    """A fresh keyframe extraction uploads the frames to object storage."""
    result = {"keyframes": [{"path": "y", "frame_id": 0}]}
    ctx = Mock()
    ctx.get_cached_keyframes = AsyncMock(return_value=None)
    ctx.set_cached_keyframes = AsyncMock()
    ctx.upload_keyframes_to_object_store = Mock()
    ctx.logger = Mock()
    ctx.profile_output_dir = tmp_path
    proc = Mock()
    proc.extract_keyframes = Mock(return_value=result)
    pm = Mock()
    pm.get_processor.return_value = proc
    video = tmp_path / "v.mp4"

    out = await ProcessingStrategySet()._process_segmentation(
        _keyframe_strategy(), video, pm, ctx
    )

    assert out == {"keyframes": result}
    ctx.upload_keyframes_to_object_store.assert_called_once_with(video, result)


@pytest.mark.asyncio
async def test_segmentation_uploads_keyframes_on_cache_hit(tmp_path):
    """A cache hit ALSO uploads — a re-ingest, or a video first ingested before
    this wiring existed, must still land its keyframes in object storage."""
    cached = {"keyframes": [{"path": "x", "frame_id": 0}]}
    ctx = Mock()
    ctx.get_cached_keyframes = AsyncMock(return_value=cached)
    ctx.upload_keyframes_to_object_store = Mock()
    ctx.logger = Mock()
    pm = Mock()
    pm.get_processor.return_value = Mock()
    video = tmp_path / "v.mp4"

    out = await ProcessingStrategySet()._process_segmentation(
        _keyframe_strategy(), video, pm, ctx
    )

    assert out == {"keyframes": cached}
    ctx.upload_keyframes_to_object_store.assert_called_once_with(video, cached)
