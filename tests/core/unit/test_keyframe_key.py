"""The shared keyframe object-key contract.

Ingestion (write) and the answer-time agent (read) both derive a keyframe's
location from this one helper. If the two ever computed the key differently,
every keyframe would be silently unfetchable — so these tests pin the exact
string both sides must agree on.
"""

from __future__ import annotations

import pytest

from cogniverse_core.common.media import keyframe_object_key, keyframe_uri


@pytest.mark.unit
@pytest.mark.ci_fast
def test_object_key_exact_string():
    assert (
        keyframe_object_key("acme:acme", "vid123", 7)
        == "acme:acme/keyframes/vid123/0007.jpg"
    )


@pytest.mark.unit
@pytest.mark.ci_fast
def test_uri_exact_string():
    assert (
        keyframe_uri("media", "acme:acme", "vid123", 7)
        == "s3://media/acme:acme/keyframes/vid123/0007.jpg"
    )


def test_uri_is_key_under_bucket():
    key = keyframe_object_key("t1", "v9", 42)
    assert keyframe_uri("bucket-x", "t1", "v9", 42) == f"s3://bucket-x/{key}"


def test_segment_id_zero_padded_to_four_digits():
    assert keyframe_object_key("t", "v", 0).endswith("/0000.jpg")
    assert keyframe_object_key("t", "v", 1234).endswith("/1234.jpg")
    # ordinal wider than four digits is not truncated
    assert keyframe_object_key("t", "v", 12345).endswith("/12345.jpg")


def test_string_segment_id_coerced_same_as_int():
    # a Vespa hit may surface segment_id as a string; both must map to one key
    assert keyframe_object_key("t", "v", "7") == keyframe_object_key("t", "v", 7)


@pytest.mark.parametrize(
    "tenant,video",
    [("", "v"), ("t", ""), ("", "")],
)
def test_missing_tenant_or_video_raises(tenant, video):
    with pytest.raises(ValueError, match="tenant_id and video_id are required"):
        keyframe_object_key(tenant, video, 1)


@pytest.mark.unit
@pytest.mark.ci_fast
def test_missing_bucket_raises():
    with pytest.raises(ValueError, match="bucket is required"):
        keyframe_uri("", "t", "v", 1)
