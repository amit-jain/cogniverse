"""Pin VespaPyClient.process() field output for every ingestion family.

process() is the write-side serializer: it turns a generic Document (whose
values live in metadata) into the exact ``fields`` dict fed to Vespa under one
schema's field names. These tests build each family's Document exactly as the
ingestion builders in embedding_generator_impl.py do and assert the full scalar
field set, the embedding field NAMES, and a millisecond ``creation_timestamp``.

The video family carries two metadata keys whose names differ from the schema
fields (``segment_index`` -> ``segment_id``, ``description`` ->
``segment_description``); those renames are declared in the schema's
``document_mapping.metadata_fields`` and applied by process(). The remaining
families key their metadata to the exact schema field names, so process()'s
schema-gated passthrough carries them unchanged. Editing a video mapping's
metadata_fields must change process()'s output — the liveness test pins that.
"""

from pathlib import Path

import numpy as np
import pytest

from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_sdk.document import (
    ContentType,
    Document,
    DocumentFieldMapping,
    ProcessingStatus,
)
from cogniverse_vespa.ingestion_client import VespaPyClient

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

_LOADER = FilesystemSchemaLoader(Path("configs/schemas"))
_MS_FLOOR = 1_000_000_000_000  # 2001-09 in ms; anything smaller is seconds


def _client(schema: str) -> VespaPyClient:
    return VespaPyClient(
        {
            "schema_name": schema,
            "url": "http://localhost",
            "port": 8080,
            "schema_loader": _LOADER,
        }
    )


def _video_doc(doc_id: str, segment_index: int, description: str | None) -> Document:
    doc = Document(
        id=doc_id,
        content_type=ContentType.VIDEO,
        content_id="vid123",
        status=ProcessingStatus.COMPLETED,
    )
    doc.add_embedding(
        "embedding",
        np.zeros((4, 128), dtype=np.float32),
        {"type": "float", "raw": True},
    )
    doc.add_metadata("start_time", 0.0)
    doc.add_metadata("end_time", 30.0)
    doc.add_metadata("segment_index", segment_index)
    doc.add_metadata("total_segments", 5)
    doc.add_metadata("audio_transcript", "hello world")
    doc.add_metadata("video_id", "vid123")
    doc.add_metadata("video_title", "vid123")
    doc.add_metadata("source_url", "s3://b/v.mp4")
    if description is not None:
        doc.add_metadata("description", description)
    return doc


def _split_timestamp(fields: dict, schema_has_ts: bool) -> dict:
    """Pop and check a stamped millisecond creation_timestamp; return the rest."""
    rest = dict(fields)
    if schema_has_ts:
        ts = rest.pop("creation_timestamp")
        assert isinstance(ts, int) and not isinstance(ts, bool)
        assert ts > _MS_FLOOR, f"creation_timestamp must be milliseconds, got {ts}"
    else:
        assert "creation_timestamp" not in rest
    return rest


class TestProcessFieldOutput:
    def test_video_colpali_frame_scalars(self):
        out = _client("video_colpali_smol500_mv_frame").process(
            _video_doc("vid123_seg_0", 0, "a cat"), "feed"
        )
        assert out["put"] == "id:content:video_colpali_smol500_mv_frame::vid123_seg_0"
        fields = out["fields"]
        # Multi-vector float + binary embeddings, patch-indexed dicts.
        assert set(fields["embedding"].keys()) == {"0", "1", "2", "3"}
        assert set(fields["embedding_binary"].keys()) == {"0", "1", "2", "3"}
        scalars = _split_timestamp(fields, schema_has_ts=True)
        scalars.pop("embedding")
        scalars.pop("embedding_binary")
        assert scalars == {
            "video_id": "vid123",
            "video_title": "vid123",
            "source_url": "s3://b/v.mp4",
            "start_time": 0.0,
            "end_time": 30.0,
            "segment_id": 0,
            "segment_description": "a cat",
            "audio_transcript": "hello world",
        }

    def test_video_chunk30s_drops_absent_segment_description(self):
        # This schema has segment_id but no segment_description field: the
        # description rename target is gated out, total_segments too.
        out = _client("video_videoprism_base_mv_chunk_30s").process(
            _video_doc("vid123_seg_1", 1, "ignored-no-field"), "feed"
        )
        assert (
            out["put"] == "id:content:video_videoprism_base_mv_chunk_30s::vid123_seg_1"
        )
        fields = out["fields"]
        scalars = _split_timestamp(fields, schema_has_ts=True)
        scalars.pop("embedding")
        scalars.pop("embedding_binary")
        assert scalars == {
            "video_id": "vid123",
            "video_title": "vid123",
            "source_url": "s3://b/v.mp4",
            "start_time": 0.0,
            "end_time": 30.0,
            "segment_id": 1,
            "audio_transcript": "hello world",
        }

    def test_audio_scalars(self):
        doc = Document(
            id="aud1_a0",
            content_type=ContentType.AUDIO,
            content_id="aud1",
            status=ProcessingStatus.COMPLETED,
        )
        doc.add_embedding(
            "embedding",
            np.zeros((4, 128), dtype=np.float32),
            {"type": "float", "raw": True},
        )
        doc.add_metadata("acoustic_embedding", [0.0, 0.0, 0.0])
        doc.add_metadata("audio_id", "aud1")
        doc.add_metadata("audio_title", "clip")
        doc.add_metadata("audio_path", "/a/clip.wav")
        doc.add_metadata("audio_transcript", "spoken words")
        doc.add_metadata("source_url", "s3://b/a.wav")

        out = _client("audio_content").process(doc, "feed")
        assert out["put"] == "id:content:audio_content::aud1_a0"
        fields = out["fields"]
        assert set(fields["semantic_embedding"].keys()) == {"0", "1", "2", "3"}
        assert set(fields["semantic_embedding_binary"].keys()) == {"0", "1", "2", "3"}
        assert fields["acoustic_embedding"] == [0.0, 0.0, 0.0]
        scalars = _split_timestamp(fields, schema_has_ts=True)
        for k in (
            "semantic_embedding",
            "semantic_embedding_binary",
            "acoustic_embedding",
        ):
            scalars.pop(k)
        assert scalars == {
            "audio_id": "aud1",
            "audio_title": "clip",
            "audio_path": "/a/clip.wav",
            "audio_transcript": "spoken words",
            "source_url": "s3://b/a.wav",
        }

    def test_document_text_scalars(self):
        doc = Document(
            id="doc1_0",
            content_type=ContentType.DOCUMENT,
            content_id="doc1",
            status=ProcessingStatus.COMPLETED,
        )
        doc.add_embedding(
            "embedding",
            np.zeros((4, 128), dtype=np.float32),
            {"type": "float", "raw": True},
        )
        for k, v in {
            "document_id": "0",
            "document_title": "report.pdf",
            "document_type": "pdf",
            "document_path": "/d/report.pdf",
            "full_text": "body",
            "page_count": 4,
            "source_url": "s3://b/d.pdf",
        }.items():
            doc.add_metadata(k, v)

        out = _client("document_text").process(doc, "feed")
        assert out["put"] == "id:content:document_text::doc1_0"
        fields = out["fields"]
        scalars = _split_timestamp(fields, schema_has_ts=True)
        scalars.pop("embedding")
        scalars.pop("embedding_binary")
        # document_text has no source_url field -> gated out.
        assert scalars == {
            "document_id": "0",
            "document_title": "report.pdf",
            "document_type": "pdf",
            "document_path": "/d/report.pdf",
            "full_text": "body",
            "page_count": 4,
        }

    def test_code_scalars(self):
        doc = Document(
            id="code1_0",
            content_type=ContentType.TEXT,
            content_id="code1",
            status=ProcessingStatus.COMPLETED,
        )
        doc.add_embedding(
            "embedding",
            np.zeros((4, 128), dtype=np.float32),
            {"type": "float", "raw": True},
        )
        for k, v in {
            "code_id": "0",
            "file_path": "/c/x.py",
            "chunk_name": "f",
            "chunk_type": "function",
            "language": "python",
            "signature": "def f()",
            "line_start": 1,
            "line_end": 9,
            "source_code": "def f(): pass",
        }.items():
            doc.add_metadata(k, v)

        out = _client("code_lateon_mv").process(doc, "feed")
        assert out["put"] == "id:content:code_lateon_mv::code1_0"
        fields = out["fields"]
        # code_lateon_mv ranks on the float multi-vector only — no binary.
        assert set(fields["embedding"].keys()) == {"0", "1", "2", "3"}
        assert "embedding_binary" not in fields
        scalars = _split_timestamp(fields, schema_has_ts=False)
        scalars.pop("embedding")
        assert scalars == {
            "code_id": "0",
            "file_path": "/c/x.py",
            "chunk_name": "f",
            "chunk_type": "function",
            "language": "python",
            "signature": "def f()",
            "line_start": 1,
            "line_end": 9,
            "source_code": "def f(): pass",
        }


class TestProcessUpdateOmitsTimestamp:
    def test_update_does_not_stamp_absent_timestamp(self):
        # A partial update with no caller timestamp must not stamp now() —
        # that would clobber the original creation time on a metadata-only write.
        out = _client("video_colpali_smol500_mv_frame").process(
            _video_doc("vid123_seg_0", 0, "a cat"), "update"
        )
        assert "creation_timestamp" not in out["fields"]

    def test_feed_honours_caller_millisecond_timestamp(self):
        doc = _video_doc("vid123_seg_0", 0, "a cat")
        doc.add_metadata("creation_timestamp", 1_700_000_000_000)
        out = _client("video_colpali_smol500_mv_frame").process(doc, "feed")
        assert out["fields"]["creation_timestamp"] == 1_700_000_000_000


class TestMappingDrivesRename:
    def test_metadata_field_rename_follows_the_mapping(self):
        # process() reads segment_index's target from the schema's declared
        # metadata_fields. Point the rename at a different real field and the
        # output must follow it — proving the mapping is live, not hardcoded.
        client = _client("video_colpali_smol500_mv_frame")
        client._doc_mapping = DocumentFieldMapping(
            metadata_fields={"segment_index": "segment_description"},
            include_metadata=False,
        )
        doc = _video_doc("vid123_seg_0", 7, None)
        out = client.process(doc, "feed")
        assert out["fields"]["segment_description"] == 7
        assert "segment_id" not in out["fields"]
