"""Tests for source_url propagation through DocumentBuilder."""

import pytest

from cogniverse_runtime.ingestion.processors.embedding_generator.document_builders import (
    DocumentBuilder,
    DocumentMetadata,
)


@pytest.fixture
def builder():
    return DocumentBuilder("test_schema")


@pytest.fixture
def base_metadata_kwargs():
    return {
        "video_id": "v_abc",
        "video_title": "Sample",
        "segment_idx": 0,
        "start_time": 0.0,
        "end_time": 5.0,
    }


class TestDocumentMetadataSourceUrl:
    def test_default_is_none(self, base_metadata_kwargs):
        meta = DocumentMetadata(**base_metadata_kwargs)
        assert meta.source_url is None

    def test_explicit_value_set(self, base_metadata_kwargs):
        meta = DocumentMetadata(**base_metadata_kwargs, source_url="s3://corpus/v.mp4")
        assert meta.source_url == "s3://corpus/v.mp4"


class TestDocumentBuilderWritesSourceUrl:
    def test_field_present_when_metadata_has_source_url(
        self, builder, base_metadata_kwargs
    ):
        meta = DocumentMetadata(
            **base_metadata_kwargs, source_url="file:///abs/v_abc.mp4"
        )
        doc = builder.build_document(meta, {}, {})
        assert doc["fields"]["source_url"] == "file:///abs/v_abc.mp4"

    def test_field_absent_when_metadata_source_url_is_none(
        self, builder, base_metadata_kwargs
    ):
        meta = DocumentMetadata(**base_metadata_kwargs)
        doc = builder.build_document(meta, {}, {})
        assert "source_url" not in doc["fields"]

    def test_field_absent_when_metadata_source_url_is_empty_string(
        self, builder, base_metadata_kwargs
    ):
        meta = DocumentMetadata(**base_metadata_kwargs, source_url="")
        doc = builder.build_document(meta, {}, {})
        assert "source_url" not in doc["fields"]

    def test_source_url_coexists_with_other_fields(self, builder, base_metadata_kwargs):
        meta = DocumentMetadata(**base_metadata_kwargs, source_url="s3://corpus/v.mp4")
        doc = builder.build_document(
            meta,
            {"float_embeddings": [0.1, 0.2]},
            {"audio_transcript": "hello", "total_segments": 5},
        )
        assert doc["fields"]["source_url"] == "s3://corpus/v.mp4"
        assert doc["fields"]["audio_transcript"] == "hello"
        assert doc["fields"]["total_segments"] == 5
        assert doc["fields"]["embedding"] == [0.1, 0.2]

    def test_various_uri_schemes(self, builder, base_metadata_kwargs):
        for uri in [
            "file:///abs/v.mp4",
            "s3://corpus/v.mp4",
            "pvc://media/videos/v.mp4",
            "https://example.com/clip.mp4",
        ]:
            meta = DocumentMetadata(**base_metadata_kwargs, source_url=uri)
            doc = builder.build_document(meta, {}, {})
            assert doc["fields"]["source_url"] == uri


class TestDocumentIdUnaffected:
    def test_doc_id_format_unchanged(self, builder, base_metadata_kwargs):
        meta = DocumentMetadata(**base_metadata_kwargs, source_url="s3://corpus/v.mp4")
        doc = builder.build_document(meta, {}, {})
        assert doc["id"] == "v_abc_segment_0"
