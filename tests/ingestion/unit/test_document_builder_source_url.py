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
        "source_url": "file:///abs/v_abc.mp4",
    }


class TestDocumentMetadataSourceUrl:
    def test_explicit_value_set(self, base_metadata_kwargs):
        meta = DocumentMetadata(
            **{**base_metadata_kwargs, "source_url": "s3://corpus/v.mp4"}
        )
        assert meta.source_url == "s3://corpus/v.mp4"

    def test_missing_source_url_raises(self, base_metadata_kwargs):
        kwargs = {k: v for k, v in base_metadata_kwargs.items() if k != "source_url"}
        with pytest.raises(TypeError):
            DocumentMetadata(**kwargs)

    def test_empty_source_url_raises(self, base_metadata_kwargs):
        with pytest.raises(ValueError, match="source_url is required"):
            DocumentMetadata(**{**base_metadata_kwargs, "source_url": ""})


class TestDocumentBuilderWritesSourceUrl:
    def test_field_always_written(self, builder, base_metadata_kwargs):
        meta = DocumentMetadata(**base_metadata_kwargs)
        doc = builder.build_document(meta, {}, {})
        assert doc["fields"]["source_url"] == base_metadata_kwargs["source_url"]

    def test_source_url_coexists_with_other_fields(self, builder, base_metadata_kwargs):
        meta = DocumentMetadata(
            **{**base_metadata_kwargs, "source_url": "s3://corpus/v.mp4"}
        )
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
            meta = DocumentMetadata(**{**base_metadata_kwargs, "source_url": uri})
            doc = builder.build_document(meta, {}, {})
            assert doc["fields"]["source_url"] == uri


class TestDocumentIdUnaffected:
    def test_doc_id_format_unchanged(self, builder, base_metadata_kwargs):
        meta = DocumentMetadata(
            **{**base_metadata_kwargs, "source_url": "s3://corpus/v.mp4"}
        )
        doc = builder.build_document(meta, {}, {})
        assert doc["id"] == "v_abc_segment_0"
