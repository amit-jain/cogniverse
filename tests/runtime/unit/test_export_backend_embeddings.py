"""Coverage for the backend-embedding exporter CLI script.

``scripts/export_backend_embeddings.py`` pulls embeddings + fields from any
SearchBackend and reduces them for visualization. It had zero test coverage.
These exercise the format-parsing and field-extraction logic that the export
depends on: multi-format embedding extraction (values/blocks/list/hex),
schema-aware field extraction, the text representation, and dimensionality
reduction shape + error handling.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "export_backend_embeddings.py"
_spec = importlib.util.spec_from_file_location("export_backend_embeddings", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
BackendEmbeddingExporter = _mod.BackendEmbeddingExporter


@pytest.fixture
def exporter():
    return BackendEmbeddingExporter(backend=MagicMock())


@pytest.mark.unit
class TestExtractEmbedding:
    def test_values_dict_format(self, exporter):
        out = exporter._extract_embedding({"embedding": {"values": [1.0, 2.0, 3.0]}})
        assert np.array_equal(out, np.array([1.0, 2.0, 3.0]))

    def test_blocks_format_mean_pools_patches(self, exporter):
        # Two patches → mean-pooled to one fixed-size vector.
        doc = {"colpali_embedding": {"blocks": {"0": [0.0, 2.0], "1": [2.0, 4.0]}}}
        out = exporter._extract_embedding(doc)
        assert np.allclose(out, np.array([1.0, 3.0]))

    def test_blocks_sorted_numerically_not_lexically(self, exporter):
        # Keys 2 and 10 must sort 2<10 (int), not "10"<"2" (str) — mean is the
        # same here but the ordering guard matters for any order-sensitive use.
        doc = {"embedding": {"blocks": {"10": [4.0], "2": [2.0]}}}
        out = exporter._extract_embedding(doc)
        assert np.allclose(out, np.array([3.0]))

    def test_list_format(self, exporter):
        out = exporter._extract_embedding({"frame_embedding": [0.5, 0.5]})
        assert np.array_equal(out, np.array([0.5, 0.5]))

    def test_hex_string_format(self, exporter):
        vec = np.array([1.0, -2.0], dtype=np.float32)
        hex_str = "0x" + vec.tobytes().hex()
        out = exporter._extract_embedding({"text_embedding": hex_str})
        assert np.array_equal(out, vec)

    def test_missing_embedding_returns_none(self, exporter):
        assert exporter._extract_embedding({"id": "x", "title": "no vec"}) is None

    def test_undecodable_hex_returns_none(self, exporter):
        assert exporter._extract_embedding({"embedding": "not-hex-zzz"}) is None


@pytest.mark.unit
class TestExtractFields:
    def test_video_frame_schema(self, exporter):
        doc = {
            "id": "f1",
            "video_id": "v_abc",
            "frame_number": 12,
            "timestamp": 3.5,
            "video_title": "Barbell lift",
            "frame_description": "person lifting",
            "embedding": [0.1],
        }
        fields = exporter._extract_fields(doc, "video_frame")
        assert fields["source_type"] == "video_frame"
        assert fields["video_id"] == "v_abc"
        assert fields["frame_number"] == 12
        assert fields["video_title"] == "Barbell lift"
        # Text representation is derived from the title + frame description.
        assert fields["text"] == "Video: Barbell lift | Frame: person lifting"
        # Raw embedding is not carried into the extracted fields.
        assert "embedding" not in fields

    def test_generic_schema_excludes_embedding_fields(self, exporter):
        doc = {
            "id": "d1",
            "custom_field": "keep me",
            "video_embedding": [0.1, 0.2],
            "text_embedding": [0.3],
        }
        fields = exporter._extract_fields(doc, "unknown_schema")
        assert fields["custom_field"] == "keep me"
        assert "video_embedding" not in fields
        assert "text_embedding" not in fields


@pytest.mark.unit
class TestTextRepresentation:
    def test_document_title_and_truncated_content(self, exporter):
        text = exporter._create_text_representation(
            {"title": "Doc A", "content": "x" * 500}
        )
        assert text.startswith("Title: Doc A | Content: ")
        # Content is truncated to 200 chars.
        assert text == f"Title: Doc A | Content: {'x' * 200}"

    def test_product_representation(self, exporter):
        text = exporter._create_text_representation(
            {"product_name": "Widget", "description": "a useful widget"}
        )
        assert text == "Product: Widget | Description: a useful widget"

    def test_empty_when_no_known_fields(self, exporter):
        assert exporter._create_text_representation({"id": "x"}) == ""


@pytest.mark.unit
class TestReduceDimensions:
    def test_pca_reduces_to_n_components(self, exporter):
        embeddings = np.random.RandomState(0).rand(10, 8).astype(np.float32)
        reduced = exporter._reduce_dimensions(embeddings, method="pca", n_components=2)
        assert reduced.shape == (10, 2)

    def test_unknown_method_raises(self, exporter):
        embeddings = np.zeros((5, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown reduction method"):
            exporter._reduce_dimensions(embeddings, method="bogus")
