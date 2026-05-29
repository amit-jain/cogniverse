"""Vespa edge-input rejections — caller mistakes must surface up-front."""

from __future__ import annotations

import numpy as np
import pytest

from cogniverse_vespa._yql import yql_quote
from cogniverse_vespa.embedding_processor import VespaEmbeddingProcessor
from cogniverse_vespa.ingestion_client import (
    _validate_ms_timestamp,
    _validate_s_timestamp,
)

# ---------------------------------------------------------------------------
# embedding_processor
# ---------------------------------------------------------------------------


def test_unsupported_payload_type_raises_type_error() -> None:
    """Pre-fix: ``process_embeddings(42)`` returned ``42`` and broke
    downstream ``"embedding" in 42`` with a confusing TypeError."""
    p = VespaEmbeddingProcessor()
    with pytest.raises(TypeError, match="Unsupported embedding payload type: int"):
        p.process_embeddings(42)


def test_unsupported_string_payload_raises_type_error() -> None:
    p = VespaEmbeddingProcessor()
    with pytest.raises(TypeError, match="Unsupported embedding payload type: str"):
        p.process_embeddings("oops")


def test_none_payload_still_returns_empty_dict() -> None:
    """Mem0 metadata-only updates path — None is the documented marker for
    "no embedding fields to add"."""
    p = VespaEmbeddingProcessor()
    assert p.process_embeddings(None) == {}


def test_nan_in_embeddings_rejected_by_binarizer() -> None:
    """Pre-fix: ``np.where(NaN > 0, 1, 0) == 0`` silently turned NaN
    rows into zero bitmaps."""
    p = VespaEmbeddingProcessor(schema_name="agent_memories_sv_x")
    arr = np.array([[float("nan"), 0.5, -0.5, 0.2]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite values"):
        p._convert_to_binary_dict(arr)


def test_inf_in_embeddings_rejected_by_binarizer() -> None:
    p = VespaEmbeddingProcessor(schema_name="agent_memories_sv_x")
    arr = np.array([[float("inf"), 0.5, -0.5, 0.2]], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite values"):
        p._convert_to_binary_dict(arr)


def test_finite_embeddings_still_binarize() -> None:
    """Happy path: finite values produce a hex bitmap."""
    p = VespaEmbeddingProcessor(schema_name="agent_memories_sv_x")
    arr = np.ones((1, 16), dtype=np.float32)
    out = p._convert_to_binary_dict(arr)
    assert isinstance(out, str)
    assert len(out) > 0


# ---------------------------------------------------------------------------
# ingestion_client timestamp magnitude validators
# ---------------------------------------------------------------------------


def test_ms_timestamp_seconds_value_rejected() -> None:
    """A seconds-shaped value (e.g. 1_700_000_000) passed to a ms field is a
    common bug — reject it so the document doesn't land at 1970-01-20."""
    # 1_700_000_000 = November 2023 in seconds; in ms it's < 1971. Both are
    # accepted as valid ms; the boundary above year 2100 in ms is the trip.
    _validate_ms_timestamp(1_700_000_000, "creation_timestamp")  # OK as ms (1970)
    # An obviously-out-of-band ms value is rejected.
    with pytest.raises(ValueError, match="seconds by mistake"):
        _validate_ms_timestamp(10**14, "creation_timestamp")


def test_s_timestamp_milliseconds_value_rejected() -> None:
    """A ms-shaped value passed to a seconds field is rejected once it
    exceeds the year-2100 ceiling in seconds (4.1e9)."""
    _validate_s_timestamp(1_700_000_000, "created_at")  # OK (2023)
    with pytest.raises(ValueError, match="milliseconds by mistake"):
        _validate_s_timestamp(1_700_000_000_000, "created_at")


def test_negative_timestamp_rejected() -> None:
    with pytest.raises(ValueError):
        _validate_ms_timestamp(-1, "creation_timestamp")
    with pytest.raises(ValueError):
        _validate_s_timestamp(-1, "created_at")


# ---------------------------------------------------------------------------
# yql_quote: newlines + NUL byte
# ---------------------------------------------------------------------------


def test_yql_quote_escapes_embedded_newline() -> None:
    assert yql_quote("a\nb") == '"a\\nb"'


def test_yql_quote_escapes_carriage_return() -> None:
    assert yql_quote("a\rb") == '"a\\rb"'


def test_yql_quote_escapes_nul_byte() -> None:
    assert yql_quote("a\x00b") == '"a\\0b"'


def test_yql_quote_escapes_combinations() -> None:
    assert yql_quote('a"\nb\\c\x00d') == '"a\\"\\nb\\\\c\\0d"'


def test_multirow_array_to_single_vector_schema_raises() -> None:
    """A single-vector schema given (3, dim) used to keep only row 0 and
    silently discard rows 1-2. It must raise instead."""
    p = VespaEmbeddingProcessor(schema_name="agent_memories_sv_768")
    with pytest.raises(ValueError, match="silently drop rows"):
        p.process_embeddings(np.ones((3, 8), dtype=np.float32))


def test_single_row_array_to_single_vector_schema_ok() -> None:
    """(1, dim) is exactly one vector — accepted, returned as a flat list."""
    p = VespaEmbeddingProcessor(schema_name="agent_memories_sv_768")
    out = p.process_embeddings(np.ones((1, 8), dtype=np.float32))
    assert out["embedding"] == [1.0] * 8
