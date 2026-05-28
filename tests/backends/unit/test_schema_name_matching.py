"""Schema-name matching in VespaEmbeddingProcessor.

The single-vector path used substring matching on ``"lvt"`` plus a case-
sensitive ``"_sv_"`` check. Any future schema name embedding ``lvt`` as
a substring (e.g. ``audio_alvtree_index``) would silently collapse a
multi-vector tensor to its first row, and an uppercase ``_SV_`` was
treated as multi-vector despite the intent. The fix uses token-bracketed
``_sv_`` / ``_lvt_`` and lower-cases both sides.
"""

from __future__ import annotations

import numpy as np
import pytest

from cogniverse_vespa.embedding_processor import (
    VespaEmbeddingProcessor,
    _is_single_vector_schema,
)


@pytest.mark.parametrize(
    "schema_name",
    [
        "video_videoprism_lvt_base_sv_chunk_6s",
        "video_videoprism_lvt_large_sv_chunk_6s",
        "video_VIDEOPRISM_SV_global",  # uppercase variant must match
        "anything_with_lvt_token",
        "anything_with_sv_token",
    ],
)
def test_single_vector_tokens_match(schema_name: str) -> None:
    assert _is_single_vector_schema(schema_name) is True


@pytest.mark.parametrize(
    "schema_name",
    [
        "audio_alvtree_index",  # 'lvt' substring without token bounds — must NOT match
        "video_colpali_smol500_mv_frame",
        "video_videoprism_large_mv_chunk_30s",
        "video_colqwen_omni_mv_chunk_30s",
        "agent_memories_tenant_acme",
        "sv_prefix_only_no_underscore",  # 'sv' at start without trailing underscore in token form
        "",
    ],
)
def test_multi_vector_or_irrelevant_names_do_not_match(schema_name: str) -> None:
    assert _is_single_vector_schema(schema_name) is False


def test_multi_vector_schema_with_substring_collision_preserves_all_patches() -> None:
    """The footgun: a hypothetical schema name with ``lvt`` as a substring
    (no token bounds) used to collapse a (4, 128) tensor to a 128-element
    list. The fix must return a 4-patch dict."""
    processor = VespaEmbeddingProcessor(schema_name="audio_alvtree_index")
    embeddings = np.arange(4 * 16, dtype=np.float32).reshape(4, 16)
    result = processor.process_embeddings(embeddings)
    assert isinstance(result["embedding"], dict)
    assert sorted(result["embedding"].keys()) == ["0", "1", "2", "3"]


def test_single_vector_token_schema_returns_flat_list() -> None:
    """A real ``_sv_`` schema collapses to one list of floats."""
    processor = VespaEmbeddingProcessor(
        schema_name="video_videoprism_lvt_base_sv_chunk_6s"
    )
    embeddings = np.arange(4 * 16, dtype=np.float32).reshape(4, 16)
    result = processor.process_embeddings(embeddings)
    assert isinstance(result["embedding"], list)
    assert len(result["embedding"]) == 16


def test_uppercase_sv_token_returns_flat_list() -> None:
    """Case-inconsistency fix: uppercase ``_SV_`` was treated as multi-vec."""
    processor = VespaEmbeddingProcessor(schema_name="video_VIDEOPRISM_SV_global")
    embeddings = np.arange(2 * 8, dtype=np.float32).reshape(2, 8)
    result = processor.process_embeddings(embeddings)
    assert isinstance(result["embedding"], list)
    assert len(result["embedding"]) == 8
