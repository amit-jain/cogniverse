"""Regression guards for ``VespaEmbeddingProcessor.process_embeddings``.

Mem0's metadata-only updates (e.g. bumping ``updated_at`` without changing
the vector) flow through ``VespaBackend.update_document`` → ingestion
client with a Document whose ``embeddings`` dict is empty. The raw
embeddings arrive at the processor as ``None``; the downstream ingestion
client then does ``"embedding" in all_processed_embeddings``, which
raised ``TypeError: argument of type 'NoneType' is not iterable`` when
the processor used to pass ``None`` through untouched. That noise
cluttered the logs whenever the orchestrator's detailed_report path ran.
"""

from __future__ import annotations

import numpy as np
import pytest

from cogniverse_vespa.embedding_processor import VespaEmbeddingProcessor


@pytest.fixture
def processor() -> VespaEmbeddingProcessor:
    return VespaEmbeddingProcessor()


def test_process_embeddings_none_returns_empty_dict(processor):
    """``None`` input must not propagate — downstream uses ``in`` on the result."""
    result = processor.process_embeddings(None)
    assert result == {}


def test_downstream_in_check_against_empty_dict_is_safe(processor):
    """Prove the regression: ``"embedding" in result`` must not raise when
    the document had no embeddings."""
    result = processor.process_embeddings(None)
    assert ("embedding" in result) is False
    assert ("embedding_binary" in result) is False


def test_process_embeddings_with_real_array_still_works(processor):
    """The None short-circuit must not regress the normal path."""
    arr = np.random.rand(4, 128).astype(np.float32)
    result = processor.process_embeddings(arr)
    assert "embedding" in result
    assert "embedding_binary" in result


def test_process_embeddings_with_empty_dict_passes_through(processor):
    """Callers that already built a partial embeddings dict get their input
    back (line 61 of the processor)."""
    result = processor.process_embeddings({})
    assert result == {}
