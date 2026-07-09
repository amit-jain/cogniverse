"""VideoPrism text encoder must not silently pad/truncate a dim mismatch.

A query embedding whose dimension differs from the configured embedding_dim was
zero-padded or truncated, producing a vector that no longer aligned with the
corpus so every score was wrong. It must fail loud instead.
"""

from __future__ import annotations

import os

# The encoder module resolves config at import time; give it a backend URL so
# the import succeeds in a unit context.
os.environ.setdefault("BACKEND_URL", "http://localhost:8080")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from cogniverse_core.common.models.videoprism_text_encoder import (  # noqa: E402
    VideoPrismTextEncoder,
)


def _encoder(embedding_dim: int) -> VideoPrismTextEncoder:
    enc = object.__new__(VideoPrismTextEncoder)
    enc.embedding_dim = embedding_dim
    enc.correlation_id = "t"
    return enc


def test_matching_dim_passes_through_unchanged():
    enc = _encoder(768)
    vec = np.ones(768, dtype=np.float32)
    out = enc._project_embeddings(vec)
    assert out is vec


def test_smaller_dim_raises_instead_of_padding():
    enc = _encoder(768)
    with pytest.raises(ValueError, match="does not match"):
        enc._project_embeddings(np.ones(512, dtype=np.float32))


def test_larger_dim_raises_instead_of_truncating():
    enc = _encoder(768)
    with pytest.raises(ValueError, match="does not match"):
        enc._project_embeddings(np.ones(1024, dtype=np.float32))
