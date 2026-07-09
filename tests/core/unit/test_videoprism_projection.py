"""VideoPrism text encoder must not silently pad/truncate a dim mismatch.

A query embedding whose dimension differs from the configured embedding_dim was
zero-padded or truncated, producing a vector that no longer aligned with the
corpus so every score was wrong. It must fail loud instead.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# The encoder module resolves config (a network call) at import time. Patch the
# config resolver before importing so this stays a hermetic unit test.
with (
    patch(
        "cogniverse_foundation.config.utils.create_default_config_manager",
        return_value=MagicMock(),
    ),
    patch("cogniverse_foundation.config.utils.get_config", return_value={}),
):
    from cogniverse_core.common.models.videoprism_text_encoder import (
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
