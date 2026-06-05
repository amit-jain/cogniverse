"""generate_acoustic_text_embedding must produce a CLAP-space 512-d vector.

Acoustic search compares a text query to stored CLAP audio embeddings, so the
query must be encoded with CLAP's text tower (shared joint space), not a
sentence-transformer whose embeddings live in a different space.
"""

from __future__ import annotations

import numpy as np
import pytest

from cogniverse_runtime.ingestion.processors.audio_embedding_generator import (
    AudioEmbeddingGenerator,
)

pytestmark = [pytest.mark.requires_models, pytest.mark.slow]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture(scope="module")
def generator():
    return AudioEmbeddingGenerator()


def test_acoustic_text_embedding_shape(generator):
    emb = generator.generate_acoustic_text_embedding("a dog barking")
    assert emb.shape == (512,)
    assert np.all(np.isfinite(emb))


def test_acoustic_text_embedding_is_semantically_structured(generator):
    # CLAP's joint space places a phrase nearer a paraphrase than an unrelated
    # concept. A sentence-transformer embedding truncated to 512 dims would not
    # share the audio space at all, so this ordering is the property the
    # acoustic query path depends on.
    dog = generator.generate_acoustic_text_embedding("a dog barking loudly")
    dog_paraphrase = generator.generate_acoustic_text_embedding(
        "the sound of dogs barking"
    )
    piano = generator.generate_acoustic_text_embedding("a calm piano melody")

    assert _cosine(dog, dog_paraphrase) > _cosine(dog, piano)
