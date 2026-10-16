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

# The cluster's CLAP service encodes the text, so the module runs where the
# cogniverse-e2e cluster is reachable, outside the CI unit selection.
pytestmark = [pytest.mark.requires_models, pytest.mark.slow, pytest.mark.local_only]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.fixture(scope="module")
def generator(remote_inference):
    generator = AudioEmbeddingGenerator(
        clap_endpoint_url=remote_inference.resolve("clap_embed").base_url
    )
    try:
        yield generator
    finally:
        generator.close()


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


def test_a_long_clip_maps_to_one_vector_the_mean_of_its_windows(generator):
    # CLAP's processor crops anything past 10 s at random; the service embeds
    # fixed windows instead, so the same 25 s clip embeds identically twice
    # and equals the normalized mean of its three windows embedded alone.
    rate = 48000
    rng = np.random.default_rng(7)
    clip = np.concatenate(
        [
            0.3 * np.sin(2 * np.pi * freq * np.arange(seconds * rate) / rate)
            + 0.02 * rng.standard_normal(seconds * rate)
            for freq, seconds in ((220, 10), (880, 10), (3520, 5))
        ]
    ).astype(np.float32)

    first = generator.generate_acoustic_embedding(audio_array=clip, sample_rate=rate)
    second = generator.generate_acoustic_embedding(audio_array=clip, sample_rate=rate)
    windows = [
        generator.generate_acoustic_embedding(
            audio_array=clip[start * rate : end * rate], sample_rate=rate
        )
        for start, end in ((0, 10), (10, 20), (20, 25))
    ]
    mean = np.mean([w / np.linalg.norm(w) for w in windows], axis=0)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, mean / np.linalg.norm(mean), atol=1e-4)
