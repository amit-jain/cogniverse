"""Real ColPali model behind ColPaliFamilyQueryEncoder.encode_image.

encode_image exists so a query IMAGE can be embedded by the same model that
embedded the stored images, making image-to-image MaxSim meaningful. The thing
it must actually do is produce embeddings that represent THAT image's content —
so these assertions compare two visually different images and require the
encoder to score each one closer to itself than to the other. An encoder that
returned a constant, ignored its input, or embedded only the image dimensions
would pass a shape check and fail here.

Uses a real ColPali model (colsmol-500m) loaded locally on CPU — no stub, no
canned vectors. The model that image search deploys is Tomoro ColQwen3, which
is remote-only and needs a GPU vLLM sidecar; colsmol is the same ColPali family
and the same code path (processor.process_images -> model forward), so it pins
encode_image's real behaviour on hardware that is available here.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from cogniverse_core.query.encoders import ColPaliFamilyQueryEncoder

pytestmark = [
    pytest.mark.integration,
    pytest.mark.local_only,
    pytest.mark.requires_colpali,
    pytest.mark.slow,
]

MODEL = "vidore/colsmol-500m"


def _max_sim(query: np.ndarray, doc: np.ndarray) -> float:
    """ColPali MaxSim: for each query patch take its best match, then sum."""
    return float(np.sum(np.max(query @ doc.T, axis=1)))


def _vertical_split() -> Image.Image:
    """Left half black, right half white — a strong vertical edge."""
    arr = np.zeros((96, 96, 3), dtype=np.uint8)
    arr[:, 48:, :] = 255
    return Image.fromarray(arr)


def _horizontal_stripes() -> Image.Image:
    """Alternating horizontal bands — different structure, same palette."""
    arr = np.zeros((96, 96, 3), dtype=np.uint8)
    for row in range(0, 96, 16):
        arr[row : row + 8, :, :] = 255
    return Image.fromarray(arr)


@pytest.fixture(scope="module")
def encoder():
    return ColPaliFamilyQueryEncoder(model_name=MODEL, model_loader="colpali")


class TestEncodeImageRealModel:
    def test_emits_multi_vector_patches_of_the_model_dim(self, encoder):
        emb = encoder.encode_image(_vertical_split())

        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 2, f"expected (patches, dim), got {emb.shape}"
        patches, dim = emb.shape
        assert dim == 128, f"colsmol emits 128-d ColPali patches, got {dim}"
        assert patches > 1, f"expected a multi-vector, got {patches} patch"
        assert emb.dtype == np.float32
        assert np.isfinite(emb).all()
        # A collapsed / constant embedding would satisfy the shape checks above.
        assert emb.std() > 0.0

    def test_same_image_encodes_deterministically(self, encoder):
        first = encoder.encode_image(_vertical_split())
        second = encoder.encode_image(_vertical_split())

        assert first.shape == second.shape
        np.testing.assert_allclose(first, second, rtol=1e-4, atol=1e-4)

    def test_each_image_scores_closer_to_itself_than_to_a_different_image(
        self, encoder
    ):
        """The property image-to-image search depends on.

        If cross-similarity matched self-similarity, retrieval could not tell
        the two images apart and query-by-image would return arbitrary results.
        """
        split = encoder.encode_image(_vertical_split())
        stripes = encoder.encode_image(_horizontal_stripes())

        split_self = _max_sim(split, split)
        split_cross = _max_sim(split, stripes)
        stripes_self = _max_sim(stripes, stripes)
        stripes_cross = _max_sim(stripes, split)

        assert split_self > split_cross, (split_self, split_cross)
        assert stripes_self > stripes_cross, (stripes_self, stripes_cross)

    def test_different_images_produce_different_embeddings(self, encoder):
        split = encoder.encode_image(_vertical_split())
        stripes = encoder.encode_image(_horizontal_stripes())

        if split.shape == stripes.shape:
            assert not np.allclose(split, stripes, rtol=1e-3, atol=1e-3), (
                "two visually different images produced the same embedding — "
                "the encoder is ignoring its input"
            )
