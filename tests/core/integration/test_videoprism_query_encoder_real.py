"""Real-model contract for ``VideoPrismQueryEncoder.encode()``.

Exercises the actual VideoPrism text encoder — no synthetic embedding. The
shape the encoder emits must match what the deployed schemas declare:

* a patch (multi-vector) model feeds the mv profiles whose ``query(qt)`` is
  ``tensor<float>(querytoken{}, v[dim])`` → ``encode()`` must return a 2D
  ``(num_query_tokens, dim)`` array so ``_build_query`` emits the mapped
  ``{"0": [...]}`` form (a flat list never binds → MaxSim scores nothing);
* an LVT (global, single-vector) model feeds the sv profiles whose
  ``query(qt)`` is ``tensor<float>(v[dim])`` → ``encode()`` must return a flat
  ``(dim,)`` vector.

Gated on the real ``videoprism`` package + ``requires_models`` + ``slow``:
loads VideoPrism weights. Run where videoprism is installed via
``uv run pytest -m requires_models tests/core/integration/test_videoprism_query_encoder_real.py``.
"""

from __future__ import annotations

import pytest

from cogniverse_core.query.encoders import VideoPrismQueryEncoder
from tests.utils.markers import skip_if_no_videoprism

pytestmark = [
    pytest.mark.requires_models,
    pytest.mark.slow,
    pytest.mark.integration,
    skip_if_no_videoprism,
]

_QUERY = "a red car driving along a coastal road at sunset"


def test_patch_model_encode_returns_mapped_query_tokens():
    """A patch (mv) model's real encode() output is 2D so it binds to the
    videoprism mv float_float profile's mapped query(qt)."""
    encoder = VideoPrismQueryEncoder(model_name="videoprism_public_v1_base_hf")
    assert encoder.is_global is False
    assert encoder.embedding_dim == 768

    out = encoder.encode(_QUERY)

    assert out.ndim == 2, (
        f"patch-model encode() must be 2D (num_query_tokens, dim) so "
        f"_build_query emits the mapped query(qt); got shape {out.shape}"
    )
    assert out.shape[0] >= 1
    assert out.shape[1] == 768


def test_lvt_global_model_encode_returns_flat_vector():
    """An LVT (global, sv) model's real encode() output is a flat 1D vector
    for the sv profile's tensor<float>(v[dim]) query(qt)."""
    encoder = VideoPrismQueryEncoder(model_name="videoprism_lvt_public_v1_base_hf")
    assert encoder.is_global is True
    assert encoder.embedding_dim == 768

    out = encoder.encode(_QUERY)

    assert out.ndim == 1, (
        f"global-model encode() must be flat 1D (dim,) for the sv schema; "
        f"got shape {out.shape}"
    )
    assert out.shape[0] == 768
