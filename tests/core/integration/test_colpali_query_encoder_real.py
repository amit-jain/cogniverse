"""Remote-only contract for the ColPali-family query encoder with Tomoro.

Tomoro (``TomoroAI/tomoro-colqwen3-embed-4b``, architecture ``qwen3_vl``) is
served exclusively via vLLM. The host's ``transformers`` is pinned 4.56.2 by
the pylate cap, which has no ``qwen3_vl`` support, and ``colpali_engine``
mis-maps it to ``idefics3`` — so an in-process local load crashes with a bare
``KeyError: 'qwen3_vl_text'``. The local encoder path must instead raise a
clear, actionable remote-only ``RuntimeError`` pointing the operator at vLLM +
``inference_service_url``.

The remote 320-d MaxSim round-trip is covered by
``test_tomoro_serving_real.py`` (RemoteColPaliLoader against a live vLLM
sidecar); this module pins the local-path guard, which needs no sidecar.
"""

from __future__ import annotations

import pytest

from cogniverse_core.query.encoders import ColPaliQueryEncoder, ColQwenQueryEncoder

pytestmark = [pytest.mark.integration]

TOMORO_MODEL = "TomoroAI/tomoro-colqwen3-embed-4b"

_REMOTE_ONLY_SUBSTRINGS = (
    "ColQwen3/Tomoro models are remote-only",
    "inference_service_url",
    "transformers>=4.57",
)


@pytest.mark.parametrize("factory", [ColPaliQueryEncoder, ColQwenQueryEncoder])
def test_local_tomoro_encoder_raises_remote_only(factory):
    """Building the local (no ``inference_service_url``) encoder for Tomoro
    raises the clear remote-only RuntimeError, never the bare arch KeyError."""
    with pytest.raises(RuntimeError) as excinfo:
        factory(TOMORO_MODEL)

    msg = str(excinfo.value)
    for substr in _REMOTE_ONLY_SUBSTRINGS:
        assert substr in msg, f"missing {substr!r} in remote-only message: {msg!r}"
    assert "KeyError" not in msg
    assert "qwen3_vl_text" not in msg
