"""``VLMProcessor.from_config`` must fail loud when no VLM endpoint is set.

With the developer's personal Modal endpoint removed from the shipped config,
a fresh deployment has an empty ``vlm_endpoint``. Description generation must
raise a clear error telling the operator to configure it, not silently POST to
an empty URL or to a stale personal endpoint.
"""

from __future__ import annotations

import logging

import pytest

from cogniverse_runtime.ingestion.processors.vlm_processor import VLMProcessor


@pytest.mark.unit
def test_from_config_raises_on_empty_endpoint():
    with pytest.raises(ValueError, match="requires 'vlm_endpoint'"):
        VLMProcessor.from_config({"vlm_endpoint": ""}, logging.getLogger("t"))


@pytest.mark.unit
def test_from_config_raises_on_missing_endpoint():
    with pytest.raises(ValueError, match="requires 'vlm_endpoint'"):
        VLMProcessor.from_config({}, logging.getLogger("t"))


@pytest.mark.unit
def test_from_config_accepts_configured_endpoint():
    proc = VLMProcessor.from_config(
        {"vlm_endpoint": "http://vlm.internal:8000/generate", "batch_size": 4},
        logging.getLogger("t"),
    )
    assert proc.vlm_endpoint == "http://vlm.internal:8000/generate"
    assert proc.batch_size == 4
