"""A down inference endpoint must trip the breaker and fail fast."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cogniverse_core.common.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
)


@pytest.fixture(autouse=True)
def _reset():
    CircuitBreaker.reset_registry()
    yield
    CircuitBreaker.reset_registry()


def _client():
    from cogniverse_core.common.models.model_loaders import RemoteInferenceClient

    c = RemoteInferenceClient(endpoint_url="http://infer:8000")
    # Tighten the shared per-endpoint breaker for the test.
    from cogniverse_core.common.utils.circuit_breaker import BreakerConfig

    c._breaker = CircuitBreaker.get(
        BreakerConfig(name="inference:test", failure_threshold=2, reset_timeout_s=999)
    )
    return c


@pytest.mark.unit
@pytest.mark.ci_fast
def test_process_images_fast_fails_after_breaker_opens():
    client = _client()
    calls = {"n": 0}

    def boom(images, **kwargs):
        calls["n"] += 1
        raise ConnectionError("inference pod down")

    client._process_images_retried = boom

    for _ in range(2):
        with pytest.raises(ConnectionError):
            client.process_images(["img"])
    assert calls["n"] == 2

    with pytest.raises(CircuitOpenError):
        client.process_images(["img"])
    assert calls["n"] == 2  # endpoint not dialed while open


@pytest.mark.unit
@pytest.mark.ci_fast
def test_success_keeps_breaker_closed():
    client = _client()
    client._process_images_retried = MagicMock(return_value={"embeddings": [1]})
    for _ in range(5):
        assert client.process_images(["img"]) == {"embeddings": [1]}
