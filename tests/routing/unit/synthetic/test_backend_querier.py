"""BackendQuerier._query_profile grounds samples in real backend content."""

from __future__ import annotations

import pytest

from cogniverse_foundation.config.unified_config import (
    BackendConfig,
    FieldMappingConfig,
)
from cogniverse_synthetic.backend_querier import BackendQuerier


class _RecordingBackend:
    """Same signature as the real ``Backend.query_metadata_documents``."""

    def __init__(self, docs: list[dict]) -> None:
        self.docs = docs
        self.calls: list[dict] = []

    def query_metadata_documents(self, schema, query=None, yql=None, **kwargs):
        self.calls.append({"schema": schema, "yql": yql, "kwargs": kwargs})
        return self.docs


def _querier(backend) -> BackendQuerier:
    return BackendQuerier(
        backend=backend,
        backend_config=BackendConfig(profiles={}, tenant_id="test:unit"),
        field_mappings=FieldMappingConfig(),
    )


async def test_query_profile_grounds_samples_without_duplicate_kwarg() -> None:
    backend = _RecordingBackend(
        [{"title": "Robots", "description": "bots play soccer"}]
    )
    querier = _querier(backend)

    samples = await querier._query_profile(
        {"schema_name": "video_frame"}, sample_size=5, strategy="diverse"
    )

    assert len(samples) == 1
    assert samples[0]["topic"] == "Robots"
    assert samples[0]["description"] == "bots play soccer"

    call = backend.calls[0]
    assert call["yql"] is not None
    assert "yql" not in call["kwargs"], "yql must not be duplicated into kwargs"
    assert call["kwargs"]["hits"] == 5
    assert call["kwargs"]["ranking"] == "random"


async def test_temporal_recent_strategy_forwards_sorting_param() -> None:
    backend = _RecordingBackend([{"title": "T"}])
    querier = _querier(backend)

    await querier._query_profile(
        {"schema_name": "video_frame"}, sample_size=3, strategy="temporal_recent"
    )

    assert backend.calls[0]["kwargs"]["ranking.sorting"] == "+creation_timestamp"
    assert "yql" not in backend.calls[0]["kwargs"]


async def test_query_profile_returns_empty_on_backend_runtime_error() -> None:
    """A genuine backend failure still degrades to [] (graceful path kept)."""

    class _Boom:
        def query_metadata_documents(self, schema, query=None, yql=None, **kwargs):
            raise RuntimeError("vespa unreachable")

    result = await _querier(_Boom())._query_profile({"schema_name": "s"}, 5, "diverse")
    assert result == []


async def test_query_profile_propagates_signature_typeerror() -> None:
    """A real argument-mismatch (programming bug) must surface, not be masked
    as an empty result."""

    class _BadSignature:
        def query_metadata_documents(self, schema, yql=None):  # rejects hits
            return []

    with pytest.raises(TypeError):
        await _querier(_BadSignature())._query_profile(
            {"schema_name": "s"}, 5, "diverse"
        )
