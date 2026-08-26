"""Unit coverage for the entity-extraction ground-truth admin upload route.

The route must persist tenant-owned ground-truth rows as a versioned blob,
activate the uploaded version, reject invalid rows before touching the store,
and preserve last-write-wins under concurrent same-tenant PUTs.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager
from cogniverse_foundation.telemetry.providers.base import DatasetNotFoundError
from cogniverse_runtime.routers import admin as admin_router

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

FIXED_NOW = datetime(2026, 8, 26, 0, 0, tzinfo=timezone.utc)


class _FrozenDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return FIXED_NOW.replace(tzinfo=None)
        return FIXED_NOW.astimezone(tz)


class _InMemoryDatasetStore:
    def __init__(
        self,
        *,
        get_error: Exception | None = None,
        block_first_active_write: bool = False,
    ) -> None:
        self.datasets: dict[str, Any] = {}
        self.create_calls: list[dict[str, Any]] = []
        self.delete_calls: list[str] = []
        self.get_calls: list[str] = []
        self._get_error = get_error
        self._block_first_active_write = block_first_active_write
        self._active_dataset_name: str | None = None
        self._first_active_write_seen = False
        self.first_active_write_started = asyncio.Event()
        self.release_first_active_write = asyncio.Event()

    def set_active_dataset_name(self, name: str) -> None:
        self._active_dataset_name = name

    async def replace_dataset(self, name, data, metadata=None):
        return await self.create_dataset(name=name, data=data, metadata=metadata)

    async def create_dataset(self, name, data, metadata=None):
        if (
            self._block_first_active_write
            and name == self._active_dataset_name
            and not self._first_active_write_seen
        ):
            self._first_active_write_seen = True
            self.first_active_write_started.set()
            await self.release_first_active_write.wait()

        frame = data.copy(deep=True)
        self.create_calls.append(
            {
                "name": name,
                "data": frame,
                "metadata": dict(metadata or {}),
            }
        )
        self.datasets[name] = frame
        return f"dataset::{len(self.create_calls)}::{name}"

    async def delete_dataset(self, name):
        self.delete_calls.append(name)
        return self.datasets.pop(name, None) is not None

    async def get_dataset(self, name):
        self.get_calls.append(name)
        if self._get_error is not None:
            raise self._get_error
        if name not in self.datasets:
            raise DatasetNotFoundError(name)
        return self.datasets[name].copy(deep=True)


class _FakeTelemetryProvider:
    def __init__(self, datasets: _InMemoryDatasetStore):
        self.datasets = datasets


def _build_app(monkeypatch, am: ArtifactManager):
    build_calls: list[str] = []
    admin_router._reset_admin_overrides_for_tests()

    def _factory(key: str):
        build_calls.append(key)
        return am

    monkeypatch.setattr(admin_router, "_build_artifact_manager", _factory)
    app = FastAPI()
    app.include_router(admin_router.router, prefix="/admin")
    return app, build_calls


def _fixed_artifact_datetime(monkeypatch):
    import cogniverse_agents.optimizer.artifact_manager as artifact_manager_module

    monkeypatch.setattr(artifact_manager_module, "datetime", _FrozenDatetime)


async def _put(app, path, body):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        return await client.put(path, json=body)


@pytest.mark.asyncio
async def test_upload_round_trip_persists_canonical_rows_and_reads_back(monkeypatch):
    _fixed_artifact_datetime(monkeypatch)
    store = _InMemoryDatasetStore()
    provider = _FakeTelemetryProvider(store)
    am = ArtifactManager(provider, tenant_id="acme")
    app, build_calls = _build_app(monkeypatch, am)

    rows = [
        {
            "query": "  find named people  ",
            "entities": [
                {"text": "  Marie Curie ", "type": "PERSON", "confidence": 0.91},
                {"text": "Paris", "type": "LOCATION"},
            ],
            "source": "tenant_upload",
        }
    ]
    expected_rows = [
        {
            "query": "find named people",
            "entities": [
                {"text": "Marie Curie", "type": "PERSON", "confidence": 0.91},
                {"text": "Paris", "type": "LOCATION"},
            ],
            "source": "tenant_upload",
        }
    ]

    response = await _put(
        app, "/admin/tenants/acme/entity_extraction_ground_truth", rows
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "tenant_id": "acme:acme",
        "row_count": 1,
        "version": 1,
        "active": {
            "version": 1,
            "activated_at": FIXED_NOW.isoformat(),
        },
    }
    assert build_calls == ["acme:acme"]

    versioned_name = am._versioned_dataset_name(
        "config", "entity_extraction_ground_truth", 1
    )
    active_name = am._blob_dataset_name("config", "entity_extraction_ground_truth")
    state_name = am._blob_dataset_name(
        "config", "blob_state_config_entity_extraction_ground_truth"
    )
    assert store.get_calls == [
        versioned_name,
        versioned_name,
        active_name,
        state_name,
    ]
    assert store.datasets[versioned_name].to_dict("records") == [
        {
            "content": json.dumps(
                expected_rows, separators=(",", ":"), ensure_ascii=False
            ),
            "ledger": json.dumps(
                {
                    "version": 1,
                    "kind": "config",
                    "key": "entity_extraction_ground_truth",
                    "consumed_example_ids": [
                        "admin_upload:entity_extraction_ground_truth"
                    ],
                    "decision": "promote",
                    "scored": False,
                    "score": None,
                    "base_score": None,
                    "candidate_score": None,
                    "created_at": FIXED_NOW.isoformat(),
                },
                ensure_ascii=False,
            ),
        }
    ]
    assert store.datasets[active_name].to_dict("records") == [
        {
            "content": json.dumps(
                expected_rows, separators=(",", ":"), ensure_ascii=False
            )
        }
    ]

    from cogniverse_agents.optimizer.entity_extraction_ground_truth import (
        load_entity_extraction_ground_truth_rows,
    )

    assert await load_entity_extraction_ground_truth_rows(am) == expected_rows


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rows", "detail"),
    [
        (
            [{"query": "   ", "entities": [{"text": "Alpha", "type": "PERSON"}]}],
            "entity_extraction_ground_truth row 1 query must be non-empty after stripping whitespace",
        ),
        (
            [{"query": "identify", "entities": []}],
            "entity_extraction_ground_truth row 1 entities must be a non-empty array",
        ),
        (
            [
                {
                    "query": "identify",
                    "entities": [{"text": "  ", "type": "PERSON"}],
                }
            ],
            "entity_extraction_ground_truth row 1 entities entry 1 text must be non-empty after stripping whitespace",
        ),
        (
            [
                {
                    "query": "identify",
                    "entities": [{"text": "Alpha", "type": "  "}],
                }
            ],
            "entity_extraction_ground_truth row 1 entities entry 1 type must be non-empty after stripping whitespace",
        ),
        (
            [
                {
                    "query": "identify",
                    "entities": [
                        {"text": "Marie Curie", "type": "PERSON"},
                        {"text": "marie curie", "type": "PERSON"},
                    ],
                }
            ],
            "entity_extraction_ground_truth row 1 entities entry 2 duplicates a prior text/type pair",
        ),
        (
            [
                {
                    "query": "identify",
                    "entities": [{"text": "Alpha", "type": "PERSON"}],
                },
                {
                    "query": " identify ",
                    "entities": [{"text": "Beta", "type": "ORG"}],
                },
            ],
            "entity_extraction_ground_truth row 2 query duplicates row 1",
        ),
    ],
)
async def test_upload_rejects_invalid_rows_before_store(monkeypatch, rows, detail):
    _fixed_artifact_datetime(monkeypatch)
    store = _InMemoryDatasetStore()
    provider = _FakeTelemetryProvider(store)
    am = ArtifactManager(provider, tenant_id="acme")
    app, build_calls = _build_app(monkeypatch, am)

    response = await _put(
        app, "/admin/tenants/acme/entity_extraction_ground_truth", rows
    )

    assert response.status_code == 400, response.text
    assert response.json() == {"detail": detail}
    assert build_calls == []
    assert store.datasets == {}


@pytest.mark.asyncio
async def test_loader_missing_returns_named_status(monkeypatch):
    _fixed_artifact_datetime(monkeypatch)
    store = _InMemoryDatasetStore(get_error=DatasetNotFoundError("missing"))
    provider = _FakeTelemetryProvider(store)
    am = ArtifactManager(provider, tenant_id="acme")

    from cogniverse_agents.optimizer.entity_extraction_ground_truth import (
        EntityExtractionGroundTruthMissingError,
        load_entity_extraction_ground_truth_rows,
    )

    with pytest.raises(EntityExtractionGroundTruthMissingError) as exc:
        await load_entity_extraction_ground_truth_rows(am)

    assert exc.value.to_result() == {
        "status": "entity_extraction_ground_truth_missing",
        "retryable": False,
        "error": "entity_extraction_ground_truth is not configured for tenant acme:acme",
    }
    assert store.get_calls == [
        am._blob_dataset_name("config", "entity_extraction_ground_truth")
    ]


@pytest.mark.asyncio
async def test_loader_store_error_raises_fault_contract(monkeypatch):
    _fixed_artifact_datetime(monkeypatch)
    store = _InMemoryDatasetStore(get_error=ConnectionError("blob store down"))
    provider = _FakeTelemetryProvider(store)
    am = ArtifactManager(provider, tenant_id="acme")

    from cogniverse_agents.optimizer.entity_extraction_ground_truth import (
        EntityExtractionGroundTruthStoreUnavailableError,
        load_entity_extraction_ground_truth_rows,
    )

    with pytest.raises(EntityExtractionGroundTruthStoreUnavailableError) as exc:
        await load_entity_extraction_ground_truth_rows(am)

    assert exc.value.to_result() == {
        "status": "entity_extraction_ground_truth_store_unavailable",
        "retryable": True,
        "error": "entity_extraction_ground_truth store unavailable",
        "cause": {
            "type": "ConnectionError",
            "message": "blob store down",
        },
    }
    assert store.get_calls == [
        am._blob_dataset_name("config", "entity_extraction_ground_truth")
    ]


@pytest.mark.asyncio
async def test_concurrent_puts_last_writer_wins(monkeypatch):
    _fixed_artifact_datetime(monkeypatch)
    store = _InMemoryDatasetStore(block_first_active_write=True)
    provider = _FakeTelemetryProvider(store)
    am = ArtifactManager(provider, tenant_id="acme")
    store.set_active_dataset_name(
        am._blob_dataset_name("config", "entity_extraction_ground_truth")
    )
    app, build_calls = _build_app(monkeypatch, am)

    first_rows = [
        {
            "query": "find scientists",
            "entities": [{"text": "Marie Curie", "type": "PERSON"}],
        }
    ]
    second_rows = [
        {
            "query": "find physicists",
            "entities": [{"text": "Albert Einstein", "type": "PERSON"}],
        }
    ]
    first_expected = [
        {
            "query": "find scientists",
            "entities": [{"text": "Marie Curie", "type": "PERSON"}],
        }
    ]
    second_expected = [
        {
            "query": "find physicists",
            "entities": [{"text": "Albert Einstein", "type": "PERSON"}],
        }
    ]

    first_task = asyncio.create_task(
        _put(
            app,
            "/admin/tenants/acme/entity_extraction_ground_truth",
            first_rows,
        )
    )
    await asyncio.wait_for(store.first_active_write_started.wait(), timeout=5)

    second_task = asyncio.create_task(
        _put(
            app,
            "/admin/tenants/acme/entity_extraction_ground_truth",
            second_rows,
        )
    )
    second_response = await asyncio.wait_for(second_task, timeout=5)
    store.release_first_active_write.set()
    first_response = await asyncio.wait_for(first_task, timeout=5)

    assert first_response.status_code == 200, first_response.text
    assert second_response.status_code == 200, second_response.text
    assert first_response.json()["version"] == 1
    assert second_response.json()["version"] == 2
    assert build_calls == ["acme:acme", "acme:acme"]

    from cogniverse_agents.optimizer.entity_extraction_ground_truth import (
        load_entity_extraction_ground_truth_rows,
    )

    loaded = await load_entity_extraction_ground_truth_rows(am)
    assert loaded == first_expected
    assert loaded != second_expected

    active_name = am._blob_dataset_name("config", "entity_extraction_ground_truth")
    assert store.datasets[active_name].to_dict("records") == [
        {
            "content": json.dumps(
                first_expected, separators=(",", ":"), ensure_ascii=False
            )
        }
    ]
