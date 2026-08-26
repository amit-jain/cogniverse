"""Unit coverage for the profile-selection ground-truth admin upload route.

The route must persist tenant-owned ground-truth rows as a versioned blob,
activate the uploaded version, and reject invalid rows before touching the
store.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cogniverse_runtime.routers import admin as admin_router

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]


class _StubArtifactManager:
    def __init__(self):
        self.save_calls = []
        self.activate_calls = []

    async def save_blob_versioned(
        self,
        kind,
        key,
        content,
        *,
        consumed_example_ids,
        decision,
        scored,
        score,
        base_score,
        candidate_score,
    ):
        self.save_calls.append(
            {
                "kind": kind,
                "key": key,
                "content": content,
                "consumed_example_ids": list(consumed_example_ids),
                "decision": decision,
                "scored": scored,
                "score": score,
                "base_score": base_score,
                "candidate_score": candidate_score,
            }
        )
        return "dataset-1", 1

    async def activate_version(self, kind, key, version):
        return await self.activate_version_guarded(kind, key, version)

    async def activate_version_guarded(self, kind, key, version):
        self.activate_calls.append((kind, key, version))
        return {
            "active": {"version": version, "activated_at": "2026-08-26T00:00:00+00:00"}
        }

    async def load_blob(self, kind, key):
        raise AssertionError("load_blob is not expected during upload")


def _build_app(monkeypatch):
    stub = _StubArtifactManager()
    admin_router._reset_admin_overrides_for_tests()
    monkeypatch.setattr(admin_router, "_build_artifact_manager", lambda key: stub)
    app = FastAPI()
    app.include_router(admin_router.router, prefix="/admin")
    return app, stub


async def _put(app, path, body):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        return await client.put(path, json=body)


@pytest.mark.asyncio
async def test_upload_persists_canonical_blob_and_activates_version(monkeypatch):
    app, stub = _build_app(monkeypatch)
    rows = [
        {
            "query": "  find basketball highlights  ",
            "expected_videos": ["v-1", " v-2 ", ""],
            "ground_truth": "basketball",
            "query_type": "question",
            "source": "tenant_upload",
        }
    ]

    try:
        response = await _put(
            app, "/admin/tenants/acme/profile_selection_ground_truth", rows
        )
    finally:
        admin_router._reset_admin_overrides_for_tests()

    assert response.status_code == 200, response.text
    assert response.json() == {
        "tenant_id": "acme:acme",
        "row_count": 1,
        "version": 1,
        "active": {
            "version": 1,
            "activated_at": "2026-08-26T00:00:00+00:00",
        },
    }

    expected_rows = [
        {
            "query": "find basketball highlights",
            "expected_videos": ["v-1", "v-2"],
            "ground_truth": "basketball",
            "query_type": "question",
            "source": "tenant_upload",
        }
    ]
    assert stub.save_calls == [
        {
            "kind": "config",
            "key": "profile_selection_ground_truth",
            "content": json.dumps(expected_rows, separators=(",", ":")),
            "consumed_example_ids": ["admin_upload:profile_selection_ground_truth"],
            "decision": "promote",
            "scored": False,
            "score": None,
            "base_score": None,
            "candidate_score": None,
        }
    ]
    assert stub.activate_calls == [("config", "profile_selection_ground_truth", 1)]


@pytest.mark.asyncio
async def test_upload_rejects_blank_query_with_400(monkeypatch):
    app, stub = _build_app(monkeypatch)
    try:
        response = await _put(
            app,
            "/admin/tenants/acme/profile_selection_ground_truth",
            [{"query": "   ", "expected_videos": ["v-1"]}],
        )
    finally:
        admin_router._reset_admin_overrides_for_tests()

    assert response.status_code == 400, response.text
    assert response.json() == {
        "detail": (
            "profile_selection_ground_truth row 1 query must be non-empty after "
            "stripping whitespace"
        )
    }
    assert stub.save_calls == []
    assert stub.activate_calls == []


@pytest.mark.asyncio
async def test_upload_rejects_empty_normalized_expected_videos_with_400(monkeypatch):
    app, stub = _build_app(monkeypatch)
    try:
        response = await _put(
            app,
            "/admin/tenants/acme/profile_selection_ground_truth",
            [{"query": "find something", "expected_videos": ["  ", "\n"]}],
        )
    finally:
        admin_router._reset_admin_overrides_for_tests()

    assert response.status_code == 400, response.text
    assert response.json() == {
        "detail": (
            "profile_selection_ground_truth row 1 expected_videos must contain at "
            "least one non-empty id after normalization"
        )
    }
    assert stub.save_calls == []
    assert stub.activate_calls == []
