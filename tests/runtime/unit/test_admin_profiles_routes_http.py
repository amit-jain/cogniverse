"""HTTP-level coverage for the admin backend-profile routes.

The list / get / update / delete / deploy / create profile routes were only
exercised behind real-Vespa integration tests, so their 2xx success bodies —
``response_model`` serialization, query/path/body binding, and the
ConfigManager call wiring — never ran in the standard pytest gate. Driving
them through the mounted FastAPI app with ``httpx.ASGITransport`` runs the full
request-parse → handler → response-model path without Docker: a stub
ConfigManager returns realistic ``BackendProfileConfig`` objects and a fake
BackendRegistry backend controls ``schema_exists`` / ``deploy_schema`` so the
tests assert the exact response shape and the exact arguments each route hands
to the store, including the canonical tenant id used for the config lookup.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cogniverse_foundation.config.unified_config import BackendProfileConfig
from cogniverse_runtime.routers import admin

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

_FIXED_CREATED_AT = datetime(2026, 1, 2, 3, 4, 5)
_STORE_VERSION = 7


def _profile(name: str, schema: str, embedding_model: str) -> BackendProfileConfig:
    return BackendProfileConfig(
        profile_name=name,
        type="video",
        description=f"desc for {name}",
        schema_name=schema,
        embedding_model=embedding_model,
        pipeline_config={"extract_keyframes": True, "keyframe_fps": 2.0},
        strategies={"segmentation": {"class": "FrameSegmentationStrategy"}},
        embedding_type="multi_vector",
        schema_config={"embedding_dim": 128, "num_patches": 1024},
        model_specific={"revision": "main"},
    )


class _FakeConfigStore:
    def __init__(self):
        self.get_config_calls = []

    def get_config(self, *, tenant_id, scope, service, config_key):
        self.get_config_calls.append(
            {
                "tenant_id": tenant_id,
                "scope": scope,
                "service": service,
                "config_key": config_key,
            }
        )
        return SimpleNamespace(version=_STORE_VERSION, created_at=_FIXED_CREATED_AT)


class _StubConfigManager:
    def __init__(self):
        self.profiles: dict[str, BackendProfileConfig] = {}
        self.store = _FakeConfigStore()
        self.calls: dict = {}

    def list_backend_profiles(self, tenant_id=None, service="backend"):
        self.calls["list"] = {"tenant_id": tenant_id, "service": service}
        return dict(self.profiles)

    def get_backend_profile(self, profile_name, tenant_id=None, service="backend"):
        self.calls.setdefault("get", []).append(
            {"profile_name": profile_name, "tenant_id": tenant_id, "service": service}
        )
        return self.profiles.get(profile_name)

    def update_backend_profile(
        self,
        profile_name,
        overrides,
        base_tenant_id="__system__",
        target_tenant_id=None,
        service="backend",
    ):
        self.calls["update"] = {
            "profile_name": profile_name,
            "overrides": overrides,
            "base_tenant_id": base_tenant_id,
            "target_tenant_id": target_tenant_id,
            "service": service,
        }
        return self.profiles.get(profile_name)

    def delete_backend_profile(self, profile_name, tenant_id=None, service="backend"):
        self.calls["delete"] = {
            "profile_name": profile_name,
            "tenant_id": tenant_id,
            "service": service,
        }
        return True

    def add_backend_profile(self, profile, tenant_id=None, service="backend"):
        self.calls["add"] = {
            "profile": profile,
            "tenant_id": tenant_id,
            "service": service,
        }
        return profile


class _StubValidator:
    def __init__(self):
        self.calls: dict = {}

    def validate_profile(self, profile, tenant_id, is_update):
        self.calls["validate_profile"] = {
            "profile": profile,
            "tenant_id": tenant_id,
            "is_update": is_update,
        }
        return []

    def validate_update_fields(self, overrides):
        self.calls["validate_update_fields"] = {"overrides": overrides}
        return []


class _FakeBackend:
    def __init__(self):
        self.deployed_schemas: set[str] = set()
        self.deleted: list = []
        self.deploy_calls: list = []
        self.schema_exists_calls: list = []
        # deploy_schema is reached via ``backend.schema_registry.deploy_schema``.
        self.schema_registry = self

    def schema_exists(self, schema_name, tenant_id):
        self.schema_exists_calls.append(
            {"schema_name": schema_name, "tenant_id": tenant_id}
        )
        return schema_name in self.deployed_schemas

    def get_tenant_schema_name(self, tenant_id, schema_name):
        return f"{tenant_id.replace(':', '_')}_{schema_name}"

    def delete_schema(self, schema_name, tenant_id):
        self.deleted.append({"schema_name": schema_name, "tenant_id": tenant_id})
        return [f"{tenant_id.replace(':', '_')}_{schema_name}"]

    def deploy_schema(self, tenant_id, base_schema_name, force=False):
        self.deploy_calls.append(
            {
                "tenant_id": tenant_id,
                "base_schema_name": base_schema_name,
                "force": force,
            }
        )


class _FakeRegistry:
    def __init__(self, backend):
        self._backend = backend
        self.calls: list = []

    def get_ingestion_backend(
        self, backend_type, *, tenant_id, config_manager, schema_loader
    ):
        self.calls.append({"backend_type": backend_type, "tenant_id": tenant_id})
        return self._backend


@dataclass
class _Env:
    app: FastAPI
    cm: _StubConfigManager
    backend: _FakeBackend
    registry: _FakeRegistry
    validator: _StubValidator


@pytest.fixture
def env(monkeypatch):
    cm = _StubConfigManager()
    backend = _FakeBackend()
    registry = _FakeRegistry(backend)
    validator = _StubValidator()

    monkeypatch.setattr(
        admin, "BackendRegistry", SimpleNamespace(get_instance=lambda: registry)
    )

    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    app.dependency_overrides[admin.get_config_manager_dependency] = lambda: cm
    app.dependency_overrides[admin.get_schema_loader_dependency] = lambda: (
        SimpleNamespace()
    )
    app.dependency_overrides[admin.get_profile_validator_dependency] = lambda: validator

    return _Env(app=app, cm=cm, backend=backend, registry=registry, validator=validator)


async def _get(app, path, **params):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        return await client.get(path, params=params)


async def _put(app, path, json, **params):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        return await client.put(path, json=json, params=params)


async def _post(app, path, json, **params):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        return await client.post(path, json=json, params=params)


async def _delete(app, path, **params):
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://t"
    ) as client:
        return await client.delete(path, params=params)


@pytest.mark.asyncio
async def test_list_profiles_returns_exact_summaries_and_deployment_flags(env):
    env.cm.profiles["prof_a"] = _profile("prof_a", "video_colpali_sv", "colpali-v1.2")
    env.cm.profiles["prof_b"] = _profile("prof_b", "video_prism_mv", "videoprism-lvt")
    # Only prof_a's base schema is deployed, so its summary flag is True.
    env.backend.deployed_schemas = {"video_colpali_sv"}

    resp = await _get(env.app, "/admin/profiles", tenant_id="acme")

    assert resp.status_code == 200
    body = resp.json()
    assert body["total_count"] == 2
    # list echoes the raw tenant id, not the canonical form.
    assert body["tenant_id"] == "acme"
    summaries = [
        {k: v for k, v in p.items() if k != "created_at"} for p in body["profiles"]
    ]
    assert summaries == [
        {
            "profile_name": "prof_a",
            "type": "video",
            "description": "desc for prof_a",
            "schema_name": "video_colpali_sv",
            "embedding_model": "colpali-v1.2",
            "schema_deployed": True,
        },
        {
            "profile_name": "prof_b",
            "type": "video",
            "description": "desc for prof_b",
            "schema_name": "video_prism_mv",
            "embedding_model": "videoprism-lvt",
            "schema_deployed": False,
        },
    ]
    assert env.cm.calls["list"] == {"tenant_id": "acme", "service": "backend"}
    assert env.backend.schema_exists_calls == [
        {"schema_name": "video_colpali_sv", "tenant_id": "acme"},
        {"schema_name": "video_prism_mv", "tenant_id": "acme"},
    ]


@pytest.mark.asyncio
async def test_get_profile_returns_full_detail_with_canonical_store_lookup(env):
    env.cm.profiles["video_colpali"] = _profile(
        "video_colpali", "video_colpali_sv", "colpali-v1.2"
    )
    env.backend.deployed_schemas = {"video_colpali_sv"}

    resp = await _get(env.app, "/admin/profiles/video_colpali", tenant_id="acme")

    assert resp.status_code == 200
    assert resp.json() == {
        "profile_name": "video_colpali",
        "tenant_id": "acme",
        "type": "video",
        "description": "desc for video_colpali",
        "schema_name": "video_colpali_sv",
        "embedding_model": "colpali-v1.2",
        "pipeline_config": {"extract_keyframes": True, "keyframe_fps": 2.0},
        "strategies": {"segmentation": {"class": "FrameSegmentationStrategy"}},
        "embedding_type": "multi_vector",
        "schema_config": {"embedding_dim": 128, "num_patches": 1024},
        "model_specific": {"revision": "main"},
        "schema_deployed": True,
        "tenant_schema_name": "acme_video_colpali_sv",
        "created_at": "2026-01-02T03:04:05",
        "version": _STORE_VERSION,
    }
    # The version/created_at lookup goes through the canonical tenant id.
    assert env.cm.store.get_config_calls[-1]["tenant_id"] == "acme:acme"


@pytest.mark.asyncio
async def test_get_profile_missing_returns_404(env):
    resp = await _get(env.app, "/admin/profiles/nope", tenant_id="acme")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Profile 'nope' not found for tenant 'acme'"


@pytest.mark.asyncio
async def test_update_profile_persists_overrides_and_echoes_updated_fields(env):
    env.cm.profiles["video_colpali"] = _profile(
        "video_colpali", "video_colpali_sv", "colpali-v1.2"
    )

    resp = await _put(
        env.app,
        "/admin/profiles/video_colpali",
        json={
            "tenant_id": "acme",
            "pipeline_config": {"keyframe_fps": 30.0},
            "description": "new desc",
        },
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "profile_name": "video_colpali",
        "tenant_id": "acme",
        # Field order follows the route's check order: pipeline_config first,
        # description second; strategies/model_specific were omitted.
        "updated_fields": ["pipeline_config", "description"],
        "version": _STORE_VERSION,
    }
    assert env.cm.calls["update"] == {
        "profile_name": "video_colpali",
        "overrides": {
            "pipeline_config": {"keyframe_fps": 30.0},
            "description": "new desc",
        },
        "base_tenant_id": "acme",
        "target_tenant_id": "acme",
        "service": "backend",
    }
    assert env.validator.calls["validate_update_fields"] == {
        "overrides": {
            "pipeline_config": {"keyframe_fps": 30.0},
            "description": "new desc",
        }
    }
    assert env.cm.store.get_config_calls[-1]["tenant_id"] == "acme:acme"


@pytest.mark.asyncio
async def test_update_profile_with_no_fields_returns_400(env):
    env.cm.profiles["video_colpali"] = _profile(
        "video_colpali", "video_colpali_sv", "colpali-v1.2"
    )

    resp = await _put(
        env.app, "/admin/profiles/video_colpali", json={"tenant_id": "acme"}
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "No fields to update provided"
    assert "update" not in env.cm.calls


@pytest.mark.asyncio
async def test_update_profile_missing_returns_404(env):
    resp = await _put(
        env.app,
        "/admin/profiles/nope",
        json={"tenant_id": "acme", "description": "x"},
    )
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Profile 'nope' not found for tenant 'acme'"


@pytest.mark.asyncio
async def test_delete_profile_without_schema_removes_config_only(env):
    env.cm.profiles["video_colpali"] = _profile(
        "video_colpali", "video_colpali_sv", "colpali-v1.2"
    )

    resp = await _delete(env.app, "/admin/profiles/video_colpali", tenant_id="acme")

    assert resp.status_code == 200
    body = resp.json()
    assert body["profile_name"] == "video_colpali"
    assert body["tenant_id"] == "acme"
    assert body["schema_deleted"] is False
    assert isinstance(body["deleted_at"], str) and body["deleted_at"]
    assert env.cm.calls["delete"] == {
        "profile_name": "video_colpali",
        "tenant_id": "acme",
        "service": "backend",
    }
    # delete_schema defaulted false, so the Vespa schema was never touched.
    assert env.backend.deleted == []


@pytest.mark.asyncio
async def test_delete_profile_with_schema_drops_vespa_schema(env):
    env.cm.profiles["video_colpali"] = _profile(
        "video_colpali", "video_colpali_sv", "colpali-v1.2"
    )

    resp = await _delete(
        env.app,
        "/admin/profiles/video_colpali",
        tenant_id="acme",
        delete_schema=True,
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["profile_name"] == "video_colpali"
    assert body["schema_deleted"] is True
    assert env.backend.deleted == [
        {"schema_name": "video_colpali_sv", "tenant_id": "acme"}
    ]
    assert env.cm.calls["delete"] == {
        "profile_name": "video_colpali",
        "tenant_id": "acme",
        "service": "backend",
    }


@pytest.mark.asyncio
async def test_deploy_schema_success_invokes_deploy_primitive(env):
    env.cm.profiles["video_prism"] = _profile(
        "video_prism", "video_prism_mv", "videoprism-lvt"
    )
    # base schema not yet deployed -> route proceeds to deploy_schema.
    env.backend.deployed_schemas = set()

    resp = await _post(
        env.app,
        "/admin/profiles/video_prism/deploy",
        json={"tenant_id": "acme", "force": False},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["profile_name"] == "video_prism"
    assert body["tenant_id"] == "acme"
    assert body["schema_name"] == "video_prism_mv"
    assert body["tenant_schema_name"] == "acme_video_prism_mv"
    assert body["deployment_status"] == "success"
    assert body["error_message"] is None
    assert env.backend.deploy_calls == [
        {"tenant_id": "acme", "base_schema_name": "video_prism_mv", "force": False}
    ]


@pytest.mark.asyncio
async def test_deploy_schema_already_deployed_skips_deploy(env):
    env.cm.profiles["video_colpali"] = _profile(
        "video_colpali", "video_colpali_sv", "colpali-v1.2"
    )
    env.backend.deployed_schemas = {"video_colpali_sv"}

    resp = await _post(
        env.app,
        "/admin/profiles/video_colpali/deploy",
        json={"tenant_id": "acme", "force": False},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["deployment_status"] == "already_deployed"
    assert body["tenant_schema_name"] == "acme_video_colpali_sv"
    assert body["schema_name"] == "video_colpali_sv"
    # force was false and the schema exists, so no deploy ran.
    assert env.backend.deploy_calls == []


@pytest.mark.asyncio
async def test_create_profile_adds_profile_without_deploy(env):
    resp = await _post(
        env.app,
        "/admin/profiles",
        json={
            "profile_name": "new_prof",
            "tenant_id": "acme",
            "type": "video",
            "description": "brand new",
            "schema_name": "video_new_sv",
            "embedding_model": "colpali-v1.2",
            "embedding_type": "single_vector",
            "pipeline_config": {"extract_keyframes": True},
            "strategies": {"embedding": {"class": "SingleVectorEmbeddingStrategy"}},
            "schema_config": {"embedding_dim": 768},
            "model_specific": {"revision": "main"},
            "deploy_schema": False,
        },
    )

    assert resp.status_code == 201
    body = resp.json()
    # created_at is datetime.now() at handler time; assert the rest exactly.
    created_at = body.pop("created_at")
    assert isinstance(created_at, str) and created_at
    assert body == {
        "profile_name": "new_prof",
        "tenant_id": "acme",
        "schema_deployed": False,
        "tenant_schema_name": None,
        "version": _STORE_VERSION,
    }
    add = env.cm.calls["add"]
    assert add["tenant_id"] == "acme"
    assert add["service"] == "backend"
    persisted = add["profile"]
    assert persisted.profile_name == "new_prof"
    assert persisted.schema_name == "video_new_sv"
    assert persisted.embedding_type == "single_vector"
    assert persisted.embedding_model == "colpali-v1.2"
    assert persisted.type == "video"
    assert persisted.description == "brand new"
    # deploy_schema false -> the Vespa deploy primitive was never invoked.
    assert env.backend.deploy_calls == []
    assert env.validator.calls["validate_profile"]["is_update"] is False
