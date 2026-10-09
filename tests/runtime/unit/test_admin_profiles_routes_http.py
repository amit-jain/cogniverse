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

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from cogniverse_foundation.common.tenant_utils import SYSTEM_TENANT_ID
from cogniverse_foundation.config.manager import BackendProfileWrite, ConfigManager
from cogniverse_foundation.config.unified_config import (
    BackendConfig,
    BackendProfileConfig,
)
from cogniverse_runtime.admin.profile_models import ProfileCreateRequest
from cogniverse_runtime.routers import admin
from cogniverse_sdk.interfaces.config_store import ConfigScope
from tests.utils.memory_store import InMemoryConfigStore

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

_FIXED_CREATED_AT = datetime(2026, 1, 2, 3, 4, 5)
_STORE_VERSION = 7
# The backend config version a stub profile write reports producing.
_WRITTEN_VERSION = 12


def test_profile_create_example_uses_deployed_visual_encoder_contract():
    example = ProfileCreateRequest.model_json_schema()["example"]

    assert example["embedding_model"] == "TomoroAI/tomoro-colqwen3-embed-4b"
    assert example["schema_config"] == {
        "schema_name": "video_colpali_smol500_mv_frame",
        "model_name": "TomoroAI/tomoro-colqwen3-embed-4b",
        "num_patches": 1024,
        "embedding_dim": 320,
        "binary_dim": 40,
    }


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
        model_loader="colpali",
        process_type="frame_based",
        extra_config={"inference_services": {"embedding": "vllm_colpali"}},
    )


class _FakeConfigStore:
    def __init__(self, profiles: dict):
        self.profiles = profiles
        self.get_config_calls = []
        self.deletion_marker_reads = []

    def get_immutable_config(self, tenant_id, scope, service, config_key):
        # No tenant is marked deleted.
        self.deletion_marker_reads.append((tenant_id, service, config_key))
        return None

    def get_config(self, *, tenant_id, scope, service, config_key):
        self.get_config_calls.append(
            {
                "tenant_id": tenant_id,
                "scope": scope,
                "service": service,
                "config_key": config_key,
            }
        )
        return SimpleNamespace(
            version=_STORE_VERSION,
            created_at=_FIXED_CREATED_AT,
            config_value=BackendConfig(
                tenant_id=tenant_id, profiles=dict(self.profiles)
            ).to_dict(),
        )


class _StubConfigManager:
    def __init__(self):
        self.profiles: dict[str, BackendProfileConfig] = {}
        self.store = _FakeConfigStore(self.profiles)
        self.calls: dict = {}

    def list_backend_profiles(self, tenant_id=None, service="backend"):
        self.calls["list"] = {"tenant_id": tenant_id, "service": service}
        return dict(self.profiles)

    def get_backend_profile(self, profile_name, tenant_id=None, service="backend"):
        self.calls.setdefault("get", []).append(
            {"profile_name": profile_name, "tenant_id": tenant_id, "service": service}
        )
        return self.profiles.get(profile_name)

    def get_stored_backend_config(self, tenant_id=None, service="backend"):
        self.calls.setdefault("stored", []).append(
            {"tenant_id": tenant_id, "service": service}
        )
        return BackendConfig(tenant_id=tenant_id, profiles=dict(self.profiles))

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
        return BackendProfileWrite(
            profile=self.profiles.get(profile_name), version=_WRITTEN_VERSION
        )

    def delete_backend_profile(self, profile_name, tenant_id=None, service="backend"):
        self.calls["delete"] = {
            "profile_name": profile_name,
            "tenant_id": tenant_id,
            "service": service,
        }
        return True

    def add_backend_profile(
        self, profile, tenant_id=None, service="backend", *, replace=True
    ):
        self.calls["add"] = {
            "profile": profile,
            "tenant_id": tenant_id,
            "service": service,
            "replace": replace,
        }
        return BackendProfileWrite(profile=profile, version=_WRITTEN_VERSION)


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
    events: object


@pytest.fixture
def env(monkeypatch, in_process_config_events):
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

    return _Env(
        app=app,
        cm=cm,
        backend=backend,
        registry=registry,
        validator=validator,
        events=in_process_config_events,
    )


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
    env.cm.profiles["prof_b"] = _profile("prof_b", "video_prism_mv", "xclip-lvt")
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
            "embedding_model": "xclip-lvt",
            "schema_deployed": False,
        },
    ]
    assert env.cm.calls["stored"] == [{"tenant_id": "acme", "service": "backend"}]
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
        "model_loader": "colpali",
        "process_type": "frame_based",
        "extra_config": {"inference_services": {"embedding": "vllm_colpali"}},
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
    # Nothing was written, so no worker is told to drop anything.
    assert env.events.published == []


@pytest.mark.asyncio
async def test_list_profiles_schema_lookup_failure_raises_500(env):
    env.cm.profiles["prof_a"] = _profile("prof_a", "video_colpali_sv", "colpali-v1.2")
    env.backend.schema_exists = MagicMock(
        side_effect=RuntimeError("schema registry unavailable")
    )

    resp = await _get(env.app, "/admin/profiles", tenant_id="acme")

    assert resp.status_code == 500
    assert resp.json() == {
        "detail": {
            "error": "profile_list_failed",
            "message": (
                "Listing profiles for tenant 'acme' failed; the runtime log "
                "names the cause."
            ),
            "failure": "RuntimeError",
            "tenant_id": "acme",
        }
    }
    assert env.backend.schema_exists.call_args_list == [
        call(schema_name="video_colpali_sv", tenant_id="acme"),
    ]


@pytest.mark.asyncio
async def test_get_profile_schema_lookup_failure_raises_500(env):
    env.cm.profiles["video_colpali"] = _profile(
        "video_colpali", "video_colpali_sv", "colpali-v1.2"
    )
    env.backend.schema_exists = MagicMock(
        side_effect=RuntimeError("schema registry unavailable")
    )

    resp = await _get(env.app, "/admin/profiles/video_colpali", tenant_id="acme")

    assert resp.status_code == 500
    assert resp.json() == {
        "detail": {
            "error": "profile_read_failed",
            "message": (
                "Reading profile 'video_colpali' failed; the runtime log names "
                "the cause."
            ),
            "failure": "RuntimeError",
            "profile_name": "video_colpali",
            "tenant_id": "acme",
        }
    }
    assert env.backend.schema_exists.call_args_list == [
        call(schema_name="video_colpali_sv", tenant_id="acme"),
    ]


def _held_copy_with_a_deleted_profile(*args, **kwargs):
    """This process's held backend config, from before another worker's
    delete: it still has the profile."""
    return {
        "video_colpali": _profile("video_colpali", "video_colpali_sv", "colpali-v1.2")
    }


@pytest.mark.asyncio
async def test_list_profiles_answers_the_store_not_the_held_copy(env):
    env.cm.list_backend_profiles = _held_copy_with_a_deleted_profile

    resp = await _get(env.app, "/admin/profiles", tenant_id="acme")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"profiles": [], "total_count": 0, "tenant_id": "acme"}
    assert env.cm.calls["stored"] == [{"tenant_id": "acme", "service": "backend"}]


@pytest.mark.asyncio
async def test_get_profile_answers_the_store_not_the_held_copy(env):
    env.cm.get_backend_profile = lambda *args, **kwargs: (
        _held_copy_with_a_deleted_profile()["video_colpali"]
    )

    resp = await _get(env.app, "/admin/profiles/video_colpali", tenant_id="acme")

    assert resp.status_code == 404
    assert resp.json() == {
        "detail": "Profile 'video_colpali' not found for tenant 'acme'"
    }
    assert env.cm.store.get_config_calls == [
        {
            "tenant_id": "acme:acme",
            "scope": ConfigScope.BACKEND,
            "service": "backend",
            "config_key": "backend_config",
        }
    ]


@pytest.mark.parametrize(
    ("path", "error", "message", "identity"),
    [
        (
            "/admin/profiles",
            "profile_list_failed",
            "Listing profiles for tenant 'acme' failed; the runtime log names "
            "the cause.",
            {"tenant_id": "acme"},
        ),
        (
            "/admin/profiles/video_colpali",
            "profile_read_failed",
            "Reading profile 'video_colpali' failed; the runtime log names the cause.",
            {"profile_name": "video_colpali", "tenant_id": "acme"},
        ),
    ],
)
@pytest.mark.asyncio
async def test_profile_reads_whose_store_cannot_be_read_raise_500_not_the_held_copy(
    env, caplog, path, error, message, identity
):
    from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError

    caplog.set_level(logging.ERROR, logger="cogniverse_runtime.http_errors")
    env.cm.list_backend_profiles = _held_copy_with_a_deleted_profile
    env.cm.get_backend_profile = lambda *args, **kwargs: (
        _held_copy_with_a_deleted_profile()["video_colpali"]
    )

    def unreadable(tenant_id=None, service="backend"):
        raise ConfigStoreUnavailableError("config store unreachable")

    def unreadable_row(**kwargs):
        raise ConfigStoreUnavailableError("config store unreachable")

    env.cm.get_stored_backend_config = unreadable
    env.cm.store.get_config = unreadable_row

    resp = await _get(env.app, path, tenant_id="acme")

    assert resp.status_code == 500
    assert resp.json() == {
        "detail": {
            "error": error,
            "message": message,
            "failure": "ConfigStoreUnavailableError",
            **identity,
        }
    }
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "cogniverse_runtime.http_errors"
    ] == [f"{error}: ConfigStoreUnavailableError: config store unreachable"]
    assert env.backend.schema_exists_calls == []


@pytest.mark.asyncio
async def test_get_profile_pairs_the_content_with_the_version_it_was_read_at(env):
    """Another worker's update lands while the get is reading the stored
    row: the answer is one stored version, its content with its number."""
    store = InMemoryConfigStore()
    serving, writing = ConfigManager(store=store), ConfigManager(store=store)
    tenant = "acme:acme"
    created = serving.add_backend_profile(
        _profile("video_colpali", "video_colpali_sv", "colpali-v1.2"),
        tenant_id=tenant,
    )
    env.app.dependency_overrides[admin.get_config_manager_dependency] = lambda: serving
    read_versions = []
    written: list = []
    landed = threading.Event()
    barrier = threading.Barrier(2)
    read_row = store.get_config

    def get_config(*args, **kwargs):
        entry = read_row(*args, **kwargs)
        if threading.current_thread() is not writer:
            read_versions.append(entry.version)
            if len(read_versions) == 1:
                barrier.wait(timeout=10)
                landed.wait(timeout=10)
        return entry

    def write() -> None:
        barrier.wait(timeout=10)
        written.append(
            writing.update_backend_profile(
                "video_colpali",
                {"description": "written between the reads"},
                base_tenant_id=tenant,
                target_tenant_id=tenant,
            ).version
        )
        landed.set()

    store.get_config = get_config
    writer = threading.Thread(target=write)
    writer.start()
    try:
        resp = await _get(env.app, "/admin/profiles/video_colpali", tenant_id=tenant)
    finally:
        writer.join(timeout=30)

    assert resp.status_code == 200, resp.text
    assert (resp.json()["description"], resp.json()["version"]) == (
        "desc for video_colpali",
        created.version,
    )
    assert read_versions == [created.version]
    assert written == [created.version + 1]
    stored = read_row(
        tenant_id=tenant,
        scope=ConfigScope.BACKEND,
        service="backend",
        config_key="backend_config",
    )
    assert (
        stored.version,
        stored.config_value["profiles"]["video_colpali"]["description"],
    ) == (created.version + 1, "written between the reads")


@pytest.mark.parametrize("path", ["/admin/profiles", "/admin/profiles/video_colpali"])
@pytest.mark.asyncio
async def test_profile_reads_keep_the_loop_serving_during_a_slow_schema_lookup(
    env, path
):
    """The schema lookup queries Vespa; while it waits, the loop serves a
    heartbeat, which is what releases the lookup."""
    env.cm.profiles["video_colpali"] = _profile(
        "video_colpali", "video_colpali_sv", "colpali-v1.2"
    )
    started = threading.Event()
    released = threading.Event()
    lookups = []

    def slow_schema_exists(schema_name, tenant_id):
        started.set()
        lookups.append(released.wait(timeout=5))
        return True

    env.backend.schema_exists = slow_schema_exists
    beats = []

    async def heartbeat() -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 30
        while not started.is_set() and loop.time() < deadline:
            await asyncio.sleep(0.01)
        beats.append(lookups == [])
        released.set()

    resp, _ = await asyncio.gather(_get(env.app, path, tenant_id="acme"), heartbeat())

    assert resp.status_code == 200, resp.text
    assert beats == [True]
    assert lookups == [True]


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
        "version": _WRITTEN_VERSION,
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
    # The version is the one the update produced, not read back afterwards.
    assert env.cm.store.get_config_calls == []
    # Every worker dropped the tenant's held profiles before the answer.
    assert env.events.published == [
        ("backend_profiles_changed", {"tenant_id": "acme:acme"})
    ]


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


def _held_copy_without_the_profile(*args, **kwargs):
    """This process's held backend config, from before another worker's
    create: it has no profile at all."""
    return None


@pytest.mark.asyncio
async def test_update_profile_decides_existence_on_the_store_not_the_held_copy(env):
    env.cm.profiles["video_colpali"] = _profile(
        "video_colpali", "video_colpali_sv", "colpali-v1.2"
    )
    env.cm.get_backend_profile = _held_copy_without_the_profile

    resp = await _put(
        env.app,
        "/admin/profiles/video_colpali",
        json={"tenant_id": "acme", "description": "new desc"},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json() == {
        "profile_name": "video_colpali",
        "tenant_id": "acme",
        "updated_fields": ["description"],
        "version": _WRITTEN_VERSION,
    }
    assert env.cm.calls["stored"] == [{"tenant_id": "acme", "service": "backend"}]


@pytest.mark.asyncio
async def test_update_profile_deleted_before_its_write_returns_404(env):
    from cogniverse_foundation.config.manager import BackendProfileNotFoundError

    env.cm.profiles["video_colpali"] = _profile(
        "video_colpali", "video_colpali_sv", "colpali-v1.2"
    )

    def deleted_meanwhile(profile_name, overrides, **kwargs):
        raise BackendProfileNotFoundError(
            f"Profile '{profile_name}' not found for tenant 'acme:acme'"
        )

    env.cm.update_backend_profile = deleted_meanwhile

    resp = await _put(
        env.app,
        "/admin/profiles/video_colpali",
        json={"tenant_id": "acme", "description": "new desc"},
    )

    assert resp.status_code == 404
    assert resp.json() == {
        "detail": "Profile 'video_colpali' not found for tenant 'acme'"
    }


@pytest.mark.asyncio
async def test_update_profile_whose_store_cannot_be_read_raises_500(env, caplog):
    from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError

    caplog.set_level(logging.ERROR, logger="cogniverse_runtime.http_errors")
    env.cm.profiles["video_colpali"] = _profile(
        "video_colpali", "video_colpali_sv", "colpali-v1.2"
    )

    def unreadable(tenant_id=None, service="backend"):
        raise ConfigStoreUnavailableError("config store unreachable")

    env.cm.get_stored_backend_config = unreadable

    resp = await _put(
        env.app,
        "/admin/profiles/video_colpali",
        json={"tenant_id": "acme", "description": "new desc"},
    )

    assert resp.status_code == 500
    assert resp.json() == {
        "detail": {
            "error": "profile_update_failed",
            "message": "Updating profile 'video_colpali' failed; the runtime log "
            "names the cause.",
            "failure": "ConfigStoreUnavailableError",
            "profile_name": "video_colpali",
            "tenant_id": "acme",
        }
    }
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "cogniverse_runtime.http_errors"
    ] == [
        "profile_update_failed: ConfigStoreUnavailableError: config store unreachable"
    ]
    assert "update" not in env.cm.calls


@pytest.mark.asyncio
async def test_delete_profile_decides_existence_on_the_store_not_the_held_copy(env):
    env.cm.profiles["video_colpali"] = _profile(
        "video_colpali", "video_colpali_sv", "colpali-v1.2"
    )
    env.cm.get_backend_profile = _held_copy_without_the_profile

    resp = await _delete(env.app, "/admin/profiles/video_colpali", tenant_id="acme")

    assert resp.status_code == 200, resp.text
    assert (resp.json()["profile_name"], resp.json()["schema_deleted"]) == (
        "video_colpali",
        False,
    )
    assert env.cm.calls["delete"] == {
        "profile_name": "video_colpali",
        "tenant_id": "acme",
        "service": "backend",
    }


@pytest.mark.asyncio
async def test_delete_profile_deleted_before_its_write_returns_404(env):
    env.cm.profiles["video_colpali"] = _profile(
        "video_colpali", "video_colpali_sv", "colpali-v1.2"
    )

    def deleted_meanwhile(profile_name, tenant_id=None, service="backend"):
        return False

    env.cm.delete_backend_profile = deleted_meanwhile

    resp = await _delete(env.app, "/admin/profiles/video_colpali", tenant_id="acme")

    assert resp.status_code == 404
    assert resp.json() == {
        "detail": "Profile 'video_colpali' not found for tenant 'acme'"
    }


@pytest.mark.asyncio
async def test_delete_schema_counts_the_stored_profiles_sharing_it(env):
    """Another worker added a profile on the same schema; this process's held
    list does not have it, and dropping the schema would strand it."""
    env.cm.profiles["video_colpali"] = _profile(
        "video_colpali", "video_colpali_sv", "colpali-v1.2"
    )
    env.cm.profiles["video_colpali_copy"] = _profile(
        "video_colpali_copy", "video_colpali_sv", "colpali-v1.2"
    )
    held = {"video_colpali": env.cm.profiles["video_colpali"]}
    env.cm.list_backend_profiles = lambda tenant_id=None, service="backend": held

    resp = await _delete(
        env.app,
        "/admin/profiles/video_colpali",
        tenant_id="acme",
        delete_schema=True,
    )

    assert resp.status_code == 409
    assert resp.json() == {
        "detail": "Cannot delete schema 'video_colpali_sv': other profiles "
        "using it: ['video_colpali_copy']"
    }
    assert env.backend.deleted == []
    assert "delete" not in env.cm.calls


@pytest.mark.asyncio
async def test_delete_profile_whose_store_cannot_be_read_raises_500(env, caplog):
    from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError

    caplog.set_level(logging.ERROR, logger="cogniverse_runtime.http_errors")
    env.cm.profiles["video_colpali"] = _profile(
        "video_colpali", "video_colpali_sv", "colpali-v1.2"
    )

    def unreadable(tenant_id=None, service="backend"):
        raise ConfigStoreUnavailableError("config store unreachable")

    env.cm.get_stored_backend_config = unreadable

    resp = await _delete(env.app, "/admin/profiles/video_colpali", tenant_id="acme")

    assert resp.status_code == 500
    assert resp.json() == {
        "detail": {
            "error": "profile_delete_failed",
            "message": "Deleting profile 'video_colpali' failed; the runtime log "
            "names the cause.",
            "failure": "ConfigStoreUnavailableError",
            "profile_name": "video_colpali",
            "tenant_id": "acme",
        }
    }
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "cogniverse_runtime.http_errors"
    ] == [
        "profile_delete_failed: ConfigStoreUnavailableError: config store unreachable"
    ]
    assert "delete" not in env.cm.calls


@pytest.mark.asyncio
async def test_deploy_decides_the_profile_on_the_store_not_the_held_copy(env):
    env.cm.profiles["video_prism"] = _profile(
        "video_prism", "video_prism_mv", "xclip-lvt"
    )
    env.cm.get_backend_profile = _held_copy_without_the_profile

    resp = await _post(
        env.app,
        "/admin/profiles/video_prism/deploy",
        json={"tenant_id": "acme", "force": False},
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["schema_name"] == "video_prism_mv"
    assert env.backend.deploy_calls == [
        {"tenant_id": "acme", "base_schema_name": "video_prism_mv", "force": False}
    ]


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
    assert env.events.published == [
        ("backend_profiles_changed", {"tenant_id": "acme:acme"})
    ]


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
        "video_prism", "video_prism_mv", "xclip-lvt"
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
async def test_deploy_schema_lookup_failure_raises_500(env):
    env.cm.profiles["video_prism"] = _profile(
        "video_prism", "video_prism_mv", "xclip-lvt"
    )
    env.backend.schema_exists = MagicMock(
        side_effect=RuntimeError("registry lookup failed")
    )

    resp = await _post(
        env.app,
        "/admin/profiles/video_prism/deploy",
        json={"tenant_id": "acme", "force": False},
    )

    assert resp.status_code == 500
    assert resp.json() == {
        "detail": {
            "error": "schema_deploy_failed",
            "message": (
                "Deploying the schema of profile 'video_prism' failed; the "
                "runtime log names the cause."
            ),
            "failure": "RuntimeError",
            "profile_name": "video_prism",
            "tenant_id": "acme",
        }
    }
    env.backend.schema_exists.assert_called_once_with(
        schema_name="video_prism_mv", tenant_id="acme"
    )
    assert env.backend.deploy_calls == []


@pytest.mark.asyncio
async def test_deploy_whose_profile_cannot_be_read_raises_500_and_deploys_nothing(
    env, caplog
):
    from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError

    caplog.set_level(logging.ERROR, logger="cogniverse_runtime.http_errors")

    def unreadable(tenant_id=None, service="backend"):
        raise ConfigStoreUnavailableError("config store unreachable")

    env.cm.get_stored_backend_config = unreadable

    resp = await _post(
        env.app,
        "/admin/profiles/video_prism/deploy",
        json={"tenant_id": "acme", "force": True},
    )

    assert resp.status_code == 500
    assert resp.json() == {
        "detail": {
            "error": "schema_deploy_failed",
            "message": "Deploying the schema of profile 'video_prism' failed; the "
            "runtime log names the cause.",
            "failure": "ConfigStoreUnavailableError",
            "profile_name": "video_prism",
            "tenant_id": "acme",
        }
    }
    assert [
        record.getMessage()
        for record in caplog.records
        if record.name == "cogniverse_runtime.http_errors"
    ] == ["schema_deploy_failed: ConfigStoreUnavailableError: config store unreachable"]
    assert env.backend.deploy_calls == []
    assert env.registry.calls == []


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
            "model_loader": "xclip",
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
        "schema_deploy_error": None,
        "version": _WRITTEN_VERSION,
    }
    assert env.cm.store.get_config_calls == []
    add = env.cm.calls["add"]
    assert add["tenant_id"] == "acme"
    assert add["service"] == "backend"
    # Create never overwrites: the uniqueness check rides the write itself.
    assert add["replace"] is False
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
    assert env.events.published == [
        ("backend_profiles_changed", {"tenant_id": "acme:acme"})
    ]


class _BackendReadRecordingStore(InMemoryConfigStore):
    """In-memory store recording the tenant of every backend-config read."""

    def __init__(self):
        super().__init__()
        self.backend_reads: list[str] = []

    def get_config(self, tenant_id, scope, service, config_key, version=None):
        if config_key == "backend_config":
            self.backend_reads.append(tenant_id)
        return super().get_config(tenant_id, scope, service, config_key, version)


@pytest.fixture
def merged(env, tmp_path, monkeypatch):
    """``env`` with a real ConfigManager over an in-memory store and a shipped
    catalog holding ``shipped_video`` and a ``video_prism`` of its own."""
    catalog = {
        "shipped_video": _profile(
            "shipped_video", "shipped_video_mv", "colpali-v1.2"
        ).to_dict(),
        "video_prism": _profile(
            "video_prism", "catalog_prism_mv", "xclip-lvt"
        ).to_dict(),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"backend": {"profiles": catalog}}))
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(config_path))
    store = _BackendReadRecordingStore()
    cm = ConfigManager(store=store)
    env.app.dependency_overrides[admin.get_config_manager_dependency] = lambda: cm
    return SimpleNamespace(env=env, cm=cm, store=store)


async def _deploy(app, profile_name: str, tenant_id: str):
    return await _post(
        app,
        f"/admin/profiles/{profile_name}/deploy",
        json={"tenant_id": tenant_id, "force": False},
    )


@pytest.mark.asyncio
async def test_deploy_resolves_a_profile_the_tenant_inherits_from_the_catalog(
    merged,
):
    resp = await _deploy(merged.env.app, "shipped_video", "acme")

    assert resp.status_code == 200, resp.text
    assert resp.json()["schema_name"] == "shipped_video_mv"
    assert merged.env.backend.deploy_calls == [
        {"tenant_id": "acme", "base_schema_name": "shipped_video_mv", "force": False}
    ]
    assert set(merged.store.backend_reads) == {"acme:acme", SYSTEM_TENANT_ID}


@pytest.mark.asyncio
async def test_deploy_of_a_profile_in_neither_source_is_404_without_a_deploy(
    merged,
):
    resp = await _deploy(merged.env.app, "no_such_profile", "acme")

    assert resp.status_code == 404
    assert resp.json() == {
        "detail": "Profile 'no_such_profile' not found for tenant 'acme'"
    }
    assert merged.env.backend.deploy_calls == []
    assert merged.env.registry.calls == []
    assert set(merged.store.backend_reads) == {"acme:acme", SYSTEM_TENANT_ID}


@pytest.mark.asyncio
async def test_deploy_prefers_the_tenant_stored_profile_over_the_catalogs(merged):
    merged.cm.add_backend_profile(
        _profile("video_prism", "tenant_prism_mv", "xclip-lvt"), tenant_id="acme"
    )
    merged.store.backend_reads.clear()

    resp = await _deploy(merged.env.app, "video_prism", "acme")

    assert resp.status_code == 200, resp.text
    assert resp.json()["schema_name"] == "tenant_prism_mv"
    assert merged.env.backend.deploy_calls == [
        {"tenant_id": "acme", "base_schema_name": "tenant_prism_mv", "force": False}
    ]
    assert merged.store.backend_reads == ["acme:acme"]


@pytest.mark.asyncio
async def test_deploy_never_resolves_a_profile_another_tenant_stored(merged):
    merged.cm.add_backend_profile(
        _profile("beta_only", "beta_only_mv", "xclip-lvt"), tenant_id="beta"
    )
    merged.store.backend_reads.clear()

    refused = await _deploy(merged.env.app, "beta_only", "acme")

    assert refused.status_code == 404
    assert refused.json() == {
        "detail": "Profile 'beta_only' not found for tenant 'acme'"
    }
    assert merged.env.backend.deploy_calls == []
    assert set(merged.store.backend_reads) == {"acme:acme", SYSTEM_TENANT_ID}

    owned = await _deploy(merged.env.app, "beta_only", "beta")

    assert owned.status_code == 200, owned.text
    assert merged.env.backend.deploy_calls == [
        {"tenant_id": "beta", "base_schema_name": "beta_only_mv", "force": False}
    ]


class _ConflictedConfigManager(ConfigManager):
    """A real ConfigManager whose store loses every compare-and-set."""

    def __init__(self):
        class _AlwaysContended(InMemoryConfigStore):
            def compare_and_set_config(self, *args, **kwargs):
                return None

        super().__init__(store=_AlwaysContended())


@pytest.mark.asyncio
async def test_profile_writes_losing_every_compare_and_set_answer_409(env):
    cm = _ConflictedConfigManager()
    cm.store.set_config(
        "acme:acme",
        ConfigScope.BACKEND,
        "backend",
        "backend_config",
        {
            "tenant_id": "acme:acme",
            "profiles": {"tuned": _profile("tuned", "video_tuned_sv", "m").to_dict()},
        },
    )
    env.app.dependency_overrides[admin.get_config_manager_dependency] = lambda: cm
    conflict = (
        "config acme:acme:backend:backend:backend_config changed under every one "
        "of 10 compare-and-set attempts; nothing was written"
    )

    created = await _post(
        env.app,
        "/admin/profiles",
        json={
            "profile_name": "new_prof",
            "tenant_id": "acme",
            "schema_name": "video_new_sv",
            "embedding_model": "colpali-v1.2",
            "embedding_type": "single_vector",
            "model_loader": "xclip",
            "deploy_schema": False,
        },
    )
    updated = await _put(
        env.app,
        "/admin/profiles/tuned",
        json={"tenant_id": "acme", "description": "changed"},
    )
    deleted = await _delete(env.app, "/admin/profiles/tuned", tenant_id="acme")

    assert [r.status_code for r in (created, updated, deleted)] == [409, 409, 409]
    assert [r.json() for r in (created, updated, deleted)] == [{"detail": conflict}] * 3
    history = cm.store.get_config_history(
        "acme:acme", ConfigScope.BACKEND, "backend", "backend_config"
    )
    assert [entry.version for entry in history] == [1]


@pytest.mark.asyncio
async def test_concurrent_creates_of_one_profile_name_store_exactly_one(env):
    """Both creates pass the validation read; the compare-and-set write lets
    exactly one store the profile and answers the other 400."""
    cm = ConfigManager(store=InMemoryConfigStore())
    env.app.dependency_overrides[admin.get_config_manager_dependency] = lambda: cm

    def create(model: str):
        return _post(
            env.app,
            "/admin/profiles",
            json={
                "profile_name": "dup_prof",
                "tenant_id": "acme",
                "schema_name": "video_dup_sv",
                "embedding_model": model,
                "embedding_type": "single_vector",
                "model_loader": "xclip",
                "deploy_schema": False,
            },
        )

    first, second = await asyncio.gather(create("model-a"), create("model-b"))

    statuses = [first.status_code, second.status_code]
    assert sorted(statuses) == [201, 400]
    loser = (first, second)[statuses.index(400)]
    assert loser.json() == {
        "detail": {
            "message": "Profile validation failed",
            "errors": ["Profile 'dup_prof' already exists for tenant 'acme:acme'"],
        }
    }
    winner_model = ("model-a", "model-b")[statuses.index(201)]
    stored = cm.store.get_config(
        "acme:acme", ConfigScope.BACKEND, "backend", "backend_config"
    )
    assert stored.version == 1
    assert stored.config_value["profiles"]["dup_prof"]["embedding_model"] == (
        winner_model
    )


@pytest.mark.asyncio
async def test_deploy_for_a_deleted_tenant_is_410_and_deploys_nothing(env):
    env.cm.profiles["video_prism"] = _profile(
        "video_prism", "video_prism_mv", "xclip-lvt"
    )
    env.backend.deployed_schemas = set()
    deleted = SimpleNamespace(config_value={"deleted": True})
    env.cm.store.get_immutable_config = lambda *coordinates: (
        deleted if coordinates[3] == "acme:acme" else None
    )

    resp = await _post(
        env.app,
        "/admin/profiles/video_prism/deploy",
        json={"tenant_id": "acme", "force": True},
    )

    assert resp.status_code == 410
    assert resp.json() == {
        "detail": {
            "error": "tenant_deleted",
            "message": "Tenant 'acme:acme' has been deleted; its schemas and "
            "memories are not written until the tenant is created again.",
            "failure": "TenantDeletedError",
            "tenant_id": "acme:acme",
            "profile_name": "video_prism",
        }
    }
    assert env.backend.deploy_calls == []
    assert env.registry.calls == []
