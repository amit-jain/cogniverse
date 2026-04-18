"""Runtime HTTP round-trip: POST /admin/profiles then query via search.

Proves the end-to-end wiring from the admin API down to the in-memory
VespaSearchBackend. The retry-storm bug was invisible at this layer
because the admin router only validated + persisted — no test ever
called search against the just-created profile in the same process.

Uses FastAPI TestClient (not k3d) so the whole chain runs in-process
against the module's shared Vespa Docker fixture.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_runtime.routers import admin, search


@pytest.fixture
def clean_backend_registry():
    """Ensure each test starts with a clean backend cache."""
    BackendRegistry._backend_instances.clear()
    yield
    BackendRegistry._backend_instances.clear()


@pytest.fixture
def runtime_app(
    vespa_instance,
    config_manager,
    schema_loader,
    real_telemetry,
    clean_backend_registry,
):
    """FastAPI app wired with admin + search routers and profile_change_listener.

    Mirrors what `libs/runtime/cogniverse_runtime/main.py` wires at
    startup: the listener propagates profile changes from ConfigManager
    into cached backends via BackendRegistry.add_profile_to_backends.
    """
    def listener(event, name, cfg):
        if event == "added" and cfg is not None:
            BackendRegistry.add_profile_to_backends(name, cfg)
        elif event == "removed":
            BackendRegistry.remove_profile_from_backends(name)

    config_manager.set_profile_change_listener(listener)

    admin.set_config_manager(config_manager)
    admin.set_schema_loader(schema_loader)

    app = FastAPI()
    app.include_router(admin.router, prefix="/admin")
    app.include_router(search.router, prefix="/search")

    app.dependency_overrides[search.get_config_manager_dependency] = (
        lambda: config_manager
    )
    app.dependency_overrides[search.get_schema_loader_dependency] = (
        lambda: schema_loader
    )

    with TestClient(app) as client:
        yield client

    config_manager.set_profile_change_listener(None)


@pytest.mark.integration
def test_dynamic_profile_is_visible_to_search_after_admin_post(runtime_app):
    """POST /admin/profiles + immediate search must not raise profile-not-found.

    Regression guard for the agent_memories retry-storm bug:
    - Before the fix: the shared VespaSearchBackend still had its boot
      snapshot, so search returned ValueError('Requested profile X not found').
    - After the fix: the profile_change_listener propagates the new profile
      to the cached backend's self.profiles, so profile resolution passes.
    """
    profile_name = "dyn_vis_probe"

    # Create a profile via the admin HTTP endpoint.
    create_resp = runtime_app.post(
        "/admin/profiles",
        json={
            "profile_name": profile_name,
            "tenant_id": "dyn_vis_tenant",
            "type": "memory",
            "schema_name": profile_name,
            "embedding_model": "nomic-embed-text",
            "pipeline_config": {},
            "strategies": {},
            "embedding_type": "dense",
            "schema_config": {"embedding_dims": 768},
            "deploy_schema": False,
        },
    )
    assert create_resp.status_code in (200, 201), (
        f"admin /profiles failed: {create_resp.status_code} {create_resp.text}"
    )

    # Search against the new profile. Profile resolution must pass even if
    # the deeper search layers return an empty/error response — what we're
    # guarding against is the "profile not found" ValueError that came
    # from the frozen in-memory dict.
    search_resp = runtime_app.post(
        "/search/",
        json={
            "query": "hello world",
            "tenant_id": "dyn_vis_tenant",
            "profile": profile_name,
            "top_k": 1,
        },
    )
    body = search_resp.text.lower()
    assert "requested profile" not in body or "not found" not in body, (
        f"Search still sees 'profile not found' — listener didn't fire:\n"
        f"status={search_resp.status_code}\nbody={search_resp.text[:500]}"
    )
