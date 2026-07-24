"""Real-Vespa proof that POST /ingestion/start combines a separately-supplied
``org_id`` with a simple ``tenant_id`` into the canonical ``org:tenant`` form
before it reaches the backend registry and the background pipeline.

Drives the real ingestion route through a real ConfigManager/SchemaLoader and
the real BackendRegistry against the shared Vespa container. The tenant
existence check reads the live ``tenant_metadata`` schema, so the route only
returns 200 when it derived the tenant that was actually registered. Only the
encoder-heavy ``VideoIngestionPipeline`` is stubbed (full video processing is
disproportionate for a tenant-routing assertion); everything that decides the
tenant — the route body, the registry lookup, and the Vespa metadata read —
is real.
"""

from __future__ import annotations

import time as _time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_runtime.admin import tenant_manager as tm
from cogniverse_runtime.routers import ingestion as ingestion_router

pytestmark = pytest.mark.integration

ORG_ID = "bigcorp"
SIMPLE_TENANT = "acme"
COMBINED_TENANT = "bigcorp:acme"


@pytest.fixture
def start_client(memory_manager, vespa_instance, config_manager):
    """Ingestion router wired to the real Vespa-backed ConfigManager, with the
    tenant_manager module seams pointed at the same real backend so
    ``assert_tenant_exists`` reads the live ``tenant_metadata`` schema."""
    tm.set_config_manager(config_manager)
    tm.set_schema_loader(FilesystemSchemaLoader(Path("configs/schemas")))

    # Register only the COMBINED tenant on real Vespa. The simple tenant
    # ``acme:acme`` (what the pre-fix route derives) is deliberately NOT
    # registered, so the route 404s there and only 200s when it combined.
    backend = tm.get_backend()
    backend.create_metadata_document(
        schema="tenant_metadata",
        doc_id=COMBINED_TENANT,
        fields={
            "tenant_full_id": COMBINED_TENANT,
            "org_id": ORG_ID,
            "tenant_name": SIMPLE_TENANT,
            "created_at": int(_time.time() * 1000),
            "created_by": "integration-test",
            "status": "active",
            "schemas_deployed": ["video_colpali_smol500_mv_frame"],
        },
    )

    from cogniverse_foundation.config.unified_config import BackendProfileConfig

    config_manager.add_backend_profile(
        BackendProfileConfig(
            profile_name="video_colpali_smol500_mv_frame",
            type="video",
            schema_name="video_colpali_smol500_mv_frame",
            embedding_model="TomoroAI/tomoro-colqwen3-embed-4b",
        ),
        tenant_id=COMBINED_TENANT,
    )

    # Drop any cached positive existence for these ids so the route re-reads
    # the live schema during this test.
    from cogniverse_core.common import tenant_utils

    tenant_utils._TENANT_EXISTS_CACHE.pop(COMBINED_TENANT, None)
    tenant_utils._TENANT_EXISTS_CACHE.pop("acme:acme", None)

    app = FastAPI()
    app.include_router(ingestion_router.router, prefix="/ingestion")
    app.dependency_overrides[ingestion_router.get_config_manager_dependency] = lambda: (
        config_manager
    )
    app.dependency_overrides[ingestion_router.get_schema_loader_dependency] = lambda: (
        FilesystemSchemaLoader(Path("configs/schemas"))
    )
    with TestClient(app) as client:
        yield client

    tenant_utils._TENANT_EXISTS_CACHE.pop(COMBINED_TENANT, None)


def test_start_routes_combined_tenant_to_real_backend(
    start_client, monkeypatch, tmp_path
):
    """The route derives ``bigcorp:acme`` (registered on real Vespa) — not
    ``acme:acme`` (unregistered) — so the live tenant check passes (200), the
    real registry resolves the ingestion backend under the combined tenant, and
    the background pipeline is built under the combined tenant.

    Pre-fix the route drops ``org_id`` and derives ``acme:acme``: the live
    ``tenant_metadata`` read misses and the route returns 404.
    """
    recorded: dict = {}

    # Capture the tenant the real BackendRegistry resolves the ingestion
    # backend for (start_ingestion) without stubbing the registry.
    from cogniverse_core.registries.backend_registry import BackendRegistry

    real_registry = BackendRegistry.get_instance()
    real_get_ingestion_backend = real_registry.get_ingestion_backend

    def _spy_get_ingestion_backend(*args, **kwargs):
        recorded["backend_tenant_id"] = kwargs.get("tenant_id")
        return real_get_ingestion_backend(*args, **kwargs)

    monkeypatch.setattr(
        real_registry, "get_ingestion_backend", _spy_get_ingestion_backend
    )

    # Stub only the encoder-heavy pipeline; capture the tenant run_ingestion
    # builds it under (the second combine site).
    class _RecordingPipeline:
        def __init__(self, **kwargs):
            recorded["pipeline_tenant_id"] = kwargs.get("tenant_id")

        async def process_videos_concurrent(self, video_files, max_concurrent):
            return {"status": "completed", "successful": 0, "failed": 0, "results": []}

    monkeypatch.setattr(
        "cogniverse_runtime.ingestion.pipeline.VideoIngestionPipeline",
        _RecordingPipeline,
    )
    monkeypatch.setattr(
        "cogniverse_runtime.ingestion.strategies.discover_ingestible_files",
        lambda video_dir, content_type: [],
    )

    resp = start_client.post(
        "/ingestion/start",
        json={
            "video_dir": str(tmp_path),
            "profile": "video_colpali_smol500_mv_frame",
            "tenant_id": SIMPLE_TENANT,
            "org_id": ORG_ID,
            "content_type": "video",
        },
    )

    assert resp.status_code == 200, (
        f"combined tenant must pass the live tenant check; got "
        f"{resp.status_code}: {resp.text}"
    )
    # The real registry resolved the backend for the combined tenant.
    assert recorded["backend_tenant_id"] == COMBINED_TENANT
    # run_ingestion built the pipeline under the same combined tenant.
    assert recorded["pipeline_tenant_id"] == COMBINED_TENANT

    # The gate hinged on the live metadata: the combined tenant exists on real
    # Vespa, the pre-fix simple tenant does not.
    import asyncio

    assert asyncio.run(tm.get_tenant_internal(COMBINED_TENANT)) is not None
    assert asyncio.run(tm.get_tenant_internal("acme:acme")) is None
