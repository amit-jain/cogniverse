"""Startup profiles reach tenant memory search with isolated overrides."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI

pytestmark = pytest.mark.integration

TENANT = "test:unit"
MEMORY_SCHEMA = "agent_memories_test_unit"
EMBEDDING_DIMS = 768


def _memory_query() -> dict:
    """The query dict BackendVectorStore.search builds for a memory read."""
    return {
        "query": "what did we learn",
        "type": "memory",
        "profile": "agent_memories",
        "schema_name": MEMORY_SCHEMA,
        "strategy": "semantic_search",
        "top_k": 5,
        "filters": {},
        "query_embeddings": np.array([0.01] * EMBEDDING_DIMS),
        "tenant_id": TENANT,
        "nearest_neighbor_approximate": False,
    }


def _schemas_queried(bodies: list) -> set[str]:
    schemas = set()
    for body in bodies:
        yql = (body or {}).get("yql", "")
        match = re.search(r"\bfrom\s+(?:sources\s+)?([A-Za-z0-9_]+)", yql)
        if match and match.group(1).startswith("agent_memories"):
            schemas.add(match.group(1))
    return schemas


def _shipped_backend_section() -> dict:
    root = Path(__file__).resolve().parents[3]
    return json.loads((root / "configs/config.json").read_text())["backend"]


async def _memory_search_schemas(monkeypatch, http_port: int) -> set[str]:
    """Run the real lifespan, then search this tenant's cached backend.

    The backend is created the way a search or ingestion request creates it,
    from the shipped config — which is why the profile set it freezes has no
    memory profile, and why the memory search must resolve one through the
    config manager.
    """
    monkeypatch.setenv("COGNIVERSE_SANDBOX_POLICY", "disabled")
    monkeypatch.setenv("COGNIVERSE_MEMORY_LIFECYCLE_DISABLED", "1")
    import dspy

    monkeypatch.setattr(dspy, "configure", lambda *a, **kw: None)

    from vespa.application import VespaSync

    bodies: list = []
    original_query = VespaSync.query

    def recording_query(self, *args, **kwargs):
        bodies.append(kwargs.get("body") or (args[0] if args else None) or kwargs)
        return original_query(self, *args, **kwargs)

    monkeypatch.setattr(VespaSync, "query", recording_query)

    from cogniverse_core.registries.backend_registry import get_backend_registry
    from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
    from cogniverse_foundation.config.utils import create_default_config_manager
    from cogniverse_runtime.main import lifespan

    shipped = _shipped_backend_section()
    async with lifespan(FastAPI()):
        backend = get_backend_registry().get_ingestion_backend(
            "vespa",
            tenant_id=TENANT,
            config={
                "url": "http://localhost",
                "port": http_port,
                "schema_name": "video_colpali_smol500_mv_frame",
                "backend": shipped,
                "profiles": shipped["profiles"],
                "default_profiles": shipped.get("default_profiles", {}),
            },
            config_manager=create_default_config_manager(),
            schema_loader=FilesystemSchemaLoader("configs/schemas"),
        )
        assert "agent_memories" not in backend.config["profiles"]
        backend.search(_memory_query())

    return _schemas_queried(bodies)


@pytest.mark.asyncio
async def test_memory_search_resolves_the_profile_a_store_already_holds(
    monkeypatch, vespa_instance
):
    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_core.memory.manager import MEMORY_BASE_SCHEMA, affirm_memory_profile
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_vespa.config.config_store import VespaConfigStore

    config_manager = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost", backend_port=vespa_instance["http_port"]
        )
    )
    # The cluster's state: the profile is already stored, so startup's
    # affirmation writes nothing new and fans nothing out.
    affirm_memory_profile(config_manager)
    assert (
        MEMORY_BASE_SCHEMA
        in config_manager.get_backend_config(SYSTEM_TENANT_ID).profiles
    )

    assert await _memory_search_schemas(monkeypatch, vespa_instance["http_port"]) == {
        MEMORY_SCHEMA
    }


@pytest.mark.asyncio
async def test_memory_search_resolves_the_profile_on_a_store_without_it(
    monkeypatch, vespa_instance
):
    from cogniverse_core.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_core.memory.manager import MEMORY_BASE_SCHEMA
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_vespa.config.config_store import VespaConfigStore

    config_manager = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost", backend_port=vespa_instance["http_port"]
        )
    )
    config_manager.delete_backend_profile(
        MEMORY_BASE_SCHEMA, tenant_id=SYSTEM_TENANT_ID
    )
    assert (
        MEMORY_BASE_SCHEMA
        not in config_manager.get_backend_config(SYSTEM_TENANT_ID).profiles
    )

    assert await _memory_search_schemas(monkeypatch, vespa_instance["http_port"]) == {
        MEMORY_SCHEMA
    }


@pytest.mark.parametrize("shipped", [False, True], ids=["stored-only", "shipped"])
def test_concurrent_tenants_merge_system_profiles_without_sharing_overrides(
    vespa_instance, monkeypatch, tmp_path, shipped
):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from cogniverse_foundation.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import (
        BackendConfig,
        BackendProfileConfig,
    )
    from cogniverse_foundation.config.utils import ConfigUtils
    from cogniverse_vespa.config.config_store import VespaConfigStore

    profile_name = "stored_memory_catalog"
    profile = BackendProfileConfig.from_dict(
        profile_name,
        {
            "type": "memory",
            "schema_name": "agent_memories",
            "embedding_model": "system-embedding",
            "description": "stored catalog",
            "pipeline_config": {"batch_size": 7},
        },
    )
    expected_profile = profile.to_dict()
    if shipped:
        expected_profile["embedding_model"] = "shipped-embedding"
        expected_profile["pipeline_config"] = {"batch_size": 11}
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {"backend": {"profiles": {profile_name: expected_profile}}}
            if shipped
            else {}
        )
    )
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(config_path))
    store = VespaConfigStore(
        backend_url="http://localhost", backend_port=vespa_instance["http_port"]
    )
    manager = ConfigManager(store=store)
    manager.add_backend_profile(profile, tenant_id=SYSTEM_TENANT_ID)
    tenants = ["catalog:alpha", "catalog:beta"]
    for tenant in tenants:
        manager.set_backend_config(
            BackendConfig(
                tenant_id=tenant,
                profiles={
                    profile_name: BackendProfileConfig.from_dict(
                        profile_name, {"type": "memory", "description": tenant}
                    )
                },
            )
        )

    reader = ConfigManager(store=store)
    ready = threading.Barrier(4)
    loaded = threading.Barrier(4)

    def read(tenant):
        ready.wait(timeout=30)
        config = ConfigUtils(tenant, reader)
        resolved = config.get("backend")["profiles"][profile_name]
        loaded.wait(timeout=30)
        snapshot = dict(resolved)
        resolved["description"] = "caller mutation"
        return snapshot

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(read, tenants * 2))
        assert actual == [
            {**expected_profile, "description": tenant} for tenant in tenants * 2
        ]
        assert (
            manager.get_backend_profile(profile_name, SYSTEM_TENANT_ID).to_dict()
            == profile.to_dict()
        )
        assert [
            reader.get_backend_profile(profile_name, tenant).description
            for tenant in tenants
        ] == tenants
    finally:
        for tenant in [SYSTEM_TENANT_ID, *tenants]:
            manager.delete_backend_profile(profile_name, tenant_id=tenant)
        store.close()


def test_unavailable_system_profile_store_raises_and_can_retry(
    vespa_instance, monkeypatch, tmp_path
):
    import socket

    from cogniverse_foundation.common.tenant_utils import SYSTEM_TENANT_ID
    from cogniverse_foundation.config.manager import ConfigManager
    from cogniverse_foundation.config.unified_config import BackendProfileConfig
    from cogniverse_foundation.config.utils import ConfigUtils
    from cogniverse_sdk.interfaces.config_store import ConfigStoreUnavailableError
    from cogniverse_vespa.config.config_store import VespaConfigStore

    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    monkeypatch.setenv("COGNIVERSE_CONFIG", str(config_path))
    store = VespaConfigStore(
        backend_url="http://localhost", backend_port=vespa_instance["http_port"]
    )
    profile = BackendProfileConfig.from_dict(
        "fault_memory_catalog",
        {"type": "memory", "schema_name": "agent_memories"},
    )
    writer = ConfigManager(store=store)
    writer.add_backend_profile(profile, tenant_id=SYSTEM_TENANT_ID)
    reader = ConfigManager(store=store, scoped_config_cache_ttl_s=300)
    assert reader.get_backend_config("catalog:fault").profiles == {}
    config = ConfigUtils("catalog:fault", reader)
    config._ensure_system_config()
    config._ensure_routing_config()
    config._ensure_telemetry_config()
    try:
        with socket.socket() as unavailable:
            unavailable.bind(("127.0.0.1", 0))
            port = unavailable.getsockname()[1]
            broken_store = VespaConfigStore(
                backend_url="http://127.0.0.1", backend_port=port
            )
            reader.store = broken_store
            try:
                with pytest.raises(ConfigStoreUnavailableError) as error:
                    config.get("backend")
                assert f"port={port}" in str(error.value)
                assert "Failed to read Vespa config visit" in str(error.value)
            finally:
                broken_store.close()
                reader.store = store
        assert (
            config.get("backend")["profiles"][profile.profile_name] == profile.to_dict()
        )
    finally:
        writer.delete_backend_profile(profile.profile_name, tenant_id=SYSTEM_TENANT_ID)
        store.close()
