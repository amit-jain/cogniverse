"""BackendVectorStore.insert / list against a real Vespa.

Two correctness pins:

* insert() must not report a dropped feed as a stored memory. The backend
  returns per-document feed failures without raising, so a wrong-dim (or
  otherwise rejected) vector left Mem0 believing the write landed.
* list() must emit a tz-aware UTC created_at (matching the sibling get()/
  search() read paths), not a host-local naive timestamp.

Vectors are supplied directly, so no embedding model / sidecar is needed —
only a real Vespa with the agent_memories schema deployed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cogniverse_core.memory._timestamps import epoch_to_iso_utc
from cogniverse_core.memory.backend_vector_store import BackendVectorStore
from cogniverse_core.registries.backend_registry import BackendRegistry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from tests.utils.vespa_test_helpers import deploy_tenant_schema, make_config_manager

pytestmark = [pytest.mark.integration, pytest.mark.requires_docker]

TENANT = "bvs_rt"
DIM = 768


@pytest.fixture(scope="module")
def memory_store(shared_vespa):
    cm = make_config_manager(shared_vespa)
    full = deploy_tenant_schema(
        shared_vespa,
        tenant_id=TENANT,
        base_schema_name="agent_memories",
        config_manager=cm,
    )
    backend = BackendRegistry.get_instance().get_ingestion_backend(
        name="vespa",
        config={
            "backend": {
                "url": "http://localhost",
                "config_port": shared_vespa["config_port"],
                "port": shared_vespa["http_port"],
            }
        },
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        tenant_id=TENANT,
    )
    return BackendVectorStore(
        collection_name=full,
        backend_client=backend,
        embedding_model_dims=DIM,
        tenant_id=TENANT,
        profile="agent_memories",
    )


def test_insert_raises_when_feed_drops_documents(memory_store):
    store = memory_store

    # Too many cells for the dense tensor<float>(d0[768]) field: Vespa rejects
    # the feed (success_count == 0) but the client swallows it into failed_docs
    # rather than raising — insert() must surface that as an error, not report
    # the dropped write as a stored memory. (A short vector would be zero-padded
    # and accepted, so the overflow direction is the reliable rejection.)
    with pytest.raises(RuntimeError, match="persisted only 0/1"):
        store.insert(
            vectors=[[0.1] * (DIM + 256)],
            payloads=[{"data": "x", "user_id": "u_fail", "agent_id": "a"}],
            ids=["mem-fail-1"],
        )
    assert store.get("mem-fail-1") is None, "a dropped feed was persisted anyway"

    # Positive control: a correctly-shaped vector stores and is retrievable.
    assert store.insert(
        vectors=[[0.1] * DIM],
        payloads=[{"data": "ok", "user_id": "u_ok", "agent_id": "a"}],
        ids=["mem-ok-1"],
    ) == ["mem-ok-1"]
    stored = store.get("mem-ok-1")
    assert stored is not None and stored.id == "mem-ok-1"


def test_list_emits_tz_aware_utc_created_at(memory_store):
    store = memory_store

    # created_at as epoch seconds; list() must render it tz-aware UTC.
    store.insert(
        vectors=[[0.2] * DIM],
        payloads=[
            {
                "data": "ts",
                "user_id": "u_ts",
                "agent_id": "a",
                "created_at": 1_700_000_000,
            }
        ],
        ids=["mem-ts-1"],
    )
    results, _next = store.list(filters={"user_id": "u_ts"})
    rec = next(r for r in results if r.id == "mem-ts-1")

    assert rec.payload["created_at"] == epoch_to_iso_utc(1_700_000_000)
    assert rec.payload["created_at"] == "2023-11-14T22:13:20+00:00"
    parsed = datetime.fromisoformat(rec.payload["created_at"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timedelta(0)
    assert parsed == datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
