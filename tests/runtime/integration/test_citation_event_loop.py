"""Citation HTTP requests traverse real Mem0 and Vespa without blocking peers."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from mem0 import Memory

from cogniverse_core.memory.backend_vector_store import BackendVectorStore
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    make_provenance,
)
from cogniverse_core.memory.provenance_store import ProvenanceStore
from cogniverse_runtime.routers import knowledge, tenant
from cogniverse_sdk.document import Document
from tests.runtime.integration.test_schema_event_loop import (  # noqa: F401
    assert_responsive,
    schema_env,
)

pytestmark = [pytest.mark.integration, pytest.mark.no_shared_vespa]


@pytest.fixture
def citation_env(schema_env, monkeypatch):  # noqa: F811
    """Wire the real read-only Mem0 adapter; citation reads need no encoder/LM."""
    env = schema_env
    for base in ("agent_memories", "provenance"):
        env.backend.schema_registry.deploy_schema(
            tenant_id=env.tenant, base_schema_name=base
        )
    import mem0.memory.telemetry as memory_telemetry

    monkeypatch.setattr(memory_telemetry, "MEM0_TELEMETRY", False)
    mm = Mem0MemoryManager(tenant_id=env.tenant)
    mm.memory = object.__new__(Memory)
    mm.memory.vector_store = BackendVectorStore(
        collection_name=f"agent_memories_{env.tenant.replace(':', '_')}",
        backend_resolver=lambda: env.backend,
        tenant_id=env.tenant,
        profile="agent_memories",
    )
    mm._provenance_store = ProvenanceStore(lambda: env.backend, env.tenant)
    for memory_id, content, kind, refs in (
        (
            "leaf",
            "The launch is on 21 September.",
            DerivationKind.DIRECT_INGEST,
            [CitationRef.external("https://source.test/launch")],
        ),
        (
            "root",
            "Launch date: 21 September.",
            DerivationKind.SYNTHESIS,
            [CitationRef.memory("leaf")],
        ),
    ):
        result = env.backend.ingest_documents(
            [
                Document(
                    id=memory_id,
                    text_content=content,
                    metadata={
                        "user_id": env.tenant,
                        "agent_id": "citation_source",
                        "memory": content,
                    },
                )
            ],
            schema_name="agent_memories",
        )
        assert result["success_count"] == 1
        mm.provenance_store.attach(
            memory_id,
            make_provenance(
                written_by="agent:citation_source",
                derivation_kind=kind,
                confidence=1.0,
                derived_from=refs,
            ),
        )
    deadline = time.monotonic() + 30
    while set(mm.provenance_store.fetch(["root", "leaf"])) != {"root", "leaf"}:
        if time.monotonic() > deadline:
            pytest.fail(
                "The exact root and leaf provenance rows did not become readable"
            )
        time.sleep(0.1)
    assert mm.memory.get("root")["memory"] == "Launch date: 21 September."
    assert mm.memory.get("leaf")["memory"] == "The launch is on 21 September."
    monkeypatch.setattr(knowledge, "_build_factory", lambda tenant_id: mm)
    monkeypatch.setattr(tenant, "_config_manager", env.cm)
    env.app.include_router(knowledge.router, prefix="/admin")
    yield SimpleNamespace(**vars(env), mm=mm)
    Mem0MemoryManager._instances.pop(env.tenant, None)


def expected_graph():
    return {
        "root_memory_id": "root",
        "nodes": [
            {
                "memory_id": "root",
                "depth": 0,
                "content_excerpt": "Launch date: 21 September.",
                "written_by": "agent:citation_source",
                "derivation_kind": "synthesis",
                "confidence": 1.0,
            },
            {
                "memory_id": "leaf",
                "depth": 1,
                "content_excerpt": "The launch is on 21 September.",
                "written_by": "agent:citation_source",
                "derivation_kind": "direct_ingest",
                "confidence": 1.0,
            },
        ],
        "primary_sources": [
            {"ref_kind": "url", "ref_id": "https://source.test/launch", "label": None},
            {"ref_kind": "memory", "ref_id": "leaf", "label": None},
        ],
        "kg_primary_sources": [],
        "truncated": False,
        "metadata": {
            "nodes_visited": 2,
            "primary_source_count": 2,
            "kg_primary_source_count": 0,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_citation_walk_keeps_loop_responsive_during_mid_graph_read(
    citation_env, failure
):
    env = citation_env
    # The root read has succeeded; hold/fail the next BFS level at real HTTP.
    env.proxy.arm(
        lambda method, path, body: (
            "/search/" in path
            and b"provenance" in body
            and b"leaf" in body
            and b"root" not in body
        ),
        failure=failure,
    )
    async with AsyncClient(
        transport=ASGITransport(app=env.app), base_url="http://test"
    ) as client:
        task = asyncio.create_task(
            client.post(
                f"/admin/tenants/{env.tenant}/knowledge/citations/trace",
                json={"memory_id": "root"},
            )
        )
        try:
            await assert_responsive(client, task, env.proxy)
        finally:
            if failure:
                with pytest.raises(
                    RuntimeError,
                    match="provenance fetch failed for schema 'provenance'.*1 memory ids",
                ):
                    await task
            else:
                response = await task
        if not failure:
            assert response.status_code == 200
            assert response.json() == expected_graph()
        recovered = await client.post(
            f"/admin/tenants/{env.tenant}/knowledge/citations/trace",
            json={"memory_id": "root"},
        )
        assert recovered.status_code == 200
        assert recovered.json() == expected_graph()
