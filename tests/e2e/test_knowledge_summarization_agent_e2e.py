"""Phase 8c — KnowledgeSummarizationAgent end-to-end.

Pins:

  * a 5-fact subject slice produces a structured summary whose
    ``source_count == 5`` and whose ``citation_refs`` cover those exact
    memory ids; ``promoted_to_org_trunk`` is False / ``promoted_memory_id``
    is None when ``promote=False``;
  * with ``promote=True`` and ``actor_role=tenant_admin``, the summary
    is written to the org trunk and the new memory id surfaces.

The agent uses the in-cluster vLLM for the actual summarisation pass —
we exercise the route end-to-end with the LLM so the contract pinned
here is the one production hits.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import httpx
import pytest

from cogniverse_core.memory.federation import org_trunk_tenant_id
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    attach_to_metadata,
    make_provenance,
)
from cogniverse_core.memory.schema import build_default_registry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.e2e.conftest import RUNTIME, skip_if_no_runtime, unique_id

VESPA_HTTP_PORT = 8080
VESPA_CONFIG_PORT = 19071
DENSEON_URL = "http://localhost:29006"


def _build_manager(tenant_id: str) -> Mem0MemoryManager:
    cm = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost", backend_port=VESPA_HTTP_PORT
        )
    )
    # In-memory only: cm.set_system_config would persist a denseon-only
    # localhost URL map into config_metadata and starve the in-cluster
    # ingestor (which reads inference_service_urls from the same store).
    cm._system_config_cache = SystemConfig(  # noqa: SLF001
        backend_url="http://localhost",
        backend_port=VESPA_HTTP_PORT,
        inference_service_urls={"denseon": DENSEON_URL},
    )
    mm = Mem0MemoryManager(tenant_id=tenant_id)
    mm.initialize(
        backend_host="http://localhost",
        backend_port=VESPA_HTTP_PORT,
        backend_config_port=VESPA_CONFIG_PORT,
        base_schema_name="agent_memories",
        llm_model="google/gemma-4-e4b-it",
        embedding_model="lightonai/DenseOn",
        llm_base_url="http://cogniverse-vllm-llm-student.cogniverse:8000/v1",
        embedder_base_url=DENSEON_URL,
        auto_create_schema=True,
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        knowledge_registry=build_default_registry(),
    )
    return mm


def _seed_facts(mm: Mem0MemoryManager, *, subject: str, count: int) -> List[str]:
    """Write ``count`` entity_facts under the same subject_key."""
    ids: List[str] = []
    for i in range(count):
        prov = make_provenance(
            written_by="agent:phase8_summary",
            derivation_kind=DerivationKind.DIRECT_INGEST,
            confidence=0.9,
            derived_from=[CitationRef.external(f"phase8://summary/{i}")],
        )
        meta = attach_to_metadata({"kind": "entity_fact", "subject_key": subject}, prov)
        mid = mm.add_memory(
            content=(
                f"fact #{i} about {subject}: detail-{i}-with-some-distinguishing-text"
            ),
            tenant_id=mm.tenant_id,
            agent_name="phase8_agent",
            metadata=meta,
            infer=False,
        )
        assert mid is not None
        ids.append(mid)
    return ids


# ---------------------------------------------------------------------------
# 1. Summarise a subject slice without promoting
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestSummarizationOverSubjectSlice:
    """5 facts under one subject_key → summary spans them, promote=False."""

    def test_summarises_five_facts_no_promotion(self) -> None:
        Mem0MemoryManager._instances.clear()
        tenant_id = unique_id("kagent_sum") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            ids = _seed_facts(mm, subject="customer.acme", count=5)

            with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/knowledge/summarize",
                    json={
                        "subject_keys": ["customer.acme"],
                        "title": "Acme summary",
                        "actor_role": "user",
                        "actor_id": "alice",
                        "promote": False,
                        # Summarizer defaults agent_name_filter to "_promoted";
                        # the test wrote under "phase8_agent" via in-process
                        # Mem0, so point the filter at that agent_name.
                        "agent_name_filter": "phase8_agent",
                    },
                )
            assert resp.status_code == 200, resp.text[:300]
            body = resp.json()

            assert body["title"] == "Acme summary"
            assert body["source_count"] == 5
            assert body["promoted_to_org_trunk"] is False
            assert body["promoted_memory_id"] is None
            # citation_refs cover exactly the 5 written memory ids.
            cited_ids = sorted(c["ref_id"] for c in body["citation_refs"])
            assert cited_ids == sorted(ids), (
                f"citation_refs drift: got {cited_ids}, expected {sorted(ids)}"
            )
            for ref in body["citation_refs"]:
                assert ref["ref_kind"] == "memory"
            # The actual summary text is LLM-generated; pin its bounds
            # tightly enough to catch a regression that returns "" or a
            # 50-KB blob, but accept the model's choice within those bounds.
            assert isinstance(body["summary"], str)
            assert 80 <= len(body["summary"]) <= 6000, len(body["summary"])
        finally:
            mm.clear_agent_memory(tenant_id, "phase8_agent")
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 2. promote=True writes the summary into the org trunk
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestSummarizationPromotesToOrgTrunk:
    """promote=True + tenant_admin → promoted_memory_id set, lands in trunk schema."""

    def test_promote_lands_summary_in_org_trunk(self) -> None:
        Mem0MemoryManager._instances.clear()
        tenant_id = unique_id("kagent_sumt") + ":t1"
        mm = _build_manager(tenant_id)
        # Pre-deploy the trunk tenant's schemas so the federation write
        # path in the agent has a target schema to land into.
        trunk_id = org_trunk_tenant_id(tenant_id)
        mm_trunk = _build_manager(trunk_id)
        try:
            ids = _seed_facts(mm, subject="customer.beta", count=4)

            with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/knowledge/summarize",
                    json={
                        "subject_keys": ["customer.beta"],
                        "title": "Beta summary",
                        "actor_role": "tenant_admin",
                        "actor_id": "boss",
                        "promote": True,
                        "agent_name_filter": "phase8_agent",
                    },
                )
            assert resp.status_code == 200, resp.text[:300]
            body = resp.json()

            assert body["title"] == "Beta summary"
            assert body["source_count"] == 4
            assert sorted(c["ref_id"] for c in body["citation_refs"]) == sorted(ids)
            assert body["promoted_to_org_trunk"] is True
            promoted_id = body["promoted_memory_id"]
            assert isinstance(promoted_id, str) and promoted_id, promoted_id
        finally:
            try:
                mm.clear_agent_memory(tenant_id, "phase8_agent")
                mm_trunk.clear_agent_memory(trunk_id, "_promoted")
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()
