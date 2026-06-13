"""MultiDocumentSynthesisAgent end-to-end.

Pins:

  * Inline-document synthesis: 5 documents → answer non-empty,
    citation_refs cover exactly the input labels (or memory ids),
    document_count metadata pinned, persisted_memory_id present;
  * From-memory-id synthesis: 3 pre-written external_doc memories →
    citation_refs reference exactly those memory ids in input order.

The agent invokes the in-cluster vLLM via dspy.Predict (no RLM unless
explicitly enabled), so the synthesis pass is a real LLM round-trip.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import httpx
import pytest

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


def _warmup_provenance_schema(mm: Mem0MemoryManager, timeout_s: float = 120.0) -> None:
    """Block until the per-tenant provenance Vespa schema accepts writes.

    Vespa's app-package re-deploy is asynchronous: the deploy returns
    immediately but content nodes pick up the new schema later. Probe
    with a real attach + read until both succeed.
    """
    import time
    import uuid

    probe_id = f"warmup-{uuid.uuid4().hex[:8]}"
    probe_prov = make_provenance(
        written_by="warmup",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.5,
        derived_from=[CitationRef.external("warmup://probe", label="probe")],
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            mm.provenance_store.attach(probe_id, probe_prov)
            time.sleep(1.0)
            if mm.provenance_store.get(probe_id) is not None:
                return
        except Exception:
            pass
        time.sleep(1.0)
    pytest.fail(
        f"provenance schema for tenant {mm.tenant_id!r} never came online "
        f"within {timeout_s}s of Mem0MemoryManager.initialize"
    )


VESPA_HTTP_PORT = 8080
VESPA_CONFIG_PORT = 19071
DENSEON_URL = "http://localhost:29006"
# The runtime route injects "multi_document_synthesis_agent" as the
# memory_agent_name; pre-seeded source memories must live under that
# agent_name so mm.memory.get(memory_id) finds them.
SYN_AGENT = "multi_document_synthesis_agent"


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
    _warmup_provenance_schema(mm)
    return mm


def _write_doc(mm: Mem0MemoryManager, *, label: str, content: str) -> str:
    """Write an external_doc with provenance under SYN_AGENT."""
    prov = make_provenance(
        written_by="agent:multidoc",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.9,
        derived_from=[CitationRef.external(f"multidoc://{label}")],
    )
    metadata = attach_to_metadata({"kind": "external_doc", "subject_key": label}, prov)
    mid = mm.add_memory(
        content=content,
        tenant_id=mm.tenant_id,
        agent_name=SYN_AGENT,
        metadata=metadata,
        infer=False,
    )
    assert mid is not None
    return mid


# ---------------------------------------------------------------------------
# 1. Inline-document synthesis
# ---------------------------------------------------------------------------


_INLINE_DOCS = [
    {
        "label": "doc1",
        "content": "Lithium-ion batteries dominate consumer electronics.",
    },
    {
        "label": "doc2",
        "content": "Lithium reserves are concentrated in South America.",
    },
    {
        "label": "doc3",
        "content": "Lithium is essential for electric vehicle adoption.",
    },
    {
        "label": "doc4",
        "content": "Recycling lithium remains an industrial challenge.",
    },
    {
        "label": "doc5",
        "content": "Demand for lithium tripled between 2020 and 2024.",
    },
]


@pytest.mark.e2e
@skip_if_no_runtime
class TestSynthesisOverInlineDocuments:
    """5 inline docs → answer + citations + persisted memory."""

    def test_inline_docs_yield_persisted_synthesis(self) -> None:
        Mem0MemoryManager._instances.clear()
        tenant_id = unique_id("kagent_syn") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/knowledge/synthesis/multi_doc",
                    json={
                        "query": "Why is lithium central to modern energy?",
                        "documents": _INLINE_DOCS,
                        "actor_role": "user",
                        "actor_id": "alice",
                    },
                )
            assert resp.status_code == 200, resp.text[:500]
            body = resp.json()

            # Answer is real LLM output; bound it tightly enough to
            # catch a regression that returns "" or a 50KB blob.
            assert isinstance(body["answer"], str)
            assert 30 <= len(body["answer"]) <= 6000, len(body["answer"])

            # citation_refs has one entry per input document.
            assert len(body["citation_refs"]) == 5
            cited_ids = sorted(c["ref_id"] for c in body["citation_refs"])
            assert cited_ids == sorted(d["label"] for d in _INLINE_DOCS), (
                f"citation_refs drift: got {cited_ids}"
            )
            # CitationRef.external() builds ref_kind="url" (see provenance.py:87)
            for ref in body["citation_refs"]:
                assert ref["ref_kind"] == "url", ref

            # persist=True is the default → a memory gets written.
            assert isinstance(body["persisted_memory_id"], str)
            assert body["persisted_memory_id"]
            assert body["used_rlm"] is False  # default RLMOptions=None
            assert body["metadata"]["document_count"] == 5
            assert body["metadata"]["derivation_kind"] == "synthesis"
            # Answer must mention the synthesis subject — catches the
            # regression where the LM returns garbage or echoes the prompt.
            assert "lithium" in body["answer"].lower(), body["answer"][:300]

            # Round-trip: re-fetch the persisted memory and verify the
            # provenance contract the agent owes (kind, derivation_kind,
            # derived_from len matches input docs).
            persisted = mm.memory.get(body["persisted_memory_id"])
            assert persisted is not None, "persisted memory missing"
            persisted_meta = persisted.get("metadata") or {}
            if isinstance(persisted_meta, str):
                import json as _json

                persisted_meta = _json.loads(persisted_meta)
            assert persisted_meta.get("kind") == "synthesis_fact", persisted_meta
            prov = persisted_meta.get("provenance") or {}
            assert prov.get("derivation_kind") == "synthesis", prov
            assert len(prov.get("derived_from") or []) == 5, prov
        finally:
            try:
                mm.clear_agent_memory(tenant_id, SYN_AGENT)
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 1b. Large-document path triggers RLM auto-detect
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestSynthesisLargeDocumentTriggersRLM:
    """50 large docs (~5KB each) with rlm.auto_detect=True → used_rlm flips True.

    Pins the agent's auto-detect contract: the dspy.Predict fast path is
    skipped when the joined-document context exceeds ``context_threshold``
    and the synthesis runs via RLMInference instead. ``context_chars`` in
    the response metadata reports the joined-prompt size and must be in a
    tight window for 50 docs of ~5_000 chars each.
    """

    @pytest.fixture(scope="class")
    def deno_path_setup(self) -> None:
        # RLMInference asserts Deno on construction; the runtime pod has
        # Deno installed at /home/cogniverse/.deno/bin so the route works,
        # but we don't need any local setup here — the RLM runs in-pod.
        return None

    def test_50_docs_5kb_each_flips_used_rlm(self, deno_path_setup: None) -> None:
        Mem0MemoryManager._instances.clear()
        tenant_id = unique_id("kagent_synl") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            big_docs = [
                {
                    "label": f"big_{i:02d}",
                    # ~5_000 chars each → 50 docs ≈ 250_000 chars before
                    # the per-document =====/label framing overhead.
                    "content": (f"Document {i} body. " * 250).strip(),
                }
                for i in range(50)
            ]
            with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/knowledge/synthesis/multi_doc",
                    json={
                        "query": "Summarise the documents.",
                        "documents": big_docs,
                        "actor_role": "user",
                        "actor_id": "alice",
                        "rlm": {
                            "auto_detect": True,
                            # Threshold below 200_000 so 50×5_000 trips it.
                            "context_threshold": 100_000,
                            "max_iterations": 1,
                            "max_llm_calls": 2,
                            "timeout_seconds": 600,
                        },
                    },
                )
            assert resp.status_code == 200, resp.text[:500]
            body = resp.json()
            # The whole point: auto_detect flipped used_rlm to True.
            assert body["used_rlm"] is True, body.get("metadata")
            assert body["metadata"]["document_count"] == 50
            # 50 docs × 5_000 chars + per-doc framing ≈ 250_000-350_000.
            ctx_chars = body["metadata"]["context_chars"]
            assert 200_000 <= ctx_chars <= 400_000, ctx_chars
            assert len(body["citation_refs"]) == 50
        finally:
            try:
                mm.clear_agent_memory(tenant_id, SYN_AGENT)
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 2. From pre-existing memory ids
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestSynthesisFromMemoryIds:
    """3 pre-written external_doc memories → citation_refs reference their ids."""

    def test_memory_id_documents_yield_matching_citations(self) -> None:
        Mem0MemoryManager._instances.clear()
        tenant_id = unique_id("kagent_synm") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            ids: List[str] = [
                _write_doc(mm, label="src1", content="Site A produces 30% of supply."),
                _write_doc(mm, label="src2", content="Site B produces 25% of supply."),
                _write_doc(mm, label="src3", content="Site C produces 20% of supply."),
            ]
            documents = [{"memory_id": mid} for mid in ids]

            with httpx.Client(base_url=RUNTIME, timeout=900.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/knowledge/synthesis/multi_doc",
                    json={
                        "query": "Summarise total supply across the three sites.",
                        "documents": documents,
                        "actor_role": "user",
                        "actor_id": "alice",
                    },
                )
            assert resp.status_code == 200, resp.text[:500]
            body = resp.json()
            assert len(body["citation_refs"]) == 3
            cited_ids = [c["ref_id"] for c in body["citation_refs"]]
            assert cited_ids == ids, (
                f"citation_refs preserve input order: expected {ids}, got {cited_ids}"
            )
            for ref in body["citation_refs"]:
                assert ref["ref_kind"] == "memory"
            assert body["metadata"]["document_count"] == 3
        finally:
            try:
                mm.clear_agent_memory(tenant_id, SYN_AGENT)
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()
