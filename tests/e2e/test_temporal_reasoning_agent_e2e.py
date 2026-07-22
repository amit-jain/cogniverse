"""TemporalReasoningAgent end-to-end.

Pins the agent's window-bucketing contract:

  * 4 entity_facts under one subject_key with provenance.written_at
    spread across 2024 Q1-Q4 → POST /knowledge/temporal/reason with H1
    + H2 windows returns one WindowViewOut per window with the exact
    matching ids; distinct_signatures_count tracks content evolution.
  * An empty window (no memories in the time range) returns an empty
    matching_memory_ids list — no error, just zero hits in that bucket.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    Provenance,
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
VESPA_CONFIG_PORT = 33071
DENSEON_URL = "http://localhost:33906"
# TemporalReasoningAgent reads via FederationService.federated_get_all
# whose default agent_name is "_promoted". Write under that name.
PROMOTED_AGENT = "_promoted"


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


def _write_at(
    mm: Mem0MemoryManager,
    *,
    subject: str,
    content: str,
    written_at_iso: str,
) -> str:
    """Write an entity_fact whose provenance.written_at is the given ISO timestamp.

    The Temporal agent buckets memories by window using
    ``provenance.written_at``; bypass ``make_provenance``'s now-stamp so
    we can place memories in deterministic time windows.
    """
    prov = Provenance(
        written_by="agent:temporal",
        written_at=written_at_iso,
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=0.9,
        derived_from=[CitationRef.external("temporal://src")],
    )
    metadata = attach_to_metadata({"kind": "entity_fact", "subject_key": subject}, prov)
    mid = mm.add_memory(
        content=content,
        tenant_id=mm.tenant_id,
        agent_name=PROMOTED_AGENT,
        metadata=metadata,
        infer=False,
    )
    assert mid is not None
    return mid


# ---------------------------------------------------------------------------
# 1. Q1-Q4 facts bucket cleanly into H1 + H2 windows
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestTemporalWindowsReturnPerWindowViews:
    """4 written_at values across 2024 → H1 holds 2, H2 holds 2."""

    def test_h1_h2_buckets_exact_ids(self) -> None:
        Mem0MemoryManager._instances.clear()
        tenant_id = unique_id("kagent_tmp") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            # Q1: 2024-02-15, Q2: 2024-05-15, Q3: 2024-08-15, Q4: 2024-11-15.
            ids = [
                _write_at(
                    mm,
                    subject="kpi.arr",
                    content="Q1 ARR = $10M",
                    written_at_iso="2024-02-15T00:00:00+00:00",
                ),
                _write_at(
                    mm,
                    subject="kpi.arr",
                    content="Q2 ARR = $15M",
                    written_at_iso="2024-05-15T00:00:00+00:00",
                ),
                _write_at(
                    mm,
                    subject="kpi.arr",
                    content="Q3 ARR = $22M",
                    written_at_iso="2024-08-15T00:00:00+00:00",
                ),
                _write_at(
                    mm,
                    subject="kpi.arr",
                    content="Q4 ARR = $28M",
                    written_at_iso="2024-11-15T00:00:00+00:00",
                ),
            ]
            q1_id, q2_id, q3_id, q4_id = ids

            with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/knowledge/temporal/reason",
                    json={
                        "subject_key": "kpi.arr",
                        "windows": [
                            {
                                "label": "H1",
                                "start": "2024-01-01T00:00:00+00:00",
                                "end": "2024-07-01T00:00:00+00:00",
                            },
                            {
                                "label": "H2",
                                "start": "2024-07-01T00:00:00+00:00",
                                "end": "2025-01-01T00:00:00+00:00",
                            },
                        ],
                    },
                )
            assert resp.status_code == 200, resp.text[:500]
            body = resp.json()
            assert body["subject_key"] == "kpi.arr"
            views = {v["label"]: v for v in body["window_views"]}
            assert sorted(views) == ["H1", "H2"]

            # H1 [Jan 1 – Jul 1) → Q1 + Q2.
            assert sorted(views["H1"]["matching_memory_ids"]) == sorted([q1_id, q2_id])
            # H2 [Jul 1 – Jan 1 next year) → Q3 + Q4.
            assert sorted(views["H2"]["matching_memory_ids"]) == sorted([q3_id, q4_id])
            # 4 distinct content strings → 2 distinct signatures (one per
            # window's content set, since H1 and H2 hold different facts).
            assert body["distinct_signatures_count"] == 2
            assert body["undated_count"] == 0
        finally:
            try:
                mm.clear_agent_memory(tenant_id, PROMOTED_AGENT)
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 2. Empty window returns zero matches without error
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestEmptyWindowReportsZeroFacts:
    """A 2025-Q1 window over a subject with only 2024 data → empty bucket."""

    def test_empty_h1_2025_returns_no_matches(self) -> None:
        Mem0MemoryManager._instances.clear()
        tenant_id = unique_id("kagent_tmpe") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            mid = _write_at(
                mm,
                subject="kpi.arr",
                content="Q1 2024 ARR = $10M",
                written_at_iso="2024-02-15T00:00:00+00:00",
            )
            with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/knowledge/temporal/reason",
                    json={
                        "subject_key": "kpi.arr",
                        "windows": [
                            {
                                "label": "Y2024",
                                "start": "2024-01-01T00:00:00+00:00",
                                "end": "2025-01-01T00:00:00+00:00",
                            },
                            {
                                "label": "Y2025",
                                "start": "2025-01-01T00:00:00+00:00",
                                "end": "2026-01-01T00:00:00+00:00",
                            },
                        ],
                    },
                )
            assert resp.status_code == 200, resp.text[:500]
            body = resp.json()
            views = {v["label"]: v for v in body["window_views"]}
            # Y2024 holds the single memory; Y2025 is empty.
            assert views["Y2024"]["matching_memory_ids"] == [mid]
            assert views["Y2025"]["matching_memory_ids"] == []
            # Two windows but only one populated → 2 signatures still
            # (empty has its own signature distinct from non-empty).
            assert body["distinct_signatures_count"] == 2
        finally:
            try:
                mm.clear_agent_memory(tenant_id, PROMOTED_AGENT)
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()
