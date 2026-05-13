"""Phase 3a — Contradiction detection + reconciliation end-to-end.

Pins the shipped ContradictionDetector + reconcile() + the
``/admin/tenants/{t}/knowledge/contradictions/reconcile`` HTTP route
against the deployed cogniverse up cluster.

Detection is purely structural (subject_key + content signature) so the
unit-shape pieces run in-process; the round-trip via the runtime route
exercises the full agent path including registry policy lookup and
per-tenant Mem0 pinning.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest

from cogniverse_core.memory.contradiction import ContradictionDetector, reconcile
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    attach_to_metadata,
    make_provenance,
)
from cogniverse_core.memory.schema import (
    ContradictionPolicy,
    build_default_registry,
)
from cogniverse_core.memory.trust import TrustRecord, attach_trust_to_metadata
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.e2e.conftest import RUNTIME, skip_if_no_runtime, unique_id

VESPA_HTTP_PORT = 8080
VESPA_CONFIG_PORT = 19071
DENSEON_URL = "http://localhost:29006"


def _build_manager(tenant_id: str) -> Mem0MemoryManager:
    """In-process Mem0MemoryManager pointing at the deployed cluster."""
    Mem0MemoryManager._instances.clear()
    cm = ConfigManager(
        store=VespaConfigStore(
            backend_url="http://localhost", backend_port=VESPA_HTTP_PORT
        )
    )
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=VESPA_HTTP_PORT,
            inference_service_urls={"denseon": DENSEON_URL},
        )
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


def _mem(
    *,
    mid: str,
    subject: str,
    content: str,
    created_at: str | None = None,
    trust_score: float | None = None,
    confidence: float | None = None,
) -> Dict[str, Any]:
    """Build a synthetic memory dict (no Vespa write) for in-process tests."""
    metadata: Dict[str, Any] = {"subject_key": subject, "kind": "entity_fact"}
    if created_at is not None:
        metadata["created_at"] = created_at
    if trust_score is not None:
        metadata = attach_trust_to_metadata(
            metadata,
            TrustRecord(
                score=trust_score,
                initial_score=trust_score,
                decayed_at="2026-05-13T00:00:00+00:00",
                endorsements=0,
            ),
        )
    if confidence is not None:
        prov = make_provenance(
            written_by="agent:phase3",
            derivation_kind=DerivationKind.DIRECT_INGEST,
            confidence=confidence,
            derived_from=[CitationRef.external("phase3://src")],
        )
        metadata = attach_to_metadata(metadata, prov)
    return {"id": mid, "memory": content, "metadata": metadata}


# ---------------------------------------------------------------------------
# 1. Detector groups by subject_key + flags conflicts on distinct content sig
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestDetectorOpensConflictSet:
    """Detector returns exactly one ConflictSet covering the conflicting pair."""

    def test_three_facts_two_conflicting_yield_one_conflict_set(self) -> None:
        # A, B share subject_key but disagree on content; C is unrelated.
        a = _mem(mid="A1", subject="paris.population", content="2.1 million")
        b = _mem(mid="B2", subject="paris.population", content="2.2 million")
        c = _mem(mid="C3", subject="berlin.population", content="3.6 million")

        conflicts = ContradictionDetector().detect([a, b, c])

        assert len(conflicts) == 1
        cs = conflicts[0]
        assert cs.subject_key == "paris.population"
        assert sorted(cs.conflicting_memory_ids) == ["A1", "B2"]
        # detected_at is a non-empty ISO-8601 — pin the year to keep a
        # tight bound; the run is during 2026 by today's date.
        assert cs.detected_at.startswith("2026-")


# ---------------------------------------------------------------------------
# 2-4. reconcile() under each ContradictionPolicy
# ---------------------------------------------------------------------------


def _two_conflicting(
    *,
    t1_iso: str,
    t2_iso: str,
    trust_a: float,
    trust_b: float,
    conf_a: float,
    conf_b: float,
) -> List[Dict[str, Any]]:
    """Return the canonical two-member conflict set used by every policy test."""
    a = _mem(
        mid="MA",
        subject="company.ceo",
        content="ceo: Alice",
        created_at=t1_iso,
        trust_score=trust_a,
        confidence=conf_a,
    )
    b = _mem(
        mid="MB",
        subject="company.ceo",
        content="ceo: Bob",
        created_at=t2_iso,
        trust_score=trust_b,
        confidence=conf_b,
    )
    return [a, b]


@pytest.mark.e2e
@skip_if_no_runtime
class TestReconcileLatestWins:
    """LATEST_WINS keeps the highest created_at member, drops the older one."""

    def test_keeps_only_latest(self) -> None:
        now = datetime.now(timezone.utc)
        older = (now - timedelta(days=2)).isoformat()
        newer = (now - timedelta(hours=1)).isoformat()
        members = _two_conflicting(
            t1_iso=older,
            t2_iso=newer,
            trust_a=0.9,
            trust_b=0.4,
            conf_a=1.0,
            conf_b=1.0,
        )
        result = reconcile(members, ContradictionPolicy.LATEST_WINS)
        assert len(result) == 1
        # MB has the newer created_at (`newer`), so it wins despite lower trust.
        assert result[0]["id"] == "MB"


@pytest.mark.e2e
@skip_if_no_runtime
class TestReconcileTrustRanked:
    """TRUST_RANKED keeps the member with the highest trust × confidence."""

    def test_high_trust_beats_low_trust(self) -> None:
        members = _two_conflicting(
            t1_iso="2026-01-01T00:00:00+00:00",
            t2_iso="2026-01-01T00:00:00+00:00",
            trust_a=0.9,
            trust_b=0.4,
            conf_a=1.0,
            conf_b=0.5,
        )
        result = reconcile(members, ContradictionPolicy.TRUST_RANKED)
        # MA composite = 0.9 * 1.0 = 0.9 vs MB = 0.4 * 0.5 = 0.20.
        assert len(result) == 1
        assert result[0]["id"] == "MA"


@pytest.mark.e2e
@skip_if_no_runtime
class TestReconcilePreserveBoth:
    """PRESERVE_BOTH returns every member, each tagged with disputed=True."""

    def test_both_returned_and_marked_disputed(self) -> None:
        members = _two_conflicting(
            t1_iso="2026-01-01T00:00:00+00:00",
            t2_iso="2026-01-02T00:00:00+00:00",
            trust_a=0.9,
            trust_b=0.4,
            conf_a=1.0,
            conf_b=1.0,
        )
        result = reconcile(members, ContradictionPolicy.PRESERVE_BOTH)
        assert len(result) == 2
        ids = sorted(m["id"] for m in result)
        assert ids == ["MA", "MB"]
        for m in result:
            # `disputed` is set on the returned member's metadata; the
            # exact value `is True` is the contract surface UIs check.
            assert m["metadata"]["disputed"] is True, m["metadata"]


# ---------------------------------------------------------------------------
# 5. End-to-end via the /reconcile HTTP route
# ---------------------------------------------------------------------------


def _write_with_trust(
    mm: Mem0MemoryManager,
    *,
    subject: str,
    content: str,
    derivation_kind: DerivationKind,
) -> str:
    """Real Vespa write: schema enforcement attaches trust automatically.

    The default registry's ``entity_fact`` schema sets default_trust=0.5;
    multiplied by the derivation weight, that fixes the initial trust
    deterministically per derivation kind.
    """
    prov = make_provenance(
        written_by="agent:phase3",
        derivation_kind=derivation_kind,
        confidence=0.9,
        derived_from=[CitationRef.external("phase3://src", label="src")],
    )
    metadata = attach_to_metadata({"subject_key": subject, "kind": "entity_fact"}, prov)
    mid = mm.add_memory(
        content=content,
        tenant_id=mm.tenant_id,
        agent_name="phase3_agent",
        metadata=metadata,
        infer=False,
    )
    assert mid is not None
    return mid


@pytest.mark.e2e
@skip_if_no_runtime
class TestReconcileViaHTTPRoute:
    """The runtime's /contradictions/reconcile route returns the canonical shape."""

    def test_high_trust_member_wins_via_route(self) -> None:
        tenant_id = unique_id("confl_http")
        mm = _build_manager(tenant_id)
        try:
            # DIRECT_INGEST → trust = 0.5 * 1.20 = 0.60
            high_id = _write_with_trust(
                mm,
                subject="company.ceo",
                content="ceo: Alice",
                derivation_kind=DerivationKind.DIRECT_INGEST,
            )
            # AGENT_INFERENCE → trust = 0.5 * 0.70 = 0.35
            low_id = _write_with_trust(
                mm,
                subject="company.ceo",
                content="ceo: Bob",
                derivation_kind=DerivationKind.AGENT_INFERENCE,
            )

            with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/knowledge/contradictions/reconcile",
                    json={
                        "target_kind": "entity_fact",
                        "conflict_member_ids": [high_id, low_id],
                        "policy_override": "trust_ranked",
                    },
                )
            assert resp.status_code == 200, (
                f"reconcile rc={resp.status_code}: {resp.text[:300]}"
            )
            body = resp.json()

            assert body["target_kind"] == "entity_fact"
            assert body["policy_used"] == "trust_ranked"
            # Survivor is the high-trust id; the other member is recorded
            # in `resolved` with survived=False.
            assert body["survivors"] == [high_id]
            resolved_by_id = {r["memory_id"]: r for r in body["resolved"]}
            assert sorted(resolved_by_id) == sorted([high_id, low_id])
            assert resolved_by_id[high_id]["survived"] is True
            assert resolved_by_id[low_id]["survived"] is False
            # Nobody is disputed under trust_ranked — disputed only fires
            # under PRESERVE_BOTH.
            assert resolved_by_id[high_id]["disputed"] is False
            assert resolved_by_id[low_id]["disputed"] is False
            assert body["metadata"]["input_count"] == 2
            assert body["metadata"]["survivor_count"] == 1
            assert body["metadata"]["policy_overridden"] is True
        finally:
            mm.clear_agent_memory(tenant_id, "phase3_agent")
            Mem0MemoryManager._instances.clear()
