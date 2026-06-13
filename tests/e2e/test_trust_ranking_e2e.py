"""Trust ranking + endorsement end-to-end.

Pins:
  * the initial-trust-from-derivation-kind multiplier table
    (``_DERIVATION_WEIGHTS``);
  * the endorsement HTTP route's exact arithmetic
    (``new_score = old + _ENDORSEMENT_DELTA[role]``) plus the persisted
    state surviving a Mem0 read-back;
  * the composite ranker's ordering (relevance × trust × confidence) on
    synthetic memories;
  * the route's input validation (unknown role => HTTP 400 with the exact
    substring the runtime emits).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

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
from cogniverse_core.memory.trust import (
    _DERIVATION_WEIGHTS,
    _ENDORSEMENT_DELTA,
    TrustRecord,
    attach_trust_to_metadata,
    extract_trust,
    rank_with_trust,
)
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.e2e.conftest import RUNTIME, skip_if_no_runtime, unique_id

VESPA_HTTP_PORT = 8080
VESPA_CONFIG_PORT = 19071
DENSEON_URL = "http://localhost:29006"


def _build_manager(tenant_id: str) -> Mem0MemoryManager:
    Mem0MemoryManager._instances.clear()
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


def _write_with_derivation(
    mm: Mem0MemoryManager,
    *,
    derivation_kind: DerivationKind,
    content: str,
) -> str:
    """Write an entity_fact memory under a fresh subject_key per derivation kind.

    Schema enforcement on the write path auto-attaches an initial trust
    record via ``compute_initial_trust``; the schema default_trust for
    ``entity_fact`` is 0.5.
    """
    prov = make_provenance(
        written_by="agent:trust",
        derivation_kind=derivation_kind,
        confidence=0.9,
        derived_from=[CitationRef.external("trust://src", label="src")],
    )
    metadata = attach_to_metadata(
        {"kind": "entity_fact", "subject_key": f"trust.{derivation_kind.value}"},
        prov,
    )
    mid = mm.add_memory(
        content=content,
        tenant_id=mm.tenant_id,
        agent_name="trust_rank_agent",
        metadata=metadata,
        infer=False,
    )
    assert mid is not None, f"add_memory returned None for {derivation_kind!r}"
    return mid


def _fetch_memory(mm: Mem0MemoryManager, memory_id: str) -> Dict[str, Any] | None:
    raw = mm.memory.get(memory_id)
    if raw is None:
        return None
    return raw if isinstance(raw, dict) else dict(raw)


# ---------------------------------------------------------------------------
# 1. Initial trust = default_trust × derivation_weight (clamped)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestInitialTrustFromDerivationKind:
    """compute_initial_trust pins (default_trust × _DERIVATION_WEIGHTS[kind])."""

    def test_each_derivation_kind_yields_expected_initial_trust(self) -> None:
        tenant_id = unique_id("trust_init")
        mm = _build_manager(tenant_id)
        # entity_fact schema's default_trust is 0.5.
        DEFAULT_TRUST = 0.5
        try:
            ids_by_kind: Dict[DerivationKind, str] = {}
            for dk in DerivationKind:
                mid = _write_with_derivation(
                    mm,
                    derivation_kind=dk,
                    content=f"fact for {dk.value}",
                )
                ids_by_kind[dk] = mid

            for dk, mid in ids_by_kind.items():
                mem = _fetch_memory(mm, mid)
                assert mem is not None, f"memory {mid} disappeared after write"
                trust = extract_trust(mem)
                assert trust is not None, (
                    f"no trust attached to memory written under {dk.value!r}"
                )
                expected = min(
                    1.0, max(0.0, DEFAULT_TRUST * _DERIVATION_WEIGHTS[dk.value])
                )
                assert trust.score == pytest.approx(expected, rel=1e-3), (
                    f"trust score drift for derivation={dk.value!r}: "
                    f"got {trust.score!r}, expected {expected!r}"
                )
                assert trust.endorsements == 0
        finally:
            mm.clear_agent_memory(tenant_id, "trust_rank_agent")
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 2. Endorsement HTTP route bumps trust by exactly the delta and persists
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestEndorsementBumpsTrust:
    """POST /endorse increments trust by the role's delta and persists it."""

    def test_user_then_org_admin_endorsement_arithmetic(self) -> None:
        tenant_id = unique_id("trust_end")
        mm = _build_manager(tenant_id)
        try:
            # EXTRACTION has weight 1.00 → initial trust = 0.5 exactly,
            # so the arithmetic below stays clean (no float drift).
            mid = _write_with_derivation(
                mm,
                derivation_kind=DerivationKind.EXTRACTION,
                content="fact whose trust we will endorse",
            )

            with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
                r1 = client.post(
                    f"/admin/tenants/{tenant_id}/memories/{mid}/endorse",
                    json={"endorser_role": "user", "actor_id": "alice"},
                )
            assert r1.status_code == 200, r1.text[:300]
            body1 = r1.json()
            assert body1["memory_id"] == mid
            assert body1["new_score"] == pytest.approx(
                0.5 + _ENDORSEMENT_DELTA["user"], rel=1e-3
            )
            assert body1["endorsements"] == 1

            # Read-back through Mem0 — the trust record must have been
            # persisted to metadata, not just returned by the endpoint.
            mem = _fetch_memory(mm, mid)
            assert mem is not None
            trust = extract_trust(mem)
            assert trust is not None
            assert trust.score == pytest.approx(0.55, rel=1e-3)
            assert trust.endorsements == 1

            with httpx.Client(base_url=RUNTIME, timeout=60.0) as client:
                r2 = client.post(
                    f"/admin/tenants/{tenant_id}/memories/{mid}/endorse",
                    json={"endorser_role": "org_admin", "actor_id": "boss"},
                )
            assert r2.status_code == 200, r2.text[:300]
            body2 = r2.json()
            # 0.55 + 0.20 = 0.75
            assert body2["new_score"] == pytest.approx(
                0.55 + _ENDORSEMENT_DELTA["org_admin"], rel=1e-3
            )
            assert body2["endorsements"] == 2
        finally:
            mm.clear_agent_memory(tenant_id, "trust_rank_agent")
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 3. Composite ranker orders by relevance × trust × confidence
# ---------------------------------------------------------------------------


def _synthetic(
    score: float, trust: float, confidence: float, mid: str
) -> Dict[str, Any]:
    """Build a synthetic Mem0-style memory dict for the ranker."""
    metadata: Dict[str, Any] = {"kind": "entity_fact", "subject_key": f"sub_{mid}"}
    metadata = attach_trust_to_metadata(
        metadata,
        TrustRecord(
            score=trust,
            initial_score=trust,
            decayed_at="2026-05-13T00:00:00+00:00",
            endorsements=0,
        ),
    )
    prov = make_provenance(
        written_by="agent:trust_syn",
        derivation_kind=DerivationKind.DIRECT_INGEST,
        confidence=confidence,
        derived_from=[CitationRef.external("trust://syn")],
    )
    metadata = attach_to_metadata(metadata, prov)
    return {"id": mid, "memory": f"content {mid}", "score": score, "metadata": metadata}


@pytest.mark.e2e
@skip_if_no_runtime
class TestRankWithTrustOrdersByComposite:
    """rank_with_trust orders by exactly (relevance × trust × confidence)."""

    def test_composite_ordering(self) -> None:
        # M2 wins composite at 0.57; M3 second at 0.56; M1 last at 0.40.
        m1 = _synthetic(score=1.0, trust=0.4, confidence=1.0, mid="M1")
        m2 = _synthetic(score=0.6, trust=0.95, confidence=1.0, mid="M2")
        m3 = _synthetic(score=0.8, trust=0.7, confidence=1.0, mid="M3")

        ranked = rank_with_trust([m1, m2, m3], apply_decay_now=False)
        assert [m["id"] for m in ranked] == ["M2", "M3", "M1"]


# ---------------------------------------------------------------------------
# 4. Endorse route rejects unknown roles with the exact 400 substring
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestEndorseUnknownRoleRejected:
    """Unknown endorser_role => HTTP 400 with substring "unknown endorser_role='random'"."""

    def test_random_role_rejected_with_substring(self) -> None:
        tenant_id = unique_id("trust_bad")
        mm = _build_manager(tenant_id)
        try:
            mid = _write_with_derivation(
                mm,
                derivation_kind=DerivationKind.EXTRACTION,
                content="fact for bad-role endorse",
            )
            with httpx.Client(base_url=RUNTIME, timeout=30.0) as client:
                resp = client.post(
                    f"/admin/tenants/{tenant_id}/memories/{mid}/endorse",
                    json={"endorser_role": "random", "actor_id": "alice"},
                )
            assert resp.status_code == 400, resp.text[:300]
            detail = resp.json().get("detail", "")
            assert "unknown endorser_role='random'" in detail, detail
        finally:
            mm.clear_agent_memory(tenant_id, "trust_rank_agent")
            Mem0MemoryManager._instances.clear()
