"""Phase 9c — ContradictionReconciliationAgent end-to-end.

Pins the runtime ``/knowledge/contradictions/reconcile`` route under
each contradiction policy:

  * default policy resolves via the schema's contradiction_policy
    (entity_fact ships TRUST_RANKED) → high-trust survivor;
  * ``policy_override="latest_wins"`` → latest created_at survives;
  * ``policy_override="preserve_both"`` → both members survive flagged
    ``disputed=True``.

Phase 3's TestReconcileViaHTTPRoute already covered the trust-ranked
path under override; this module adds the schema-default + the other
two override branches with the same ResolvedMemberOut shape pinned.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Tuple

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

VESPA_HTTP_PORT = 8080
VESPA_CONFIG_PORT = 19071
DENSEON_URL = "http://localhost:29006"
# The reconciliation route injects "contradiction_reconciliation_agent"
# as the memory_agent_name; the agent reads via mm.get_all_memories
# scoped to that name. Write the conflicting facts under the same name.
RECON_AGENT = "contradiction_reconciliation_agent"


def _build_manager(tenant_id: str) -> Mem0MemoryManager:
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


def _write_conflicting(
    mm: Mem0MemoryManager,
    *,
    subject: str,
    high_trust_content: str,
    low_trust_content: str,
    high_trust_age_days: float = 0.0,
    low_trust_age_days: float = 0.0,
) -> Tuple[str, str]:
    """Write two same-subject entity_facts; one with DIRECT_INGEST (trust 0.6),
    the other with AGENT_INFERENCE (trust 0.35). Returns (high, low) ids."""
    now = datetime.now(timezone.utc)
    high_meta = attach_to_metadata(
        {"kind": "entity_fact", "subject_key": subject},
        make_provenance(
            written_by="agent:phase9_recon",
            derivation_kind=DerivationKind.DIRECT_INGEST,
            confidence=0.95,
            derived_from=[CitationRef.external("phase9://high")],
        ),
    )
    if high_trust_age_days > 0:
        high_meta["created_at"] = (
            now - timedelta(days=high_trust_age_days)
        ).isoformat()
    high_id = mm.add_memory(
        content=high_trust_content,
        tenant_id=mm.tenant_id,
        agent_name=RECON_AGENT,
        metadata=high_meta,
        infer=False,
    )
    assert high_id is not None

    low_meta = attach_to_metadata(
        {"kind": "entity_fact", "subject_key": subject},
        make_provenance(
            written_by="agent:phase9_recon",
            derivation_kind=DerivationKind.AGENT_INFERENCE,
            confidence=0.5,
            derived_from=[CitationRef.external("phase9://low")],
        ),
    )
    if low_trust_age_days > 0:
        low_meta["created_at"] = (now - timedelta(days=low_trust_age_days)).isoformat()
    low_id = mm.add_memory(
        content=low_trust_content,
        tenant_id=mm.tenant_id,
        agent_name=RECON_AGENT,
        metadata=low_meta,
        infer=False,
    )
    assert low_id is not None
    return high_id, low_id


def _post_reconcile(
    tenant_id: str,
    *,
    target_kind: str,
    member_ids: list[str],
    policy_override: str | None = None,
) -> httpx.Response:
    body = {
        "target_kind": target_kind,
        "conflict_member_ids": member_ids,
    }
    if policy_override is not None:
        body["policy_override"] = policy_override
    with httpx.Client(base_url=RUNTIME, timeout=120.0) as client:
        return client.post(
            f"/admin/tenants/{tenant_id}/knowledge/contradictions/reconcile",
            json=body,
        )


# ---------------------------------------------------------------------------
# 1. Default policy = schema's contradiction_policy (TRUST_RANKED)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestAgentResolvesConflictUsingSchemaPolicy:
    """No override → schema default (TRUST_RANKED for entity_fact) picks high trust."""

    def test_schema_default_picks_high_trust(self) -> None:
        Mem0MemoryManager._instances.clear()
        tenant_id = unique_id("kagent_rc") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            high, low = _write_conflicting(
                mm,
                subject="company.ceo",
                high_trust_content="ceo: Alice",
                low_trust_content="ceo: Bob",
            )
            resp = _post_reconcile(
                tenant_id, target_kind="entity_fact", member_ids=[high, low]
            )
            assert resp.status_code == 200, resp.text[:500]
            body = resp.json()
            assert body["target_kind"] == "entity_fact"
            # entity_fact's seeded contradiction_policy is TRUST_RANKED.
            assert body["policy_used"] == "trust_ranked"
            assert body["survivors"] == [high]
            resolved_by_id = {r["memory_id"]: r for r in body["resolved"]}
            assert sorted(resolved_by_id) == sorted([high, low])
            assert resolved_by_id[high]["survived"] is True
            assert resolved_by_id[low]["survived"] is False
            # Disputed only fires under preserve_both.
            assert resolved_by_id[high]["disputed"] is False
            assert resolved_by_id[low]["disputed"] is False
            # policy_overridden=False (we didn't pass an override).
            assert body["metadata"]["policy_overridden"] is False
            assert body["metadata"]["input_count"] == 2
            assert body["metadata"]["survivor_count"] == 1
        finally:
            try:
                mm.clear_agent_memory(tenant_id, RECON_AGENT)
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 2. policy_override="latest_wins" → newer created_at survives
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestPolicyOverrideRespected:
    """latest_wins picks the member with the higher created_at, ignoring trust."""

    def test_latest_wins_overrides_trust(self) -> None:
        Mem0MemoryManager._instances.clear()
        tenant_id = unique_id("kagent_rc2") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            # high-trust is older (5 days), low-trust is fresh (now).
            # Under TRUST_RANKED high would win; under latest_wins LOW
            # wins because its created_at is newer.
            high, low = _write_conflicting(
                mm,
                subject="company.hq",
                high_trust_content="hq: New York",
                low_trust_content="hq: London",
                high_trust_age_days=5.0,
                low_trust_age_days=0.0,
            )
            resp = _post_reconcile(
                tenant_id,
                target_kind="entity_fact",
                member_ids=[high, low],
                policy_override="latest_wins",
            )
            assert resp.status_code == 200, resp.text[:500]
            body = resp.json()
            assert body["policy_used"] == "latest_wins"
            assert body["metadata"]["policy_overridden"] is True
            assert body["survivors"] == [low]
            resolved_by_id = {r["memory_id"]: r for r in body["resolved"]}
            assert resolved_by_id[low]["survived"] is True
            assert resolved_by_id[high]["survived"] is False
        finally:
            try:
                mm.clear_agent_memory(tenant_id, RECON_AGENT)
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()


# ---------------------------------------------------------------------------
# 3. policy_override="preserve_both" → both survive, both disputed
# ---------------------------------------------------------------------------


@pytest.mark.e2e
@skip_if_no_runtime
class TestPreserveBothPolicy:
    """preserve_both keeps every member with disputed=True flagged."""

    def test_preserve_both_keeps_both_with_disputed_flag(self) -> None:
        Mem0MemoryManager._instances.clear()
        tenant_id = unique_id("kagent_rc3") + ":t1"
        mm = _build_manager(tenant_id)
        try:
            high, low = _write_conflicting(
                mm,
                subject="paris.population",
                high_trust_content="paris pop: 2.1 million",
                low_trust_content="paris pop: 2.2 million",
            )
            resp = _post_reconcile(
                tenant_id,
                target_kind="entity_fact",
                member_ids=[high, low],
                policy_override="preserve_both",
            )
            assert resp.status_code == 200, resp.text[:500]
            body = resp.json()
            assert body["policy_used"] == "preserve_both"
            assert sorted(body["survivors"]) == sorted([high, low])
            resolved_by_id = {r["memory_id"]: r for r in body["resolved"]}
            assert resolved_by_id[high]["survived"] is True
            assert resolved_by_id[low]["survived"] is True
            assert resolved_by_id[high]["disputed"] is True
            assert resolved_by_id[low]["disputed"] is True
            assert body["metadata"]["survivor_count"] == 2
        finally:
            try:
                mm.clear_agent_memory(tenant_id, RECON_AGENT)
            except Exception:
                pass
            Mem0MemoryManager._instances.clear()
