"""Unit tests for federation read + admin-gated promote."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from cogniverse_core.memory.federation import (
    FederationDeniedError,
    FederationService,
    PromotionResult,
    org_trunk_tenant_id,
)
from cogniverse_core.memory.schema import (
    KnowledgeRegistry,
    KnowledgeSchema,
    Pinnable,
    Sensitivity,
    build_default_registry,
)


def _row(
    mid: str,
    content: str,
    *,
    subject_key: str = "",
    kind: str = "external_doc",
    extra_meta: Dict[str, Any] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {"kind": kind}
    if subject_key:
        meta["subject_key"] = subject_key
    if extra_meta:
        meta.update(extra_meta)
    return {"id": mid, "memory": content, "metadata": meta}


def _factory_for(per_tenant: Dict[str, List[Dict[str, Any]]]):
    """Return a memory_manager factory that yields a stub MM with rows from ``per_tenant``."""
    instances: Dict[str, MagicMock] = {}

    def _factory(tenant_id: str) -> MagicMock:
        if tenant_id in instances:
            return instances[tenant_id]
        mm = MagicMock()
        mm.memory = MagicMock()  # truthy so federation _fetch proceeds
        mm.tenant_id = tenant_id
        rows = list(per_tenant.get(tenant_id, []))
        mm.get_all_memories = lambda *, tenant_id=tenant_id, agent_name: list(rows)
        # Capture promotion writes.
        adds: List[Dict[str, Any]] = []

        def _add(*, content, tenant_id, agent_name, metadata=None, infer=False):
            adds.append(
                {
                    "content": content,
                    "tenant_id": tenant_id,
                    "agent_name": agent_name,
                    "metadata": dict(metadata or {}),
                }
            )
            return f"m_promoted_{len(adds)}"

        mm.add_memory = _add
        mm._adds = adds  # exposed for assertions
        instances[tenant_id] = mm
        return mm

    return _factory, instances


class TestOrgTrunkId:
    def test_simple_tenant_yields_org_trunk(self):
        assert org_trunk_tenant_id("acme") == "acme:_org_trunk"

    def test_org_tenant_yields_org_trunk(self):
        assert org_trunk_tenant_id("acme:production") == "acme:_org_trunk"


@pytest.mark.unit
@pytest.mark.ci_fast
class TestFederatedRead:
    def test_tenant_overlay_wins_on_subject_key_collision(self):
        per_tenant = {
            "acme:production": [
                _row("m_t", "tenant override of paris", subject_key="france:capital"),
            ],
            "acme:_org_trunk": [
                _row("m_o", "org-trunk says paris", subject_key="france:capital"),
                _row(
                    "m_o2",
                    "org-trunk france is in europe",
                    subject_key="france:continent",
                ),
            ],
        }
        factory, _ = _factory_for(per_tenant)
        svc = FederationService(factory, build_default_registry())
        rows = svc.federated_get_all("acme:production", "search_agent")

        # Tenant override of france:capital wins; org's france:continent kept.
        ids = {r["id"] for r in rows}
        assert "m_t" in ids
        assert "m_o" not in ids
        assert "m_o2" in ids

    def test_tenant_origin_tag_set(self):
        per_tenant = {
            "acme:production": [_row("m_t", "tenant content", subject_key="x")],
            "acme:_org_trunk": [_row("m_o", "org content", subject_key="y")],
        }
        factory, _ = _factory_for(per_tenant)
        svc = FederationService(factory, build_default_registry())
        rows = svc.federated_get_all("acme:production", "search_agent")
        by_id = {r["id"]: r for r in rows}
        assert by_id["m_t"]["_federation_origin"] == "tenant"
        assert by_id["m_o"]["_federation_origin"] == "org_trunk"

    def test_no_subject_key_kept_unique(self):
        per_tenant = {
            "acme:production": [_row("m_t", "anonymous tenant fact")],
            "acme:_org_trunk": [_row("m_o", "anonymous org fact")],
        }
        factory, _ = _factory_for(per_tenant)
        svc = FederationService(factory, build_default_registry())
        rows = svc.federated_get_all("acme:production", "search_agent")
        ids = {r["id"] for r in rows}
        # Both kept — no subject_key to dedup on.
        assert ids == {"m_t", "m_o"}


@pytest.mark.unit
@pytest.mark.ci_fast
class TestPromotionAuthority:
    def test_tenant_private_cannot_be_promoted(self):
        registry = KnowledgeRegistry()
        registry.register(
            KnowledgeSchema(
                kind="entity_fact",
                sensitivity=Sensitivity.TENANT_PRIVATE,
                pinnable_by=Pinnable.USER,
                provenance_required=False,
            ),
            replace=True,
        )
        per_tenant = {"acme:production": [], "acme:_org_trunk": []}
        factory, _ = _factory_for(per_tenant)
        svc = FederationService(factory, registry)

        source = _row("m_a", "tenant-private fact", kind="entity_fact")
        with pytest.raises(FederationDeniedError, match="tenant_private"):
            svc.promote_to_org_trunk(
                source_tenant_id="acme:production",
                source_memory=source,
                actor_role=Pinnable.ORG_ADMIN,  # even org admin can't promote private
                actor_id="oadm",
            )

    def test_user_role_below_pin_floor_rejected(self):
        registry = KnowledgeRegistry()
        registry.register(
            KnowledgeSchema(
                kind="external_doc",
                sensitivity=Sensitivity.ORG_SHARED,
                pinnable_by=Pinnable.TENANT_ADMIN,
                provenance_required=False,
            ),
            replace=True,
        )
        per_tenant = {"acme:production": [], "acme:_org_trunk": []}
        factory, _ = _factory_for(per_tenant)
        svc = FederationService(factory, registry)
        source = _row("m_b", "doc", kind="external_doc")

        with pytest.raises(FederationDeniedError):
            svc.promote_to_org_trunk(
                source_tenant_id="acme:production",
                source_memory=source,
                actor_role=Pinnable.USER,  # user < tenant_admin floor
                actor_id="alice",
            )

    def test_tenant_admin_can_promote_org_shared(self):
        registry = KnowledgeRegistry()
        registry.register(
            KnowledgeSchema(
                kind="external_doc",
                sensitivity=Sensitivity.ORG_SHARED,
                pinnable_by=Pinnable.TENANT_ADMIN,
                provenance_required=False,
            ),
            replace=True,
        )
        per_tenant = {"acme:production": [], "acme:_org_trunk": []}
        factory, instances = _factory_for(per_tenant)
        svc = FederationService(factory, registry)
        source = _row("m_c", "shareable doc", kind="external_doc")

        out = svc.promote_to_org_trunk(
            source_tenant_id="acme:production",
            source_memory=source,
            actor_role=Pinnable.TENANT_ADMIN,
            actor_id="admin_alpha",
        )
        assert isinstance(out, PromotionResult)
        assert out.source_memory_id == "m_c"
        assert out.org_trunk_tenant_id == "acme:_org_trunk"

        # The org trunk got the new memory with promotion stamps.
        org_mm = instances["acme:_org_trunk"]
        assert len(org_mm._adds) == 1
        promoted_meta = org_mm._adds[0]["metadata"]
        assert promoted_meta["promoted_from_tenant"] == "acme:production"
        assert promoted_meta["promoted_by"] == "admin_alpha"
        assert promoted_meta["promoted_by_role"] == "tenant_admin"
        assert promoted_meta["kind"] == "external_doc"


class TestPromotionRoundsTrip:
    def test_promoted_memory_visible_via_federated_read_in_other_tenant(self):
        """An admin promotes; a sibling tenant under the same org sees it."""
        registry = KnowledgeRegistry()
        registry.register(
            KnowledgeSchema(
                kind="external_doc",
                sensitivity=Sensitivity.ORG_SHARED,
                pinnable_by=Pinnable.TENANT_ADMIN,
                provenance_required=False,
            ),
            replace=True,
        )
        # Two tenants under acme; sibling has nothing of its own.
        per_tenant = {
            "acme:alpha": [
                _row(
                    "m_alpha", "alpha local doc", subject_key="x:1", kind="external_doc"
                ),
            ],
            "acme:beta": [],
            "acme:_org_trunk": [],
        }
        factory, instances = _factory_for(per_tenant)
        svc = FederationService(factory, registry)

        # Alpha admin promotes m_alpha to the org trunk.
        svc.promote_to_org_trunk(
            source_tenant_id="acme:alpha",
            source_memory=per_tenant["acme:alpha"][0],
            actor_role=Pinnable.TENANT_ADMIN,
            actor_id="alpha_admin",
        )

        # The trunk MM now has a promoted record. To make the federated read
        # see it, mirror the recorded add into the trunk's get_all source.
        trunk = instances["acme:_org_trunk"]
        trunk_added = trunk._adds[0]
        promoted_row = {
            "id": "m_promoted",
            "memory": trunk_added["content"],
            "metadata": trunk_added["metadata"],
        }
        # Refresh trunk's list to include it.
        trunk.get_all_memories = lambda *, tenant_id="acme:_org_trunk", agent_name: [
            promoted_row
        ]

        # Beta does its own federated read; should see the promoted row.
        rows = svc.federated_get_all("acme:beta", "_promoted")
        ids = {r["id"] for r in rows}
        assert "m_promoted" in ids
        promoted = next(r for r in rows if r["id"] == "m_promoted")
        assert promoted["_federation_origin"] == "org_trunk"


class TestPromotionStorageFailure:
    """A promotion whose org-trunk write returns no id must fail loudly, not
    return a PromotionResult with an empty promoted_memory_id."""

    def test_promote_raises_when_storage_returns_no_id(self):
        registry = KnowledgeRegistry()
        registry.register(
            KnowledgeSchema(
                kind="external_doc",
                sensitivity=Sensitivity.ORG_SHARED,
                pinnable_by=Pinnable.TENANT_ADMIN,
                provenance_required=False,
            ),
            replace=True,
        )

        def factory(tenant_id):
            mm = MagicMock()
            mm.memory = MagicMock()
            mm.add_memory = lambda **kwargs: None  # deduplicated / dropped
            return mm

        svc = FederationService(factory, registry)
        source = _row("m_c", "shareable doc", kind="external_doc")

        with pytest.raises(RuntimeError, match="not persisted"):
            svc.promote_to_org_trunk(
                source_tenant_id="acme:production",
                source_memory=source,
                actor_role=Pinnable.TENANT_ADMIN,
                actor_id="admin_alpha",
            )


class TestBackendFailureVisibility:
    """A backend read failure must surface at WARNING, not be silently
    swallowed at DEBUG — otherwise a Mem0/Vespa outage is invisible at the
    default INFO level and the agent answers with missing memory context."""

    def _service(self, factory):
        from unittest.mock import MagicMock

        from cogniverse_core.memory.federation import FederationService

        return FederationService(memory_manager_factory=factory, registry=MagicMock())

    def test_factory_failure_logs_warning_and_returns_empty(self, caplog):
        import logging

        def boom(tenant_id):
            raise ConnectionError("mem0 down")

        svc = self._service(boom)
        with caplog.at_level(
            logging.WARNING, logger="cogniverse_core.memory.federation"
        ):
            result = svc._fetch("acme:acme", "search")

        assert result == []
        assert any(
            r.levelno >= logging.WARNING and "factory" in r.getMessage()
            for r in caplog.records
        ), "backend factory failure must log at WARNING"

    def test_get_all_failure_logs_warning_and_returns_empty(self, caplog):
        import logging
        from unittest.mock import MagicMock

        mm = MagicMock()
        mm.memory = object()
        mm.get_all_memories.side_effect = TimeoutError("vespa timeout")

        svc = self._service(lambda tenant_id: mm)
        with caplog.at_level(
            logging.WARNING, logger="cogniverse_core.memory.federation"
        ):
            result = svc._fetch("acme:acme", "search")

        assert result == []
        assert any(
            r.levelno >= logging.WARNING and "get_all_memories" in r.getMessage()
            for r in caplog.records
        )
