"""Unit tests for the knowledge schema registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from cogniverse_core.memory.schema import (
    ContradictionPolicy,
    KnowledgeRegistry,
    KnowledgeSchema,
    Pinnable,
    Retention,
    SchemaViolationError,
    Sensitivity,
    build_default_registry,
)


@dataclass
class FakeProvenance:
    derived_from: List[str]


@pytest.mark.unit
@pytest.mark.ci_fast
class TestKnowledgeSchemaConstruction:
    def test_defaults_are_conservative(self):
        s = KnowledgeSchema(kind="custom")
        assert s.retention is Retention.PERMANENT
        assert s.sensitivity is Sensitivity.TENANT_PRIVATE
        assert s.pinnable_by is Pinnable.TENANT_ADMIN
        assert s.provenance_required is True
        assert s.contradiction_policy is ContradictionPolicy.LATEST_WINS
        assert s.default_trust == 0.5

    def test_empty_kind_rejected(self):
        with pytest.raises(ValueError):
            KnowledgeSchema(kind="")
        with pytest.raises(ValueError):
            KnowledgeSchema(kind="   ")

    def test_default_trust_must_be_in_unit_interval(self):
        with pytest.raises(ValueError):
            KnowledgeSchema(kind="x", default_trust=-0.1)
        with pytest.raises(ValueError):
            KnowledgeSchema(kind="x", default_trust=1.1)

    def test_ephemeral_days_requires_positive_retention_days(self):
        with pytest.raises(ValueError):
            KnowledgeSchema(kind="x", retention=Retention.EPHEMERAL_DAYS)
        with pytest.raises(ValueError):
            KnowledgeSchema(
                kind="x", retention=Retention.EPHEMERAL_DAYS, retention_days=0
            )
        # Valid
        KnowledgeSchema(kind="x", retention=Retention.EPHEMERAL_DAYS, retention_days=7)

    def test_ephemeral_session_forbids_pinning(self):
        # Pinning a session-scoped memory is a foot-gun: drop_session
        # hard-deletes regardless of pin, so the schema gates non-NOBODY
        # pinning at construction.
        for role in (Pinnable.USER, Pinnable.TENANT_ADMIN, Pinnable.ORG_ADMIN):
            with pytest.raises(ValueError, match="ephemeral_session"):
                KnowledgeSchema(
                    kind="x",
                    retention=Retention.EPHEMERAL_SESSION,
                    pinnable_by=role,
                )
        # NOBODY is the only allowed pin role for session memories.
        KnowledgeSchema(
            kind="x",
            retention=Retention.EPHEMERAL_SESSION,
            pinnable_by=Pinnable.NOBODY,
        )


@pytest.mark.unit
@pytest.mark.ci_fast
class TestValidateWriteProvenance:
    def test_provenance_required_rejects_missing(self):
        s = KnowledgeSchema(kind="entity_fact", provenance_required=True)
        with pytest.raises(SchemaViolationError) as exc:
            s.validate_write(provenance=None)
        assert "entity_fact" in str(exc.value)
        assert "provenance" in str(exc.value).lower()

    def test_provenance_required_rejects_empty_derived_from(self):
        s = KnowledgeSchema(kind="entity_fact", provenance_required=True)
        with pytest.raises(SchemaViolationError):
            s.validate_write(provenance=FakeProvenance(derived_from=[]))

    def test_provenance_required_accepts_non_empty(self):
        s = KnowledgeSchema(kind="entity_fact", provenance_required=True)
        s.validate_write(provenance=FakeProvenance(derived_from=["src://doc/1"]))

    def test_provenance_optional_accepts_missing(self):
        s = KnowledgeSchema(kind="conversation_turn", provenance_required=False)
        s.validate_write(provenance=None)


class TestValidateWritePinAuthority:
    @pytest.mark.parametrize(
        "schema_min,requester,allowed",
        [
            (Pinnable.TENANT_ADMIN, Pinnable.USER, False),
            (Pinnable.TENANT_ADMIN, Pinnable.TENANT_ADMIN, True),
            (Pinnable.TENANT_ADMIN, Pinnable.ORG_ADMIN, True),
            (Pinnable.USER, Pinnable.USER, True),
            (Pinnable.NOBODY, Pinnable.USER, True),  # NOBODY = floor; anyone above
            (Pinnable.ORG_ADMIN, Pinnable.TENANT_ADMIN, False),
        ],
    )
    def test_pin_authority_matrix(
        self, schema_min: Pinnable, requester: Pinnable, allowed: bool
    ):
        s = KnowledgeSchema(
            kind="x",
            pinnable_by=schema_min,
            provenance_required=False,
        )
        if allowed:
            s.validate_write(pinned_by=requester)
        else:
            with pytest.raises(SchemaViolationError) as exc:
                s.validate_write(pinned_by=requester)
            assert "pin" in str(exc.value).lower()


class TestValidateSessionMembership:
    def test_session_kind_rejects_missing_session_id(self):
        s = KnowledgeSchema(
            kind="session_scratch",
            retention=Retention.EPHEMERAL_SESSION,
            pinnable_by=Pinnable.NOBODY,
            provenance_required=False,
        )
        with pytest.raises(SchemaViolationError, match="session_id"):
            s.validate_session_membership({})
        with pytest.raises(SchemaViolationError, match="session_id"):
            s.validate_session_membership({"session_id": ""})
        with pytest.raises(SchemaViolationError, match="session_id"):
            s.validate_session_membership({"session_id": "   "})
        with pytest.raises(SchemaViolationError, match="session_id"):
            s.validate_session_membership({"session_id": 42})

    def test_session_kind_accepts_non_empty_session_id(self):
        s = KnowledgeSchema(
            kind="session_scratch",
            retention=Retention.EPHEMERAL_SESSION,
            pinnable_by=Pinnable.NOBODY,
            provenance_required=False,
        )
        # No exception.
        s.validate_session_membership({"session_id": "s_abc"})

    def test_non_session_kind_ignores_session_id(self):
        # Non-session kinds never gate on session_id.
        s = KnowledgeSchema(kind="entity_fact")
        s.validate_session_membership({})
        s.validate_session_membership({"session_id": ""})


class TestDefaultRegistryHasSessionScratch:
    def test_session_scratch_is_registered_with_session_retention(self):
        from cogniverse_core.memory.schema import build_default_registry

        reg = build_default_registry()
        assert reg.is_registered("session_scratch")
        s = reg.get("session_scratch")
        assert s.retention is Retention.EPHEMERAL_SESSION
        assert s.pinnable_by is Pinnable.NOBODY
        # Provenance not required for session scratch — it's transient.
        assert s.provenance_required is False


@pytest.mark.unit
@pytest.mark.ci_fast
class TestKnowledgeRegistry:
    def test_register_and_get(self):
        reg = KnowledgeRegistry()
        s = KnowledgeSchema(kind="my_kind")
        reg.register(s)
        assert reg.is_registered("my_kind")
        assert reg.get("my_kind") is s

    def test_get_unknown_returns_safe_default(self):
        reg = KnowledgeRegistry()
        unknown = reg.get("nonexistent")
        assert unknown.retention is Retention.PERMANENT
        assert unknown.sensitivity is Sensitivity.TENANT_PRIVATE
        assert unknown.provenance_required is True

    def test_re_register_same_schema_is_idempotent(self):
        reg = KnowledgeRegistry()
        s1 = KnowledgeSchema(kind="dup")
        s2 = KnowledgeSchema(kind="dup")
        reg.register(s1)
        reg.register(s2)  # equal; must not raise
        assert reg.get("dup") == s1

    def test_re_register_different_schema_rejected_without_replace(self):
        reg = KnowledgeRegistry()
        reg.register(KnowledgeSchema(kind="dup", default_trust=0.5))
        with pytest.raises(ValueError):
            reg.register(KnowledgeSchema(kind="dup", default_trust=0.9))

    def test_replace_flag_overrides(self):
        reg = KnowledgeRegistry()
        reg.register(KnowledgeSchema(kind="dup", default_trust=0.5))
        reg.register(KnowledgeSchema(kind="dup", default_trust=0.9), replace=True)
        assert reg.get("dup").default_trust == 0.9

    def test_all_kinds_returns_sorted(self):
        reg = KnowledgeRegistry()
        for k in ["zebra", "apple", "monkey"]:
            reg.register(KnowledgeSchema(kind=k))
        assert reg.all_kinds() == ["apple", "monkey", "zebra"]


class TestDefaultRegistrySeed:
    def test_seed_includes_all_planned_kinds(self):
        reg = build_default_registry()
        expected = {
            # User-facing knowledge kinds.
            "conversation_turn",
            "learned_strategy",
            "tenant_instruction",
            "external_doc",
            "entity_fact",
            "kg_node",
            "kg_edge",
            "session_scratch",
            # System-internal sentinel kinds (writes by the platform itself,
            # not callers — registered with provenance_required=False so
            # internal writes don't trip the schema gate).
            "pin_record",
            "conflict_set",
        }
        assert set(reg.all_kinds()) == expected

    def test_conversation_turn_is_ephemeral_14_days(self):
        s = build_default_registry().get("conversation_turn")
        assert s.retention is Retention.EPHEMERAL_DAYS
        assert s.retention_days == 14
        assert s.provenance_required is False

    def test_tenant_instruction_is_admin_pinned_high_trust(self):
        s = build_default_registry().get("tenant_instruction")
        assert s.pinnable_by is Pinnable.TENANT_ADMIN
        assert s.default_trust == 0.95
        assert s.retention is Retention.PERMANENT

    def test_kg_edge_preserves_both_on_contradiction(self):
        s = build_default_registry().get("kg_edge")
        assert s.contradiction_policy is ContradictionPolicy.PRESERVE_BOTH

    def test_learned_strategy_uses_schema_driven_retention(self):
        # The schema-driven lifecycle hook reads this — strategies decay based on confirmation_count,
        # not simple age, so retention is schema-driven.
        s = build_default_registry().get("learned_strategy")
        assert s.retention is Retention.SCHEMA_DRIVEN
