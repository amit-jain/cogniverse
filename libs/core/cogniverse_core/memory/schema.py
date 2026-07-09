"""Knowledge schema registry.

Cogniverse's memory layer is the spine of a knowledge management system, not
enrichment for retrieval. Every write is a fact-shaped artefact whose lifetime,
sharing rules, and trust expectations differ by *kind*. The registry models
these per-kind expectations:

  * **retention** — does this memory live forever, expire after a session,
    expire after N days, or follow a schema-defined custom rule?
  * **sensitivity** — is it tenant-private, shareable across an org, or
    explicitly global?
  * **pinnable_by** — who is allowed to pin a memory of this kind?
  * **provenance_required** — must the writer attach `derived_from`/source?
  * **contradiction_policy** — when two memories of the same subject disagree,
    which wins?
  * **default_trust** — initial trust score before user/admin endorsement
    or trust decay updates.

Defaults are conservative: a kind that has not been explicitly registered is
treated as ``permanent`` + ``tenant_private`` + ``provenance_required=True``
(no auto-cleanup, no leakage, must cite a source).

The pinning service, lifecycle scheduler, provenance store, and contradiction
detector all read this registry at validation time.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Dict, Optional


class Retention(str, enum.Enum):
    PERMANENT = "permanent"
    EPHEMERAL_SESSION = "ephemeral_session"
    EPHEMERAL_DAYS = "ephemeral_days"
    SCHEMA_DRIVEN = "schema_driven"


class Sensitivity(str, enum.Enum):
    TENANT_PRIVATE = "tenant_private"
    ORG_SHARED = "org_shared"
    GLOBAL_SHARED = "global_shared"


class Pinnable(str, enum.Enum):
    NOBODY = "nobody"
    USER = "user"
    TENANT_ADMIN = "tenant_admin"
    ORG_ADMIN = "org_admin"


class ContradictionPolicy(str, enum.Enum):
    LATEST_WINS = "latest_wins"
    TRUST_RANKED = "trust_ranked"
    PRESERVE_BOTH = "preserve_both"


class SchemaViolationError(ValueError):
    """Raised when a memory write violates its kind's schema policy."""


@dataclass(frozen=True)
class KnowledgeSchema:
    """Per-kind policy applied at write/read time.

    Frozen dataclass — schemas are registered once at boot and never mutated.
    A future loader will read these from ``configs/knowledge_schemas/``
    rather than relying only on the seed defaults below.
    """

    kind: str
    retention: Retention = Retention.PERMANENT
    sensitivity: Sensitivity = Sensitivity.TENANT_PRIVATE
    pinnable_by: Pinnable = Pinnable.TENANT_ADMIN
    provenance_required: bool = True
    contradiction_policy: ContradictionPolicy = ContradictionPolicy.LATEST_WINS
    default_trust: float = 0.5
    # Used only when retention == EPHEMERAL_DAYS. Ignored otherwise.
    retention_days: Optional[int] = None
    # Used only when retention == SCHEMA_DRIVEN. Receives the candidate
    # memory dict (with id, metadata, created_at, etc.) and returns True
    # iff the cleanup tick should delete it. Pinned memories are filtered
    # out before this is called, so the hook does NOT need to re-check pins.
    cleanup_hook: Optional[Callable[[Dict[str, Any], "KnowledgeSchema"], bool]] = None

    def __post_init__(self) -> None:
        if not self.kind or not self.kind.strip():
            raise ValueError("KnowledgeSchema.kind must be a non-empty string")
        if not 0.0 <= self.default_trust <= 1.0:
            raise ValueError(
                f"default_trust must be in [0.0, 1.0]; got {self.default_trust}"
            )
        if self.retention is Retention.EPHEMERAL_DAYS and (
            self.retention_days is None or self.retention_days <= 0
        ):
            raise ValueError("retention=EPHEMERAL_DAYS requires retention_days > 0")
        if (
            self.retention is Retention.EPHEMERAL_SESSION
            and self.pinnable_by is not Pinnable.NOBODY
        ):
            # Pinning a session-scoped memory is a foot-gun: drop_session
            # hard-deletes every match regardless of pin, so a tenant
            # admin pinning a session memory expecting durability would
            # silently lose it on session end. Refuse the schema instead.
            raise ValueError(
                f"kind={self.kind!r} retention=ephemeral_session requires "
                f"pinnable_by=nobody (session memories cannot be pinned — "
                f"got pinnable_by={self.pinnable_by.value!r})"
            )

    def validate_provenance(
        self, provenance: Optional["ProvenanceLike"] = None
    ) -> None:
        """Raise SchemaViolationError if provenance policy is violated."""
        if not self.provenance_required:
            return
        if provenance is None or not getattr(provenance, "derived_from", []):
            raise SchemaViolationError(
                f"kind={self.kind!r} requires provenance.derived_from to be "
                "non-empty; this write would not be auditable"
            )

    def validate_pin_authority(self, pinned_by: Pinnable) -> None:
        """Raise SchemaViolationError if requester's role is below the floor."""
        if not _can_pin(pinned_by, self.pinnable_by):
            raise SchemaViolationError(
                f"kind={self.kind!r} forbids pin from role={pinned_by.value} "
                f"(minimum required: {self.pinnable_by.value})"
            )

    def validate_session_membership(self, metadata: Dict[str, Any]) -> None:
        """Reject EPHEMERAL_SESSION writes that don't carry session_id.

        EPHEMERAL_SESSION memories are cleaned by an explicit session-end
        event (Mem0MemoryManager.drop_session). Without a session_id the
        cleanup path has nothing to match — the memory would silently
        outlive its session. Refuse the write.
        """
        if self.retention is not Retention.EPHEMERAL_SESSION:
            return
        sid = (metadata or {}).get("session_id")
        if not isinstance(sid, str) or not sid.strip():
            raise SchemaViolationError(
                f"kind={self.kind!r} retention=ephemeral_session requires "
                "metadata.session_id (non-empty string) so session-end "
                "cleanup can find and drop it"
            )

    def validate_write(
        self,
        provenance: Optional["ProvenanceLike"] = None,
        pinned_by: Optional[Pinnable] = None,
    ) -> None:
        """Composite check used by the write path.

        Pin-only flows (where the original write already happened earlier)
        should use ``validate_pin_authority`` directly to avoid re-asserting
        the provenance constraint.
        """
        self.validate_provenance(provenance)
        if pinned_by is not None:
            self.validate_pin_authority(pinned_by)


# Provenance is defined in cogniverse_core.memory.provenance; here we
# accept any object exposing `derived_from` via duck typing.
class ProvenanceLike:  # pragma: no cover — typing helper
    derived_from: list


_PIN_AUTHORITY: Dict[Pinnable, int] = {
    Pinnable.NOBODY: 0,
    Pinnable.USER: 1,
    Pinnable.TENANT_ADMIN: 2,
    Pinnable.ORG_ADMIN: 3,
}


def _can_pin(requester: Pinnable, minimum: Pinnable) -> bool:
    """True iff requester's authority is >= the schema's required minimum."""
    return _PIN_AUTHORITY[requester] >= _PIN_AUTHORITY[minimum]


@dataclass
class _RegistryState:
    schemas: Dict[str, KnowledgeSchema] = field(default_factory=dict)


class KnowledgeRegistry:
    """Process-wide registry of knowledge schemas.

    Thread-safe register/get; reads are lock-free after registration since
    the underlying dict supports atomic single-key reads in CPython. The
    registry is a singleton — instantiate once at boot and inject through
    config managers, do not paper over with module globals.
    """

    def __init__(self) -> None:
        self._state = _RegistryState()
        self._lock = Lock()

    def register(self, schema: KnowledgeSchema, *, replace: bool = False) -> None:
        with self._lock:
            existing = self._state.schemas.get(schema.kind)
            if existing is not None and not replace:
                if existing == schema:
                    return  # Idempotent re-register of the same definition.
                raise ValueError(
                    f"kind={schema.kind!r} already registered with different "
                    f"definition; pass replace=True to override"
                )
            self._state.schemas[schema.kind] = schema

    def get(self, kind: str) -> KnowledgeSchema:
        """Return the schema for a kind, falling back to a safe default.

        Conservative default: permanent + tenant_private + provenance_required.
        Any unknown kind is treated as private and durable so a forgotten
        registration never leaks data or quietly drops history.
        """
        schema = self._state.schemas.get(kind)
        if schema is not None:
            return schema
        return KnowledgeSchema(kind=kind)

    def is_registered(self, kind: str) -> bool:
        return kind in self._state.schemas

    def all_kinds(self) -> list[str]:
        with self._lock:
            return sorted(self._state.schemas.keys())


_LEARNED_STRATEGY_RETIRE_AGE_DAYS = 30
_LEARNED_STRATEGY_MIN_CONFIRMATIONS = 3


def _retire_unconfirmed_strategy(
    memory: Dict[str, Any], schema: "KnowledgeSchema"
) -> bool:
    """Cleanup hook: retire a learned_strategy that never gained traction.

    Returns True (delete) when ``confirmation_count`` is still below the
    minimum AND the record's ``created_at`` is older than the retire-age
    threshold. ``last_confirmed_at`` (set in :class:`Strategy`) is reset
    each time a duplicate is rediscovered, so the age check naturally
    measures "time since last confirmation" — the bump-on-dedup path in
    StrategyLearner replaces the old record with a fresh one.
    """
    import json as _json
    from datetime import datetime as _dt
    from datetime import timezone as _tz

    meta = memory.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = _json.loads(meta)
        except (ValueError, TypeError):
            return False
    if not isinstance(meta, dict):
        return False
    if meta.get("kind") != "learned_strategy":
        return False

    confirmation_count = int(meta.get("confirmation_count", 1) or 1)
    if confirmation_count >= _LEARNED_STRATEGY_MIN_CONFIRMATIONS:
        return False

    created_at = meta.get("created_at") or memory.get("created_at")
    if not created_at:
        return False
    try:
        cdt = _dt.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False
    age_days = (
        _dt.now(_tz.utc).replace(tzinfo=None) - cdt.replace(tzinfo=None)
    ).total_seconds() / 86400.0
    return age_days > _LEARNED_STRATEGY_RETIRE_AGE_DAYS


def build_default_registry() -> KnowledgeRegistry:
    """Build a registry seeded with cogniverse's known knowledge kinds.

    Returns a fresh registry — callers (runtime boot, tests) decide how to
    share it. The seed list mirrors the kinds named in the plan:
      * conversation_turn — short-lived dialog history
      * learned_strategy — distilled patterns from optimization traces
      * tenant_instruction — admin-curated, pinned by tenant admin
      * external_doc — ingested external corpus
      * entity_fact — atomic facts about an entity
      * kg_node / kg_edge — knowledge graph
    """
    registry = KnowledgeRegistry()

    registry.register(
        KnowledgeSchema(
            kind="conversation_turn",
            retention=Retention.EPHEMERAL_DAYS,
            retention_days=14,
            sensitivity=Sensitivity.TENANT_PRIVATE,
            pinnable_by=Pinnable.USER,
            provenance_required=False,  # session text is its own provenance
            contradiction_policy=ContradictionPolicy.LATEST_WINS,
            default_trust=0.4,
        )
    )
    registry.register(
        KnowledgeSchema(
            kind="learned_strategy",
            retention=Retention.SCHEMA_DRIVEN,
            sensitivity=Sensitivity.TENANT_PRIVATE,
            pinnable_by=Pinnable.TENANT_ADMIN,
            provenance_required=True,
            contradiction_policy=ContradictionPolicy.TRUST_RANKED,
            default_trust=0.6,
            # Retirement: a learned_strategy whose confirmation_count is
            # still below 3 after 30 days never proved itself worth keeping.
            # The hook reads metadata.confirmation_count + created_at on each
            # candidate; pinned strategies are filtered out upstream by the
            # lifecycle scheduler before this hook runs.
            cleanup_hook=_retire_unconfirmed_strategy,
        )
    )
    registry.register(
        KnowledgeSchema(
            kind="tenant_instruction",
            retention=Retention.PERMANENT,
            sensitivity=Sensitivity.TENANT_PRIVATE,
            pinnable_by=Pinnable.TENANT_ADMIN,
            provenance_required=False,
            contradiction_policy=ContradictionPolicy.LATEST_WINS,
            default_trust=0.95,  # admin-curated; trusted highly
        )
    )
    registry.register(
        KnowledgeSchema(
            kind="external_doc",
            retention=Retention.PERMANENT,
            sensitivity=Sensitivity.TENANT_PRIVATE,
            pinnable_by=Pinnable.TENANT_ADMIN,
            provenance_required=True,
            contradiction_policy=ContradictionPolicy.PRESERVE_BOTH,
            default_trust=0.7,
        )
    )
    registry.register(
        KnowledgeSchema(
            kind="entity_fact",
            retention=Retention.PERMANENT,
            sensitivity=Sensitivity.TENANT_PRIVATE,
            pinnable_by=Pinnable.TENANT_ADMIN,
            provenance_required=True,
            contradiction_policy=ContradictionPolicy.TRUST_RANKED,
            default_trust=0.5,
        )
    )
    registry.register(
        KnowledgeSchema(
            kind="kg_node",
            retention=Retention.PERMANENT,
            sensitivity=Sensitivity.TENANT_PRIVATE,
            pinnable_by=Pinnable.TENANT_ADMIN,
            provenance_required=True,
            contradiction_policy=ContradictionPolicy.PRESERVE_BOTH,
            default_trust=0.6,
        )
    )
    registry.register(
        KnowledgeSchema(
            kind="kg_edge",
            retention=Retention.PERMANENT,
            sensitivity=Sensitivity.TENANT_PRIVATE,
            pinnable_by=Pinnable.TENANT_ADMIN,
            provenance_required=True,
            contradiction_policy=ContradictionPolicy.PRESERVE_BOTH,
            default_trust=0.6,
        )
    )
    # Session-scoped scratch: anything an agent stores under this kind
    # vanishes when the session closes (Mem0MemoryManager.drop_session,
    # invoked via POST /admin/sessions/{session_id}/close or DELETE
    # /admin/tenants/{tenant_id}/sessions/{session_id}). Writes MUST
    # carry metadata.session_id; the schema validator rejects writes
    # that don't. Pinning is forbidden by the EPHEMERAL_SESSION schema
    # gate so a pinned session memory can't silently outlive its session.
    registry.register(
        KnowledgeSchema(
            kind="session_scratch",
            retention=Retention.EPHEMERAL_SESSION,
            sensitivity=Sensitivity.TENANT_PRIVATE,
            pinnable_by=Pinnable.NOBODY,
            provenance_required=False,
            contradiction_policy=ContradictionPolicy.LATEST_WINS,
            default_trust=0.3,  # transient, low base trust
        )
    )
    # Sentinel kinds the system itself writes — no caller-supplied
    # provenance is possible (the system IS the provenance). Without
    # explicit registration, the registry's default returns
    # provenance_required=True and every internal write blows up.
    registry.register(
        KnowledgeSchema(
            kind="pin_record",
            retention=Retention.PERMANENT,
            sensitivity=Sensitivity.TENANT_PRIVATE,
            pinnable_by=Pinnable.NOBODY,  # users don't pin pin_records
            provenance_required=False,
            contradiction_policy=ContradictionPolicy.LATEST_WINS,
            default_trust=1.0,
        )
    )
    registry.register(
        KnowledgeSchema(
            kind="conflict_set",
            retention=Retention.PERMANENT,
            sensitivity=Sensitivity.TENANT_PRIVATE,
            pinnable_by=Pinnable.NOBODY,
            provenance_required=False,
            contradiction_policy=ContradictionPolicy.LATEST_WINS,
            default_trust=1.0,
        )
    )
    return registry


SUMMARY_KIND = "knowledge_summary"


def register_summary_kind(registry: "KnowledgeRegistry") -> None:
    """Register the org-shared ``knowledge_summary`` kind on ``registry``.

    This is the one kind eligible for org-trunk promotion: KnowledgeSummarizationAgent
    produces it and the admin promote endpoint copies it into the trunk. Defined
    once here so the producer and the promotion gate agree on its sensitivity.
    """
    if registry.is_registered(SUMMARY_KIND):
        return
    registry.register(
        KnowledgeSchema(
            kind=SUMMARY_KIND,
            retention=Retention.PERMANENT,
            sensitivity=Sensitivity.ORG_SHARED,
            pinnable_by=Pinnable.TENANT_ADMIN,
            provenance_required=True,
            contradiction_policy=ContradictionPolicy.LATEST_WINS,
            default_trust=0.7,
        )
    )


def build_promotable_registry() -> KnowledgeRegistry:
    """Default registry plus the org-shared kinds eligible for org-trunk promotion."""
    registry = build_default_registry()
    register_summary_kind(registry)
    return registry
