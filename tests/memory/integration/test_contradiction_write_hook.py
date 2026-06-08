"""ContradictionDetector runs on every write, persists conflict_set.

The detector existed and ran at READ time inside
MemoryAwareMixin._apply_trust_and_reconcile, but the contract requires
it to run on every WRITE so a conflict_set memory gets persisted for
downstream agents (ContradictionReconciliationAgent) to consume.
Without this, ContradictionReconciliationAgent had nothing to reconcile
in production.

This test verifies, against real Vespa:

  * a single write with a subject_key produces NO conflict_set
    (nothing to disagree with);
  * a second write with the same subject_key and DIFFERENT content
    produces exactly one conflict_set memory under sentinel agent
    ``_conflict_store`` whose metadata names both members;
  * a third write with the same subject_key + same content as one
    existing member does NOT produce another conflict_set (the
    detector's signature matched an existing entry);
  * writing a memory of kind ``conflict_set`` itself does NOT recurse
    into the detector (no infinite loop);
  * writing without a subject_key skips the hook (no conflict_set
    can ever be produced — the detector has nothing to group on).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from cogniverse_core.memory.contradiction import (
    CONFLICT_AGENT_NAME,
    CONFLICT_RECORD_KIND,
)
from cogniverse_core.memory.manager import Mem0MemoryManager
from cogniverse_core.memory.schema import build_default_registry
from cogniverse_core.schemas.filesystem_loader import FilesystemSchemaLoader
from cogniverse_foundation.config.manager import ConfigManager
from cogniverse_foundation.config.unified_config import SystemConfig
from cogniverse_vespa.config.config_store import VespaConfigStore
from tests.utils.llm_config import get_llm_base_url, get_llm_model

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.integration

TENANT = "test_tenant"
AGENT = "h5_writer"


@pytest.fixture(scope="module")
def manager_with_registry(shared_memory_vespa, shared_denseon):
    """Mem0 manager wired with the schema registry — enables contradiction detection."""
    Mem0MemoryManager._instances.clear()
    config_store = VespaConfigStore(
        backend_url="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
    )
    cm = ConfigManager(store=config_store)
    cm.set_system_config(
        SystemConfig(
            backend_url="http://localhost",
            backend_port=shared_memory_vespa["http_port"],
            inference_service_urls={"denseon": shared_denseon},
        )
    )
    mm = Mem0MemoryManager(tenant_id=TENANT)
    mm.initialize(
        backend_host="http://localhost",
        backend_port=shared_memory_vespa["http_port"],
        backend_config_port=shared_memory_vespa["config_port"],
        base_schema_name="agent_memories",
        llm_model=get_llm_model(),
        embedding_model="lightonai/DenseOn",
        llm_base_url=get_llm_base_url(),
        embedder_base_url=shared_denseon,
        auto_create_schema=False,
        config_manager=cm,
        schema_loader=FilesystemSchemaLoader(Path("configs/schemas")),
        knowledge_registry=build_default_registry(),
    )
    yield mm
    try:
        mm.clear_agent_memory(TENANT, AGENT)
        mm.clear_agent_memory(TENANT, CONFLICT_AGENT_NAME)
    except Exception:
        pass
    Mem0MemoryManager._instances.clear()


def _conflict_records_for_subject(mm, subject_key: str) -> list:
    rows = mm.get_all_memories(tenant_id=TENANT, agent_name=CONFLICT_AGENT_NAME)
    out = []
    for row in rows:
        md = row.get("metadata") or {}
        if (
            md.get("kind") == CONFLICT_RECORD_KIND
            and md.get("subject_key") == subject_key
        ):
            out.append(row)
    return out


class TestWriteHookProducesConflictSet:
    def test_single_write_no_conflict_set(self, manager_with_registry):
        subject = "h5:single_write"
        mm = manager_with_registry
        # Make sure we start clean for this subject.
        for r in _conflict_records_for_subject(mm, subject):
            mm.delete_memory(r["id"], TENANT, CONFLICT_AGENT_NAME)

        mid = mm.add_memory(
            content="Paris is the capital of France",
            tenant_id=TENANT,
            agent_name=AGENT,
            metadata={"kind": "tenant_instruction", "subject_key": subject},
            infer=False,
        )
        assert mid
        # Single member → no contradiction, no conflict_set persisted.
        assert _conflict_records_for_subject(mm, subject) == [], (
            "single-member subject must not produce a conflict_set"
        )

    def test_disagreeing_writes_produce_conflict_set(self, manager_with_registry):
        subject = "h5:capital_dispute"
        mm = manager_with_registry
        # Cleanup any pre-existing conflict_set for this subject (test
        # isolation — module-scoped fixture re-uses the manager).
        for r in _conflict_records_for_subject(mm, subject):
            mm.delete_memory(r["id"], TENANT, CONFLICT_AGENT_NAME)

        a = mm.add_memory(
            content="Paris is the capital of France",
            tenant_id=TENANT,
            agent_name=AGENT,
            metadata={"kind": "tenant_instruction", "subject_key": subject},
            infer=False,
        )
        # Same subject, DIFFERENT content → triggers detection on the second write.
        b = mm.add_memory(
            content="Lyon is the capital of France",
            tenant_id=TENANT,
            agent_name=AGENT,
            metadata={"kind": "tenant_instruction", "subject_key": subject},
            infer=False,
        )
        assert a and b

        records = _conflict_records_for_subject(mm, subject)
        assert len(records) == 1, (
            f"second disagreeing write must persist exactly one "
            f"conflict_set; got {len(records)} records for subject={subject}"
        )
        meta = records[0]["metadata"]
        members = sorted(meta.get("conflicting_memory_ids") or [])
        assert members == sorted([a, b]), (
            f"conflict_set members must reference both writers; "
            f"got members={members!r}, expected={sorted([a, b])!r}"
        )
        assert meta.get("kind") == CONFLICT_RECORD_KIND

    def test_repeat_same_content_does_not_duplicate_conflict_set(
        self, manager_with_registry
    ):
        subject = "h5:capital_dispute"
        mm = manager_with_registry
        # Run the disagreement first if not already present.
        if not _conflict_records_for_subject(mm, subject):
            mm.add_memory(
                content="Paris is the capital of France",
                tenant_id=TENANT,
                agent_name=AGENT,
                metadata={"kind": "tenant_instruction", "subject_key": subject},
                infer=False,
            )
            mm.add_memory(
                content="Lyon is the capital of France",
                tenant_id=TENANT,
                agent_name=AGENT,
                metadata={"kind": "tenant_instruction", "subject_key": subject},
                infer=False,
            )
        before = len(_conflict_records_for_subject(mm, subject))

        # Re-write the SAME content as one existing member — the detector's
        # signature must match the existing record and skip the duplicate.
        mm.add_memory(
            content="Paris is the capital of France",
            tenant_id=TENANT,
            agent_name=AGENT,
            metadata={"kind": "tenant_instruction", "subject_key": subject},
            infer=False,
        )
        after = len(_conflict_records_for_subject(mm, subject))
        assert after == before, (
            f"re-writing same content should not produce another "
            f"conflict_set; before={before} after={after}"
        )


class TestRecursionAndScopeGuards:
    def test_writing_conflict_set_kind_does_not_recurse(self, manager_with_registry):
        """Direct write of kind=conflict_set must not re-trigger detection
        (would infinite-loop). Hook short-circuits on this kind."""
        subject = "h5:no_recurse"
        mm = manager_with_registry
        mid = mm.add_memory(
            content="conflict_set: subject_key=h5:no_recurse members=a,b",
            tenant_id=TENANT,
            agent_name=CONFLICT_AGENT_NAME,
            metadata={
                "kind": CONFLICT_RECORD_KIND,
                "subject_key": subject,
                "conflicting_memory_ids": ["a", "b"],
            },
            infer=False,
        )
        assert mid
        # The hook guard means writing a conflict_set kind does not itself
        # produce a NEW conflict_set on top — the test reaches here
        # without recursing.

    def test_no_subject_key_skips_hook(self, manager_with_registry):
        mm = manager_with_registry
        # Two memories, no subject_key → detector has nothing to group on.
        mid_a = mm.add_memory(
            content="loose claim A",
            tenant_id=TENANT,
            agent_name=AGENT,
            metadata={"kind": "tenant_instruction"},
            infer=False,
        )
        mid_b = mm.add_memory(
            content="loose claim B",
            tenant_id=TENANT,
            agent_name=AGENT,
            metadata={"kind": "tenant_instruction"},
            infer=False,
        )
        assert mid_a and mid_b

        # The two subject-less writes must not appear in ANY conflict_set —
        # the hook skipped them, so neither id is groupable. Other tests in
        # the module produce conflicts, so assert by id membership (bounded)
        # rather than a global count.
        referenced = {
            member
            for r in mm.get_all_memories(
                tenant_id=TENANT, agent_name=CONFLICT_AGENT_NAME
            )
            for member in (r.get("metadata") or {}).get("conflicting_memory_ids") or []
        }
        assert mid_a not in referenced
        assert mid_b not in referenced


class TestSubjectKeyServerSideFilter:
    """subject_key is promoted to a top-level Vespa field, so get_all can
    filter same-subject peers server-side instead of pulling and filtering in
    Python (and missing peers past Mem0's limit)."""

    def test_get_all_subject_key_filter_returns_only_same_subject(
        self, manager_with_registry
    ):
        mm = manager_with_registry
        alpha = "subj:alpha_filter"
        beta = "subj:beta_filter"
        a1 = mm.add_memory(
            content="alpha subject fact",
            tenant_id=TENANT,
            agent_name=AGENT,
            metadata={"kind": "tenant_instruction", "subject_key": alpha},
            infer=False,
        )
        b1 = mm.add_memory(
            content="beta subject fact",
            tenant_id=TENANT,
            agent_name=AGENT,
            metadata={"kind": "tenant_instruction", "subject_key": beta},
            infer=False,
        )
        assert a1 and b1

        raw = mm.memory.get_all(user_id=TENANT, filters={"subject_key": alpha})
        rows = raw.get("results", []) if isinstance(raw, dict) else (raw or [])
        subjects = {(r.get("metadata") or {}).get("subject_key") for r in rows}

        # Server-side filter on the promoted field: only the alpha subject is
        # returned — the beta write is excluded by the backend, not in Python.
        assert subjects == {alpha}
        assert any(r.get("id") == a1 for r in rows)
