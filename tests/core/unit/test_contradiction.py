"""Unit tests for A.3 — ContradictionDetector + reconciliation policies."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from cogniverse_core.memory.contradiction import (
    CONFLICT_AGENT_NAME,
    CONFLICT_RECORD_KIND,
    ConflictSet,
    ContradictionDetector,
    _content_signature,
    reconcile,
)
from cogniverse_core.memory.schema import ContradictionPolicy
from cogniverse_core.memory.trust import (
    TrustRecord,
    attach_trust_to_metadata,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed(
    mid: str,
    content: str,
    subject_key: str,
    *,
    created_at: Optional[str] = None,
    trust_score: float = 0.5,
    confidence: float = 1.0,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "kind": "entity_fact",
        "subject_key": subject_key,
        "created_at": created_at or _now_iso(),
        "provenance": {
            "written_by": "agent:test",
            "written_at": _now_iso(),
            "derivation_kind": "synthesis",
            "confidence": confidence,
            "derived_from": [],
        },
    }
    meta = attach_trust_to_metadata(
        meta,
        TrustRecord(
            score=trust_score,
            initial_score=trust_score,
            decayed_at=_now_iso(),
            endorsements=0,
        ),
    )
    return {"id": mid, "memory": content, "metadata": meta}


class TestContentSignature:
    def test_whitespace_normalised(self):
        a = _content_signature("Paris is the capital of France")
        b = _content_signature("  paris  is\tthe\ncapital   of france  ")
        assert a == b

    def test_different_content_different_signature(self):
        assert _content_signature("X is Y") != _content_signature("X is Z")


class TestContradictionDetector:
    def test_no_conflict_when_all_members_agree(self):
        memories = [
            _seed("m1", "Paris is the capital of France", subject_key="france:capital"),
            _seed("m2", "paris is the capital of france", subject_key="france:capital"),
        ]
        conflicts = ContradictionDetector().detect(memories)
        assert conflicts == []

    def test_conflict_when_two_members_disagree(self):
        memories = [
            _seed("m1", "Paris is the capital of France", subject_key="france:capital"),
            _seed("m2", "Lyon is the capital of France", subject_key="france:capital"),
        ]
        conflicts = ContradictionDetector().detect(memories)
        assert len(conflicts) == 1
        cs = conflicts[0]
        assert cs.subject_key == "france:capital"
        assert sorted(cs.conflicting_memory_ids) == ["m1", "m2"]

    def test_three_distinct_signatures_one_conflict_set(self):
        memories = [
            _seed("m1", "Paris", subject_key="france:capital"),
            _seed("m2", "Lyon", subject_key="france:capital"),
            _seed("m3", "Marseille", subject_key="france:capital"),
        ]
        conflicts = ContradictionDetector().detect(memories)
        assert len(conflicts) == 1
        assert sorted(conflicts[0].conflicting_memory_ids) == ["m1", "m2", "m3"]

    def test_separate_subjects_separate_conflict_sets(self):
        memories = [
            _seed("m1", "Paris", subject_key="france:capital"),
            _seed("m2", "Lyon", subject_key="france:capital"),
            _seed("m3", "Berlin", subject_key="germany:capital"),
            _seed("m4", "Munich", subject_key="germany:capital"),
        ]
        conflicts = ContradictionDetector().detect(memories)
        assert len(conflicts) == 2
        keys = {c.subject_key for c in conflicts}
        assert keys == {"france:capital", "germany:capital"}

    def test_memories_without_subject_key_ignored(self):
        memories = [
            _seed("m1", "x", subject_key="france:capital"),
            {"id": "m_unsubject", "memory": "anything", "metadata": {"kind": "x"}},
        ]
        conflicts = ContradictionDetector().detect(memories)
        assert conflicts == []

    def test_conflict_set_to_metadata_payload_round_trip(self):
        cs = ConflictSet(
            subject_key="x:y",
            conflicting_memory_ids=["m1", "m2"],
            detected_at=_now_iso(),
        )
        payload = cs.to_metadata_payload()
        assert payload["kind"] == CONFLICT_RECORD_KIND
        assert payload["subject_key"] == "x:y"
        assert payload["conflicting_memory_ids"] == ["m1", "m2"]


class TestReconcileLatestWins:
    def test_latest_wins_picks_most_recent(self):
        old_ts = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        new_ts = datetime.now(timezone.utc).isoformat()
        memories = [
            _seed("m_old", "Lyon", subject_key="france:capital", created_at=old_ts),
            _seed("m_new", "Paris", subject_key="france:capital", created_at=new_ts),
        ]
        out = reconcile(memories, ContradictionPolicy.LATEST_WINS)
        assert len(out) == 1
        assert out[0]["id"] == "m_new"

    def test_no_conflict_passes_through(self):
        memories = [
            _seed("m1", "alpha", subject_key="topic:a"),
            _seed("m2", "beta", subject_key="topic:b"),
        ]
        out = reconcile(memories, ContradictionPolicy.LATEST_WINS)
        assert {m["id"] for m in out} == {"m1", "m2"}

    def test_memories_without_subject_pass_through(self):
        plain = {"id": "m_x", "memory": "hi", "metadata": {"kind": "x"}}
        out = reconcile([plain], ContradictionPolicy.LATEST_WINS)
        assert out == [plain]


class TestReconcileTrustRanked:
    def test_higher_trust_wins(self):
        memories = [
            _seed(
                "m_low",
                "Lyon",
                subject_key="france:capital",
                trust_score=0.4,
                confidence=0.9,
            ),
            _seed(
                "m_high",
                "Paris",
                subject_key="france:capital",
                trust_score=0.95,
                confidence=0.9,
            ),
        ]
        out = reconcile(memories, ContradictionPolicy.TRUST_RANKED)
        assert out[0]["id"] == "m_high"

    def test_confidence_factor_applied(self):
        # Same trust, but one has higher provenance confidence.
        memories = [
            _seed(
                "m_a",
                "Lyon",
                subject_key="france:capital",
                trust_score=0.7,
                confidence=0.3,
            ),
            _seed(
                "m_b",
                "Paris",
                subject_key="france:capital",
                trust_score=0.7,
                confidence=0.95,
            ),
        ]
        out = reconcile(memories, ContradictionPolicy.TRUST_RANKED)
        assert out[0]["id"] == "m_b"


class TestReconcilePreserveBoth:
    def test_preserve_both_returns_all_with_disputed_flag(self):
        memories = [
            _seed("m1", "Lyon", subject_key="france:capital"),
            _seed("m2", "Paris", subject_key="france:capital"),
        ]
        out = reconcile(memories, ContradictionPolicy.PRESERVE_BOTH)
        assert {m["id"] for m in out} == {"m1", "m2"}
        for m in out:
            assert m["metadata"]["disputed"] is True

    def test_preserve_both_does_not_mark_singletons(self):
        memories = [_seed("m1", "Paris", subject_key="france:capital")]
        out = reconcile(memories, ContradictionPolicy.PRESERVE_BOTH)
        assert len(out) == 1
        assert out[0]["metadata"].get("disputed") is None


class TestSentinelConstants:
    def test_sentinel_kind_and_agent_name(self):
        # Pin matches A.6 pattern: a sentinel agent_name keeps these
        # records out of normal-agent search results.
        assert CONFLICT_RECORD_KIND == "conflict_set"
        assert CONFLICT_AGENT_NAME == "_conflict_store"
