"""Unit tests for TrustRecord, decay, endorsement, retrieval ranking."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from cogniverse_core.memory.provenance import (
    CitationRef,
    DerivationKind,
    Provenance,
    attach_to_metadata,
    make_provenance,
)
from cogniverse_core.memory.schema import KnowledgeSchema
from cogniverse_core.memory.trust import (
    TrustRecord,
    apply_decay,
    apply_endorsement,
    attach_trust_to_metadata,
    compute_initial_trust,
    extract_trust,
    rank_with_trust,
)


def _prov(kind: DerivationKind, refs=None) -> Provenance:
    return make_provenance(
        written_by="agent:test",
        derivation_kind=kind,
        confidence=0.7,
        derived_from=refs or [CitationRef.external("https://wiki/x")],
    )


@pytest.mark.unit
class TestComputeInitialTrust:
    @pytest.mark.ci_fast
    def test_no_provenance_returns_schema_default(self):
        schema = KnowledgeSchema(kind="external_doc", default_trust=0.6)
        rec = compute_initial_trust(schema, provenance=None)
        assert rec.score == 0.6
        assert rec.initial_score == 0.6
        assert rec.endorsements == 0

    def test_direct_ingest_boosts_score(self):
        schema = KnowledgeSchema(kind="external_doc", default_trust=0.6)
        rec = compute_initial_trust(
            schema, provenance=_prov(DerivationKind.DIRECT_INGEST)
        )
        # 0.6 * 1.20 = 0.72 (clamped to <= 1.0).
        assert rec.score == pytest.approx(0.72, rel=1e-3)

    def test_agent_inference_lowers_score(self):
        schema = KnowledgeSchema(kind="entity_fact", default_trust=0.7)
        rec = compute_initial_trust(
            schema, provenance=_prov(DerivationKind.AGENT_INFERENCE)
        )
        # 0.7 * 0.70 = 0.49.
        assert rec.score == pytest.approx(0.49, rel=1e-3)

    def test_score_clamped_to_unit_interval(self):
        schema = KnowledgeSchema(kind="external_doc", default_trust=0.95)
        # 0.95 * 1.20 = 1.14 → clamped to 1.0.
        rec = compute_initial_trust(
            schema, provenance=_prov(DerivationKind.DIRECT_INGEST)
        )
        assert rec.score == 1.0


@pytest.mark.unit
class TestEndorsement:
    @pytest.mark.ci_fast
    def test_user_endorsement_adds_005(self):
        rec = TrustRecord(
            score=0.5, initial_score=0.5, decayed_at=_now_iso(), endorsements=0
        )
        out = apply_endorsement(rec, "user")
        assert out.score == pytest.approx(0.55, rel=1e-3)
        assert out.endorsements == 1
        assert out.initial_score == 0.5  # baseline unchanged

    def test_org_admin_endorsement_adds_020(self):
        rec = TrustRecord(
            score=0.5, initial_score=0.5, decayed_at=_now_iso(), endorsements=0
        )
        out = apply_endorsement(rec, "org_admin")
        assert out.score == pytest.approx(0.70, rel=1e-3)

    @pytest.mark.ci_fast
    def test_unknown_role_rejected(self):
        rec = TrustRecord(
            score=0.5, initial_score=0.5, decayed_at=_now_iso(), endorsements=0
        )
        with pytest.raises(ValueError, match="unknown endorser"):
            apply_endorsement(rec, "stranger")

    def test_clamps_to_one(self):
        rec = TrustRecord(
            score=0.95, initial_score=0.5, decayed_at=_now_iso(), endorsements=0
        )
        out = apply_endorsement(rec, "org_admin")
        assert out.score == 1.0


@pytest.mark.unit
class TestDecay:
    @pytest.mark.ci_fast
    def test_decay_reduces_score_above_initial(self):
        # Endorsed score 0.8, initial baseline 0.5; after 10 days expect
        # 0.8 - 10*0.005 = 0.75.
        rec = TrustRecord(
            score=0.8,
            initial_score=0.5,
            decayed_at=(datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
            endorsements=2,
        )
        out = apply_decay(rec)
        assert out.score == pytest.approx(0.75, rel=1e-3)

    def test_decay_floor_is_initial_score(self):
        # 100 days -> would drop by 0.5 to 0.3, but floor is 0.5 (initial).
        rec = TrustRecord(
            score=0.8,
            initial_score=0.5,
            decayed_at=(datetime.now(timezone.utc) - timedelta(days=100)).isoformat(),
            endorsements=2,
        )
        out = apply_decay(rec)
        assert out.score == 0.5

    def test_decay_is_no_op_at_or_before_decayed_at(self):
        rec = TrustRecord(
            score=0.7,
            initial_score=0.5,
            decayed_at=datetime.now(timezone.utc).isoformat(),
            endorsements=1,
        )
        out = apply_decay(rec, now=datetime.now(timezone.utc))
        assert out.score == pytest.approx(0.7, rel=1e-9)


class TestAttachExtract:
    def test_attach_preserves_other_metadata(self):
        rec = TrustRecord(
            score=0.7, initial_score=0.7, decayed_at=_now_iso(), endorsements=0
        )
        meta = attach_trust_to_metadata({"kind": "entity_fact"}, rec)
        assert meta["kind"] == "entity_fact"
        assert meta["trust"]["score"] == 0.7

    def test_extract_round_trip(self):
        rec = TrustRecord(
            score=0.7, initial_score=0.6, decayed_at=_now_iso(), endorsements=2
        )
        meta = attach_trust_to_metadata({}, rec)
        memory = {"id": "m1", "metadata": meta}
        out = extract_trust(memory)
        assert out == rec

    def test_extract_from_json_string_metadata(self):
        rec = TrustRecord(
            score=0.7, initial_score=0.6, decayed_at=_now_iso(), endorsements=2
        )
        meta = json.dumps(attach_trust_to_metadata({}, rec))
        memory = {"id": "m1", "metadata": meta}
        out = extract_trust(memory)
        assert out is not None
        assert out.score == 0.7

    def test_extract_returns_none_when_absent(self):
        memory = {"id": "m1", "metadata": {"kind": "x"}}
        assert extract_trust(memory) is None


@pytest.mark.unit
class TestRankWithTrust:
    def _seed(self, score, trust_score, confidence=1.0):
        prov = make_provenance(
            written_by="agent:x",
            derivation_kind=DerivationKind.SYNTHESIS,
            confidence=confidence,
            derived_from=[CitationRef.external("https://wiki/x")],
        )
        meta = attach_to_metadata({"kind": "external_doc"}, prov)
        meta = attach_trust_to_metadata(
            meta,
            TrustRecord(
                score=trust_score,
                initial_score=trust_score,
                decayed_at=_now_iso(),
                endorsements=0,
            ),
        )
        return {
            "id": f"m_{trust_score}_{score}",
            "memory": "x",
            "score": score,
            "metadata": meta,
        }

    @pytest.mark.ci_fast
    def test_higher_trust_outranks_higher_relevance_when_close(self):
        # Two memories with similar relevance but one has much higher trust.
        low_trust_high_relevance = self._seed(score=0.65, trust_score=0.4)
        high_trust_lower_relevance = self._seed(score=0.55, trust_score=0.95)

        ranked = rank_with_trust(
            [low_trust_high_relevance, high_trust_lower_relevance],
            apply_decay_now=False,
        )
        # high_trust * conf=0.7: 0.55 * 0.95 * 0.7 = 0.366
        # low_trust * conf=0.7: 0.65 * 0.4 * 0.7 = 0.182
        # high_trust must rank first.
        assert ranked[0]["id"] == high_trust_lower_relevance["id"]

    def test_confidence_factor_applied(self):
        # Same trust + relevance; confidence breaks the tie.
        a = self._seed(score=0.6, trust_score=0.6, confidence=1.0)
        b = self._seed(score=0.6, trust_score=0.6, confidence=0.5)
        ranked = rank_with_trust([b, a], apply_decay_now=False)
        assert ranked[0]["id"] == a["id"]

    def test_missing_trust_treated_as_neutral(self):
        # No trust metadata; default 0.5 used.
        no_trust = {
            "id": "m_no_trust",
            "memory": "x",
            "score": 0.6,
            "metadata": {"kind": "external_doc"},  # no trust key
        }
        with_trust = self._seed(score=0.6, trust_score=0.9)
        ranked = rank_with_trust([no_trust, with_trust], apply_decay_now=False)
        assert ranked[0]["id"] == with_trust["id"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
