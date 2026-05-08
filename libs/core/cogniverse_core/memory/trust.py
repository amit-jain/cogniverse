"""Trust / source ranking (A.4) over the knowledge layer.

Each memory carries a ``trust_score`` derived at write time from:

  * the schema's ``default_trust``;
  * the provenance ``derivation_kind`` (direct ingest > synthesis > inference);
  * explicit user / admin endorsement (``apply_endorsement``);
  * slow age-based decay.

Retrieval ranking combines ``relevance × trust × confidence`` so that
two memories with the same semantic match still surface in trust order:
the one whose source we trust more wins.

Storage: trust lives inside ``metadata["trust"]`` as a small dict
(``score``, ``decayed_at``, ``endorsements`` count). Endorsements use the
delete-and-readd pattern (matching A.6 pinning + A.8 strategy decay)
since Mem0's ``update`` only changes content, not metadata.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from cogniverse_core.memory.provenance import Provenance
    from cogniverse_core.memory.schema import KnowledgeSchema

logger = logging.getLogger(__name__)


# Per-derivation-kind multipliers applied to the schema's default_trust.
# Designed so that:
#   direct_ingest (a primary source) > user_assert > extraction > summarization
#   > synthesis > agent_inference
# Multipliers are bounded such that the final score stays in [0.0, 1.0].
_DERIVATION_WEIGHTS: Dict[str, float] = {
    "direct_ingest": 1.20,
    "user_assert": 1.10,
    "extraction": 1.00,
    "summarization": 0.90,
    "synthesis": 0.85,
    "agent_inference": 0.70,
}

# Endorsement deltas (per_call) — additive on top of the existing score.
_ENDORSEMENT_DELTA = {
    "user": 0.05,
    "tenant_admin": 0.10,
    "org_admin": 0.20,
}

# Decay: a memory's trust loses this much per day above its initial value
# UNTIL it reaches the schema's default. Decay never pushes below default.
_DAILY_DECAY = 0.005

_MIN_SCORE = 0.0
_MAX_SCORE = 1.0


@dataclass(frozen=True)
class TrustRecord:
    """Serialisable trust state stored in memory metadata."""

    score: float
    initial_score: float  # baseline at write time; floor for decay
    decayed_at: str  # ISO-8601 of last decay computation
    endorsements: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "initial_score": self.initial_score,
            "decayed_at": self.decayed_at,
            "endorsements": self.endorsements,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrustRecord":
        return cls(
            score=float(d.get("score", 0.5)),
            initial_score=float(d.get("initial_score", d.get("score", 0.5))),
            decayed_at=str(d.get("decayed_at") or _now_iso()),
            endorsements=int(d.get("endorsements", 0)),
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(score: float) -> float:
    return max(_MIN_SCORE, min(_MAX_SCORE, score))


def compute_initial_trust(
    schema: "KnowledgeSchema",
    provenance: Optional["Provenance"] = None,
) -> TrustRecord:
    """Compute the initial trust score at write time.

    Combines the schema's ``default_trust`` with a per-derivation-kind
    weight from the provenance. Memories without provenance keep the
    schema default (clamped).
    """
    base = float(schema.default_trust)
    weight = 1.0
    if provenance is not None:
        kind = getattr(
            provenance.derivation_kind, "value", str(provenance.derivation_kind)
        )
        weight = _DERIVATION_WEIGHTS.get(kind, 1.0)
    score = _clamp(base * weight)
    return TrustRecord(
        score=score,
        initial_score=score,
        decayed_at=_now_iso(),
        endorsements=0,
    )


def attach_trust_to_metadata(
    metadata: Optional[Dict[str, Any]],
    trust: TrustRecord,
) -> Dict[str, Any]:
    """Merge a TrustRecord into a memory's metadata under the ``trust`` key."""
    out = dict(metadata or {})
    out["trust"] = trust.to_dict()
    return out


def extract_trust(memory: Dict[str, Any]) -> Optional[TrustRecord]:
    """Read a TrustRecord from a memory dict; tolerates JSON-string metadata."""
    meta = memory.get("metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except (ValueError, TypeError):
            return None
    if not isinstance(meta, dict):
        return None
    payload = meta.get("trust")
    if not isinstance(payload, dict):
        return None
    try:
        return TrustRecord.from_dict(payload)
    except (KeyError, ValueError):
        return None


def apply_endorsement(
    trust: TrustRecord,
    endorser: str,
) -> TrustRecord:
    """Return a new TrustRecord with the endorsement delta applied.

    ``endorser`` matches the keys in ``_ENDORSEMENT_DELTA``. Unknown roles
    raise ValueError so callers cannot silently miss an endorsement.
    """
    if endorser not in _ENDORSEMENT_DELTA:
        raise ValueError(
            f"unknown endorser role={endorser!r}; valid: {sorted(_ENDORSEMENT_DELTA)}"
        )
    new_score = _clamp(trust.score + _ENDORSEMENT_DELTA[endorser])
    return TrustRecord(
        score=new_score,
        initial_score=trust.initial_score,
        decayed_at=_now_iso(),
        endorsements=trust.endorsements + 1,
    )


def apply_decay(
    trust: TrustRecord,
    *,
    now: Optional[datetime] = None,
) -> TrustRecord:
    """Return a TrustRecord with age-based decay applied.

    The decay is bounded by ``initial_score`` — a memory never loses more
    trust than it had originally gained above the schema baseline.
    Endorsements raise the floor implicitly: each endorsement bumped the
    score above ``initial_score``, but ``initial_score`` itself stays as
    the post-write baseline. Callers that want endorsements to also raise
    the decay floor should set ``initial_score`` to the post-endorsement
    value via a follow-up call (left to higher-level orchestration).
    """
    now = now or datetime.now(timezone.utc)
    try:
        last = datetime.fromisoformat(trust.decayed_at.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return trust
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)

    age_days = max(0.0, (now - last).total_seconds() / 86400.0)
    if age_days <= 0:
        return trust

    decayed = trust.score - age_days * _DAILY_DECAY
    floor = trust.initial_score  # never drop below the post-write baseline
    decayed = max(floor, decayed)
    return TrustRecord(
        score=decayed,
        initial_score=trust.initial_score,
        decayed_at=now.isoformat(),
        endorsements=trust.endorsements,
    )


def rank_with_trust(
    memories: List[Dict[str, Any]],
    *,
    apply_decay_now: bool = True,
) -> List[Dict[str, Any]]:
    """Re-rank ``memories`` by ``relevance × trust × confidence``.

    The ``relevance`` is read from each memory's ``score`` (from semantic
    search), ``trust`` from the metadata trust record (default 0.5 when
    absent), and ``confidence`` from the metadata's provenance record
    (default 1.0 when absent so memories without provenance are not
    penalised twice).

    Memories without scores keep their original positional order at the
    bottom of the result list.
    """
    annotated = []
    for memory in memories:
        meta = memory.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (ValueError, TypeError):
                meta = {}

        relevance = float(memory.get("score", 0.0) or 0.0)
        trust_record = None
        trust_dict = meta.get("trust") if isinstance(meta, dict) else None
        if isinstance(trust_dict, dict):
            try:
                trust_record = TrustRecord.from_dict(trust_dict)
            except (KeyError, ValueError):
                trust_record = None
        if trust_record is None:
            trust_score = 0.5
        else:
            if apply_decay_now:
                trust_record = apply_decay(trust_record)
            trust_score = trust_record.score

        confidence = 1.0
        prov = meta.get("provenance") if isinstance(meta, dict) else None
        if isinstance(prov, dict):
            try:
                confidence = float(prov.get("confidence", 1.0) or 1.0)
            except (TypeError, ValueError):
                confidence = 1.0

        composite = relevance * trust_score * confidence
        annotated.append((composite, memory))

    annotated.sort(key=lambda pair: pair[0], reverse=True)
    return [m for _, m in annotated]
