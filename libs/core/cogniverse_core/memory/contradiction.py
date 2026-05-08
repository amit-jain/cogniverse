"""Contradiction detection + reconciliation (A.3).

Two memories about the same *subject* can disagree. The
``ContradictionDetector`` groups candidate memories by a
``subject_key`` carried in metadata. Memories that share a subject_key
but differ on content form a ``ConflictSet``. Downstream agents then
choose which side to surface based on the schema's
``contradiction_policy``:

  * ``latest_wins`` — return the most recently written memory
  * ``trust_ranked`` — return the highest-trust × confidence memory
  * ``preserve_both`` — return both with a ``disputed=True`` flag

Detection is content-based (cheap, no embedding lookup): two memories
with the same subject_key are treated as agreeing iff their normalised
content matches. Stronger semantic comparison can plug in later by
swapping ``_content_signature``.

Conflict sets are themselves first-class memories of kind
``conflict_set`` (sentinel agent_name ``_conflict_store``) so a future
ContradictionReconciliationAgent (C3.4) can read them without polluting
normal-agent search results — the same pattern PinService (A.6) uses.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from cogniverse_core.memory.schema import ContradictionPolicy
from cogniverse_core.memory.trust import extract_trust

logger = logging.getLogger(__name__)

CONFLICT_RECORD_KIND = "conflict_set"
CONFLICT_AGENT_NAME = "_conflict_store"


@dataclass(frozen=True)
class ConflictSet:
    """A group of memories that share a subject_key but disagree on content.

    All memory ids in ``conflicting_memory_ids`` share the same
    ``subject_key`` and represent distinct content signatures (per
    ``_content_signature``). The set's own id is what gets persisted as a
    sentinel ``conflict_set`` memory; that record carries pointers to the
    conflicting members so a reconciliation agent can fetch and resolve.
    """

    subject_key: str
    conflicting_memory_ids: List[str]
    detected_at: str  # ISO-8601 UTC

    def to_metadata_payload(self) -> Dict[str, Any]:
        return {
            "kind": CONFLICT_RECORD_KIND,
            "subject_key": self.subject_key,
            "conflicting_memory_ids": list(self.conflicting_memory_ids),
            "detected_at": self.detected_at,
        }

    def to_memory_content(self) -> str:
        """Human-readable line for the persisted conflict_set memory."""
        return (
            f"conflict_set: subject_key={self.subject_key} "
            f"members={','.join(self.conflicting_memory_ids)}"
        )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_signature(content: str) -> str:
    """Normalised content fingerprint for conflict detection.

    Strips whitespace + lowercases + sha1s the result. Two memories with
    the same fingerprint are considered to agree; different fingerprints
    on the same subject_key form a conflict.
    """
    if content is None:
        return ""
    normalised = re.sub(r"\s+", " ", str(content).strip().lower())
    return hashlib.sha1(normalised.encode("utf-8")).hexdigest()


def _read_metadata(memory: Dict[str, Any]) -> Dict[str, Any]:
    meta = memory.get("metadata") or {}
    if isinstance(meta, str):
        try:
            return json.loads(meta) or {}
        except (ValueError, TypeError):
            return {}
    return meta if isinstance(meta, dict) else {}


def _subject_key(memory: Dict[str, Any]) -> Optional[str]:
    meta = _read_metadata(memory)
    val = meta.get("subject_key")
    return str(val) if val else None


class ContradictionDetector:
    """Stateless contradiction detector.

    Grouped by ``metadata.subject_key`` from the candidate set; emits one
    ``ConflictSet`` per subject_key that has more than one distinct
    content signature.
    """

    def detect(self, candidates: Iterable[Dict[str, Any]]) -> List[ConflictSet]:
        """Return a list of conflict sets across the candidate memories.

        Memories without a ``subject_key`` are ignored — the detector has
        no way to know what they are claims *about*.
        """
        groups: Dict[str, Dict[str, List[str]]] = {}
        # subject_key -> {signature -> [memory_id, ...]}
        for memory in candidates:
            subject = _subject_key(memory)
            if not subject:
                continue
            mid = memory.get("id")
            if not mid:
                continue
            content = memory.get("memory") or memory.get("content") or ""
            sig = _content_signature(content)
            groups.setdefault(subject, {}).setdefault(sig, []).append(str(mid))

        conflicts: List[ConflictSet] = []
        for subject, by_sig in groups.items():
            if len(by_sig) <= 1:
                continue  # all members agree on content
            members = sorted({mid for ids in by_sig.values() for mid in ids})
            conflicts.append(
                ConflictSet(
                    subject_key=subject,
                    conflicting_memory_ids=members,
                    detected_at=_now_iso(),
                )
            )
        return conflicts


def reconcile(
    candidates: List[Dict[str, Any]],
    policy: ContradictionPolicy,
) -> List[Dict[str, Any]]:
    """Apply the policy to a candidate list and return the resolved view.

    Memories are grouped by ``subject_key`` (memories without one pass
    through unchanged). Within each subject group, the policy decides:

      * ``latest_wins`` → keep the highest ``created_at`` member;
      * ``trust_ranked`` → keep the highest ``trust × confidence`` member;
      * ``preserve_both`` → keep all members but tag each with
        ``metadata['disputed'] = True`` so callers / UI surface the
        disagreement.
    """
    by_subject: Dict[Optional[str], List[Dict[str, Any]]] = {}
    for m in candidates:
        by_subject.setdefault(_subject_key(m), []).append(m)

    resolved: List[Dict[str, Any]] = []
    for subject, members in by_subject.items():
        if subject is None or len(members) == 1:
            resolved.extend(members)
            continue

        # Distinct content signatures decide whether members truly conflict.
        sigs = {
            _content_signature(m.get("memory") or m.get("content") or "")
            for m in members
        }
        if len(sigs) <= 1:
            # All members agree — pick any one (preserve relevance order
            # by keeping the earliest in input order).
            resolved.append(members[0])
            continue

        if policy is ContradictionPolicy.LATEST_WINS:
            resolved.append(_pick_latest(members))
        elif policy is ContradictionPolicy.TRUST_RANKED:
            resolved.append(_pick_trust_ranked(members))
        elif policy is ContradictionPolicy.PRESERVE_BOTH:
            resolved.extend(_mark_disputed(members))
        else:
            # Unknown policy — fall back to latest_wins.
            logger.warning(
                "Unknown contradiction policy %r; defaulting to latest_wins",
                policy,
            )
            resolved.append(_pick_latest(members))
    return resolved


def _pick_latest(members: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _ts(m: Dict[str, Any]) -> str:
        meta = _read_metadata(m)
        ts = meta.get("created_at") or m.get("created_at") or ""
        return str(ts)

    return max(members, key=_ts)


def _pick_trust_ranked(members: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _score(m: Dict[str, Any]) -> float:
        trust = extract_trust(m)
        trust_score = trust.score if trust is not None else 0.5
        meta = _read_metadata(m)
        prov = meta.get("provenance") or {}
        confidence = 1.0
        if isinstance(prov, dict):
            try:
                confidence = float(prov.get("confidence", 1.0) or 1.0)
            except (TypeError, ValueError):
                confidence = 1.0
        return trust_score * confidence

    return max(members, key=_score)


def _mark_disputed(members: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for m in members:
        meta = dict(_read_metadata(m))
        meta["disputed"] = True
        out.append({**m, "metadata": meta})
    return out
