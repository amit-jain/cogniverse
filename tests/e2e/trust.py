"""Trust scores a fresh memory write is expected to persist."""

from __future__ import annotations

from cogniverse_core.memory.provenance import DerivationKind


def expected_initial_trust(kind: str, derivation_kind: DerivationKind) -> float:
    """The trust score a fresh ``kind`` write under ``derivation_kind`` persists.

    Written out from the shipped schema defaults and derivation weights rather
    than imported from the product, so an e2e expectation cannot drift along
    with the table it is meant to pin. Unknown keys raise.
    """
    default_trust = {
        "conversation_turn": 0.4,
        "learned_strategy": 0.6,
        "tenant_instruction": 0.95,
        "external_doc": 0.7,
        "entity_fact": 0.5,
        "kg_node": 0.6,
        "kg_edge": 0.6,
        "session_scratch": 0.3,
    }[kind]
    weight = {
        DerivationKind.DIRECT_INGEST: 1.20,
        DerivationKind.USER_ASSERT: 1.10,
        DerivationKind.EXTRACTION: 1.00,
        DerivationKind.SUMMARIZATION: 0.90,
        DerivationKind.SYNTHESIS: 0.85,
        DerivationKind.AGENT_INFERENCE: 0.70,
    }[derivation_kind]
    return min(1.0, max(0.0, default_trust * weight))
