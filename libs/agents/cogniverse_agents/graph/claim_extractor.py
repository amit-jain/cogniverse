"""ClaimExtractor — DSPy ChainOfThought / InstrumentedRLM over the
ClaimExtractionSignature, producing real SPO Edge objects anchored to
a per-segment Mention.

Replaces the old co-occurrence "mentioned_with" edges from DocExtractor.
Compiled DSPy state persists as a JSON blob under ``("model",
"claim_extraction")`` via ``ArtifactManager.save_blob`` and is restored
with ``ArtifactManager.load_blob`` + ``module.load_state``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
from typing import Any, List, Optional

import dspy

from cogniverse_agents._confidence import parse_confidence
from cogniverse_agents.graph.dspy_signatures import ClaimExtractionSignature
from cogniverse_agents.graph.graph_schema import Edge, Mention
from cogniverse_core.common.utils.async_bridge import run_coro_blocking

logger = logging.getLogger(__name__)

# Threshold above which the input is routed through the recursive LM
# path instead of a single ChainOfThought call. Tuned for typical
# transcript-segment sizes; long PDF / code chunks trip this.
RLM_PROMOTION_TOKENS = 3000

# Claim output budget: 4 claims * 80 tokens/claim + 192 reasoning = 512
# tokens. 512 / 11 tok/s ≈ 46.5s, which leaves margin inside the 120s
# request timeout while still failing fast on runaway completions.
CLAIM_EXTRACTION_MAX_CLAIMS = 4
CLAIM_EXTRACTION_TOKENS_PER_CLAIM = 80
CLAIM_EXTRACTION_REASONING_TOKENS = 192
CLAIM_EXTRACTION_MAX_OUTPUT_TOKENS = (
    CLAIM_EXTRACTION_MAX_CLAIMS * CLAIM_EXTRACTION_TOKENS_PER_CLAIM
    + CLAIM_EXTRACTION_REASONING_TOKENS
)

# Hard cap on the verbatim evidence_span length stored on each Edge.
_MAX_EVIDENCE_CHARS = 200

# The locked predicate vocabulary. Edges whose ``relation`` does not
# fall into this set after normalization are dropped — the KG only
# carries claims expressed in the canonical 16 relations. Loose LLM
# outputs like ``yellow``, ``in``, ``glass``, ``feels``, ``looks_like``
# get filtered here instead of leaking into the graph as one-off edges.
PREDICATE_VOCABULARY = frozenset(
    {
        "born_in",
        "discovered",
        "discovered_in",
        "located_at",
        "won",
        "worked_at",
        "wrote",
        "invented",
        "studied",
        "contains",
        "occurred_at",
        "part_of",
        "preceded_by",
        "followed_by",
        "caused_by",
        "contradicts",
    }
)

# Map the most common LLM-emitted predicate variants into the locked
# 16-element vocabulary. The LM (especially small models like
# gemma-4-e4b-it) tends to copy verb tense and prepositions from the
# source text — "was born in" → "was_born_in" rather than the canonical
# "born_in". Without normalization the KG is fractured across surface
# variants of the same relation.
_PREDICATE_ALIASES = {
    "was_born_in": "born_in",
    "is_born_in": "born_in",
    "born_at": "born_in",
    "was_born_at": "born_in",
    "was_discovered_by": "discovered",
    "is_discovered_by": "discovered",
    "was_discovered_in": "discovered_in",
    "discovered_at": "worked_at",
    "discovered_during": "discovered_in",
    "was_invented_by": "invented",
    "was_written_by": "wrote",
    "wrote_by": "wrote",
    "was_won_by": "won",
    "won_by": "won",
    "was_studied_by": "studied",
    "is_located_at": "located_at",
    "was_located_at": "located_at",
    "is_at": "located_at",
    "located_in": "located_at",
    "is_part_of": "part_of",
    "was_part_of": "part_of",
    "contained_in": "part_of",
    "containing": "contains",
    "occurred_in": "occurred_at",
    "took_place_at": "occurred_at",
    "took_place_in": "occurred_at",
    "was_preceded_by": "preceded_by",
    "was_followed_by": "followed_by",
    "caused": "caused_by",
    "was_caused_by": "caused_by",
    "contradicted_by": "contradicts",
}


# Pronouns whose subject form starts a sentence. Lowercase comparison;
# substitution preserves the original casing of the antecedent.
_LEADING_PRONOUNS = ("she", "he", "they", "it")


def _resolve_leading_pronoun(text: str, entity_hints: List[str]) -> str:
    """Replace a leading subject pronoun with the first multi-word
    capitalized name in ``entity_hints`` that doesn't already appear in
    the text (heuristic antecedent — usually a Person introduced in an
    earlier segment).

    Conservative: only fires when the chunk starts with a pronoun AND
    a plausible Person-shaped name from a PRIOR segment is in the hints.
    Picking an antecedent already present in the text would substitute
    nonsense like "Nobel Prize later won the Nobel Prize in Physics."
    """
    if not text or not entity_hints:
        return text
    stripped = text.lstrip()
    if not stripped:
        return text
    first_word = stripped.split()[0].rstrip(",.;:!?").lower()
    if first_word not in _LEADING_PRONOUNS:
        return text
    text_lower = text.lower()
    antecedent = next(
        (
            h
            for h in entity_hints
            if " " in h and h[0].isupper() and h.lower() not in text_lower
        ),
        None,
    )
    if antecedent is None:
        return text
    leading_ws_len = len(text) - len(stripped)
    return text[:leading_ws_len] + antecedent + stripped[len(stripped.split()[0]) :]


def _normalize_predicate(raw: str) -> str:
    """Map LLM-emitted predicate variants to the locked vocabulary."""
    p = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if p in _PREDICATE_ALIASES:
        return _PREDICATE_ALIASES[p]
    # Strip a leading "was_" / "is_" / "were_" / "are_" auxiliary that the
    # LM copies from the source text but the canonical predicate omits.
    for prefix in ("was_", "is_", "were_", "are_", "has_been_", "have_been_"):
        if p.startswith(prefix):
            stripped = p[len(prefix) :]
            if stripped in _PREDICATE_ALIASES:
                return _PREDICATE_ALIASES[stripped]
            return stripped
    return p


class ClaimExtractor:
    """Extract SPO edges from a text segment using a compiled DSPy module."""

    def __init__(
        self,
        artifact_manager: Optional[Any] = None,
        rlm_promotion_chars: int = RLM_PROMOTION_TOKENS,
        llm_config: Optional[Any] = None,
        config_manager: Optional[Any] = None,
    ) -> None:
        self._artifact_manager = artifact_manager
        self._rlm_promotion_chars = rlm_promotion_chars
        self._cot_module: Optional[dspy.ChainOfThought] = None
        self._rlm_module: Optional[dspy.RLM] = None
        # Guards the lazy module build below: the per-segment KG claim pass
        # invokes one shared extractor from several threads at once, so an
        # unguarded ``if self._cot_module is None`` would double-build the
        # module (and double-load its compiled state) on the first touch.
        self._module_lock = threading.Lock()
        # When set, every module invocation runs inside ``dspy.context(lm=...)``
        # bound from this config. None means the call falls through to the
        # ambient ``dspy.settings.lm`` (the worker-startup default).
        # The signature emits reasoning + claims JSON, with claims as the final
        # output field. The derived 512-token cap keeps the response short
        # enough to fit inside the 120s timeout at the measured ~11 tok/s.
        if llm_config is not None and dataclasses.is_dataclass(llm_config):
            llm_config = dataclasses.replace(
                llm_config, max_tokens=CLAIM_EXTRACTION_MAX_OUTPUT_TOKENS
            )
        # Claim extraction is a structured task: sampling temperatures let the
        # model mis-attribute claim subjects (a 4B model at 0.1 swaps the SPO
        # subject on real transcript segments), so this path decodes greedily
        # regardless of the tenant's conversational temperature.
        current_temperature = getattr(llm_config, "temperature", None)
        if (
            llm_config is not None
            and dataclasses.is_dataclass(llm_config)
            and isinstance(current_temperature, (int, float))
            and current_temperature != 0.0
        ):
            llm_config = dataclasses.replace(llm_config, temperature=0.0)
        self._llm_config = llm_config
        # Needed to resolve semantic routing per request; None keeps the direct
        # (direct) path.
        self._config_manager = config_manager

    def extract(
        self,
        *,
        text: str,
        entity_hints: List[str],
        modality_hint: str,
        segment_anchor: Mention,
        tenant_id: str,
        source_doc_id: str,
    ) -> List[Edge]:
        """Run the compiled claim extractor and return Edge objects."""
        if not text.strip():
            return []

        try:
            prediction = self._invoke(
                text=text,
                entity_hints=entity_hints,
                modality_hint=modality_hint,
                tenant_id=tenant_id,
                source_doc_id=source_doc_id,
                segment_anchor=segment_anchor,
            )
            claims = self._coerce_claims(prediction)
        except Exception as exc:
            raise RuntimeError(
                f"Claim extraction failed for source {source_doc_id!r} "
                f"segment {segment_anchor.segment_id!r}: {exc}"
            ) from exc
        return self._claims_to_edges(
            claims=claims,
            segment_anchor=segment_anchor,
            tenant_id=tenant_id,
            source_doc_id=source_doc_id,
            full_text=text,
        )

    def _invoke(
        self,
        *,
        text: str,
        entity_hints: List[str],
        modality_hint: str,
        tenant_id: str,
        source_doc_id: str,
        segment_anchor: Mention,
    ) -> dspy.Prediction:
        module = self._select_module(text=text, tenant_id=tenant_id)
        # Substitute leading subject pronouns with the most plausible
        # prior entity. Small LMs (gemma, Llama-3-8B) don't reliably
        # resolve coreference even when given the entity list — but they
        # do extract claims correctly when the explicit name is in the
        # surface text. The first multi-word capitalized name in
        # ``entity_hints`` is the heuristic antecedent (usually a Person).
        text_for_lm = _resolve_leading_pronoun(text, entity_hints)
        if self._llm_config is not None:
            from cogniverse_foundation.config.semantic_router import (
                ingest_lm_context_for,
            )

            with ingest_lm_context_for(self._llm_config):
                prediction = module(
                    text_segment=text_for_lm,
                    entity_hints=entity_hints,
                    modality_hint=modality_hint,
                )
                self._raise_if_length_truncated(
                    dspy.settings.lm,
                    source_doc_id=source_doc_id,
                    segment_anchor=segment_anchor,
                )
                return prediction
        prediction = module(
            text_segment=text_for_lm,
            entity_hints=entity_hints,
            modality_hint=modality_hint,
        )
        self._raise_if_length_truncated(
            dspy.settings.lm,
            source_doc_id=source_doc_id,
            segment_anchor=segment_anchor,
        )
        return prediction

    def _select_module(self, *, text: str, tenant_id: str):
        """Pick ChainOfThought for short text, RLM for long text.

        The build is double-checked-locked and only publishes the module after
        its compiled state is loaded, so concurrent first-touches build it once
        and no thread ever sees a half-loaded module.
        """
        if len(text) > self._rlm_promotion_chars:
            if self._rlm_module is None:
                with self._module_lock:
                    if self._rlm_module is None:
                        # Bound the REPL output folded into follow-up prompts:
                        # the dspy default (100k chars) packs the next request
                        # to the serving window's boundary, which 400s on
                        # smaller-window deployments (the server's tokenizer
                        # counts a token or two more than the client's trim
                        # math reserved).
                        module = dspy.RLM(
                            ClaimExtractionSignature, max_output_chars=16_000
                        )
                        self._load_compiled_state(module, tenant_id)
                        self._rlm_module = module
            return self._rlm_module

        if self._cot_module is None:
            with self._module_lock:
                if self._cot_module is None:
                    module = dspy.ChainOfThought(ClaimExtractionSignature)
                    self._load_compiled_state(module, tenant_id)
                    self._cot_module = module
        return self._cot_module

    def _load_compiled_state(self, module: dspy.Module, tenant_id: str) -> None:
        """Restore compiled DSPy state from the ArtifactManager if present.

        Mirrors EntityExtractionAgent / QueryEnhancementAgent: the compiled
        module state is persisted as a JSON blob under
        ``("model", "claim_extraction")`` and restored via
        ``module.load_state``. The ArtifactManager is already bound to its
        tenant at construction, so ``tenant_id`` here is only for logging.
        """
        if self._artifact_manager is None:
            return
        try:
            blob = run_coro_blocking(
                self._artifact_manager.load_blob("model", "claim_extraction")
            )
            if not blob:
                return
            module.load_state(json.loads(blob))
            logger.info("ClaimExtractor loaded compiled state for tenant %s", tenant_id)
        except Exception as exc:
            logger.warning(
                "No claim_extraction artifact loaded for tenant %s (using defaults): %s",
                tenant_id,
                exc,
            )

    @staticmethod
    def _coerce_claims(prediction: dspy.Prediction) -> List[dict]:
        """Normalize the LM output to a list of claim dicts.

        The model either returns an empty list or a list of dicts. Any
        missing or non-list claims payload is a failed completion.
        """
        claims = getattr(prediction, "claims", None)
        if claims is None:
            raise RuntimeError("LM response omitted claims")
        if isinstance(claims, list):
            if any(not isinstance(c, dict) for c in claims):
                raise RuntimeError("LM claims output must be a list of dicts")
            return claims
        raise RuntimeError(
            f"LM claims output must be a list, got {type(claims).__name__}"
        )

    @staticmethod
    def _response_finish_reason(lm: object) -> Optional[str]:
        """Return the finish reason for the latest LM response, when available."""
        history = getattr(lm, "history", None)
        if not history:
            return None
        entry = history[-1]
        if not isinstance(entry, dict):
            return None
        response = entry.get("response")
        if response is None:
            return None
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            return None
        for choice in choices:
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason is None and isinstance(choice, dict):
                finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                return finish_reason
        return None

    @classmethod
    def _raise_if_length_truncated(
        cls,
        lm: object,
        *,
        source_doc_id: str,
        segment_anchor: Mention,
    ) -> None:
        finish_reason = cls._response_finish_reason(lm)
        if finish_reason == "length":
            raise RuntimeError(
                f"LM response hit max_tokens for source {source_doc_id!r} "
                f"segment {segment_anchor.segment_id!r}"
            )

    @staticmethod
    def _claim_field_text(value: object) -> str:
        """LMs emit JSON scalars where the signature asks for strings
        ("object": 1867 for born_in). Numbers carry meaning — keep their
        text form. Structured values (list/dict/bool/None) have no usable
        node text — map to "" so only that claim drops."""
        if isinstance(value, str):
            return value.strip()
        if value is None or isinstance(value, bool):
            return ""
        if isinstance(value, (int, float)):
            return str(value)
        return ""

    def _claims_to_edges(
        self,
        *,
        claims: List[dict],
        segment_anchor: Mention,
        tenant_id: str,
        source_doc_id: str,
        full_text: str,
    ) -> List[Edge]:
        edges: List[Edge] = []
        for claim in claims:
            subject = self._claim_field_text(claim.get("subject"))
            predicate_raw = self._claim_field_text(claim.get("predicate"))
            obj = self._claim_field_text(claim.get("object"))
            if not subject or not predicate_raw or not obj:
                continue
            predicate = _normalize_predicate(predicate_raw)
            if predicate not in PREDICATE_VOCABULARY:
                # Drop edges whose predicate falls outside the locked
                # vocabulary. The LM occasionally invents one-off relations
                # ("yellow", "in", "glass", "feels", "looks_like") on
                # content that doesn't carry the canonical predicates.
                # Keeping them would fracture the KG.
                continue

            evidence = self._claim_field_text(claim.get("evidence_span"))
            if not evidence or evidence not in full_text:
                evidence = full_text[:_MAX_EVIDENCE_CHARS]
            evidence = evidence[:_MAX_EVIDENCE_CHARS]

            confidence = parse_confidence(claim.get("confidence"), default=1.0)
            edges.append(
                Edge(
                    tenant_id=tenant_id,
                    source=subject,
                    target=obj,
                    relation=predicate,
                    evidence_span=evidence,
                    segment_id=segment_anchor.segment_id,
                    ts_start=segment_anchor.ts_start,
                    ts_end=segment_anchor.ts_end,
                    modality=segment_anchor.modality,
                    provenance="EXTRACTED",
                    source_doc_id=source_doc_id,
                    confidence=confidence,
                )
            )
        return edges
