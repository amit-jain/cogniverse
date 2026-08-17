"""Document extractor — GLiNER entities anchored to segment Mentions.

For text-based documents (`.md`, `.txt`, `.rst`, `.html`, `.pdf` after
text extraction) and per-segment text inputs (transcripts, VLM
descriptions, OCR, etc.). GLiNER is already in the cogniverse stack via
the routing agent.

This extractor produces nodes only. SPO edges are produced by
``ClaimExtractor`` (DSPy ChainOfThought + RLM-promoted) — co-occurrence
"mentioned_with" edges have been removed.
"""

import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

from cogniverse_agents.graph.graph_schema import (
    Edge,
    ExtractionResult,
    Mention,
    Node,
)
from cogniverse_foundation.config.utils import get_config_manager_singleton

logger = logging.getLogger(__name__)


@dataclass
class SegmentEntities:
    """Pass-1 output for one segment of the two-pass extraction.

    ``nodes`` are the entity nodes; ``per_chunk_entity_names`` records, per text
    chunk, the entity names GLiNER found in that chunk — the exact per-chunk
    hints the claim pass feeds the ``ClaimExtractor``. Carrying it lets the
    entity pass run for every segment first (no cross-segment coreference
    dependency, so parallelisable) while the claim pass still reproduces the
    identical chunk-level hints.
    """

    nodes: List[Node] = field(default_factory=list)
    per_chunk_entity_names: List[List[str]] = field(default_factory=list)


_TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".html", ".htm"}
_PDF_EXTENSIONS = {".pdf"}

_DEFAULT_LABELS = [
    "Person",
    "Organization",
    "Location",
    "Date",
    "Substance",
    "Award",
    "Field",
    "Event",
    "Concept",
    "Technology",
    "Product",
    "Algorithm",
    "Model",
    "Framework",
    "Language",
]

# Pronouns and common stop-verbs that GLiNER occasionally emits as
# Concept/Person entities. Filtered out before nodes are created so the
# KG isn't polluted with "She", "He", "discovered", "made" etc.
_PRONOUN_BLOCKLIST = frozenset(
    {
        "he",
        "she",
        "it",
        "they",
        "we",
        "i",
        "you",
        "him",
        "her",
        "his",
        "hers",
        "its",
        "their",
        "theirs",
        "them",
        "us",
        "our",
        "ours",
        "my",
        "mine",
        "your",
        "yours",
        "this",
        "that",
        "these",
        "those",
    }
)
_COMMON_VERB_BLOCKLIST = frozenset(
    {
        "discovered",
        "made",
        "found",
        "created",
        "wrote",
        "won",
        "born",
        "died",
        "is",
        "was",
        "were",
        "are",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "said",
        "say",
        "says",
        "go",
        "goes",
        "went",
        "gone",
        "come",
        "came",
        "take",
        "took",
        "taken",
        "get",
        "got",
        "gotten",
        "give",
        "gave",
        "given",
        "see",
        "saw",
        "seen",
        "know",
        "knew",
        "known",
        "think",
        "thought",
        "show",
        "shown",
        "showed",
    }
)


# Adverbs/fillers GLiNER sometimes glues onto a verb to form a span like
# "later won" or "then discovered". On their own they're never entities;
# combined with a blocked verb the whole span is verb-phrase noise.
_ADVERB_BLOCKLIST = frozenset(
    {
        "later",
        "then",
        "also",
        "now",
        "soon",
        "again",
        "once",
        "ever",
        "never",
        "just",
        "still",
        "yet",
        "already",
        "only",
        "subsequently",
        "eventually",
        "recently",
    }
)

_NOISE_TOKENS = _PRONOUN_BLOCKLIST | _COMMON_VERB_BLOCKLIST | _ADVERB_BLOCKLIST


def _is_blocked_entity(name: str) -> bool:
    """Return True for entity candidates that carry no real noun content.

    Blocks bare pronouns/verbs ("She", "discovered") and multi-word spans
    whose every token is a pronoun, common verb, or adverb ("later won",
    "then discovered") — GLiNER emits these as Event/Concept entities but
    they pollute the KG.
    """
    lower = name.strip().lower()
    if lower in _PRONOUN_BLOCKLIST or lower in _COMMON_VERB_BLOCKLIST:
        return True
    tokens = lower.split()
    return len(tokens) > 1 and all(t in _NOISE_TOKENS for t in tokens)


_MAX_CHARS_PER_CHUNK = 2000
_MAX_EVIDENCE_CHARS = 200


def supported_extensions() -> Set[str]:
    return _TEXT_EXTENSIONS | _PDF_EXTENSIONS


class DocExtractor:
    """Extract entities (and via ClaimExtractor, SPO edges) from text segments."""

    def __init__(
        self,
        labels: Optional[List[str]] = None,
        claim_extractor: Optional["ClaimExtractorProtocol"] = None,
        gliner_inference_url: Optional[str] = None,
    ) -> None:
        self._labels = labels or list(_DEFAULT_LABELS)
        self._gliner = None
        self._gliner_failed = False
        self._gliner_load_lock = threading.Lock()
        self._claim_extractor = claim_extractor
        self._gliner_inference_url = gliner_inference_url

    def _get_gliner(self):
        """Lazily load the GLiNER model, caching the instance.

        Uses the explicitly injected GLiNER inference service URL when one
        was provided; otherwise resolves it from validated system
        configuration (``_discover_gliner_url``) so the slim runtime image
        routes through the GLiNER inference service instead of the heavy
        local gliner+torch stack. The canonical server is
        ``cogniverse_cli.modal_inference.servers.gliner``. Only when no URL
        is configured anywhere does the model load in-process.
        """
        if self._gliner is not None:
            return self._gliner
        if self._gliner_failed:
            raise RuntimeError("GLiNER model is unavailable after a failed load")
        with self._gliner_load_lock:
            if self._gliner is not None:
                return self._gliner
            if self._gliner_failed:
                raise RuntimeError("GLiNER model is unavailable after a failed load")
            from cogniverse_core.common.models import get_or_load_gliner

            inference_url = self._gliner_inference_url
            if inference_url is None:
                inference_url = self._discover_gliner_url()
            try:
                model = get_or_load_gliner(
                    "urchade/gliner_large-v2.1",
                    logger=logger,
                    inference_url=inference_url,
                )
            except Exception as exc:
                self._gliner_failed = True
                raise RuntimeError("GLiNER model failed to load") from exc
            if model is None:
                self._gliner_failed = True
                raise RuntimeError("GLiNER model is unavailable")
            self._gliner = model
            return model

    @staticmethod
    def _discover_gliner_url():
        """Return the GLiNER inference service URL from validated system configuration."""
        try:
            sys_cfg = get_config_manager_singleton().get_system_config()
        except Exception as exc:
            raise RuntimeError(
                "failed to resolve GLiNER inference service URL from system "
                "configuration"
            ) from exc
        return (sys_cfg.inference_service_urls or {}).get("gliner")

    def extract(
        self,
        file_path: Path,
        tenant_id: str,
        source_doc_id: str,
    ) -> Optional[ExtractionResult]:
        """Extract from a text or PDF file — produces a doc-level Mention anchor."""
        ext = file_path.suffix.lower()
        if ext not in supported_extensions():
            return None

        text = self._load_text(file_path, ext)
        if not text:
            return None

        file_anchor = Mention(
            source_doc_id=source_doc_id,
            segment_id="file",
            ts_start=0.0,
            ts_end=0.0,
            modality="document",
            evidence_span=_truncate(text, _MAX_EVIDENCE_CHARS),
        )
        return self.extract_from_text(text, tenant_id, source_doc_id, file_anchor)

    def extract_from_text(
        self,
        text: str,
        tenant_id: str,
        source_doc_id: str,
        segment_anchor: Mention,
        prior_entities: Optional[List[str]] = None,
    ) -> ExtractionResult:
        """Per-segment entity extraction. ``segment_anchor`` is required.

        ``prior_entities`` carries names already seen in earlier segments of
        the same ``source_doc_id`` so the ClaimExtractor can resolve
        pronoun-style coreferences (``"She later won the Nobel Prize."``
        binds ``She`` → ``Marie Curie`` when Marie Curie was extracted
        from an earlier segment).
        """
        return self._extract_from_text(
            text, tenant_id, source_doc_id, segment_anchor, prior_entities or []
        )

    def _load_text(self, file_path: Path, ext: str) -> str:
        if ext in _PDF_EXTENSIONS:
            return self._load_pdf(file_path)
        try:
            return file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _load_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF using PyPDF2 (already a runtime dep)."""
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(str(file_path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            logger.warning("PDF text extraction failed for %s: %s", file_path, exc)
            return ""

    def extract_entities_from_text(
        self,
        text: str,
        tenant_id: str,
        source_doc_id: str,
        segment_anchor: Mention,
    ) -> SegmentEntities:
        """Pass 1 of the two-pass extraction: entity nodes for one segment.

        Runs GLiNER (or the fallback) over the segment's chunks. Has NO
        cross-segment coreference dependency, so it is safe to run for every
        segment concurrently before any claim extraction begins.
        """
        return self._extract_entities(text, tenant_id, source_doc_id, segment_anchor)

    def extract_claims_from_text(
        self,
        text: str,
        segment_entities: SegmentEntities,
        prior_entities: Optional[List[str]],
        tenant_id: str,
        source_doc_id: str,
        segment_anchor: Mention,
    ) -> List[Edge]:
        """Pass 2 of the two-pass extraction: claim edges for one segment.

        Given ``segment_entities`` (this segment's Pass-1 output) and
        ``prior_entities`` (entity names from the other segments used for
        coreference), extracts SPO claim edges. With the prior pool precomputed,
        this too runs concurrently across segments.
        """
        return self._extract_claims(
            text,
            segment_entities.per_chunk_entity_names,
            prior_entities,
            tenant_id,
            source_doc_id,
            segment_anchor,
        )

    def _extract_from_text(
        self,
        text: str,
        tenant_id: str,
        source_doc_id: str,
        segment_anchor: Mention,
        prior_entities: Optional[List[str]] = None,
    ) -> ExtractionResult:
        # Serial composition of the two passes — the entity pass then the claim
        # pass over the same chunks. Byte-identical to running them interleaved
        # per chunk, since claim extraction for a chunk depends only on that
        # chunk's entities plus prior_entities, both available up front.
        ents = self._extract_entities(text, tenant_id, source_doc_id, segment_anchor)
        edges = self._extract_claims(
            text,
            ents.per_chunk_entity_names,
            prior_entities,
            tenant_id,
            source_doc_id,
            segment_anchor,
        )
        return ExtractionResult(
            source_doc_id=source_doc_id,
            nodes=ents.nodes,
            edges=edges,
        )

    def _extract_entities(
        self,
        text: str,
        tenant_id: str,
        source_doc_id: str,
        segment_anchor: Mention,
    ) -> SegmentEntities:
        gliner = self._get_gliner()
        if gliner is None:
            raise RuntimeError("GLiNER model is unavailable")
        nodes: List[Node] = []
        seen: Set[str] = set()
        per_chunk_entity_names: List[List[str]] = []
        gliner_chunks = 0
        gliner_failures = 0

        for chunk_index, chunk in enumerate(self._chunk_text(text), start=1):
            entities_in_chunk: List[Tuple[str, str]] = []

            gliner_chunks += 1
            try:
                # 0.3 chosen empirically against the production
                # gliner_large-v2.1 inference service: at 0.5 the model
                # silently drops named entities scoring well above
                # the threshold (observed against a real video
                # transcript — 'Bear Grylls' at 0.917 was returned
                # at 0.3 but not at 0.5). 0.3 preserves real
                # entities; pronoun + verb noise is filtered
                # downstream by _PRONOUN_BLOCKLIST + _COMMON_VERB_
                # BLOCKLIST.
                raw = gliner.predict_entities(chunk, self._labels, threshold=0.3)
                logger.info(
                    "GLiNER returned %d raw entities for chunk (len=%d): %s",
                    len(raw),
                    len(chunk),
                    [
                        (e.get("text"), e.get("label"), round(e.get("score", 0), 3))
                        for e in raw[:8]
                    ],
                )
                for ent in raw:
                    name = ent.get("text", "").strip()
                    label = ent.get("label", "Concept")
                    if not name or len(name) < 2:
                        continue
                    if _is_blocked_entity(name):
                        continue
                    entities_in_chunk.append((name, label))
                logger.info(
                    "After filtering: %d entities → %s",
                    len(entities_in_chunk),
                    entities_in_chunk[:8],
                )
            except Exception as exc:
                gliner_failures += 1
                raise RuntimeError(
                    f"GLiNER prediction failed for chunk {chunk_index} "
                    f"of source {source_doc_id!r}"
                ) from exc

            chunk_evidence = _truncate(chunk, _MAX_EVIDENCE_CHARS)
            for name, label in entities_in_chunk:
                normalized = name.strip()
                if normalized.lower() in seen:
                    continue
                seen.add(normalized.lower())
                nodes.append(
                    Node(
                        tenant_id=tenant_id,
                        name=normalized,
                        description=f"{label} mentioned in {source_doc_id}",
                        kind="concept",
                        label=label or "Concept",
                        mentions=[
                            Mention(
                                source_doc_id=segment_anchor.source_doc_id,
                                segment_id=segment_anchor.segment_id,
                                ts_start=segment_anchor.ts_start,
                                ts_end=segment_anchor.ts_end,
                                modality=segment_anchor.modality,
                                evidence_span=chunk_evidence,
                            )
                        ],
                    )
                )

            # Record the chunk's raw entity names (pre node-dedup), which are the
            # hints the claim pass merges with prior_entities for this chunk.
            per_chunk_entity_names.append([name for name, _ in entities_in_chunk])

        logger.debug(
            "GLiNER processed %d chunks with %d failures",
            gliner_chunks,
            gliner_failures,
        )

        return SegmentEntities(
            nodes=nodes, per_chunk_entity_names=per_chunk_entity_names
        )

    def _extract_claims(
        self,
        text: str,
        per_chunk_entity_names: List[List[str]],
        prior_entities: Optional[List[str]],
        tenant_id: str,
        source_doc_id: str,
        segment_anchor: Mention,
    ) -> List[Edge]:
        if self._claim_extractor is None:
            return []

        edges: List[Edge] = []
        prior = prior_entities or []

        for chunk_i, chunk in enumerate(self._chunk_text(text)):
            chunk_hints = (
                per_chunk_entity_names[chunk_i]
                if chunk_i < len(per_chunk_entity_names)
                else []
            )
            merged_hints: List[str] = []
            seen_hints: Set[str] = set()
            for n in chunk_hints + prior:
                if n.lower() not in seen_hints:
                    merged_hints.append(n)
                    seen_hints.add(n.lower())
            if merged_hints:
                claim_edges = self._claim_extractor.extract(
                    text=chunk,
                    entity_hints=merged_hints,
                    modality_hint=segment_anchor.modality,
                    segment_anchor=segment_anchor,
                    tenant_id=tenant_id,
                    source_doc_id=source_doc_id,
                )
                edges.extend(claim_edges)

        return edges

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into paragraph-aware chunks of at most _MAX_CHARS_PER_CHUNK."""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        chunks: List[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 > _MAX_CHARS_PER_CHUNK:
                if current:
                    chunks.append(current)
                if len(para) > _MAX_CHARS_PER_CHUNK:
                    for i in range(0, len(para), _MAX_CHARS_PER_CHUNK):
                        chunks.append(para[i : i + _MAX_CHARS_PER_CHUNK])
                    current = ""
                else:
                    current = para
            else:
                current = f"{current}\n\n{para}" if current else para
        if current:
            chunks.append(current)
        return chunks


class ClaimExtractorProtocol:
    """Structural protocol — ClaimExtractor satisfies this without inheriting."""

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
        raise NotImplementedError


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"
