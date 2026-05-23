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
from pathlib import Path
from typing import List, Optional, Set, Tuple

from cogniverse_agents.graph.graph_schema import (
    Edge,
    ExtractionResult,
    Mention,
    Node,
)

logger = logging.getLogger(__name__)

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


def _is_blocked_entity(name: str) -> bool:
    """Return True for pronouns and common verbs that aren't real entities."""
    lower = name.strip().lower()
    return lower in _PRONOUN_BLOCKLIST or lower in _COMMON_VERB_BLOCKLIST


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
    ) -> None:
        self._labels = labels or list(_DEFAULT_LABELS)
        self._gliner = None
        self._gliner_failed = False
        self._claim_extractor = claim_extractor

    def _get_gliner(self):
        """Lazily load the GLiNER model, caching the instance.

        Prefers the deployed ``deploy/gliner`` sidecar URL when one is
        registered in ``SystemConfig.inference_service_urls`` so the
        runtime image doesn't need the heavy local gliner+torch stack.
        Falls back to a local-load (which usually fails in the slim
        production image) only when the env doesn't expose the URL.
        """
        if self._gliner is not None:
            return self._gliner
        if self._gliner_failed:
            return None
        from cogniverse_core.common.models import get_or_load_gliner

        inference_url = self._discover_gliner_url()
        self._gliner = get_or_load_gliner(
            "urchade/gliner_large-v2.1",
            logger=logger,
            inference_url=inference_url,
        )
        if self._gliner is None:
            self._gliner_failed = True
        return self._gliner

    @staticmethod
    def _discover_gliner_url():
        """Return the GLiNER sidecar URL from SystemConfig or env."""
        import os as _os

        try:
            from cogniverse_foundation.config.utils import (  # noqa: PLC0415
                get_config_manager_singleton,
            )

            sys_cfg = get_config_manager_singleton().get_system_config()
            url = (sys_cfg.inference_service_urls or {}).get("gliner")
            if url:
                return url
        except Exception as exc:  # noqa: BLE001 — log + degrade
            # Fall through to env / None — the loader handles both.
            # Log at debug so misconfig is investigable but doesn't
            # spam the worker log every entity extraction.
            logger.debug(
                "GLiNER URL discovery via ConfigManager singleton failed "
                "(falling back to env / default): %s",
                exc,
            )
        # Env-backed override for processes that don't share the
        # SystemConfig singleton (e.g. the ingestor worker, which builds
        # its own ConfigManager). Reads INFERENCE_SERVICE_URLS or the
        # dedicated GLINER_INFERENCE_URL var.
        import json as _json

        raw = _os.environ.get("INFERENCE_SERVICE_URLS")
        if raw:
            try:
                parsed = _json.loads(raw)
            except _json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                url = parsed.get("gliner")
                if url:
                    return url
        return _os.environ.get("GLINER_INFERENCE_URL")

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

    def _extract_from_text(
        self,
        text: str,
        tenant_id: str,
        source_doc_id: str,
        segment_anchor: Mention,
        prior_entities: Optional[List[str]] = None,
    ) -> ExtractionResult:
        gliner = self._get_gliner()
        nodes: List[Node] = []
        edges: List[Edge] = []
        seen: Set[str] = set()

        chunks = self._chunk_text(text)

        for chunk in chunks:
            entities_in_chunk: List[Tuple[str, str]] = []

            if gliner is not None:
                try:
                    # 0.3 chosen empirically against the production
                    # gliner_large-v2.1 sidecar: at 0.5 the model
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
                    logger.warning("GLiNER prediction failed on chunk: %s", exc)

            if not entities_in_chunk:
                entities_in_chunk = self._fallback_extract(chunk)

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

            if self._claim_extractor is not None:
                chunk_hints = [name for name, _ in entities_in_chunk]
                prior = prior_entities or []
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

        return ExtractionResult(
            source_doc_id=source_doc_id,
            nodes=nodes,
            edges=edges,
        )

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

    def _fallback_extract(self, chunk: str) -> List[Tuple[str, str]]:
        """Cheap fallback when GLiNER is unavailable: capitalized phrases.

        Splits on sentence-initial capitalized articles so "The ColPali" →
        "ColPali". Only accepts phrases that start with a non-stopword
        capitalized token.
        """
        stopwords = {
            "the",
            "this",
            "that",
            "these",
            "those",
            "there",
            "here",
            "when",
            "where",
            "how",
            "why",
            "what",
            "which",
            "who",
            "it",
            "its",
            "a",
            "an",
            "and",
            "but",
            "or",
            "so",
        }
        candidates = re.findall(
            r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+){0,3})\b", chunk
        )
        seen: Set[str] = set()
        out: List[Tuple[str, str]] = []
        for raw in candidates:
            parts = raw.split()
            while parts and parts[0].lower() in stopwords:
                parts.pop(0)
            if not parts:
                continue
            name = " ".join(parts)
            if len(name) < 3:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append((name, "Concept"))
        return out


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
