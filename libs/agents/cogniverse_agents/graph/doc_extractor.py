"""Document extractor — GLiNER for entities + heuristic relationship mining.

For text-based documents (`.md`, `.txt`, `.rst`, `.html`, `.pdf` after
text extraction). GLiNER is already in the cogniverse stack via the
routing agent, so no new heavy deps.

Extracts named entities and concepts, emits them as nodes with
INFERRED provenance (LLM-based, not structural). Edges link entities
mentioned in the same chunk of text.
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Set

from cogniverse_agents.graph.graph_schema import Edge, ExtractionResult, Node

logger = logging.getLogger(__name__)

_TEXT_EXTENSIONS = {".md", ".txt", ".rst", ".html", ".htm"}
_PDF_EXTENSIONS = {".pdf"}

_DEFAULT_LABELS = [
    "Person",
    "Organization",
    "Technology",
    "Concept",
    "Location",
    "Product",
    "Algorithm",
    "Model",
    "Framework",
    "Language",
]

_MAX_CHARS_PER_CHUNK = 2000


def supported_extensions() -> Set[str]:
    return _TEXT_EXTENSIONS | _PDF_EXTENSIONS


class DocExtractor:
    """Extract entities and relationships from text documents."""

    def __init__(self, labels: Optional[List[str]] = None) -> None:
        self._labels = labels or list(_DEFAULT_LABELS)
        self._gliner = None
        self._gliner_failed = False

    def _get_gliner(self):
        """Lazily load the GLiNER model, caching the instance."""
        if self._gliner is not None:
            return self._gliner
        if self._gliner_failed:
            return None
        from cogniverse_core.common.models import get_or_load_gliner

        self._gliner = get_or_load_gliner("urchade/gliner_large-v2.1", logger=logger)
        if self._gliner is None:
            self._gliner_failed = True
        return self._gliner

    def extract(
        self,
        file_path: Path,
        tenant_id: str,
        source_doc_id: str,
    ) -> Optional[ExtractionResult]:
        """Extract from a text or PDF file."""
        ext = file_path.suffix.lower()
        if ext not in supported_extensions():
            return None

        text = self._load_text(file_path, ext)
        if not text:
            return None

        return self._extract_from_text(text, tenant_id, source_doc_id)

    def extract_from_text(
        self,
        text: str,
        tenant_id: str,
        source_doc_id: str,
    ) -> ExtractionResult:
        """Public entry point for callers that already have extracted text
        (e.g. video transcriptions, image captions).
        """
        return self._extract_from_text(text, tenant_id, source_doc_id)

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
        self, text: str, tenant_id: str, source_doc_id: str
    ) -> ExtractionResult:
        gliner = self._get_gliner()
        nodes: List[Node] = []
        edges: List[Edge] = []
        seen: Set[str] = set()

        chunks = self._chunk_text(text)

        for chunk in chunks:
            entities_in_chunk: List[tuple] = []

            if gliner is not None:
                try:
                    raw = gliner.predict_entities(chunk, self._labels, threshold=0.5)
                    for ent in raw:
                        name = ent.get("text", "").strip()
                        label = ent.get("label", "Concept")
                        if not name or len(name) < 2:
                            continue
                        entities_in_chunk.append((name, label))
                except Exception as exc:
                    logger.warning("GLiNER prediction failed on chunk: %s", exc)

            if not entities_in_chunk:
                entities_in_chunk = self._fallback_extract(chunk)

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
                        mentions=[source_doc_id],
                    )
                )

            chunk_names = [name for name, _ in entities_in_chunk]
            for i, src in enumerate(chunk_names):
                for tgt in chunk_names[i + 1 :]:
                    edges.append(
                        Edge(
                            tenant_id=tenant_id,
                            source=src,
                            target=tgt,
                            relation="mentioned_with",
                            provenance="INFERRED",
                            source_doc_id=source_doc_id,
                            confidence=0.5,
                        )
                    )

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

    def _fallback_extract(self, chunk: str) -> List[tuple]:
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
        out: List[tuple] = []
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
