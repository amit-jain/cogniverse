"""Prose extraction for the JSON evaluation corpus.

Document profiles embed prose. The evaluation corpus stores its prose inside
JSON structures, so the text is extracted before ingestion: the raw file would
embed brace syntax, dict keys and per-word timing records as though they were
content.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

__all__ = [
    "corpus_prose",
    "has_substantive_prose",
    "materialize_corpus_text",
]

# Transcripts of silent or music-only footage reduce to a handful of repeated
# filler tokens ("Oh", "Music Music", runs of "."). Distinct-word count
# separates those from real speech without enumerating filler words.
_MIN_DISTINCT_WORDS = 5

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def has_substantive_prose(text: str) -> bool:
    """Return whether ``text`` carries enough distinct words to be content."""
    return len({word.lower() for word in _WORD.findall(text)}) >= _MIN_DISTINCT_WORDS


def _frame_descriptions_prose(payload: dict[str, Any], path: Path) -> str:
    """Join frame descriptions in frame order.

    Shape: ``{"0": "...", "1": "..."}`` keyed by stringified frame index.
    """
    try:
        ordered = sorted(payload.items(), key=lambda kv: int(kv[0]))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Frame-description keys must be integer indices: {path}"
        ) from exc

    parts = [value.strip() for _, value in ordered if isinstance(value, str)]
    return "\n\n".join(part for part in parts if part)


def _transcript_prose(payload: list[Any], path: Path) -> str:
    """Join transcript segment text, dropping timings and per-word records.

    Shape: ``[{"start": .., "end": .., "text": "..", "words": [..]}, ..]``.
    """
    parts: list[str] = []
    for index, segment in enumerate(payload):
        if not isinstance(segment, dict):
            raise ValueError(f"Transcript segment {index} is not an object: {path}")
        text = segment.get("text")
        if text is None:
            raise ValueError(f"Transcript segment {index} has no text: {path}")
        stripped = str(text).strip()
        if stripped:
            parts.append(stripped)
    return " ".join(parts)


def _retrieval_queries_prose(payload: list[Any], path: Path) -> str:
    """Join retrieval queries and their ground-truth passages.

    Shape: ``[{"query": "..", "ground_truth": "..", ..}, ..]``. Both fields
    repeat across records for the same video, so each distinct value is
    emitted once in first-seen order.
    """
    queries: list[str] = []
    passages: list[str] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise ValueError(f"Query record {index} is not an object: {path}")
        query = str(record.get("query") or "").strip()
        ground_truth = str(record.get("ground_truth") or "").strip()
        if query and query not in queries:
            queries.append(query)
        if ground_truth and ground_truth not in passages:
            passages.append(ground_truth)

    return "\n\n".join(queries + passages)


def corpus_prose(path: Path) -> str:
    """Return the prose carried by one evaluation-corpus JSON file."""
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        return _frame_descriptions_prose(payload, path)

    if isinstance(payload, list):
        if not payload:
            # A video with no speech has an empty transcript; that is content,
            # not a malformed file.
            return ""
        head = payload[0]
        if isinstance(head, dict) and "text" in head:
            return _transcript_prose(payload, path)
        if isinstance(head, dict) and "query" in head:
            return _retrieval_queries_prose(payload, path)

    raise ValueError(f"Unrecognized evaluation corpus shape: {path}")


def materialize_corpus_text(
    path: Path, relative_key: str, dest_dir: Path
) -> Path | None:
    """Write ``path``'s prose to a ``.txt`` file under ``dest_dir``.

    Returns ``None`` when the file carries no substantive prose, so the caller
    can skip it rather than ingest filler as a document.

    The destination name is derived from ``relative_key`` so the same corpus
    file always materializes to the same path, and the write is skipped when
    the content already matches — ingestion is content-addressed, so a stable
    body keeps the content id stable across sessions.
    """
    prose = corpus_prose(path)
    if not has_substantive_prose(prose):
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (relative_key.replace("/", "__").removesuffix(".json") + ".txt")
    if not dest.exists() or dest.read_text(encoding="utf-8") != prose:
        dest.write_text(prose, encoding="utf-8")
    return dest
