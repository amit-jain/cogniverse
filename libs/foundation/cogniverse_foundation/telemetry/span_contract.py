"""Canonical telemetry span I/O contract.

One shape for every operation span — search, query_enhancement,
entity_extraction, routing, orchestration, profile_selection, gateway. The
operation's input goes on ``input.value``, its output goes on ``output.value``
as JSON, its type goes on ``operation`` (and the span name). Every consumer
(eval, dataset, experiment, optimize, annotation) reads back through
``read_span_io`` and dispatches on the operation, so a single pipeline serves
every operation type instead of a bespoke read path per span kind.

``record_span_io`` is the sole writer; ``read_span_io`` is the sole reader.
"""

from __future__ import annotations

import ast
import json
from typing import Any, Mapping, Optional

# Operation type values — the discriminator written to the `operation` attribute
# (also carried by the span name). Consumers filter on these.
OP_SEARCH = "search"
OP_QUERY_ENHANCEMENT = "query_enhancement"
OP_ENTITY_EXTRACTION = "entity_extraction"
OP_ROUTING = "routing"
OP_ORCHESTRATION = "orchestration"
OP_PROFILE_SELECTION = "profile_selection"
OP_GATEWAY = "gateway"

# Annotation contract — one home for the names, metadata key, and thresholds
# every consumer of result_click / result_relevance / preference pairs shares.
RESULT_RELEVANCE = "result_relevance"
RESULT_CLICK = "result_click"
RESULT_ID_META_KEY = "result_id"
RELEVANCE_POSITIVE_THRESHOLD = 0.7
PREFERENCE_CHOSEN_THRESHOLD = 0.5

_ATTR_PREFIX = "attributes."


def record_span_io(
    span: Any,
    *,
    input_value: Optional[str],
    output: Any,
    operation: Optional[str] = None,
    modality: Optional[str] = None,
) -> None:
    """Write the canonical input/output slots on an active span.

    ``input_value`` → ``input.value`` (clean text — the query / source text).
    ``output`` → ``output.value`` = ``json.dumps(output)`` (a list for search,
    a dict for the domain operations); the SLOT is uniform even though the
    payload shape varies by operation. ``operation`` → ``operation`` attribute
    (type discriminator, alongside the span name). ``modality`` → ``modality``.
    """
    if span is None:
        return
    if input_value is not None:
        span.set_attribute(
            "input.value",
            input_value if isinstance(input_value, str) else json.dumps(input_value),
        )
    span.set_attribute("output.value", json.dumps(output, default=str))
    if operation:
        span.set_attribute("operation", operation)
    if modality:
        span.set_attribute("modality", modality)


def search_result_row(result: Any) -> dict:
    """Canonical search-result row for ``output.value``.

    Accepts either a backend ``SearchResult`` object (``.document.id`` /
    ``.score`` / ``.document.metadata``) or an already-built result dict
    (``{"id","score",**metadata}``) and returns the superset id shape every
    search consumer reads — ``document_id`` / ``video_id`` / ``source_id`` / ``id``
    all populated so a consumer's preferred key always resolves.
    """
    if isinstance(result, dict):
        d = result
        _id = d.get("id") or d.get("document_id") or d.get("documentid")
        doc_id = d.get("document_id") or d.get("documentid") or _id
        source = d.get("source_id") or d.get("video_id")
        content = (
            d.get("content")
            or d.get("text_content")
            or d.get("description")
            or d.get("text")
            or d.get("title")
            or ""
        )
        score = d.get("score")
    else:
        doc = getattr(result, "document", None)
        meta = getattr(doc, "metadata", None) or {}
        _id = getattr(doc, "id", None)
        doc_id = _id
        source = meta.get("source_id") or meta.get("video_id")
        content = (
            meta.get("content")
            or meta.get("text_content")
            or meta.get("description")
            or meta.get("title")
            or ""
        )
        score = getattr(result, "score", None)
    video_id = source or _id
    return {
        "document_id": doc_id,
        "video_id": video_id,
        "source_id": source or video_id,
        "id": _id,
        "score": float(score) if score is not None else 0.0,
        "content": content,
    }


def _reconstruct_attributes(row: Any) -> dict:
    """Flatten a Phoenix span row into a plain attribute dict.

    ``get_spans`` has no bare ``attributes`` column — attributes live in dotted
    ``attributes.<key>`` columns, some leaf scalars, some nested dicts. Strip
    the prefix, drop NaNs, and expand nested dicts to ``<key>.<sub>`` so callers
    read ``input.value`` / ``output.value`` / ``operation`` uniformly whichever
    way Phoenix surfaced them. A pre-stripped mapping passes through unchanged.
    """
    items = row.items() if hasattr(row, "items") else dict(row).items()
    attrs: dict = {}
    for col, val in items:
        if not isinstance(col, str):
            continue
        try:
            import pandas as pd

            if pd.isna(val):
                continue
        except (TypeError, ValueError, ImportError):
            pass  # dicts / lists / arrays are not NaN
        key = col[len(_ATTR_PREFIX) :] if col.startswith(_ATTR_PREFIX) else col
        if isinstance(val, dict):
            for k, v in val.items():
                attrs[f"{key}.{k}"] = v
        attrs[key] = val
    return attrs


def _first(attrs: Mapping, *keys: str, default: Any = None) -> Any:
    for k in keys:
        v = attrs.get(k)
        if v is not None and v != "":
            return v
    return default


def _parse_output(raw: Any) -> Any:
    """output.value is a JSON string; tolerate a literal or an already-parsed value."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        return raw
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(raw)
        except (ValueError, SyntaxError):
            continue
    return raw


def read_span_io(row: Any) -> dict:
    """Read the canonical input/output/operation/modality from a Phoenix span row.

    Returns ``{"input", "output", "operation", "modality"}``. ``output`` is the
    JSON-decoded ``output.value`` (a list for search, a dict for domain spans).
    Tolerant of the legacy ``input.query`` / ``output.results`` / ``search_type``
    keys during migration; the writer always emits the canonical slots.
    """
    attrs = _reconstruct_attributes(row)
    return {
        "input": _first(attrs, "input.value", "input.query", "query", default=None),
        "output": _parse_output(
            _first(attrs, "output.value", "output.results", "results", default=None)
        ),
        "operation": attrs.get("operation"),
        "modality": _first(
            attrs, "modality", "input.modality", "search_type", default=None
        ),
    }
