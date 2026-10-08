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
import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

# Operation type values — the discriminator written to the `operation` attribute
# (also carried by the span name). Consumers filter on these.
OP_SEARCH = "search"
OP_QUERY_ENHANCEMENT = "query_enhancement"
OP_ENTITY_EXTRACTION = "entity_extraction"
OP_ROUTING = "routing"
OP_ORCHESTRATION = "orchestration"
OP_PROFILE_SELECTION = "profile_selection"
OP_GATEWAY = "gateway"

# The model the backend reports having served a routed LM call, as the
# completion's ``model`` field; stamped on the span the call ran under.
LLM_SERVED_MODEL_ATTRIBUTE = "llm.served_model"
LLM_TIER_DEGRADED_ATTRIBUTE = "tier_degraded"
LLM_UPSTREAM_STATUS_ATTRIBUTE = "upstream_status"
LLM_UPSTREAM_EXCEPTION_TYPE_ATTRIBUTE = "upstream_exception_type"
PRO_MODEL_UNAVAILABLE = "pro_model_unavailable"

# Stamped on the span of an LM call whose endpoint answered 404
# (``LMEndpointNotServing``): the state, whether the call was refused without
# being sent, and the seconds until one call rechecks the endpoint.
LLM_ENDPOINT_STATE_ATTRIBUTE = "llm.endpoint.state"
LLM_ENDPOINT_FAILED_FAST_ATTRIBUTE = "llm.endpoint.failed_fast"
LLM_ENDPOINT_RECHECK_IN_S_ATTRIBUTE = "llm.endpoint.recheck_in_s"

# Query enhancement path marker — every query_enhancement span sets this so
# served rows stay machine-readable even when the LM falls back.
QUERY_ENHANCEMENT_PATH_ATTRIBUTE = "enhancement.path"
QUERY_ENHANCEMENT_PATH_LM = "lm"
QUERY_ENHANCEMENT_PATH_HEURISTIC_FALLBACK = "heuristic_fallback"
QUERY_ENHANCEMENT_PATH_VALUES = frozenset(
    {
        QUERY_ENHANCEMENT_PATH_LM,
        QUERY_ENHANCEMENT_PATH_HEURISTIC_FALLBACK,
    }
)
QUERY_ENHANCEMENT_SPAN_ATTRIBUTE_KEYS = frozenset(
    {
        "input.value",
        "input.source_text",
        "input.grounding_context",
        "operation",
        "output.value",
        QUERY_ENHANCEMENT_PATH_ATTRIBUTE,
    }
)

# Entity extraction fallback marker — an entity_extraction span that served the
# fast path names why in an attribute, so an engine answering outside the
# signature's enforced output schema is queryable rather than one warning line
# indistinguishable from an LM outage.
ENTITY_EXTRACTION_FALLBACK_ATTRIBUTE = "entity_extraction.fallback_reason"
ENTITY_EXTRACTION_FALLBACK_ERROR_ATTRIBUTE = "entity_extraction.fallback_error"
ENTITY_EXTRACTION_FALLBACK_SCHEMA_REFUSED = "schema_refused"
ENTITY_EXTRACTION_FALLBACK_LM_UNAVAILABLE = "lm_unavailable"
# The engine was reachable and REFUSED the request (4xx). Distinct from both of
# the above: nothing was generated, so it is not a schema refusal, and the
# provider answered, so it is not an outage. The status is carried because it
# names the operator action — 400 is a malformed request cogniverse (or a hop
# on the way) built, 401/403 a credential, 429 a quota.
ENTITY_EXTRACTION_FALLBACK_REQUEST_REJECTED = "request_rejected"
# The LM answered and the answer was dropped locally: an entity it returned is
# not a span of the query the spans are grounded against. Distinct from the
# three above, which are all the engine's doing, and it points at cogniverse's
# own grounding rather than at the provider.
ENTITY_EXTRACTION_FALLBACK_GROUNDING_FAILED = "grounding_failed"
# A DSPy answer that partly survived grounding: the mentions the raw query does
# not contain are dropped on their own and counted here, so a served extraction
# says how much of the LM's answer it discarded. ``grounding_failed`` above
# remains the state only when nothing survived.
ENTITY_EXTRACTION_GROUNDING_DROPPED_COUNT_ATTRIBUTE = (
    "entity_extraction.grounding_dropped_count"
)
ENTITY_EXTRACTION_GROUNDING_DROPPED_ATTRIBUTE = "entity_extraction.grounding_dropped"


# The entity extractor itself could not answer -- an unprovisioned sidecar, a
# missing dependency, an unreachable inference service. Distinct from the LM
# reasons above: no extraction was attempted, so whether the text holds
# entities is unknown rather than known to be nothing.
ENTITY_EXTRACTION_FALLBACK_EXTRACTOR_UNAVAILABLE = "extractor_unavailable"


# The base reasons; ``request_rejected`` is served with its status appended.
ENTITY_EXTRACTION_FALLBACK_VALUES = frozenset(
    {
        ENTITY_EXTRACTION_FALLBACK_SCHEMA_REFUSED,
        ENTITY_EXTRACTION_FALLBACK_LM_UNAVAILABLE,
        ENTITY_EXTRACTION_FALLBACK_REQUEST_REJECTED,
        ENTITY_EXTRACTION_FALLBACK_GROUNDING_FAILED,
        ENTITY_EXTRACTION_FALLBACK_EXTRACTOR_UNAVAILABLE,
    }
)


def entity_extraction_request_rejected(status_code: int) -> str:
    """The fallback reason for an LM that refused the request: ``<reason>:<status>``."""
    return f"{ENTITY_EXTRACTION_FALLBACK_REQUEST_REJECTED}:{int(status_code)}"


# Annotation contract — one home for the names, metadata key, and thresholds
# every consumer of result_click / result_relevance / preference pairs shares.
RESULT_RELEVANCE = "result_relevance"
RESULT_CLICK = "result_click"
RESULT_ID_META_KEY = "result_id"
RELEVANCE_POSITIVE_THRESHOLD = 0.7
PREFERENCE_CHOSEN_THRESHOLD = 0.5

# A reviewer's relevance label and the score it is stored with. Only "Highly
# Relevant" clears RELEVANCE_POSITIVE_THRESHOLD, so it is the label the triplet
# miner counts as a positive.
RELEVANCE_SCORES = {
    "Highly Relevant": 1.0,
    "Somewhat Relevant": 0.5,
    "Not Relevant": 0.0,
}


# A reviewer's verdict on a whole conversation, read back by the trajectory
# converter (cogniverse_finetuning trace_converter) under this name.
SESSION_EVALUATION = "session_evaluation"
SESSION_ID_META_KEY = "session_id"
SESSION_OUTCOMES = ("success", "partial", "failure")


class SpanNotInProjectError(LookupError):
    """The span to annotate is not a span of the project being written."""


async def persist_result_relevance(
    provider: Any,
    project: str,
    span_id: Optional[str],
    result_id: str,
    relevance_label: str,
) -> float:
    """Write a ``result_relevance`` annotation on search span ``span_id`` of
    ``project`` and return its score. Each result keeps its own annotation;
    rating a result again replaces its earlier rating.

    The span is read back from ``project`` first: the backend keys
    annotations by span id alone, so writing without that check would let a
    caller annotate another project's span. Raises ``ValueError`` on a missing
    span id or an unknown label and ``SpanNotInProjectError`` when the project
    holds no such span.
    """
    if not span_id:
        raise ValueError(
            "no search span_id — cannot annotate this result "
            "(telemetry disabled or span not captured)"
        )
    if relevance_label not in RELEVANCE_SCORES:
        raise ValueError(f"unknown relevance label: {relevance_label!r}")
    spans = await provider.traces.get_spans(
        project=project,
        filters={"span_id": [span_id]},
        limit=1,
    )
    if spans.empty or span_id not in set(spans["context.span_id"]):
        raise SpanNotInProjectError(f"span {span_id} is not in project {project}")

    score = RELEVANCE_SCORES[relevance_label]
    await provider.annotations.add_annotation(
        span_id=span_id,
        name=RESULT_RELEVANCE,
        label=relevance_label,
        score=score,
        metadata={RESULT_ID_META_KEY: str(result_id)},
        project=project,
        identifier=str(result_id),
    )
    return score


async def persist_session_evaluation(
    provider: Any,
    project: str,
    session_id: str,
    span_ids: list,
    outcome: str,
    score: float,
) -> list:
    """Write a ``session_evaluation`` annotation (``outcome`` as its label,
    ``score`` in 0-1) on each of ``span_ids``, the spans of one conversation,
    in ``project`` and return the span ids written, sorted. Evaluating the
    conversation again replaces its earlier verdict on each span.

    Every span is read back from ``project`` before any is written, for the
    reason ``persist_result_relevance`` gives. Raises ``ValueError`` on an
    unknown outcome, a score outside 0-1 or no span ids, and
    ``SpanNotInProjectError`` naming the spans the project does not hold.
    """
    if outcome not in SESSION_OUTCOMES:
        raise ValueError(f"unknown session outcome: {outcome!r}")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"session score must be between 0 and 1, got {score}")
    wanted = sorted(set(span_ids))
    if not wanted:
        raise ValueError("no spans name the conversation to evaluate")
    spans = await provider.traces.get_spans(
        project=project,
        filters={"span_id": wanted},
        limit=len(wanted),
    )
    found = set() if spans.empty else set(spans["context.span_id"])
    missing = [span_id for span_id in wanted if span_id not in found]
    if missing:
        raise SpanNotInProjectError(
            f"spans {', '.join(missing)} are not in project {project}"
        )
    for span_id in wanted:
        await provider.annotations.add_annotation(
            span_id=span_id,
            name=SESSION_EVALUATION,
            label=outcome,
            score=score,
            metadata={SESSION_ID_META_KEY: session_id, "num_spans": len(wanted)},
            project=project,
            identifier=session_id,
        )
    return wanted


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
    all populated so a consumer's preferred key always resolves — plus the
    source's stored ``source_title`` (``None`` when the hit carries none),
    which golden evaluation matches on.
    """
    if isinstance(result, dict):
        d = result
        _id = d.get("id") or d.get("document_id") or d.get("documentid")
        doc_id = d.get("document_id") or d.get("documentid") or _id
        source = d.get("source_id") or d.get("video_id")
        source_title = d.get("source_title")
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
        source_title = meta.get("source_title")
        content = (
            meta.get("content")
            or meta.get("text_content")
            or meta.get("description")
            or meta.get("title")
            or ""
        )
        score = getattr(result, "score", None)
    video_id = source or _id
    # Coerce defensively: a non-numeric score must not raise, and NaN/inf must
    # not reach output.value — json.dumps would emit invalid JSON (NaN) that
    # strict consumers (Phoenix UI, JSON.parse) reject.
    import math

    try:
        score_f = float(score) if score is not None else 0.0
    except (TypeError, ValueError):
        score_f = 0.0
    if not math.isfinite(score_f):
        score_f = 0.0
    return {
        "document_id": doc_id,
        "video_id": video_id,
        "source_id": source or video_id,
        "source_title": source_title,
        "id": _id,
        "score": score_f,
        "content": content,
    }


def current_span_id() -> Optional[str]:
    """16-hex id of the active telemetry span, or None when none is active
    (telemetry off leaves an invalid span current).

    A search stamps it on its output so a client can annotate this exact
    search span (``result_relevance``, ``result_click``). Read it on the
    coroutine that holds the span: a worker thread started by ``to_thread``
    does not carry the span context.
    """
    from opentelemetry import trace

    context = trace.get_current_span().get_span_context()
    if context and context.is_valid:
        return f"{context.span_id:016x}"
    return None


def record_search_io_on_current_span(query: str, results: list, modality: str) -> None:
    """Record a search's query, modality and result rows on the active span.

    The span whose id a search hands its client: the triplet miner reads the
    anchor (``input.value``), the candidates (``output.value``, one
    ``search_result_row`` per result) and the client's relevance annotations
    from that one span. Modality rides on a plain ``modality`` attribute;
    Phoenix folds ``input.*`` sub-keys into ``input.value``. Nothing is
    recorded when no span is active, and a failure to record is logged rather
    than failing the search.
    """
    from opentelemetry import trace

    span = trace.get_current_span()
    if not span.get_span_context().is_valid:
        return
    try:
        record_span_io(
            span,
            input_value=query,
            output=[search_result_row(result) for result in results],
            operation=OP_SEARCH,
            modality=modality,
        )
    except Exception as exc:
        logger.warning("search span %s io not recorded: %s", modality, exc)


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
        if v is None:
            continue
        # Only strings are "empty" — comparing a numpy array against ""
        # yields an elementwise result whose truthiness raises.
        if isinstance(v, str) and v == "":
            continue
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


def read_span_id(row: Any) -> Optional[str]:
    """Return a Phoenix span row's stable span id (``context.span_id``).

    This is the id the optimizer's ledger records as the source of a consumed
    training example. Returns ``None`` only when the column is absent — real
    Phoenix rows always carry it.
    """
    span_id = _reconstruct_attributes(row).get("context.span_id")
    return str(span_id) if span_id is not None else None


def read_span_attributes(row: Any) -> dict:
    """Return a Phoenix span row's attributes as a flat ``{key: value}`` dict.

    Keys are the writer's dotted names (``input.value``,
    ``input.grounding_context``, ``operation``, ...) whichever way Phoenix
    surfaced them (leaf columns or nested dicts).
    """
    return _reconstruct_attributes(row)


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
