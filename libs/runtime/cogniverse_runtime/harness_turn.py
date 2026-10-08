"""Transport-neutral helpers for one harness turn.

The dispatch envelope, the wiki auto-file hook and the harness transports all
need the same three things from a turn: the per-conversation seed used for
canary/variant bucketing, the human-facing answer text of a dispatch result,
and the OpenAI assistant tool-call shape for an agent's pending tool calls.
They live here so no consumer re-derives them from each executor's envelope.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any, Dict, List, Optional, Tuple


class NoAnswerError(RuntimeError):
    """A dispatch result carries no human-facing answer.

    Raised for an error envelope and for an empty result: rendering either as
    assistant text would present a failure as a reply.
    """

    def __init__(self, message: str, *, status: Optional[str] = None) -> None:
        super().__init__(message)
        self.status = status


class ToolCallShapeError(ValueError):
    """A pending tool call is missing a field the OpenAI shape requires."""


# Ordered rules mapping the fields an agent output declares to the fields whose
# text forms the answer. The first field decides the rule; the rest are appended
# when present and non-empty (a report's findings follow its summary).
_ANSWER_FIELD_RULES: Tuple[Tuple[str, ...], ...] = (
    ("answer",),
    ("executive_summary", "detailed_findings"),
    ("summary",),
    ("final_output",),
    ("aggregated_content",),
    ("response",),
    ("report",),
    ("analysis",),
    ("explanation",),
)

# Keys the dispatcher stamps on every envelope; they describe the turn, not the
# answer, so the structured fallback renders the payload without them.
_ENVELOPE_KEYS = frozenset(
    {
        "answer",
        "agent",
        "downstream_result",
        "gateway",
        "gateway_context",
        "message",
        "status",
    }
)

# Where a dispatch envelope nests an agent's own output.
_PAYLOAD_KEYS = ("orchestration_result", "result")

# A hit is named by its title: a search hit's own, else its source's.
_HIT_TITLE_KEYS = (
    "title",
    "video_title",
    "source_title",
    "document_title",
    "audio_title",
    "image_title",
    "filename",
)
# What a hit shows, in the order a reader wants it: a preview, a video
# frame's description, a transcript, then the raw text fields.
_HIT_TEXT_KEYS = (
    "content_preview",
    "segment_description",
    "description",
    "transcript",
    "audio_transcript",
    "full_text",
    "image_description",
    "text",
    "content",
)
HIT_SNIPPET_CHARS = 200

# A supporting answer field (a report's findings) is a list of entries, each
# either a line of text or a record labelling one.
_ENTRY_LABEL_KEYS = ("category", "title", "name")
_ENTRY_TEXT_KEYS = ("finding", "summary", "text", "description", "content")


def answer_fields_for(field_names: Iterable[str]) -> Tuple[str, ...]:
    """Fields carrying the answer text for a payload declaring ``field_names``.

    Empty when the payload declares none: such a result is rendered from the
    dispatch envelope's message and hits, or as structured JSON.
    """
    declared = set(field_names)
    for rule in _ANSWER_FIELD_RULES:
        if rule[0] in declared:
            return tuple(name for name in rule if name in declared)
    return ()


def is_answer_field(field_name: str) -> bool:
    """Whether tokens streamed for ``field_name`` are the turn's answer text.

    A streaming transport sees one field name per token event and no payload,
    so it decides from the same ordered rules ``answer_fields_for`` applies to
    a finished payload: a field is answer text when it heads a rule. An
    agent's working fields (a decomposition, a gap list, a plan) head none.
    """
    return answer_fields_for((field_name,)) == (field_name,)


def derive_request_seed(query: str, history: List[Dict[str, Any]]) -> str:
    """Stable per-conversation seed for canary/variant bucketing.

    Anchored on the conversation's first user message: it is the only part of a
    replayed transcript that never changes turn-to-turn, and it is the first
    part that differs between two conversations from a client that prefixes the
    same system prompt to both. Only role and content are hashed, so per-turn
    metadata a client attaches to the replayed message keeps the seed stable.
    """
    anchor: Dict[str, Any] = {"role": "user", "content": query}
    for message in history or []:
        if not isinstance(message, Mapping):
            continue
        if message.get("role") != "user":
            continue
        anchor = {"role": "user", "content": message.get("content")}
        break
    return hashlib.sha256(
        json.dumps(anchor, sort_keys=True, default=str).encode()
    ).hexdigest()[:32]


def _first_text(item: Mapping[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _clock(seconds: float) -> str:
    """75.4 -> "1:15", as the web client's result cards show a time."""
    whole = int(seconds)
    return f"{whole // 60}:{whole % 60:02d}"


def _snippet(text: str) -> str:
    """The first line of a hit's text, cut at a word under the snippet cap."""
    line = next((part.strip() for part in text.splitlines() if part.strip()), "")
    if len(line) <= HIT_SNIPPET_CHARS:
        return line
    cut = line[:HIT_SNIPPET_CHARS].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{cut}…"


def _hit_field(item: Mapping[str, Any], keys: Iterable[str]) -> Optional[str]:
    """A hit's field, read on the hit first and then on its metadata."""
    found = _first_text(item, keys)
    if found is None:
        metadata = item.get("metadata")
        if isinstance(metadata, Mapping):
            found = _first_text(metadata, keys)
    return found


def _format_search_results(results: List[Any], limit: int = 10) -> str:
    """Render search hits as lines a person reads: each hit by its title (the
    video, document, image or clip it is from) and, for a video segment, its
    time range, then the first line of what it shows. A hit's backend
    document id is never shown."""
    lines: List[str] = []
    for position, item in enumerate(results[:limit], start=1):
        if not isinstance(item, Mapping):
            continue
        title = _hit_field(item, _HIT_TITLE_KEYS)
        line = title or f"Result {position}"
        temporal = item.get("temporal_info")
        if isinstance(temporal, Mapping):
            start, end = temporal.get("start_time"), temporal.get("end_time")
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                line += f" ({_clock(start)}–{_clock(end)})"
        text = _hit_field(item, _HIT_TEXT_KEYS)
        if text is not None:
            snippet = _snippet(text)
            if snippet and snippet != title:
                line += f": {snippet}"
        lines.append(f"- {line}")
    return "\n".join(lines)


def _describe_entities(payload: Mapping[str, Any]) -> str:
    entities = [
        entity
        for entity in payload.get("entities") or []
        if isinstance(entity, Mapping) and _first_text(entity, ("text",))
    ]
    if not entities:
        return "Found no entities."
    named = ", ".join(
        f"{entity['text'].strip()} ({str(entity.get('type') or 'entity').lower()})"
        for entity in entities
    )
    noun = "entity" if len(entities) == 1 else "entities"
    sentence = f"Found {len(entities)} {noun}: {named}."
    relations = [
        " ".join(
            str(relation.get(key, "")).strip()
            for key in ("subject", "relation", "object")
        )
        for relation in payload.get("relationships") or []
        if isinstance(relation, Mapping)
    ]
    if relations:
        sentence += f" Relationships: {'; '.join(relations)}."
    return sentence


def _describe_enhancement(payload: Mapping[str, Any]) -> str:
    original = str(payload.get("original_query") or "").strip()
    enhanced = str(payload.get("enhanced_query") or "").strip()
    if not enhanced or enhanced == original:
        return f'Kept the query as asked: "{original}".'
    return f'Enhanced "{original}" to "{enhanced}".'


def _describe_profile_selection(payload: Mapping[str, Any]) -> str:
    sentence = f"Selected profile {payload['selected_profile']}"
    intent = payload.get("query_intent")
    if isinstance(intent, str) and intent.strip():
        sentence += f" for a {intent.strip().replace('_', ' ')}"
    confidence = payload.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        sentence += f" (confidence {float(confidence):.2f})"
    return f"{sentence}."


# Payloads that declare no answer field but whose result a person reads as a
# sentence: the key set that identifies each, and its sentence. The web client
# shows the full result beside it.
_STRUCTURED_REPLIES = (
    (("entities", "entity_count"), _describe_entities),
    (("original_query", "enhanced_query"), _describe_enhancement),
    (("selected_profile",), _describe_profile_selection),
)


def _describe_structured(payload: Mapping[str, Any]) -> Optional[str]:
    for keys, describe in _STRUCTURED_REPLIES:
        if all(key in payload for key in keys):
            return describe(payload)
    return None


def _render_entries(value: Any) -> Optional[str]:
    """Render a supporting answer field: free text, or one line per entry."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    if not isinstance(value, list):
        return None
    lines: List[str] = []
    for entry in value:
        if isinstance(entry, str) and entry.strip():
            lines.append(f"- {entry.strip()}")
        elif isinstance(entry, Mapping):
            label = _first_text(entry, _ENTRY_LABEL_KEYS)
            text = _first_text(entry, _ENTRY_TEXT_KEYS)
            body = ": ".join(part for part in (label, text) if part)
            if body:
                lines.append(f"- {body}")
    return "\n".join(lines) if lines else None


#: Statuses that carry no answer. ``partial`` is deliberately absent: a
#: partial result has a real answer from the steps that did complete.
TERMINAL_FAILURE_STATUSES = frozenset({"error", "failed"})


def _raise_if_error(payload: Mapping[str, Any], where: str) -> None:
    status = payload.get("status")
    if isinstance(status, str) and status.lower() in TERMINAL_FAILURE_STATUSES:
        detail = payload.get("error") or payload.get("message") or "no detail"
        raise NoAnswerError(
            f"{where} reported status={status}: {detail}", status=status
        )


def _from_payload(payload: Mapping[str, Any]) -> Optional[str]:
    fields = answer_fields_for(payload.keys())
    if not fields:
        return None
    primary = payload.get(fields[0])
    if isinstance(primary, Mapping):
        # The orchestrator's final_output is itself a payload: deep synthesis
        # writes ``answer``, cross-modal fusion writes ``aggregated_content``.
        _raise_if_error(primary, f"'{fields[0]}'")
        return _from_payload(primary)
    if not isinstance(primary, str) or not primary.strip():
        return None
    parts = [primary.strip()]
    for name in fields[1:]:
        rendered = _render_entries(payload.get(name))
        if rendered is not None:
            parts.append(rendered)
    return "\n\n".join(parts)


def _extract(result: Mapping[str, Any]) -> Optional[str]:
    _raise_if_error(result, f"agent '{result.get('agent', 'unknown')}'")

    text = _from_payload(result)
    if text is not None:
        return text

    downstream = result.get("downstream_result")
    if isinstance(downstream, Mapping):
        nested = _extract(downstream)
        if nested is not None:
            return nested

    for key in _PAYLOAD_KEYS:
        payload = result.get(key)
        if isinstance(payload, Mapping):
            _raise_if_error(payload, f"'{key}'")
            nested = _from_payload(payload)
            if nested is None:
                nested = _extract(payload)
            if nested is not None:
                return nested
        elif isinstance(payload, str) and payload.strip():
            return payload.strip()

    message = result.get("message")
    if isinstance(message, str) and message.strip():
        hits = result.get("results")
        if isinstance(hits, list) and hits:
            body = _format_search_results(hits)
            if body:
                return f"{message.strip()}\n{body}"
        return message.strip()

    described = _describe_structured(result)
    if described is not None:
        return described

    payload = {k: v for k, v in result.items() if k not in _ENVELOPE_KEYS}
    if payload:
        return json.dumps(payload, sort_keys=True, default=str)
    return None


def extract_answer_text(result: Mapping[str, Any]) -> str:
    """Human-facing text of a dispatch result.

    An agent's own output is read first (nested under ``result`` /
    ``orchestration_result``, or flat for the generic path), then a gateway
    wrapper's downstream result, then the envelope's message plus its hits
    (each by title and time range). An entity extraction, query enhancement or
    profile selection result reads as one sentence; any other result declaring
    no text at all renders as structured JSON of its payload.

    Raises:
        NoAnswerError: the result is an error envelope or carries no payload.
    """
    if not isinstance(result, Mapping):
        raise NoAnswerError(
            f"dispatch result is {type(result).__name__}, not a mapping"
        )
    found = _extract(result)
    if found is None:
        status = result.get("status")
        raise NoAnswerError(
            f"no answer text in result with keys {sorted(result)}",
            status=status if isinstance(status, str) else None,
        )
    return found


def to_openai_tool_calls(pending: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert an agent's ``pending_tool_calls`` ({id, name, arguments}) into
    the OpenAI assistant tool-call shape the harness transports emit and the
    coding agent's resume path parses (arguments JSON-encoded)."""
    calls: List[Dict[str, Any]] = []
    for index, call in enumerate(pending):
        if not isinstance(call, Mapping):
            raise ToolCallShapeError(
                f"pending_tool_calls[{index}] is {type(call).__name__}, "
                f"expected a mapping with 'id' and 'name'"
            )
        for key in ("id", "name"):
            if key not in call:
                raise ToolCallShapeError(
                    f"pending_tool_calls[{index}] is missing '{key}'; "
                    f"has {sorted(call)}"
                )
        calls.append(
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call.get("arguments", {})),
                },
            }
        )
    return calls
