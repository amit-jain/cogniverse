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

_HIT_ID_KEYS = ("document_id", "image_id", "audio_id", "video_id", "id")
_HIT_SCORE_KEYS = ("score", "relevance_score")
_HIT_TEXT_KEYS = (
    "title",
    "description",
    "content_preview",
    "transcript",
    "text",
    "content",
)

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


def _format_search_results(results: List[Any], limit: int = 10) -> str:
    """Render search hits as lines, so an answer carries the hits themselves
    rather than only the "Found N results" counter. Hit shape varies per
    modality (video/image/audio/document), hence the key precedence."""
    lines: List[str] = []
    for item in results[:limit]:
        if not isinstance(item, Mapping):
            continue
        parts = [str(_first_text(item, _HIT_ID_KEYS) or "?")]
        for key in _HIT_SCORE_KEYS:
            score = item.get(key)
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                parts.append(f"score {float(score):.3f}")
                break
        temporal = item.get("temporal_info")
        if isinstance(temporal, Mapping) and temporal.get("start_time") is not None:
            parts.append(f"{temporal.get('start_time')}s-{temporal.get('end_time')}s")
        line = " · ".join(parts)
        text = _first_text(item, _HIT_TEXT_KEYS)
        if text is None:
            metadata = item.get("metadata")
            if isinstance(metadata, Mapping):
                text = _first_text(metadata, _HIT_TEXT_KEYS)
        if text is not None:
            line += f": {text[:200]}"
        lines.append(f"- {line}")
    return "\n".join(lines)


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


def _raise_if_error(payload: Mapping[str, Any], where: str) -> None:
    status = payload.get("status")
    if isinstance(status, str) and status.lower() == "error":
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

    payload = {k: v for k, v in result.items() if k not in _ENVELOPE_KEYS}
    if payload:
        return json.dumps(payload, sort_keys=True, default=str)
    return None


def extract_answer_text(result: Mapping[str, Any]) -> str:
    """Human-facing text of a dispatch result.

    An agent's own output is read first (nested under ``result`` /
    ``orchestration_result``, or flat for the generic path), then a gateway
    wrapper's downstream result, then the envelope's message plus its hits. A
    result declaring no text at all renders as structured JSON of its payload.

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
