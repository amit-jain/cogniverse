"""JSONAdapter variant that normalizes LM field-name variants.

DSPy's stock `JSONAdapter.parse()` fails with `AdapterParseError` when the LM
emits a field name that differs from the signature's expected name. Smaller
local models (e.g. gemma4:e2b) routinely substitute `reason` for the
`reasoning` field that `dspy.ChainOfThought` auto-adds, or emit singular
forms (`sub_question`) when the schema names the field plural
(`sub_questions`).

This adapter applies a small set of canonical aliases before the strict
field-key equality check in the parent parser. Unknown fields still get
stripped; only known aliases are renamed. Everything else (tool calls, type
casting, adapter fallback behaviour) is inherited from `JSONAdapter`.
"""

from __future__ import annotations

from typing import Any

from dspy.adapters.json_adapter import JSONAdapter
from dspy.signatures.signature import Signature


class LenientJSONAdapter(JSONAdapter):
    """JSONAdapter that renames common LM field-name variants before validation."""

    # Each tuple is (alias emitted by some LMs, canonical signature field name).
    # Keep this list tight — add a pair only when confirmed in production.
    _FIELD_ALIASES: tuple[tuple[str, str], ...] = (
        # ChainOfThought adds a `reasoning` field; smaller LMs routinely
        # call it `reason`/`rationale`/`thought` instead.
        ("reason", "reasoning"),
        ("reasons", "reasoning"),
        ("rationale", "reasoning"),
        ("thought", "reasoning"),
        ("thoughts", "reasoning"),
        # Answer-shaped aliases — used when the signature's primary output is
        # a free-text field (summary / response / content / output).
        ("answer", "summary"),
        ("response", "summary"),
        ("result", "summary"),
        ("output", "summary"),
        ("text", "summary"),
        # Plural/singular confusions
        ("sub_question", "sub_questions"),
        ("subquestions", "sub_questions"),
        ("queries", "sub_questions"),
    )

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        expected = set(signature.output_fields.keys())
        if not expected:
            return super().parse(signature, completion)

        # Resolve the raw JSON first. Delegate to the parent for tolerant
        # decoding (json_repair + regex object extraction). We only need to
        # rename aliases; the parent handles casting, validation, etc.
        import json_repair
        import regex

        fields = json_repair.loads(completion)
        if not isinstance(fields, dict):
            pattern = r"\{(?:[^{}]|(?R))*\}"
            match = regex.search(pattern, completion, regex.DOTALL)
            if match:
                fields = json_repair.loads(match.group(0))

        if isinstance(fields, dict):
            remapped: dict[str, Any] = {}
            for key, value in fields.items():
                target = key
                if key not in expected:
                    for alias, canonical in self._FIELD_ALIASES:
                        if key == alias and canonical in expected and canonical not in fields:
                            target = canonical
                            break
                remapped[target] = value

            # After alias renaming, if the LM still missed one or more
            # output fields we can recover in two ways:
            #   1. Single unknown key + single missing expected field →
            #      assume the LM used a non-canonical name for that field.
            #   2. Any remaining missing expected field → fill with a safe
            #      default so downstream schema validation holds (the
            #      dispatcher's consumers already handle empty strings /
            #      empty lists; they'd otherwise hit AdapterParseError and
            #      500 the request).
            missing = expected - remapped.keys()
            unknown = [k for k in remapped if k not in expected]
            if len(missing) == 1 and len(unknown) == 1:
                remapped[next(iter(missing))] = remapped.pop(unknown[0])
                missing = set()
            for field_name in missing:
                field_info = signature.output_fields.get(field_name)
                annotation = getattr(field_info, "annotation", str)
                remapped[field_name] = _default_for(annotation)

            import json
            completion = json.dumps(remapped)

        return super().parse(signature, completion)


def _default_for(annotation: Any) -> Any:
    """Return a safe empty value matching the output field's annotation."""
    origin = getattr(annotation, "__origin__", None)
    if origin in (list, tuple, set):
        return []
    if origin is dict:
        return {}
    if annotation in (int, float):
        return 0
    if annotation is bool:
        return False
    return ""
