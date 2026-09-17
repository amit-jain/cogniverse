"""JSONAdapter variant that normalizes LM field-name variants.

DSPy's stock `JSONAdapter.parse()` fails with `AdapterParseError` when the LM
emits a field name that differs from the signature's expected name. Smaller
local models (e.g. gemma4:e2b) routinely substitute `reason` for the
`reasoning` field that `dspy.ChainOfThought` auto-adds, or emit singular
forms (`sub_question`) when the schema names the field plural
(`sub_questions`).

This adapter applies a small set of canonical aliases before the strict
field-key equality check in the parent parser. Unknown fields still get
stripped; only known aliases are renamed. An output the LM left unfilled —
absent, `null`, or a blank string — is incomplete generation, and raises
`LMOutputIncomplete` naming those fields. An empty collection is a value and
passes. A field declared with a default (`dspy.OutputField(default="")`) takes
that default for a `null` or blank answer; it is still incomplete when absent.
Everything else (tool calls, type casting, adapter fallback behaviour) is
inherited from `JSONAdapter`.
"""

from __future__ import annotations

from typing import Any, Iterable

from dspy.adapters.json_adapter import JSONAdapter
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError


def _is_filled(value: Any) -> bool:
    """Whether the LM put a value in the field it named.

    ``null`` and a blank string name the field without answering it. An empty
    list, dict, ``0`` and ``False`` are answers.
    """
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


class LMOutputIncomplete(AdapterParseError):
    """The LM stopped before producing every output field its signature requires."""

    def __init__(
        self,
        *,
        adapter_name: str,
        signature: type[Signature],
        lm_response: str,
        missing_fields: Iterable[str],
        parsed_result: dict[str, Any],
    ) -> None:
        self.missing_fields: tuple[str, ...] = tuple(sorted(missing_fields))
        super().__init__(
            adapter_name=adapter_name,
            signature=signature,
            lm_response=lm_response,
            message=(
                "The LM produced no "
                f"{', '.join(self.missing_fields)} for this signature."
            ),
            parsed_result=parsed_result,
        )


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
                        if (
                            key == alias
                            and canonical in expected
                            and canonical not in fields
                        ):
                            target = canonical
                            break
                remapped[target] = value

            produced: dict[str, Any] = {}
            for name, value in remapped.items():
                if name not in expected:
                    continue
                if _is_filled(value):
                    produced[name] = value
                elif not signature.output_fields[name].is_required():
                    produced[name] = signature.output_fields[name].default
            missing = expected - produced.keys()
            if missing:
                raise LMOutputIncomplete(
                    adapter_name=type(self).__name__,
                    signature=signature,
                    lm_response=completion,
                    missing_fields=missing,
                    parsed_result=produced,
                )

            import json

            completion = json.dumps(produced)

        return super().parse(signature, completion)
