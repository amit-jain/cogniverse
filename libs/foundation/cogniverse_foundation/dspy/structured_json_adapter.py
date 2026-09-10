"""JSON adapter that binds a signature's output fields as a server-enforced schema.

DSPy's stock `JSONAdapter` only asks the server for structured output when
litellm's model registry claims the model supports a response schema. A
self-hosted OpenAI-compatible model is absent from that registry, so the
adapter degrades to `{"type": "json_object"}` — which constrains the response
to *valid JSON* and nothing more. A bare `{}` satisfies it and then fails
DSPy's field check, so a sampling miss surfaces as `AdapterParseError`.

This adapter always sends the signature's output fields as a JSON schema, so
the engine's guided decoding can only emit an object carrying every output
field.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import pydantic
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.lm import LM
from dspy.signatures.signature import Signature

_NON_SCHEMA_NAME = re.compile(r"[^A-Za-z0-9_-]")

# dspy builds a signature from a string, and rebuilds one under
# `with_instructions()` or `ChainOfThought`, under these placeholder class
# names. They identify nothing: every such signature in the process shares
# one, so the class name cannot name the schema.
_GENERIC_SIGNATURE_NAMES = frozenset({"Signature", "StringSignature"})
_SCHEMA_NAME_MAX = 64
_DIGEST_CHARS = 8


def _field_declaration(signature: type[Signature]) -> tuple[str, ...]:
    """The signature's fields as the schema sees them: name, kind, type."""
    return tuple(
        f"{kind}:{name}:{getattr(info.annotation, '__name__', info.annotation)}"
        for kind, fields in (
            ("in", signature.input_fields),
            ("out", signature.output_fields),
        )
        for name, info in fields.items()
    )


def _schema_name(signature: type[Signature]) -> str:
    """A name that identifies THIS signature's schema, never a placeholder.

    Two signatures declaring different fields get different names, so a
    server-side log of enforced schemas stays readable when several
    signatures are in flight.
    """
    declared = _NON_SCHEMA_NAME.sub("_", signature.__name__)
    if declared and declared not in _GENERIC_SIGNATURE_NAMES:
        return declared[:_SCHEMA_NAME_MAX]

    declaration = _field_declaration(signature)
    digest = hashlib.sha256("|".join(declaration).encode()).hexdigest()[:_DIGEST_CHARS]
    stem = _NON_SCHEMA_NAME.sub("_", "_".join(signature.output_fields)) or "Output"
    return f"{stem[: _SCHEMA_NAME_MAX - _DIGEST_CHARS - 1]}_{digest}"


def _close_objects(schema: dict[str, Any]) -> dict[str, Any]:
    """Forbid unlisted properties on every object in the schema.

    OpenAI's strict json_schema mode requires it, and it keeps a permissive
    engine from inventing fields the signature never declared.
    """
    if schema.get("type") == "object" or "properties" in schema:
        schema["additionalProperties"] = False
    for key in ("properties", "$defs", "definitions"):
        for value in schema.get(key, {}).values():
            if isinstance(value, dict):
                _close_objects(value)
    for key in ("items", "additionalItems"):
        value = schema.get(key)
        if isinstance(value, dict):
            _close_objects(value)
    for key in ("anyOf", "oneOf", "allOf"):
        for value in schema.get(key, []):
            if isinstance(value, dict):
                _close_objects(value)
    return schema


def signature_response_format(signature: type[Signature]) -> dict[str, Any]:
    """The OpenAI ``json_schema`` response_format for a signature's outputs.

    Every output field is required, so an object missing one cannot be
    generated.
    """
    fields: dict[str, Any] = {
        name: (info.annotation if info.annotation is not None else str, ...)
        for name, info in signature.output_fields.items()
    }
    if not fields:
        raise ValueError(f"{signature.__name__} declares no output fields")
    name = _schema_name(signature)
    model = pydantic.create_model(name, **fields)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": _close_objects(model.model_json_schema()),
        },
    }


class StructuredJSONAdapter(JSONAdapter):
    """JSONAdapter that always constrains the response to the signature's schema.

    ChatAdapter's call is invoked directly so the response_format survives:
    `JSONAdapter.__call__` rewrites it to `{"type": "json_object"}` for any
    model litellm does not recognise. A parse failure raises rather than
    retrying under a second adapter, so a server that ignored the schema is
    visible instead of silently reprompted.
    """

    def _schema_kwargs(
        self, lm_kwargs: dict[str, Any], signature: type[Signature]
    ) -> dict[str, Any]:
        return {**lm_kwargs, "response_format": signature_response_format(signature)}

    def __call__(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return ChatAdapter.__call__(
            self,
            lm,
            self._schema_kwargs(lm_kwargs, signature),
            signature,
            demos,
            inputs,
        )

    async def acall(
        self,
        lm: LM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return await ChatAdapter.acall(
            self,
            lm,
            self._schema_kwargs(lm_kwargs, signature),
            signature,
            demos,
            inputs,
        )
