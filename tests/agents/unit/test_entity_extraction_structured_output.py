"""The entity signature's output fields reach the server as an enforced schema.

The optimizer's bootstrap and the served agent both run
``EntityExtractionModule``. Both must send one prompt and one
``response_format`` carrying every output field, so the engine's guided
decoding cannot answer with an object that omits them.
"""

from __future__ import annotations

import json
from typing import Any

import dspy
import pydantic
import pytest
from dspy.utils.exceptions import AdapterParseError

from cogniverse_agents.entity_extraction_agent import (
    ENTITY_TYPES,
    EntityExtractionModule,
    EntityExtractionSignature,
    EntityMention,
)
from cogniverse_core.agents.base import _register_stream_adapter
from cogniverse_foundation.dspy import (
    LenientJSONAdapter,
    StructuredJSONAdapter,
    signature_response_format,
)

pytestmark = [pytest.mark.unit, pytest.mark.ci_fast]

QUERY = "What are the people doing behind the car at the beginning of the video?"

# The completion a schema-constrained engine can produce for this signature.
CONFORMING_COMPLETION = json.dumps(
    {
        "reasoning": "people and car are the spans, in first-appearance order.",
        "entities": [
            {"text": "people", "type": "PERSON"},
            {"text": "car", "type": "CONCEPT"},
        ],
    }
)

# Completions a server that ignored the schema could return: a type outside
# the vocabulary, an item missing a key, an item with an extra key, and the
# free-text form entities had before it was typed.
SCHEMA_VIOLATING_COMPLETIONS = {
    "type_outside_vocabulary": json.dumps(
        {"reasoning": "r", "entities": [{"text": "people", "type": "Person"}]}
    ),
    "item_missing_type": json.dumps(
        {"reasoning": "r", "entities": [{"text": "people"}]}
    ),
    "item_with_extra_key": json.dumps(
        {
            "reasoning": "r",
            "entities": [{"text": "people", "type": "PERSON", "confidence": 0.9}],
        }
    ),
    "entities_as_text": json.dumps(
        {"reasoning": "r", "entities": "people|PERSON|1.0\ncar|CONCEPT|1.0"}
    ),
}

# What the teacher returned on the run that motivated the schema: valid JSON,
# no output fields. json_object mode accepts it; a json_schema cannot emit it.
EMPTY_OBJECT_COMPLETION = "{}"


class _Item(pydantic.BaseModel):
    text: str


class _RecordingLM(dspy.BaseLM):
    """An LM that records the kwargs the adapter puts on the wire."""

    def __init__(self, completion: str = CONFORMING_COMPLETION):
        super().__init__(model="openai/Qwen/Qwen3-14B-AWQ")
        self.completion = completion
        self.calls: list[dict[str, Any]] = []

    def __call__(self, prompt=None, messages=None, **kwargs):
        self.calls.append({"prompt": prompt, "messages": messages, "kwargs": kwargs})
        return [self.completion]


def _run(module: EntityExtractionModule, lm: _RecordingLM, **context: Any):
    with dspy.context(lm=lm, **context):
        return module(query=QUERY)


class TestEntitySignatureSchemaOnTheWire:
    def test_response_format_is_the_signature_derived_json_schema(self):
        lm = _RecordingLM()
        module = EntityExtractionModule()

        _run(module, lm)

        assert len(lm.calls) == 1
        response_format = lm.calls[0]["kwargs"]["response_format"]
        assert response_format == signature_response_format(
            module.extractor.predict.signature
        )
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True

        schema = response_format["json_schema"]["schema"]
        assert tuple(schema["properties"]) == ("reasoning", "entities")
        assert {
            name: field["type"] for name, field in schema["properties"].items()
        } == {"reasoning": "string", "entities": "array"}
        assert schema["properties"]["entities"]["items"] == {
            "$ref": "#/$defs/EntityMention"
        }
        assert schema["required"] == ["reasoning", "entities"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["$defs"] == {
            "EntityMention": {
                "additionalProperties": False,
                "description": (
                    "One entity as the extraction signature's output schema carries it."
                ),
                "properties": {
                    "text": {
                        "description": "Verbatim span of the query",
                        "title": "Text",
                        "type": "string",
                    },
                    "type": {
                        "enum": [
                            "CONCEPT",
                            "EVENT",
                            "ORGANIZATION",
                            "PERSON",
                            "PLACE",
                            "TECHNOLOGY",
                        ],
                        "title": "Type",
                        "type": "string",
                    },
                },
                "required": ["text", "type"],
                "title": "EntityMention",
                "type": "object",
            }
        }
        assert (
            set(schema["$defs"]["EntityMention"]["properties"]["type"]["enum"])
            == ENTITY_TYPES
        )

    def test_schema_properties_track_the_production_signature(self):
        """A rename of the signature's output field moves the enforced schema."""
        lm = _RecordingLM()

        _run(EntityExtractionModule(), lm)

        schema = lm.calls[0]["kwargs"]["response_format"]["json_schema"]["schema"]
        # `reasoning` is what dspy.ChainOfThought prepends; the rest is the
        # signature's own declaration, read from production, not restated.
        assert set(schema["properties"]) == {
            "reasoning",
            *EntityExtractionSignature.output_fields,
        }
        assert set(schema["required"]) == set(schema["properties"])

    def test_stock_json_adapter_degrades_to_json_object_for_this_model(self):
        """Control: what DSPy sends without the fix, on the same signature."""
        lm = _RecordingLM()
        module = EntityExtractionModule()

        with dspy.context(lm=lm):
            dspy.JSONAdapter()(
                lm, {}, module.extractor.predict.signature, [], {"query": QUERY}
            )

        assert lm.calls[0]["kwargs"]["response_format"] == {"type": "json_object"}

    def test_bootstrap_and_serving_send_the_same_prompt_and_schema(self):
        """The optimizer configures no adapter; the runtime configures a lenient
        one. The module's own binding makes both calls identical."""
        bootstrap_lm = _RecordingLM()
        serving_lm = _RecordingLM()

        _run(EntityExtractionModule(), bootstrap_lm)
        _run(EntityExtractionModule(), serving_lm, adapter=LenientJSONAdapter())

        assert bootstrap_lm.calls[0]["messages"] == serving_lm.calls[0]["messages"]
        assert (
            bootstrap_lm.calls[0]["kwargs"]["response_format"]
            == serving_lm.calls[0]["kwargs"]["response_format"]
        )

    def test_empty_object_raises_instead_of_being_filled_with_defaults(self):
        """A server that ignored the schema must be visible, not defaulted away."""
        lm = _RecordingLM(completion=EMPTY_OBJECT_COMPLETION)

        with pytest.raises(AdapterParseError) as excinfo:
            _run(EntityExtractionModule(), lm, adapter=LenientJSONAdapter())

        assert excinfo.value.lm_response == EMPTY_OBJECT_COMPLETION
        assert len(lm.calls) == 1

    def test_conforming_completion_parses_into_the_prediction(self):
        lm = _RecordingLM()

        prediction = _run(EntityExtractionModule(), lm)

        assert prediction.entities == [
            EntityMention(text="people", type="PERSON"),
            EntityMention(text="car", type="CONCEPT"),
        ]
        assert (
            prediction.reasoning
            == "people and car are the spans, in first-appearance order."
        )

    @pytest.mark.parametrize("shape", sorted(SCHEMA_VIOLATING_COMPLETIONS))
    def test_schema_violating_item_raises_adapter_parse_error(self, shape):
        """An item the schema forbids is the same failure as a missing field:
        AdapterParseError carrying the response, after exactly one call."""
        completion = SCHEMA_VIOLATING_COMPLETIONS[shape]
        lm = _RecordingLM(completion=completion)

        with pytest.raises(AdapterParseError) as excinfo:
            _run(EntityExtractionModule(), lm, adapter=LenientJSONAdapter())

        assert excinfo.value.lm_response == completion
        assert excinfo.value.adapter_name == "StructuredJSONAdapter"
        assert str(excinfo.value).splitlines()[0] == (
            "LM response violates the output schema: 1 validation error for "
            "list[EntityMention]"
        )
        assert len(lm.calls) == 1


class TestSignatureResponseFormat:
    def test_typed_output_fields_become_their_json_types(self):
        class _Typed(dspy.Signature):
            query: str = dspy.InputField()
            count: int = dspy.OutputField()
            labels: list[str] = dspy.OutputField()

        schema = signature_response_format(_Typed)["json_schema"]["schema"]

        assert schema["properties"]["count"]["type"] == "integer"
        assert schema["properties"]["labels"] == {
            "items": {"type": "string"},
            "title": "Labels",
            "type": "array",
        }
        assert schema["required"] == ["count", "labels"]
        assert schema["additionalProperties"] is False

    def test_nested_objects_are_closed_too(self):
        class _Nested(dspy.Signature):
            query: str = dspy.InputField()
            items: list[_Item] = dspy.OutputField()

        schema = signature_response_format(_Nested)["json_schema"]["schema"]

        assert schema["additionalProperties"] is False
        assert schema["$defs"]["_Item"]["additionalProperties"] is False
        assert schema["$defs"]["_Item"]["required"] == ["text"]
        assert schema["properties"]["items"]["items"] == {"$ref": "#/$defs/_Item"}

    def test_signature_without_outputs_is_refused(self):
        class _NoOutputs(dspy.Signature):
            query: str = dspy.InputField()

        assert _NoOutputs.output_fields == {}
        with pytest.raises(ValueError, match="declares no output fields"):
            signature_response_format(_NoOutputs)


class TestSchemaNameIdentifiesTheSignature:
    """dspy names a signature built from a string, rebuilt by
    ``with_instructions()`` or wrapped by ``ChainOfThought``, ``StringSignature``
    — every one of them. A schema named by that class name identifies nothing,
    so two signatures in flight are indistinguishable in the engine's log.
    """

    class _Alpha(dspy.Signature):
        query: str = dspy.InputField()
        alpha: str = dspy.OutputField()

    class _Beta(dspy.Signature):
        query: str = dspy.InputField()
        beta: str = dspy.OutputField()

    @staticmethod
    def _name(signature) -> str:
        return signature_response_format(signature)["json_schema"]["name"]

    def test_two_generic_signatures_get_two_names(self):
        """Both carry dspy's placeholder class name; the schemas must not."""
        alpha = dspy.ChainOfThought(self._Alpha).predict.signature
        beta = dspy.ChainOfThought(self._Beta).predict.signature

        assert alpha.__name__ == beta.__name__ == "StringSignature"
        assert self._name(alpha) != self._name(beta)
        assert (self._name(alpha), self._name(beta)) == (
            "reasoning_alpha_7e6b5bd8",
            "reasoning_beta_1dff36e1",
        )

    def test_the_entity_schema_is_named_for_its_own_fields(self):
        module = EntityExtractionModule()

        name = self._name(module.extractor.predict.signature)

        assert module.extractor.predict.signature.__name__ == "StringSignature"
        assert name == "reasoning_entities_23b2f964"
        assert name == self._name(module.extractor.predict.signature)

    def test_a_named_signature_keeps_its_own_name(self):
        class NamedOutputs(dspy.Signature):
            query: str = dspy.InputField()
            answer: str = dspy.OutputField()

        assert self._name(NamedOutputs) == "NamedOutputs"

    def test_the_name_tracks_the_declaration_not_the_instructions(self):
        """An optimizer rewriting instructions must not rename the schema."""
        rewritten = self._Alpha.with_instructions("a rewritten instruction")

        assert rewritten.instructions != self._Alpha.instructions
        assert self._name(rewritten) == self._name(
            self._Alpha.with_instructions(self._Alpha.instructions)
        )


class TestAdapterName:
    def test_module_binds_the_structured_adapter(self):
        assert isinstance(EntityExtractionModule().dspy_adapter, StructuredJSONAdapter)


class TestStreamListenerKnowsTheBoundAdapter:
    """StreamListener keys support by exact class name and raises on a subclass.

    The module binds its own adapter, so that class — not the ambient one — is
    what `settings.adapter` holds while the listener consumes chunks.
    """

    @staticmethod
    def _chunk(content: str):
        from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

        return ModelResponseStream(
            choices=[StreamingChoices(index=0, delta=Delta(content=content))]
        )

    def test_unregistered_adapter_class_breaks_the_listener(self):
        """Control: what the listener does with a subclass it was not taught."""
        listener = dspy.streaming.StreamListener("entities")

        with dspy.context(adapter=StructuredJSONAdapter()):
            with pytest.raises(ValueError) as excinfo:
                listener.receive(self._chunk('{"reasoning": "x"'))

        assert "Unsupported adapter for streaming: StructuredJSONAdapter" in str(
            excinfo.value
        )

    def test_call_dspy_registers_the_modules_adapter(self):
        listener = dspy.streaming.StreamListener("entities")
        module = EntityExtractionModule()

        _register_stream_adapter(listener, LenientJSONAdapter())
        _register_stream_adapter(listener, module.dspy_adapter)

        assert (
            listener.adapter_identifiers["StructuredJSONAdapter"]
            == listener.adapter_identifiers["JSONAdapter"]
        )
        assert (
            listener.adapter_identifiers["LenientJSONAdapter"]
            == listener.adapter_identifiers["JSONAdapter"]
        )

        received = []
        with dspy.context(adapter=module.dspy_adapter):
            for piece in (
                '{"reasoning": "x", ',
                '"entities": [{"text": "peo',
                'ple", "type": "PERSON"}',
                "]}",
            ):
                response = listener.receive(self._chunk(piece))
                received.append(None if response is None else response.chunk)

        assert received == [None, '[{"text": "peo', None, 'ple", "type": "PERSON"}]']
        assert json.loads("".join(chunk for chunk in received if chunk)) == [
            {"text": "people", "type": "PERSON"}
        ]
        assert listener.stream_end is True

    def test_an_adapter_with_no_supported_ancestor_is_left_unregistered(self):
        class _Foreign:
            pass

        listener = dspy.streaming.StreamListener("entities")
        before = dict(listener.adapter_identifiers)

        _register_stream_adapter(listener, _Foreign())
        _register_stream_adapter(listener, None)

        assert listener.adapter_identifiers == before
