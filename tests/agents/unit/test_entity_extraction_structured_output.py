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
    EntityExtractionModule,
    EntityExtractionSignature,
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
        "entities": "people|PERSON|1.0\ncar|CONCEPT|1.0",
    }
)

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
        } == {"reasoning": "string", "entities": "string"}
        assert schema["required"] == ["reasoning", "entities"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False

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

        assert prediction.entities == "people|PERSON|1.0\ncar|CONCEPT|1.0"
        assert (
            prediction.reasoning
            == "people and car are the spans, in first-appearance order."
        )


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
                '"entities": "peo',
                "ple|PERSON",
                '|1.0"}',
            ):
                response = listener.receive(self._chunk(piece))
                received.append(None if response is None else response.chunk)

        assert received == [None, '"peo', "ple|PERSON", '|1.0"']
        assert listener.stream_end is True

    def test_an_adapter_with_no_supported_ancestor_is_left_unregistered(self):
        class _Foreign:
            pass

        listener = dspy.streaming.StreamListener("entities")
        before = dict(listener.adapter_identifiers)

        _register_stream_adapter(listener, _Foreign())
        _register_stream_adapter(listener, None)

        assert listener.adapter_identifiers == before
