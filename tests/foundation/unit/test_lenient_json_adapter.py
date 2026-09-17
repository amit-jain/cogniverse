"""LenientJSONAdapter — the process-wide DSPy adapter's rewrite contract.

Every structured LM output at runtime flows through parse() (installed
globally via dspy.configure in the runtime lifespan). It renames known
field-name aliases and rejects responses missing any required output.
"""

from __future__ import annotations

import dspy
import pytest
from dspy.utils.exceptions import AdapterParseError

from cogniverse_foundation.dspy.lenient_json_adapter import (
    LenientJSONAdapter,
    LMOutputIncomplete,
)


class PlanSignature(dspy.Signature):
    """Plan sub-questions for a query."""

    query: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    sub_questions: list[str] = dspy.OutputField()


class SummarySignature(dspy.Signature):
    """Summarize content."""

    content: str = dspy.InputField()
    summary: str = dspy.OutputField()


class PlanStepsSignature(dspy.Signature):
    """Plan agents with steps that may run together."""

    query: str = dspy.InputField()
    agent_sequence: str = dspy.OutputField()
    parallel_steps: str = dspy.OutputField(default="")


class BareCollectionsSignature(dspy.Signature):
    """Signature with bare (unparameterized) collection annotations."""

    query: str = dspy.InputField()
    items: list = dspy.OutputField()
    mapping: dict = dspy.OutputField()


@pytest.fixture
def adapter() -> LenientJSONAdapter:
    return LenientJSONAdapter()


class TestAliasRemap:
    def test_reason_remaps_to_reasoning(self, adapter):
        out = adapter.parse(
            PlanSignature, '{"reason": "because", "sub_questions": ["a", "b"]}'
        )
        assert out == {"reasoning": "because", "sub_questions": ["a", "b"]}

    @pytest.mark.parametrize("alias", ["rationale", "thought", "thoughts", "reasons"])
    def test_reasoning_alias_family(self, adapter, alias):
        out = adapter.parse(PlanSignature, f'{{"{alias}": "r", "sub_questions": []}}')
        assert out == {"reasoning": "r", "sub_questions": []}

    @pytest.mark.parametrize(
        "alias", ["answer", "response", "result", "output", "text"]
    )
    def test_summary_alias_family(self, adapter, alias):
        out = adapter.parse(SummarySignature, f'{{"{alias}": "the gist"}}')
        assert out == {"summary": "the gist"}

    def test_alias_skipped_when_canonical_already_present(self, adapter):
        out = adapter.parse(
            PlanSignature,
            '{"reason": "loser", "reasoning": "winner", "sub_questions": []}',
        )
        assert out == {"reasoning": "winner", "sub_questions": []}

    def test_correct_payload_passes_through_unchanged(self, adapter):
        out = adapter.parse(PlanSignature, '{"reasoning": "r", "sub_questions": ["x"]}')
        assert out == {"reasoning": "r", "sub_questions": ["x"]}


class TestIncompleteOutputs:
    def test_single_unknown_does_not_supply_missing_output(self, adapter):
        with pytest.raises(LMOutputIncomplete) as error:
            adapter.parse(PlanSignature, '{"reasoning": "r", "weird_key": ["x", "y"]}')
        assert error.value.missing_fields == ("sub_questions",)
        assert error.value.parsed_result == {"reasoning": "r"}
        assert error.value.signature is PlanSignature
        assert error.value.lm_response == '{"reasoning": "r", "weird_key": ["x", "y"]}'
        assert "The LM produced no sub_questions" in str(error.value)

    def test_missing_fields_raise(self, adapter):
        with pytest.raises(LMOutputIncomplete) as error:
            adapter.parse(PlanSignature, '{"reasoning": "r"}')
        assert error.value.missing_fields == ("sub_questions",)
        assert error.value.parsed_result == {"reasoning": "r"}
        assert error.value.signature is PlanSignature

    def test_every_missing_field_is_named(self, adapter):
        with pytest.raises(LMOutputIncomplete) as error:
            adapter.parse(PlanSignature, '{"query": "ignored"}')
        assert error.value.missing_fields == ("reasoning", "sub_questions")
        assert error.value.parsed_result == {}
        assert "The LM produced no reasoning, sub_questions" in str(error.value)

    def test_bare_collection_annotations_require_outputs(self, adapter):
        with pytest.raises(LMOutputIncomplete) as error:
            adapter.parse(BareCollectionsSignature, "{}")
        assert error.value.missing_fields == ("items", "mapping")
        assert error.value.parsed_result == {}
        assert error.value.signature is BareCollectionsSignature

    def test_null_value_is_not_a_produced_output(self, adapter):
        with pytest.raises(LMOutputIncomplete) as error:
            adapter.parse(SummarySignature, '{"summary": null}')
        assert error.value.missing_fields == ("summary",)
        assert error.value.parsed_result == {}
        assert error.value.signature is SummarySignature
        assert error.value.lm_response == '{"summary": null}'
        assert "The LM produced no summary" in str(error.value)

    @pytest.mark.parametrize(
        "blank", ['""', '"   "', r'"\n\t"'], ids=["empty", "spaces", "whitespace"]
    )
    def test_blank_string_is_not_a_produced_output(self, adapter, blank):
        with pytest.raises(LMOutputIncomplete) as error:
            adapter.parse(SummarySignature, '{"summary": %s}' % blank)
        assert error.value.missing_fields == ("summary",)
        assert error.value.parsed_result == {}

    @pytest.mark.parametrize(
        "blank", ['""', '"   "', "null"], ids=["empty", "spaces", "null"]
    )
    def test_defaulted_output_takes_its_default_for_a_blank_answer(
        self, adapter, blank
    ):
        assert adapter.parse(
            PlanStepsSignature,
            '{"agent_sequence": "search_agent,summarizer_agent", '
            '"parallel_steps": %s}' % blank,
        ) == {"agent_sequence": "search_agent,summarizer_agent", "parallel_steps": ""}

    def test_defaulted_output_keeps_a_filled_answer(self, adapter):
        assert adapter.parse(
            PlanStepsSignature,
            '{"agent_sequence": "a,b,c", "parallel_steps": "0,1|2"}',
        ) == {"agent_sequence": "a,b,c", "parallel_steps": "0,1|2"}

    def test_defaulted_output_is_incomplete_when_absent(self, adapter):
        with pytest.raises(LMOutputIncomplete) as error:
            adapter.parse(PlanStepsSignature, '{"agent_sequence": "search_agent"}')
        assert error.value.missing_fields == ("parallel_steps",)
        assert error.value.parsed_result == {"agent_sequence": "search_agent"}

    def test_required_output_stays_incomplete_beside_a_defaulted_blank(self, adapter):
        with pytest.raises(LMOutputIncomplete) as error:
            adapter.parse(
                PlanStepsSignature, '{"agent_sequence": "", "parallel_steps": ""}'
            )
        assert error.value.missing_fields == ("agent_sequence",)
        assert error.value.parsed_result == {"parallel_steps": ""}

    def test_unfilled_fields_are_named_beside_absent_ones(self, adapter):
        with pytest.raises(LMOutputIncomplete) as error:
            adapter.parse(PlanSignature, '{"reasoning": ""}')
        assert error.value.missing_fields == ("reasoning", "sub_questions")
        assert error.value.parsed_result == {}
        assert "The LM produced no reasoning, sub_questions" in str(error.value)

    def test_null_alias_does_not_fill_its_canonical_field(self, adapter):
        with pytest.raises(LMOutputIncomplete) as error:
            adapter.parse(SummarySignature, '{"answer": null}')
        assert error.value.missing_fields == ("summary",)
        assert error.value.parsed_result == {}

    def test_empty_collections_are_produced_outputs(self, adapter):
        assert adapter.parse(
            BareCollectionsSignature, '{"items": [], "mapping": {}}'
        ) == {"items": [], "mapping": {}}

    def test_false_and_zero_are_produced_outputs(self, adapter):
        signature = dspy.Signature("query -> flag, score").with_updated_fields(
            "flag", type_=bool
        )
        signature = signature.with_updated_fields("score", type_=int)
        assert adapter.parse(signature, '{"flag": false, "score": 0}') == {
            "flag": False,
            "score": 0,
        }

    def test_incomplete_output_is_an_adapter_parse_error(self, adapter):
        with pytest.raises(AdapterParseError) as error:
            adapter.parse(SummarySignature, '{"note": "nothing usable"}')
        assert type(error.value) is LMOutputIncomplete
        assert error.value.missing_fields == ("summary",)


class TestRequiredTypes:
    @pytest.mark.parametrize(
        "annotation",
        [str, int, float, bool, list, dict, tuple, set, list[str], dict[str, int]],
        ids=[
            "str",
            "int",
            "float",
            "bool",
            "bare_list",
            "bare_dict",
            "bare_tuple",
            "bare_set",
            "parameterized_list",
            "parameterized_dict",
        ],
    )
    def test_missing_typed_output_raises(self, adapter, annotation):
        signature = dspy.Signature("query -> value").with_updated_fields(
            "value", type_=annotation
        )
        with pytest.raises(LMOutputIncomplete) as error:
            adapter.parse(signature, "{}")
        assert error.value.missing_fields == ("value",)
        assert error.value.parsed_result == {}
        assert set(error.value.signature.output_fields) == {"value"}

    def test_truncated_reasoning_does_not_create_summary(self, adapter):
        signature = dspy.ChainOfThought(SummarySignature).predict.signature
        with pytest.raises(LMOutputIncomplete) as error:
            adapter.parse(signature, '{"reasoning":"Reading the source')
        assert error.value.missing_fields == ("summary",)
        assert error.value.parsed_result == {"reasoning": "Reading the source"}
        assert set(
            error.value.signature.output_fields
        ) - error.value.parsed_result.keys() == {"summary"}
