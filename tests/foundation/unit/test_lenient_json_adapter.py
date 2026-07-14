"""LenientJSONAdapter — the process-wide DSPy adapter's rewrite contract.

Every structured LM output at runtime flows through parse() (installed
globally via dspy.configure in the runtime lifespan). It renames known
field-name aliases, recovers a single unknown/missing pair, and fills
missing output fields with type-shaped defaults so downstream pydantic
validation holds instead of 500ing the request.
"""

from __future__ import annotations

import dspy
import pytest

from cogniverse_foundation.dspy.lenient_json_adapter import (
    LenientJSONAdapter,
    _default_for,
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


class TestRecovery:
    def test_single_unknown_swaps_into_single_missing(self, adapter):
        out = adapter.parse(
            PlanSignature, '{"reasoning": "r", "weird_key": ["x", "y"]}'
        )
        assert out == {"reasoning": "r", "sub_questions": ["x", "y"]}

    def test_missing_fields_fill_with_typed_defaults(self, adapter):
        out = adapter.parse(PlanSignature, '{"reasoning": "r"}')
        assert out == {"reasoning": "r", "sub_questions": []}

    def test_bare_collection_annotations_fill_with_typed_defaults(self, adapter):
        """Bare ``list``/``dict`` annotations have no ``__origin__`` — the
        default-fill must still produce []/{} (a str default fails the
        parent's validation and 500s the request)."""
        out = adapter.parse(BareCollectionsSignature, "{}")
        assert out == {"items": [], "mapping": {}}


class TestDefaultFor:
    @pytest.mark.parametrize(
        "annotation,expected",
        [
            (str, ""),
            (int, 0),
            (float, 0),
            (bool, False),
            (list, []),
            (dict, {}),
            (tuple, []),
            (set, []),
            (list[str], []),
            (dict[str, int], {}),
        ],
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
    def test_type_shaped_defaults(self, annotation, expected):
        assert _default_for(annotation) == expected
