"""The provider's span-attribute unflattening matches Phoenix 20.3.0's.

``_normalize_span_page`` nests dotted span attributes the way
``phoenix.trace.attributes.unflatten`` does, without importing the Phoenix
server package. Each case pins the output Phoenix 20.3.0 gives and compares
against that function's output recorded in
``fixtures/phoenix_20_3_0_unflatten.json``; a seeded batch of generated
inputs is compared the same way.
"""

import json
import random
from pathlib import Path

import pytest

from cogniverse_telemetry_phoenix.provider import _unflatten_attributes

CASES = [
    (
        [("llm.token_count.completion", 123), ("llm.token_count.prompt", 4)],
        {"llm": {"token_count": {"completion": 123, "prompt": 4}}},
    ),
    (
        [("documents.0.content", "A"), ("documents.1.content", "B")],
        {"documents": [{"content": "A"}, {"content": "B"}]},
    ),
    ([("tags.0", "python"), ("tags.1", "ai")], {"tags": {"0": "python", "1": "ai"}}),
    ([("a", {"b": 1}), ("a.c", 2)], {"a": {"b": 1}, "a.c": 2}),
    ([("a.c", 2), ("a", {"b": 1})], {"a": {"b": 1}, "a.c": 2}),
    ([("a", None), ("b", 1)], {"b": 1}),
    ([("a.b", None)], {}),
    ([("a.b", None), ("a.c", 1)], {"a": {"c": 1}}),
    ([("a.00.b", 1), ("a.0.c", 2)], {"a": [{"b": 1, "c": 2}]}),
    ([("a.-1.b", 1)], {"a": {"-1": {"b": 1}}}),
    ([("a..b", 1)], {"a": {"b": 1}}),
    ([("a.", 1)], {"a": 1}),
    ([(".a", 1)], {"a": 1}),
    ([("a.0a.b", 1)], {"a": {"0a": {"b": 1}}}),
    ([(" a . b ", 1)], {"a": {"b": 1}}),
    ([("", 1)], {"": 1}),
    ([("a", 1), ("a", 2)], {"a": 2}),
    ([("docs.2.x", 1), ("docs.0.x", 2)], {"docs": [{"x": 2}, {"x": 1}]}),
    ([("a.0.b", 1), ("a.0", 2)], {"a": {"0": 2, "0.b": 1}}),
    ([("a.0", 2), ("a.0.b", 1)], {"a": {"0": 2, "0.b": 1}}),
    ([("a.0.b", 1), ("a.x", 2)], {"a": [{"b": 1}], "a.x": 2}),
    ([("a.x", 2), ("a.0.b", 1)], {"a": [{"b": 1}], "a.x": 2}),
    ([("0.a", 1), ("1.a", 2)], {"0": {"a": 1}, "1": {"a": 2}}),
    ([("0", 1), ("1", 2)], {"0": 1, "1": 2}),
    (
        [("a.0.b.1.c", 1), ("a.0.b.0.c", 2), ("a.1.d", 3)],
        {"a": [{"b": [{"c": 2}, {"c": 1}]}, {"d": 3}]},
    ),
    ([("a.b", 1), ("a.b.c", 2)], {"a": {"b": 1, "b.c": 2}}),
    ([("a.b.c", 2), ("a.b", 1)], {"a": {"b": 1, "b.c": 2}}),
    ([("a", [1, 2]), ("a.0.b", 3)], {"a": [1, 2], "a.0": {"b": 3}}),
    ([("x.1.y", 1)], {"x": [{"y": 1}]}),
    ([("a", 0), ("b", False), ("c", "")], {"a": 0, "b": False, "c": ""}),
    ([("a.٣.b", 1)], {"a": [{"b": 1}]}),
]


_SEGMENTS = ["a", "b", "x", "0", "1", "2", "00", " c ", "-1", "0a"]
_VALUES = [1, "s", None, {"k": 1}, [1, 2], [{"k": 2}], 0, False, ""]


def _restore_values(node):
    """Map a recorded ``{"$value": i}`` back to the very ``_VALUES[i]`` object.

    Phoenix passes a list or dict attribute value through as the same
    object, so the recording marks those by index and loading restores
    their identity.
    """
    if set(node) == {"$value"}:
        return _VALUES[node["$value"]]
    return node


_PHOENIX = json.loads(
    (Path(__file__).parent / "fixtures" / "phoenix_20_3_0_unflatten.json").read_text(
        encoding="utf-8"
    ),
    object_hook=_restore_values,
)


@pytest.mark.parametrize(
    ("index", "pairs", "expected"),
    [(index, pairs, expected) for index, (pairs, expected) in enumerate(CASES)],
)
def test_unflatten_matches_phoenix_on_documented_cases(index, pairs, expected):
    assert _PHOENIX["cases"][index] == expected
    assert _unflatten_attributes(pairs) == expected


def test_unflatten_rejects_a_non_decimal_digit_segment_as_phoenix_does():
    pairs = [("a.²", 1)]

    with pytest.raises(ValueError) as ours:
        _unflatten_attributes(pairs)

    assert str(ours.value) == _PHOENIX["non_decimal_digit_error"]
    assert str(ours.value) == "invalid literal for int() with base 10: '²'"


def _generated_pairs(rng: random.Random) -> list[tuple[str, object]]:
    return [
        (
            ".".join(rng.choice(_SEGMENTS) for _ in range(rng.randint(1, 4))),
            rng.choice(_VALUES),
        )
        for _ in range(rng.randint(1, 6))
    ]


def _nodes(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _nodes(child)


def test_unflatten_matches_phoenix_on_generated_inputs():
    rng = random.Random(20260928)
    inputs = [_generated_pairs(rng) for _ in range(5000)]
    expected = _PHOENIX["generated"]

    actual = [_unflatten_attributes(pairs) for pairs in inputs]

    assert [
        (pairs, ours, theirs)
        for pairs, ours, theirs in zip(inputs, actual, expected, strict=True)
        if ours != theirs
    ] == []
    built_lists = [
        output
        for output in expected
        if any(
            isinstance(node, list) and all(node is not v for v in _VALUES)
            for node in _nodes(output)
        )
    ]
    dotted_keys = [
        output
        for output in expected
        if any(
            isinstance(node, dict) and any("." in key for key in node)
            for node in _nodes(output)
        )
    ]
    assert (len(built_lists), len(dotted_keys), expected.count({})) == (2690, 1288, 90)
