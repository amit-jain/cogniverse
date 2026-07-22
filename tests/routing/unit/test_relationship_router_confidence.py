"""ComposableQueryAnalysisModule._parse_confidence handles LM label/percent
shapes, not just plain floats."""

from __future__ import annotations

import pytest

from cogniverse_agents.routing.dspy_relationship_router import (
    ComposableQueryAnalysisModule,
)

pytestmark = [pytest.mark.unit]


@pytest.fixture
def module():
    return object.__new__(ComposableQueryAnalysisModule)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("high", 0.9),
        ("85%", 0.85),
        ("0.9", 0.9),
        (1.5, 1.0),
        ("not a number", 0.5),
        (0.42, 0.42),
    ],
)
def test_parse_confidence(module, value, expected):
    assert module._parse_confidence(value) == expected
