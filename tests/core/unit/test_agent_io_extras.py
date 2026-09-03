"""Contract of the AgentBase input/output validation seam.

Every agent's raw request dict enters through ``AgentBase.validate_input``
(``self._input_type.model_validate``).  Context fields an upstream threads
through that an input model does not declare must survive validation.
Outputs stay closed: an undeclared output field is a programming error and
must raise.
"""

import pytest
from pydantic import ValidationError

from cogniverse_core.agents.base import (
    AgentBase,
    AgentDeps,
    AgentInput,
    AgentOutput,
    AgentValidationError,
)


class _ProbeInput(AgentInput):
    query: str


class _ProbeOutput(AgentOutput):
    answer: str


class _ProbeAgent(AgentBase[_ProbeInput, _ProbeOutput, AgentDeps]):
    async def _process_impl(self, input: _ProbeInput) -> _ProbeOutput:
        return _ProbeOutput(answer=input.query)


class TestAgentInputRetainsExtras:
    def test_validate_input_retains_undeclared_context_fields(self):
        agent = _ProbeAgent(deps=AgentDeps())

        validated = agent.validate_input(
            {
                "query": "robot arm assembly",
                "enhanced_query": "robot arm assembly steps",
                "search_results": [{"id": "video_1", "score": 0.9}],
            }
        )

        assert validated.query == "robot arm assembly"
        assert validated.enhanced_query == "robot arm assembly steps"
        assert validated.search_results == [{"id": "video_1", "score": 0.9}]
        assert validated.model_dump() == {
            "query": "robot arm assembly",
            "enhanced_query": "robot arm assembly steps",
            "search_results": [{"id": "video_1", "score": 0.9}],
        }

    def test_declared_fields_still_validate(self):
        agent = _ProbeAgent(deps=AgentDeps())

        with pytest.raises(AgentValidationError) as exc_info:
            agent.validate_input({"enhanced_query": "no query supplied"})

        errors = exc_info.value.validation_error.errors()
        assert errors[0]["loc"] == ("query",)
        assert errors[0]["type"] == "missing"


class TestAgentOutputStaysClosed:
    def test_undeclared_output_field_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            _ProbeOutput.model_validate({"answer": "a", "confidence": 0.9})

        assert exc_info.value.errors()[0]["loc"] == ("confidence",)
        assert exc_info.value.errors()[0]["type"] == "extra_forbidden"
