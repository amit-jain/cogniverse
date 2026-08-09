"""Contract tests for relationship-aware agent routing inputs."""

from unittest.mock import Mock, patch

import dspy
import pytest

from cogniverse_agents.optimizer.dspy_agent_optimizer import (
    DSPyAgentOptimizerPipeline,
    DSPyAgentPromptOptimizer,
)
from cogniverse_agents.routing.dspy_relationship_router import (
    DSPyAdvancedRoutingModule,
    DSPyBasicRoutingModule,
)


def test_optimizer_routing_examples_use_agent_name_lists():
    pipeline = DSPyAgentOptimizerPipeline(DSPyAgentPromptOptimizer())

    examples = pipeline.load_training_data()["agent_routing"]

    assert [example.available_agents for example in examples] == [
        ["video_search"],
        ["detailed_report"],
        ["summarizer"],
    ]


def test_optimizer_compile_receives_agent_name_lists():
    pipeline = DSPyAgentOptimizerPipeline(DSPyAgentPromptOptimizer())
    pipeline.initialize_modules()
    examples = pipeline.load_training_data()["agent_routing"]
    compiled_module = Mock()

    with patch(
        "cogniverse_agents.optimizer.dspy_agent_optimizer.BootstrapFewShot"
    ) as bootstrap:
        bootstrap.return_value.compile.return_value = compiled_module
        result = pipeline.optimize_module("agent_routing", examples)

    assert result is compiled_module
    assert bootstrap.return_value.compile.call_args.kwargs["trainset"] == examples
    assert [example.available_agents for example in examples] == [
        ["video_search"],
        ["detailed_report"],
        ["summarizer"],
    ]


def test_basic_router_selects_exact_agent_from_list():
    router = DSPyBasicRoutingModule()
    available_agents = ["summarizer_agent", "search_agent"]

    search = router.forward(
        "Find the robot soccer clip", available_agents=available_agents
    )
    summary = router.forward(
        "Summarize the robot soccer clip", available_agents=available_agents
    )

    assert search.recommended_agent == "search_agent"
    assert summary.recommended_agent == "summarizer_agent"


@pytest.mark.parametrize("available_agents", ["search_agent", 42, ("search_agent",)])
def test_basic_router_rejects_non_list_available_agents(available_agents):
    router = DSPyBasicRoutingModule()

    with pytest.raises(
        TypeError, match=r"available_agents must be a list\[str\] or None"
    ):
        router.forward("Find the robot soccer clip", available_agents=available_agents)


def test_basic_router_rejects_empty_agent_name():
    router = DSPyBasicRoutingModule()

    with pytest.raises(
        ValueError, match="available_agents must not contain empty agent names"
    ):
        router.forward(
            "Find the robot soccer clip", available_agents=["search_agent", ""]
        )


def test_basic_router_rejects_empty_available_agents_list():
    router = DSPyBasicRoutingModule()

    with pytest.raises(
        ValueError, match="available_agents must contain at least one agent name"
    ):
        router.forward("Find the robot soccer clip", available_agents=[])


@pytest.mark.parametrize("agent_name", [" search_agent", "search_agent "])
def test_basic_router_rejects_agent_name_with_surrounding_whitespace(agent_name):
    router = DSPyBasicRoutingModule()

    with pytest.raises(
        ValueError,
        match=r"available_agents\[0\] must not contain surrounding whitespace",
    ):
        router.forward("Find the robot soccer clip", available_agents=[agent_name])


def test_basic_router_rejects_non_string_agent_name():
    router = DSPyBasicRoutingModule()

    with pytest.raises(
        TypeError, match=r"available_agents\[1\] must be a non-empty str"
    ):
        router.forward(
            "Find the robot soccer clip",
            available_agents=["search_agent", 7],
        )


def test_advanced_router_propagates_available_agents_contract_error():
    analysis_module = Mock()
    analysis_module.forward.return_value = dspy.Prediction()
    router = DSPyAdvancedRoutingModule(analysis_module=analysis_module)

    with pytest.raises(
        TypeError, match=r"available_agents must be a list\[str\] or None"
    ):
        router.forward("Find the robot soccer clip", available_agents="search_agent")

    analysis_module.forward.assert_not_called()
