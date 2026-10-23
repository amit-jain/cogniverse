"""An orchestrated follow-up reaches the orchestrator resolved against the
conversation.

The orchestrator's planner reads the history, but the agents it runs receive
only the query, so ``_execute_orchestration_task`` rewrites a follow-up such as
"Tell me more about the first one" with the real LM before the orchestrator
sees it. A stand-in ``OrchestratorAgent._process_impl`` records the input it is
handed; everything before it — the dispatcher, the rewrite module and the LM
— is the production path.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import litellm
import pytest

from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_foundation.config.body_bounded_lm import BodyBoundedLM
from cogniverse_foundation.config.utils import create_default_config_manager
from cogniverse_runtime.agent_dispatcher import AgentDispatcher
from tests.runtime.integration.conftest import skip_if_no_lm

pytestmark = [pytest.mark.integration, skip_if_no_lm]

TENANT = "followup_org:production"
FOLLOW_UP = "Tell me more about the first one"


def _first_turn(video: str) -> list[dict]:
    return [
        {"role": "user", "content": "search for sports clips"},
        {
            "role": "assistant",
            "content": (
                "Found 1 results for 'search for sports clips'\n"
                f"- {video} (0:05–0:06): This video frame captures an outdoor "
                "scene, likely a sports event or practice, viewed through a "
                "chain-link fence."
            ),
        },
    ]


@pytest.fixture
def orchestrated():
    """A dispatcher whose orchestrator records each input and holds it at
    ``held`` (when set) until every concurrent caller has arrived."""
    from cogniverse_agents.orchestrator_agent import OrchestratorOutput

    config_manager = create_default_config_manager()
    registry = AgentRegistry(tenant_id=TENANT, config_manager=config_manager)
    registry.register_agent(
        AgentEndpoint(
            name="orchestrator_agent",
            url="http://localhost:8011",
            capabilities=["orchestration"],
        )
    )
    dispatcher = AgentDispatcher(
        agent_registry=registry, config_manager=config_manager, schema_loader=None
    )
    state = {"inputs": [], "held": None}

    async def _record(self, input_data):
        state["inputs"].append(input_data)
        if state["held"] is not None:
            await state["held"].wait()
        return OrchestratorOutput(
            query=input_data.query,
            workflow_id="recorded",
            plan_steps=[],
            plan_reasoning="recorded",
            agent_results={},
            final_output={"status": "success", "message": input_data.query},
        )

    with (
        patch(
            "cogniverse_agents.orchestrator_agent.OrchestratorAgent._process_impl",
            new=_record,
        ),
        patch(
            "cogniverse_runtime.agent_dispatcher.AgentDispatcher._init_agent_memory",
            new=lambda *a, **kw: None,
        ),
        patch(
            "cogniverse_agents.orchestrator_agent.OrchestratorAgent._load_artifact",
            new=lambda self: None,
        ),
    ):
        yield dispatcher, state


async def _orchestrate(dispatcher, query, history=None):
    context = {"tenant_id": TENANT}
    if history is not None:
        context["conversation_history"] = history
    return await dispatcher.dispatch(
        agent_name="orchestrator_agent", query=query, context=context
    )


class TestOrchestratedFollowUp:
    async def test_the_orchestrator_gets_the_follow_up_resolved_to_the_video(
        self, orchestrated, dspy_lm
    ):
        dispatcher, state = orchestrated
        history = _first_turn("v_-6dz6tBH77I.mp4")

        response = await _orchestrate(dispatcher, FOLLOW_UP, history)

        [received] = state["inputs"]
        assert "v_-6dz6tBH77I" in received.query, received.query
        assert received.conversation_history == history
        assert (response["original_query"], response["rewritten_query"]) == (
            FOLLOW_UP,
            received.query,
        )
        assert response["message"] == received.query

    async def test_a_first_turn_reaches_the_orchestrator_as_asked(
        self, orchestrated, dspy_lm
    ):
        dispatcher, state = orchestrated

        response = await _orchestrate(dispatcher, "search for sports clips")

        assert [received.query for received in state["inputs"]] == [
            "search for sports clips"
        ]
        assert state["inputs"][0].conversation_history is None
        assert {"original_query", "rewritten_query"} & set(response) == set()

    async def test_an_unreachable_lm_refuses_the_follow_up_before_orchestration(
        self, orchestrated
    ):
        from cogniverse_agents.search_agent import ConversationalQueryRewriteModule

        dispatcher, state = orchestrated
        dispatcher._query_rewriter = ConversationalQueryRewriteModule()
        dispatcher._query_rewriter.set_lm(
            BodyBoundedLM(
                "openai/followup-unreachable",
                api_base="http://127.0.0.1:29071/v1",
                api_key="test",
                num_retries=0,
                timeout=1,
                cache=False,
            )
        )

        with pytest.raises(litellm.InternalServerError) as refused:
            await _orchestrate(dispatcher, FOLLOW_UP, _first_turn("v_-6dz6tBH77I.mp4"))

        assert str(refused.value) == (
            "litellm.InternalServerError: InternalServerError: OpenAIException "
            "- Connection error."
        )
        assert state["inputs"] == []

    async def test_concurrent_follow_ups_each_reach_the_orchestrator_with_their_own_video(
        self, orchestrated, dspy_lm
    ):
        dispatcher, state = orchestrated
        videos = [
            "v_-6dz6tBH77I.mp4",
            "v_J0nA4VgnoCo.mp4",
            "v_2Wd9Yr8hS0k.mp4",
            "v_Kt3pQ7mLx1E.mp4",
        ]
        state["held"] = asyncio.Event()

        async def _release_when_all_arrive():
            while len(state["inputs"]) < len(videos):
                await asyncio.sleep(0.05)
            state["held"].set()

        responses, _ = await asyncio.gather(
            asyncio.gather(
                *(
                    _orchestrate(dispatcher, FOLLOW_UP, _first_turn(video))
                    for video in videos
                )
            ),
            asyncio.wait_for(_release_when_all_arrive(), timeout=300),
        )

        named = [
            [video for video in videos if video.removesuffix(".mp4") in r.query]
            for r in state["inputs"]
        ]
        assert sorted(named) == sorted([video] for video in videos), [
            r.query for r in state["inputs"]
        ]
        assert [
            [
                video
                for video in videos
                if video.removesuffix(".mp4") in r["rewritten_query"]
            ]
            for r in responses
        ] == [[video] for video in videos], responses
