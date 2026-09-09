"""Workspace and token dispatch preserve request state through real runtime wiring."""

import asyncio
import threading
from types import ModuleType

import dspy
import pytest

from cogniverse_core.agents.base import AgentBase, AgentDeps, AgentInput, AgentOutput
from cogniverse_core.common.agent_models import AgentEndpoint
from cogniverse_core.registries.agent_registry import AgentRegistry
from cogniverse_runtime.agent_dispatcher import (
    AgentDispatcher,
    _scan_module_for_generic_classes,
)

pytestmark = [pytest.mark.integration, pytest.mark.ci_fast]


@pytest.fixture
def stream_dispatcher(config_manager, schema_loader):
    registry = AgentRegistry(tenant_id="test:unit", config_manager=config_manager)
    registry.register_agent(
        AgentEndpoint(
            "deep_research_agent",
            "http://localhost:8000",
            ["deep_research"],
            streams_answer_tokens=True,
        )
    )
    registry.register_agent(
        AgentEndpoint("coding_agent", "http://localhost:8000", ["coding"])
    )
    return AgentDispatcher(registry, config_manager, schema_loader)


def test_stem_classes_win_over_imported_types():
    module = ModuleType("multiple_agents")

    class AardvarkDeps(AgentDeps):
        pass

    class AardvarkInput(AgentInput):
        pass

    class ResearchDeps(AgentDeps):
        pass

    class ResearchInput(AgentInput):
        pass

    class ResearchAgent:
        pass

    for cls in (
        AardvarkDeps,
        AardvarkInput,
        ResearchDeps,
        ResearchInput,
        ResearchAgent,
    ):
        setattr(module, cls.__name__, cls)
    assert _scan_module_for_generic_classes(module, "ResearchAgent") == (
        ResearchAgent,
        ResearchDeps,
        ResearchInput,
    )


def test_each_agent_in_a_module_binds_its_own_declared_classes():
    """Two agents in one module keep their own Deps/Input.

    Neither agent's classes carry its name stem, so a scan of ``dir(module)``
    hands both of them the alphabetically first pair and the second agent
    dispatches on the first agent's input shape.
    """
    module = ModuleType("two_parameterised_agents")

    class AlphaDeps(AgentDeps):
        pass

    class AlphaInput(AgentInput):
        pass

    class AlphaOutput(AgentOutput):
        pass

    class BetaDeps(AgentDeps):
        pass

    class BetaInput(AgentInput):
        pass

    class BetaOutput(AgentOutput):
        pass

    class FirstTurnAgent(AgentBase[AlphaInput, AlphaOutput, AlphaDeps]):
        async def _process_impl(self, input: AlphaInput) -> AlphaOutput:
            return AlphaOutput()

    class SecondTurnAgent(AgentBase[BetaInput, BetaOutput, BetaDeps]):
        async def _process_impl(self, input: BetaInput) -> BetaOutput:
            return BetaOutput()

    for cls in (
        AlphaDeps,
        AlphaInput,
        AlphaOutput,
        BetaDeps,
        BetaInput,
        BetaOutput,
        FirstTurnAgent,
        SecondTurnAgent,
    ):
        setattr(module, cls.__name__, cls)

    assert [
        _scan_module_for_generic_classes(module, name)
        for name in ("FirstTurnAgent", "SecondTurnAgent")
    ] == [
        (FirstTurnAgent, AlphaDeps, AlphaInput),
        (SecondTurnAgent, BetaDeps, BetaInput),
    ]


def test_declared_parameters_win_over_a_stem_named_sibling():
    """The agent's own type parameters decide, not a same-stem name.

    A module can hold a class whose name matches the agent's stem without
    being what the agent is parameterised on; binding by name would hand the
    dispatcher an input the agent never declared.
    """
    module = ModuleType("stem_collision")

    class DeclaredDeps(AgentDeps):
        pass

    class DeclaredInput(AgentInput):
        pass

    class DeclaredOutput(AgentOutput):
        pass

    class ReportDeps(AgentDeps):
        pass

    class ReportInput(AgentInput):
        pass

    class ReportAgent(AgentBase[DeclaredInput, DeclaredOutput, DeclaredDeps]):
        async def _process_impl(self, input: DeclaredInput) -> DeclaredOutput:
            return DeclaredOutput()

    for cls in (
        DeclaredDeps,
        DeclaredInput,
        DeclaredOutput,
        ReportDeps,
        ReportInput,
        ReportAgent,
    ):
        setattr(module, cls.__name__, cls)

    assert _scan_module_for_generic_classes(module, "ReportAgent") == (
        ReportAgent,
        DeclaredDeps,
        DeclaredInput,
    )


def test_stream_declaration(stream_dispatcher):
    assert [
        stream_dispatcher.supports_token_stream(name)
        for name in ("deep_research_agent", "coding_agent", "unknown")
    ] == [True, False, False]


@pytest.mark.asyncio
async def test_generic_stream_builder_wires_search_and_context(
    stream_dispatcher, monkeypatch
):
    seen = []

    async def search(query, tenant_id, top_k):
        seen.append((query, tenant_id, top_k))
        return {"results": [{"id": "evidence-a"}]}

    monkeypatch.setattr(stream_dispatcher, "_execute_search_task", search)
    agent, request = stream_dispatcher._build_generic_streaming_agent(
        "deep_research_agent",
        "question",
        "test:unit",
        {
            "max_iterations": 2,
            "attachments": ["data:image/png;base64,AA=="],
            "rlm": {"max_iterations": 4},
        },
    )
    assert agent._config_manager is stream_dispatcher._config_manager
    assert await agent._search_fn("subquestion", "test:unit") == [{"id": "evidence-a"}]
    assert seen == [("subquestion", "test:unit", 10)]
    assert (
        request.max_iterations,
        request.attachments,
        request.rlm.max_iterations,
    ) == (2, ["data:image/png;base64,AA=="], 4)


class TurnInput(AgentInput):
    query: str


class TurnOutput(AgentOutput):
    answer: str


class TurnAgent(AgentBase[TurnInput, TurnOutput, AgentDeps]):
    def __init__(self, gate):
        super().__init__(deps=AgentDeps())
        self.gate = gate

    async def _process_impl(self, input):
        self.emit_progress(
            "token",
            input.query,
            data={"accumulated": input.query, "output_field": "answer"},
        )
        await self.gate.wait()
        return TurnOutput(answer=input.query)


@pytest.mark.asyncio
async def test_dispatch_stream_checks_egress_and_isolates_concurrent_turns(
    stream_dispatcher, monkeypatch
):
    gate = asyncio.Event()
    agent = TurnAgent(gate)
    prepared = []
    calls = []
    loop_thread = threading.get_ident()

    def consult(name):
        calls.append(("consult", name, threading.get_ident() == loop_thread))

    def verify(name, tenant):
        calls.append(("verify", name, tenant, threading.get_ident() == loop_thread))

    async def factory(name, query, tenant_id, context=None):
        prepared.append((query, tenant_id, context["attachments"]))
        if len(prepared) == 2:
            gate.set()
        return agent, TurnInput(query=query)

    monkeypatch.setattr(stream_dispatcher, "consult_egress_policy", consult)
    monkeypatch.setattr(stream_dispatcher, "_verify_egress", verify)
    monkeypatch.setattr(stream_dispatcher, "create_streaming_agent", factory)

    async def drain(tag):
        return [
            event
            async for event in stream_dispatcher.dispatch_stream(
                "deep_research_agent",
                tag,
                {"tenant_id": "test:unit", "attachments": [tag]},
            )
        ]

    first, second = await asyncio.wait_for(
        asyncio.gather(drain("first"), drain("second")), 10
    )
    assert sorted(prepared) == [
        ("first", "test:unit", ["first"]),
        ("second", "test:unit", ["second"]),
    ]
    assert calls.count(("consult", "deep_research_agent", False)) == 2
    assert calls.count(("verify", "deep_research_agent", "test:unit", False)) == 2
    assert [
        (event["message"], event["data"])
        for event in first
        if event.get("phase") == "token"
    ] == [("first", {"accumulated": "first", "output_field": "answer"})]
    assert [
        (event["message"], event["data"])
        for event in second
        if event.get("phase") == "token"
    ] == [("second", {"accumulated": "second", "output_field": "answer"})]
    assert [event["data"]["answer"] for event in first if event["type"] == "final"] == [
        "first"
    ]
    assert [
        event["data"]["answer"] for event in second if event["type"] == "final"
    ] == ["second"]


@pytest.mark.asyncio
async def test_dispatch_stream_propagates_egress_fault_before_build(
    stream_dispatcher, monkeypatch
):
    builds = []

    async def factory(*args, **kwargs):
        builds.append(args)
        raise AssertionError("agent built before egress verification")

    def fail(name, tenant):
        raise RuntimeError(f"egress unavailable: {name}/{tenant}")

    monkeypatch.setattr(stream_dispatcher, "create_streaming_agent", factory)
    monkeypatch.setattr(stream_dispatcher, "_verify_egress", fail)
    with pytest.raises(
        RuntimeError, match="^egress unavailable: deep_research_agent/test:unit$"
    ):
        _ = [
            event
            async for event in stream_dispatcher.dispatch_stream(
                "deep_research_agent", "q", {"tenant_id": "test:unit"}
            )
        ]
    assert builds == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["summary", "report", "research"])
async def test_dispatch_prepares_attachment_at_answer_module(
    stream_dispatcher, monkeypatch, kind
):
    from PIL import Image

    from cogniverse_agents import (
        deep_research_agent,
        detailed_report_agent,
        summarizer_agent,
    )
    from cogniverse_foundation.config.agent_config import (
        AgentConfig,
        DSPyModuleType,
        ModuleConfig,
    )
    from tests.agents.unit.test_answer_attachments import ANSWER, ContentDecision
    from tests.agents.unit.test_multimodal_attachments import _data_uri, _decoded_jpeg

    module, cls, predictor = {
        "summary": (
            summarizer_agent,
            summarizer_agent.SummarizerAgent,
            "summarization_module",
        ),
        "report": (
            detailed_report_agent,
            detailed_report_agent.DetailedReportAgent,
            "report_module",
        ),
        "research": (
            deep_research_agent,
            deep_research_agent.DeepResearchAgent,
            "_synthesizer",
        ),
    }[kind]
    decision = ContentDecision()
    original = cls.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        setattr(self, predictor, decision)
        if kind == "research":
            self._decomposer = decision
            self._evaluator = decision

    monkeypatch.setattr(cls, "__init__", initialize)

    async def search(query, tenant_id, top_k):
        return {"results": []}

    monkeypatch.setattr(stream_dispatcher, "_execute_search_task", search)
    agent_name = {
        "summary": "summarizer_agent",
        "report": "detailed_report_agent",
        "research": "deep_research_agent",
    }[kind]
    cm = stream_dispatcher._config_manager
    config = AgentConfig(
        agent_name=agent_name,
        agent_version="1.0",
        agent_description="Answer content",
        agent_url="http://localhost:8000",
        capabilities=[],
        skills=[],
        module_config=ModuleConfig(
            module_type=DSPyModuleType.PREDICT, signature="query -> answer"
        ),
        thinking_enabled=False,
    )
    cm.set_agent_config("test:unit", agent_name, config)
    uri = _data_uri(Image.new("RGB", (100, 40), "blue"), "PNG")
    execute = {
        "summary": stream_dispatcher._execute_summarization_task,
        "report": stream_dispatcher._execute_detailed_report_task,
        "research": stream_dispatcher._execute_deep_research_task,
    }[kind]
    result = await execute(
        "The Eiffel Tower was completed in 1889. Summarize this text.",
        "test:unit",
        {
            "attachments": [uri],
            "search_results": [],
            "max_iterations": 1,
            "rlm": {"enabled": False, "max_iterations": 4},
        },
    )
    expected_field = "executive_summary" if kind == "report" else "summary"
    assert result["result"][expected_field] == ANSWER
    assert [
        [_decoded_jpeg(image).size for image in item["keyframes"]]
        for item in decision.inputs
        if "keyframes" in item
    ] == [[(100, 40)]]
    if kind == "research":
        assert result["result"]["iterations_used"] == 1
    config.visual_analysis_enabled = False
    cm.set_agent_config("test:unit", agent_name, config)
    with pytest.raises(ValueError) as error:
        await execute("q", "test:unit", {"attachments": [uri], "search_results": []})
    assert (
        str(error.value)
        == "attachments_disabled: visual inputs are disabled for this request"
    )
    config.visual_analysis_enabled = True
    cm.set_agent_config("test:unit", agent_name, config)


@pytest.mark.asyncio
async def test_coding_suspension_and_failed_step_envelopes(
    stream_dispatcher, monkeypatch
):
    import uuid

    from cogniverse_agents import coding_agent as coding
    from tests.agents.unit.test_coding_workspace_turns import TOOLS, envelope

    decisions = []

    async def decide(self, module, **kwargs):
        decisions.append(kwargs)
        return dspy.Prediction(
            tool_name="read_file", tool_args_json='{"path":"x.py"}', summary="read file"
        )

    monkeypatch.setattr(coding.CodingAgent, "call_dspy", decide)
    monkeypatch.setattr(coding.uuid, "uuid4", lambda: uuid.UUID(int=1))
    context = {
        "tenant_id": "test:unit",
        "external_tools": TOOLS,
        "continuation_state": {"plan": "read plan", "step": 0},
        "conversation_history": [],
    }
    result = await stream_dispatcher.dispatch("coding_agent", "read x", context)
    pending = [
        {"id": "call_000000000000", "name": "read_file", "arguments": {"path": "x.py"}}
    ]
    continuation = {"mode": "workspace", "plan": "read plan", "step": 1}
    assert result == {
        "status": "input_required",
        "agent": "coding_agent",
        "message": "Workspace tool execution required",
        "pending_tool_calls": pending,
        "continuation_state": continuation,
        "result": envelope(
            iterations_used=1,
            pending_tool_calls=pending,
            continuation_state=continuation,
        ),
    }
    assert len(decisions) == 1

    async def invalid(self, module, **kwargs):
        decisions.append(kwargs)
        return dspy.Prediction(
            tool_name="unknown", tool_args_json="{}", summary="not done"
        )

    monkeypatch.setattr(coding.CodingAgent, "call_dspy", invalid)
    result = await stream_dispatcher.dispatch("coding_agent", "read x", context)
    error = "Invalid workspace action after 3 attempt(s): unknown tool 'unknown'"
    assert result == {
        "status": "error",
        "agent": "coding_agent",
        "error": error,
        "result": envelope(success=False, iterations_used=3, error=error),
    }
    assert len(decisions) == 4


class UnavailableAnswerAgent(AgentBase[TurnInput, TurnOutput, AgentDeps]):
    def __init__(self):
        super().__init__(deps=AgentDeps())
        self._dspy_lm = dspy.LM(
            "openai/workspace-test",
            api_base="http://127.0.0.1:29071/v1",
            api_key="test",
            timeout=1,
            num_retries=0,
            cache=False,
        )
        self.predict = dspy.Predict("query -> answer")

    async def _process_impl(self, input):
        prediction = await self.call_dspy(
            self.predict, output_field="answer", query=input.query
        )
        return TurnOutput(answer=prediction.answer)


@pytest.mark.asyncio
async def test_dispatch_stream_dead_lm_has_exact_terminal_error(
    stream_dispatcher, monkeypatch
):
    async def factory(*args, **kwargs):
        return UnavailableAnswerAgent(), TurnInput(query="q")

    monkeypatch.setattr(stream_dispatcher, "create_streaming_agent", factory)
    events = [
        event
        async for event in stream_dispatcher.dispatch_stream(
            "deep_research_agent", "q", {"tenant_id": "test:unit"}
        )
    ]
    assert events == [
        {
            "type": "error",
            "agent": "UnavailableAnswerAgent",
            "error_type": "InternalServerError",
            "message": "UnavailableAnswerAgent streaming failed with InternalServerError. See server logs for detail.",
        }
    ]


@pytest.mark.asyncio
async def test_closing_dispatch_stream_joins_agent_cleanup(
    stream_dispatcher, monkeypatch
):
    cleaned = asyncio.Event()
    blocked = asyncio.Event()

    class CancellableAgent(AgentBase[TurnInput, TurnOutput, AgentDeps]):
        async def _process_impl(self, input):
            try:
                self.emit_progress(
                    "token",
                    "first",
                    data={"accumulated": "first", "output_field": "answer"},
                )
                await blocked.wait()
                return TurnOutput(answer="must not complete")
            finally:
                cleaned.set()

    async def factory(*args, **kwargs):
        return CancellableAgent(deps=AgentDeps()), TurnInput(query="q")

    monkeypatch.setattr(stream_dispatcher, "create_streaming_agent", factory)
    stream = stream_dispatcher.dispatch_stream(
        "deep_research_agent", "q", {"tenant_id": "test:unit"}
    )
    first = await anext(stream)
    assert first == {
        "type": "partial",
        "phase": "token",
        "message": "first",
        "data": {"accumulated": "first", "output_field": "answer"},
    }
    assert cleaned.is_set() is False
    await stream.aclose()
    assert cleaned.is_set() is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent_name,capability",
    [
        ("summarizer_agent", "summarization"),
        ("detailed_report_agent", "detailed_report"),
        ("deep_research_agent", "deep_research"),
    ],
)
async def test_disabled_visuals_rejected_before_stream(
    stream_dispatcher, agent_name, capability
):
    from cogniverse_foundation.config.agent_config import (
        AgentConfig,
        DSPyModuleType,
        ModuleConfig,
    )

    config = AgentConfig(
        agent_name=agent_name,
        agent_version="1.0",
        agent_description="Answer content",
        agent_url="http://localhost:8000",
        capabilities=[capability],
        skills=[],
        module_config=ModuleConfig(
            module_type=DSPyModuleType.PREDICT, signature="query -> answer"
        ),
        visual_analysis_enabled=False,
    )
    stream_dispatcher._config_manager.set_agent_config("test:unit", agent_name, config)
    stream_dispatcher._registry.register_agent(
        AgentEndpoint(
            agent_name,
            "http://localhost:8000",
            [capability],
            streams_answer_tokens=True,
        )
    )
    try:
        with pytest.raises(ValueError) as error:
            _ = [
                event
                async for event in stream_dispatcher.dispatch_stream(
                    agent_name,
                    "q",
                    {
                        "tenant_id": "test:unit",
                        "attachments": ["http://127.0.0.1:29071/photo.png"],
                        "search_results": [],
                    },
                )
            ]
        assert (
            str(error.value)
            == "attachments_disabled: visual inputs are disabled for this request"
        )
    finally:
        config.visual_analysis_enabled = True
        stream_dispatcher._config_manager.set_agent_config(
            "test:unit", agent_name, config
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["stream", "task"])
async def test_research_config_read_yields_serving_loop(
    stream_dispatcher, monkeypatch, path
):
    from cogniverse_agents.deep_research_agent import DeepResearchAgent
    from tests.agents.unit.test_answer_attachments import ContentDecision

    cm = stream_dispatcher._config_manager
    original = cm.get_agent_config
    entered = threading.Event()
    released = threading.Event()
    seen = []
    loop_thread = threading.get_ident()

    def blocked_read(*args, **kwargs):
        entered.set()
        unblocked = released.wait(2)
        seen.append((threading.get_ident() == loop_thread, unblocked))
        return original(*args, **kwargs)

    monkeypatch.setattr(cm, "get_agent_config", blocked_read)
    original_init = DeepResearchAgent.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        decision = ContentDecision()
        self._decomposer = decision
        self._evaluator = decision
        self._synthesizer = decision

    monkeypatch.setattr(DeepResearchAgent, "__init__", initialize)

    async def search(*args, **kwargs):
        return {"results": []}

    monkeypatch.setattr(stream_dispatcher, "_execute_search_task", search)
    if path == "stream":
        turn = asyncio.create_task(
            stream_dispatcher.create_streaming_agent(
                "deep_research_agent", "q", "test:unit"
            )
        )
    else:
        turn = asyncio.create_task(
            stream_dispatcher._execute_deep_research_task(
                "q", "test:unit", {"max_iterations": 1}
            )
        )
    await asyncio.to_thread(entered.wait, 5)
    released.set()
    await asyncio.wait_for(turn, 10)
    assert seen == [(False, True)]
