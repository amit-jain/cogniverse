"""Unit tests for OrchestratorAgent"""

from contextlib import nullcontext
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import dspy
import pytest

from cogniverse_agents.orchestrator_agent import (
    AgentStep,
    FusionStrategy,
    OrchestrationModule,
    OrchestrationPlan,
    OrchestrationResult,
    OrchestratorAgent,
    OrchestratorDeps,
    OrchestratorInput,
    OrchestratorOutput,
    _search_results_from_completed,
)
from cogniverse_foundation.config.unified_config import (
    LLMEndpointConfig,
    SemanticRouterConfig,
    SystemConfig,
)


def _plan_executes_to_completion(plan) -> bool:
    """Mirror _execute_plan's dependency-readiness loop without any I/O.

    Returns True iff every step eventually becomes ready (the topological
    order is satisfiable). Catches the failure modes: a self-referential
    depends_on stalls the loop ("No steps ready" -> False), and an
    out-of-range depends_on raises IndexError on executed[dep_idx].
    """
    executed = [False] * len(plan.steps)
    while not all(executed):
        ready = [
            i
            for i, step in enumerate(plan.steps)
            if not executed[i] and all(executed[d] for d in step.depends_on)
        ]
        if not ready:
            return False
        for i in ready:
            executed[i] = True
    return True


def _make_mock_config_manager() -> Mock:
    """Mock ConfigManager whose ``get_system_config()`` returns a real
    SystemConfig dataclass. The orchestrator reads
    ``iter_retrieval_max_iter`` / ``_token_budget`` /
    ``_wall_clock_ms`` / ``redis_url`` off it; bare ``Mock()`` would
    return ``Mock`` instances that can't be ``range()``'d / compared.
    """
    cm = Mock()
    cm.get_system_config = Mock(return_value=SystemConfig())
    return cm


@pytest.fixture
def mock_dspy_lm():
    """Mock DSPy language model"""
    lm = Mock()
    lm.return_value = dspy.Prediction(
        agent_sequence="query_enhancement_agent,entity_extraction_agent,profile_selection_agent,search_agent",
        parallel_steps="0,1",
        reasoning="Enhance query and extract entities in parallel, then select profile and search sequentially",
    )
    return lm


@pytest.fixture
def mock_agent_registry():
    """Mock AgentRegistry with test agents and A2A-compatible interface"""
    registry = Mock()

    # Map agent names to mock endpoints
    agent_endpoints = {}
    for agent_name in [
        "entity_extraction_agent",
        "profile_selection_agent",
        "query_enhancement_agent",
        "search_agent",
        "summarizer_agent",
        "video_search_agent",
        "image_search_agent",
        "audio_analysis_agent",
    ]:
        endpoint = Mock()
        endpoint.name = agent_name
        endpoint.url = f"http://localhost:800{len(agent_endpoints)}"
        endpoint.capabilities = [agent_name]
        agent_endpoints[agent_name] = endpoint

    registry.get_agent = Mock(side_effect=lambda name: agent_endpoints.get(name))
    registry.find_agents_by_capability = Mock(
        side_effect=lambda cap: [
            ep for ep in agent_endpoints.values() if cap in ep.capabilities
        ]
    )
    registry.list_agents = Mock(return_value=list(agent_endpoints.keys()))
    registry.agents = agent_endpoints

    return registry


@pytest.fixture
def orchestrator_agent(mock_agent_registry):
    """Create OrchestratorAgent for testing"""
    with patch("dspy.ChainOfThought"):
        deps = OrchestratorDeps()
        mock_config_manager = _make_mock_config_manager()
        agent = OrchestratorAgent(
            deps=deps,
            registry=mock_agent_registry,
            config_manager=mock_config_manager,
            port=8013,
        )
        return agent


@pytest.mark.unit
class TestSearchResultsFromCompleted:
    """The orchestrator threads only a real video-search step's hits to an
    answer step — not image/audio search results, and not an empty set."""

    def test_returns_search_agent_hits(self):
        hits = [{"document_id": "v_0", "metadata": {"source_url": "s3://b/t/v.mp4"}}]
        results = {
            "query_enhancement_agent": {"enhanced_query": "x"},
            "search_agent": {"agent": "search_agent", "results": hits},
        }
        assert _search_results_from_completed(results) == hits

    def test_ignores_non_search_agent_results(self):
        results = {
            "image_search_agent": {
                "agent": "image_search_agent",
                "results": [{"a": 1}],
            },
            "summarizer_agent": {"agent": "summarizer_agent", "results": [{"b": 2}]},
        }
        assert _search_results_from_completed(results) is None

    def test_empty_search_results_returns_none(self):
        results = {"search_agent": {"agent": "search_agent", "results": []}}
        assert _search_results_from_completed(results) is None

    def test_no_search_step_returns_none(self):
        results = {"query_enhancement_agent": {"enhanced_query": "x"}}
        assert _search_results_from_completed(results) is None


class TestOrchestratorInputValidation:
    """Pydantic-level contract tests for OrchestratorInput."""

    def test_rejects_missing_tenant_id(self):
        """Constructing OrchestratorInput without tenant_id must raise — no silent 'default'."""
        import pydantic

        with pytest.raises(pydantic.ValidationError) as excinfo:
            OrchestratorInput(query="find ML videos")

        errors = excinfo.value.errors()
        tenant_errors = [e for e in errors if "tenant_id" in e.get("loc", ())]
        assert tenant_errors, f"Expected a validation error on tenant_id, got: {errors}"

    def test_accepts_explicit_tenant_id(self):
        """Explicit tenant_id round-trips through the model unchanged."""
        inp = OrchestratorInput(query="find ML videos", tenant_id="acme:production")
        assert inp.tenant_id == "acme:production"
        assert inp.query == "find ML videos"


class TestOrchestrationModule:
    """Test DSPy module for orchestration"""

    def test_module_initialization(self):
        """Test OrchestrationModule initializes correctly"""
        with patch("dspy.ChainOfThought") as mock_cot:
            module = OrchestrationModule()
            assert module.planner is not None
            mock_cot.assert_called_once()

    def test_forward_success(self, mock_dspy_lm):
        """Test successful orchestration planning"""
        module = OrchestrationModule()
        module.planner = mock_dspy_lm

        result = module.forward(
            query="Show me machine learning videos",
            available_agents="query_enhancement_agent,entity_extraction_agent,profile_selection_agent,search_agent",
        )

        assert (
            result.agent_sequence
            == "query_enhancement_agent,entity_extraction_agent,profile_selection_agent,search_agent"
        )
        assert result.parallel_steps == "0,1"
        assert "parallel" in result.reasoning.lower()

    def test_forward_with_gateway_context(self, mock_dspy_lm):
        """Test forward passes gateway_context to planner"""
        module = OrchestrationModule()
        module.planner = mock_dspy_lm

        module.forward(
            query="Show me videos",
            available_agents="search_agent",
            gateway_context="intent: video_search, modality: VIDEO",
        )

        # Planner was called with gateway_context
        mock_dspy_lm.assert_called_once()
        call_kwargs = mock_dspy_lm.call_args
        assert (
            call_kwargs.kwargs.get("gateway_context")
            == "intent: video_search, modality: VIDEO"
        )

    def test_forward_propagates_error(self):
        """Test that DSPy failure propagates instead of silently falling back"""
        module = OrchestrationModule()
        module.planner = Mock(side_effect=Exception("DSPy failed"))

        with pytest.raises(Exception, match="DSPy failed"):
            module.forward(
                query="Show me videos",
                available_agents="query_enhancement_agent,entity_extraction_agent,profile_selection_agent,search_agent",
            )


class TestOrchestratorAgent:
    """Test OrchestratorAgent core functionality"""

    def test_agent_initialization(self, orchestrator_agent):
        """Test agent initializes with correct configuration"""
        assert orchestrator_agent.agent_name == "orchestrator_agent"
        assert "orchestration" in orchestrator_agent.capabilities
        assert "planning" in orchestrator_agent.capabilities
        assert len(orchestrator_agent.registry.agents) == 8

    def test_agent_initialization_with_optional_deps(self, mock_agent_registry):
        """Test agent initializes with event_queue, workflow_intelligence"""
        with patch("dspy.ChainOfThought"):
            deps = OrchestratorDeps()
            mock_config_manager = _make_mock_config_manager()
            mock_event_queue = Mock()
            mock_workflow_intelligence = Mock()

            agent = OrchestratorAgent(
                deps=deps,
                registry=mock_agent_registry,
                config_manager=mock_config_manager,
                port=8013,
                event_queue=mock_event_queue,
                workflow_intelligence=mock_workflow_intelligence,
            )

            assert agent.event_queue is mock_event_queue
            assert agent.workflow_intelligence is mock_workflow_intelligence

    @pytest.mark.asyncio
    async def test_create_plan(self, orchestrator_agent):
        """Test planning phase with dynamic agent discovery"""
        orchestrator_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="query_enhancement_agent,profile_selection_agent,search_agent",
                parallel_steps="",
                reasoning="Sequential execution: enhance, select, search",
            )
        )

        plan = await orchestrator_agent._create_plan("Show me ML videos")

        assert isinstance(plan, OrchestrationPlan)
        assert plan.query == "Show me ML videos"
        assert len(plan.steps) == 3
        assert plan.steps[0].agent_name == "query_enhancement_agent"
        assert plan.steps[1].agent_name == "profile_selection_agent"
        assert plan.steps[2].agent_name == "search_agent"

    @pytest.mark.asyncio
    async def test_create_plan_skips_unknown_agents(self, orchestrator_agent):
        """Test that unknown agents from LLM are skipped"""
        orchestrator_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="query_enhancement_agent,nonexistent_agent,search_agent",
                parallel_steps="",
                reasoning="Test plan",
            )
        )

        plan = await orchestrator_agent._create_plan("Test query")

        # nonexistent_agent should be skipped
        agent_names = [step.agent_name for step in plan.steps]
        assert "nonexistent_agent" not in agent_names
        assert "query_enhancement_agent" in agent_names
        assert "search_agent" in agent_names

        # depends_on must index the SURVIVING step list, not the
        # raw agent_sequence. With the mid-sequence agent filtered out, the two
        # surviving steps are [query_enhancement(0), search_agent(1)]; search must
        # depend on query_enhancement (step 0), NOT on itself (the old code passed
        # the unfiltered enumerate index, yielding depends_on=[1] -> deadlock).
        assert len(plan.steps) == 2
        assert plan.steps[0].agent_name == "query_enhancement_agent"
        assert plan.steps[0].depends_on == []
        assert plan.steps[1].agent_name == "search_agent"
        assert plan.steps[1].depends_on == [0]
        # No step may depend on itself or on an out-of-range index.
        for idx, step in enumerate(plan.steps):
            assert idx not in step.depends_on
            assert all(0 <= d < len(plan.steps) for d in step.depends_on)
        # The readiness loop in _execute_plan must terminate (no deadlock).
        assert _plan_executes_to_completion(plan)

    @pytest.mark.asyncio
    async def test_create_plan_filtered_agent_remaps_parallel_groups(
        self, orchestrator_agent
    ):
        """A filtered mid-sequence agent must remap parallel_steps indices.

        Raw sequence indices 0,1,2,3 with agent 1 (nonexistent) filtered: the
        parallel group "0,2" (enhance + extract) must remap to surviving-step
        positions [0,1], and the trailing search step must depend on that group.
        """
        orchestrator_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence=(
                    "query_enhancement_agent,nonexistent_agent,"
                    "entity_extraction_agent,search_agent"
                ),
                parallel_steps="0,2",
                reasoning="Enhance + extract in parallel after dropping the bogus agent",
            )
        )

        plan = await orchestrator_agent._create_plan("Test query")

        agent_names = [step.agent_name for step in plan.steps]
        assert agent_names == [
            "query_enhancement_agent",
            "entity_extraction_agent",
            "search_agent",
        ]
        # Raw group [0,2] remaps to surviving-step positions [0,1].
        assert plan.parallel_groups == [[0, 1]]
        assert plan.steps[0].depends_on == []
        assert plan.steps[1].depends_on == []
        assert set(plan.steps[2].depends_on) == {0, 1}
        for idx, step in enumerate(plan.steps):
            assert idx not in step.depends_on
            assert all(0 <= d < len(plan.steps) for d in step.depends_on)
        assert _plan_executes_to_completion(plan)

    @pytest.mark.asyncio
    async def test_create_plan_with_parallel_groups(self, orchestrator_agent):
        """Test planning with parallel execution groups"""
        orchestrator_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="query_enhancement_agent,entity_extraction_agent,search_agent",
                parallel_steps="0,1",
                reasoning="Enhance and extract in parallel, then search",
            )
        )

        plan = await orchestrator_agent._create_plan("Test query")

        assert len(plan.parallel_groups) == 1
        assert plan.parallel_groups[0] == [0, 1]
        # Steps 0 and 1 should have no dependencies
        assert plan.steps[0].depends_on == []
        assert plan.steps[1].depends_on == []
        # Step 2 should depend on steps 0 and 1
        assert set(plan.steps[2].depends_on) == {0, 1}

    @pytest.mark.asyncio
    async def test_create_plan_multiple_parallel_groups(self, orchestrator_agent):
        """Test planning with multiple parallel groups"""
        orchestrator_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="query_enhancement_agent,entity_extraction_agent,profile_selection_agent,search_agent",
                parallel_steps="0,1|2,3",
                reasoning="Two parallel groups",
            )
        )

        plan = await orchestrator_agent._create_plan("Test query")

        assert len(plan.parallel_groups) == 2
        assert plan.parallel_groups[0] == [0, 1]
        assert plan.parallel_groups[1] == [2, 3]
        # Second group should depend on first group
        assert set(plan.steps[2].depends_on) == {0, 1}
        assert set(plan.steps[3].depends_on) == {0, 1}

    def test_calculate_dependencies_sequential(self, orchestrator_agent):
        """Test dependency calculation for sequential steps"""
        parallel_groups = []

        deps_0 = orchestrator_agent._calculate_dependencies(0, parallel_groups)
        deps_1 = orchestrator_agent._calculate_dependencies(1, parallel_groups)
        deps_2 = orchestrator_agent._calculate_dependencies(2, parallel_groups)

        assert deps_0 == []  # First step has no dependencies
        assert deps_1 == [0]  # Second step depends on first
        assert deps_2 == [1]  # Third step depends on second

    def test_calculate_dependencies_parallel(self, orchestrator_agent):
        """Test dependency calculation for parallel groups"""
        parallel_groups = [[0, 1], [2, 3]]

        # Steps in first group have no dependencies
        assert orchestrator_agent._calculate_dependencies(0, parallel_groups) == []
        assert orchestrator_agent._calculate_dependencies(1, parallel_groups) == []

        # Steps in second group depend on first group
        assert set(orchestrator_agent._calculate_dependencies(2, parallel_groups)) == {
            0,
            1,
        }
        assert set(orchestrator_agent._calculate_dependencies(3, parallel_groups)) == {
            0,
            1,
        }

    @pytest.mark.asyncio
    async def test_execute_plan(self, orchestrator_agent):
        """Test action phase execution via HTTP calls"""
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "status": "success",
            "result": "Mock result",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        plan = OrchestrationPlan(
            query="test query",
            steps=[
                AgentStep(
                    agent_name="query_enhancement_agent",
                    input_data={"query": "test query"},
                    depends_on=[],
                    reasoning="Enhance query",
                ),
                AgentStep(
                    agent_name="search_agent",
                    input_data={"query": "test query"},
                    depends_on=[0],
                    reasoning="Search",
                ),
            ],
            parallel_groups=[],
            reasoning="Test plan",
        )

        with patch(
            "cogniverse_agents.orchestrator_agent._get_http_client",
            new=AsyncMock(return_value=mock_client),
        ):
            results = await orchestrator_agent._execute_plan(
                plan, tenant_id="test:unit"
            )

        assert len(results) == 2
        assert "query_enhancement_agent" in results
        assert "search_agent" in results
        assert results["query_enhancement_agent"]["status"] == "success"
        assert results["search_agent"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_plan_threads_search_hits_to_answer_step(
        self, orchestrator_agent
    ):
        """A completed search step's hits reach a dependent summary/report step
        through its context, so the answer agent reuses them (and the keyframes
        derived from them) instead of running a second search."""
        s3_hit = {
            "document_id": "v_0",
            "score": 0.9,
            "metadata": {
                "source_url": "s3://cogniverse-ingest/test:unit/vid.mp4",
                "video_id": "vid",
                "segment_id": 0,
            },
        }
        captured = {}

        async def _post(url, json=None, **kwargs):
            resp = Mock()
            resp.raise_for_status = Mock()
            if json.get("agent_name") == "search_agent":
                resp.json = Mock(
                    return_value={
                        "status": "success",
                        "agent": "search_agent",
                        "results": [s3_hit],
                    }
                )
            else:
                captured["payload"] = json
                resp.json = Mock(return_value={"status": "success"})
            return resp

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=_post)

        plan = OrchestrationPlan(
            query="report on the clip",
            steps=[
                AgentStep(
                    agent_name="search_agent",
                    input_data={"query": "report on the clip"},
                    depends_on=[],
                    reasoning="Search",
                ),
                AgentStep(
                    agent_name="summarizer_agent",
                    input_data={"query": "report on the clip"},
                    depends_on=[0],
                    reasoning="Summarize the hits",
                ),
            ],
            parallel_groups=[],
            reasoning="Search then summarize",
        )

        with patch(
            "cogniverse_agents.orchestrator_agent._get_http_client",
            new=AsyncMock(return_value=mock_client),
        ):
            await orchestrator_agent._execute_plan(plan, tenant_id="test:unit")

        assert captured["payload"]["context"]["search_results"] == [s3_hit], (
            "the summarizer step's context must carry the search step's hits"
        )

    @pytest.mark.asyncio
    async def test_execute_plan_agent_not_found(self, orchestrator_agent):
        """Test execution when agent not in registry"""
        plan = OrchestrationPlan(
            query="test query",
            steps=[
                AgentStep(
                    agent_name="summarizer_agent",
                    input_data={"query": "test query"},
                    depends_on=[],
                    reasoning="Summarize",
                )
            ],
            parallel_groups=[],
            reasoning="Test plan",
        )

        results = await orchestrator_agent._execute_plan(plan, tenant_id="test:unit")

        # summarizer_agent IS in the registry, but its A2A endpoint is
        # unreachable in this unit test, so _execute_plan records a failed
        # dispatch as status "error" (not a silent pass).
        assert "summarizer_agent" in results
        assert results["summarizer_agent"]["status"] == "error"

    @pytest.mark.asyncio
    async def test_execute_plan_agent_error(self, orchestrator_agent):
        """Test execution when HTTP call to agent raises exception"""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ConnectionError("Agent failed"))

        plan = OrchestrationPlan(
            query="test query",
            steps=[
                AgentStep(
                    agent_name="search_agent",
                    input_data={"query": "test query"},
                    depends_on=[],
                    reasoning="Search",
                )
            ],
            parallel_groups=[],
            reasoning="Test plan",
        )

        with patch(
            "cogniverse_agents.orchestrator_agent._get_http_client",
            new=AsyncMock(return_value=mock_client),
        ):
            results = await orchestrator_agent._execute_plan(
                plan, tenant_id="test:unit"
            )

        assert "search_agent" in results
        assert results["search_agent"]["status"] == "error"
        assert "Agent failed" in results["search_agent"]["message"]

    @pytest.mark.asyncio
    async def test_execute_plan_empty_str_exception_includes_type(
        self, orchestrator_agent
    ):
        """Empty-``str()`` exceptions (e.g. ``httpx.ReadTimeout()``) used to
        log ``"Agent X failed:"`` with no context. The enriched handler
        must surface the exception type in both the log and the returned
        error message so timeouts don't look identical to other failures.
        """
        import httpx

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ReadTimeout(""))

        plan = OrchestrationPlan(
            query="test query",
            steps=[
                AgentStep(
                    agent_name="search_agent",
                    input_data={"query": "test query"},
                    depends_on=[],
                    reasoning="Search",
                )
            ],
            parallel_groups=[],
            reasoning="Test plan",
        )

        with patch(
            "cogniverse_agents.orchestrator_agent._get_http_client",
            new=AsyncMock(return_value=mock_client),
        ):
            results = await orchestrator_agent._execute_plan(
                plan, tenant_id="test:unit"
            )

        assert results["search_agent"]["status"] == "error"
        # Old behavior: message == "" for empty-str exceptions.
        # New behavior: message surfaces the exception type.
        assert "ReadTimeout" in results["search_agent"]["message"], (
            "Empty-str exception should still carry the type name in the "
            f"returned error, got {results['search_agent']['message']!r}"
        )

    @pytest.mark.asyncio
    async def test_process_full_workflow(self, orchestrator_agent):
        """Test complete orchestration workflow"""
        orchestrator_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="query_enhancement_agent,search_agent",
                parallel_steps="",
                reasoning="Enhance then search",
            )
        )
        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {
            "status": "success",
            "result": "Mock result",
        }
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch(
            "cogniverse_agents.orchestrator_agent._get_http_client",
            new=AsyncMock(return_value=mock_client),
        ):
            result = await orchestrator_agent._process_impl(
                OrchestratorInput(
                    query="Show me machine learning videos", tenant_id="test:unit"
                )
            )

        assert isinstance(result, OrchestratorOutput)
        assert result.query == "Show me machine learning videos"
        assert result.workflow_id.startswith("workflow_")
        assert len(result.plan_steps) == 2
        assert "query_enhancement_agent" in result.agent_results
        assert "search_agent" in result.agent_results
        assert result.final_output["status"] == "success"
        assert "Executed" in result.execution_summary

    @pytest.mark.asyncio
    async def test_process_empty_query(self, orchestrator_agent):
        """Test processing empty query"""
        result = await orchestrator_agent._process_impl(
            OrchestratorInput(query="", tenant_id="test:unit")
        )

        assert result.query == ""
        assert len(result.plan_steps) == 0
        assert result.final_output["status"] == "error"
        assert "Empty query" in result.final_output["message"]

    def test_generate_summary(self, orchestrator_agent):
        """Test execution summary generation"""
        plan = OrchestrationPlan(
            query="test",
            steps=[
                AgentStep(
                    agent_name="search_agent",
                    input_data={},
                    depends_on=[],
                    reasoning="Search",
                ),
                AgentStep(
                    agent_name="summarizer_agent",
                    input_data={},
                    depends_on=[],
                    reasoning="Summarize",
                ),
            ],
            parallel_groups=[],
            reasoning="Test plan",
        )

        agent_results = {
            "search_agent": {"status": "success"},
            "summarizer_agent": {"status": "error", "message": "Failed"},
        }

        summary = orchestrator_agent._generate_summary(plan, agent_results)

        assert "2/2" in summary  # 2 executed out of 2 planned
        assert "1 successful" in summary  # 1 successful out of 2 executed
        assert "Test plan" in summary

    def test_dspy_to_a2a_output(self, orchestrator_agent):
        """Test conversion to A2A output format"""
        result = OrchestrationResult(
            query="test query",
            plan=OrchestrationPlan(
                query="test query",
                steps=[
                    AgentStep(
                        agent_name="search_agent",
                        input_data={"query": "test"},
                        depends_on=[],
                        reasoning="Search step",
                    )
                ],
                parallel_groups=[],
                reasoning="Test plan",
            ),
            agent_results={"search_agent": {"status": "success"}},
            final_output={"status": "success"},
            execution_summary="Executed 1/1 steps (1 successful). Plan: Test plan",
        )

        a2a_output = orchestrator_agent._dspy_to_a2a_output(result)

        assert a2a_output["status"] == "success"
        assert a2a_output["agent"] == "orchestrator_agent"
        assert a2a_output["query"] == "test query"
        assert len(a2a_output["plan"]["steps"]) == 1
        assert a2a_output["plan"]["steps"][0]["agent_name"] == "search_agent"
        assert (
            a2a_output["execution_summary"]
            == "Executed 1/1 steps (1 successful). Plan: Test plan"
        )

    def test_get_agent_skills(self, orchestrator_agent):
        """Test agent skills definition"""
        skills = orchestrator_agent._get_agent_skills()

        assert len(skills) == 1
        assert skills[0]["name"] == "orchestrate"
        assert "query" in skills[0]["input_schema"]
        assert "plan" in skills[0]["output_schema"]
        assert "agent_results" in skills[0]["output_schema"]
        assert len(skills[0]["examples"]) > 0


class TestOrchestratorStreaming:
    """Test streaming progress events"""

    @pytest.mark.asyncio
    async def test_emits_progress_events(self, orchestrator_agent):
        """Test that _execute_plan emits progress events for each step"""
        progress_events = []
        original_emit = orchestrator_agent.emit_progress

        def capture_progress(phase, message, data=None):
            progress_events.append({"phase": phase, "message": message, "data": data})
            original_emit(phase, message, data)

        orchestrator_agent.emit_progress = capture_progress

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"status": "success", "result": "ok"}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        plan = OrchestrationPlan(
            query="test",
            steps=[
                AgentStep(
                    agent_name="search_agent",
                    input_data={"query": "test"},
                    depends_on=[],
                    reasoning="Search",
                ),
            ],
            parallel_groups=[],
            reasoning="Test",
        )

        with patch(
            "cogniverse_agents.orchestrator_agent._get_http_client",
            new=AsyncMock(return_value=mock_client),
        ):
            await orchestrator_agent._execute_plan(plan, tenant_id="test:unit")

        # Should have "executing" and "step_complete" events
        phases = [e["phase"] for e in progress_events]
        assert "executing" in phases
        assert "step_complete" in phases


class TestOrchestratorFusion:
    """Test cross-modal fusion"""

    def test_cross_modal_fusion(self, orchestrator_agent):
        """Test that multi-modality results use fusion"""
        agent_results = {
            "video_search_agent": {
                "status": "success",
                "results": ["video1"],
                "confidence": 0.9,
            },
            "image_search_agent": {
                "status": "success",
                "results": ["image1"],
                "confidence": 0.7,
            },
        }

        output = orchestrator_agent._aggregate_results("find dogs", agent_results)

        assert output["status"] == "success"
        assert output["fusion_strategy"] == "score"  # Multi-modal -> SCORE_BASED
        assert "fusion_quality" in output
        assert output["fusion_quality"]["modality_count"] == 2

    def test_fusion_handles_label_confidence(self, orchestrator_agent):
        """A real LM can return confidence as 'high'/'85%'; aggregation must
        parse it rather than feed a string into the fusion math."""
        agent_results = {
            "video_search_agent": {
                "status": "success",
                "results": ["v1"],
                "confidence": "high",
            },
            "image_search_agent": {
                "status": "success",
                "results": ["i1"],
                "confidence": "85%",
            },
        }

        output = orchestrator_agent._aggregate_results("find dogs", agent_results)
        assert output["status"] == "success"

    def test_fusion_strategy_selection_simple(self, orchestrator_agent):
        """Test simple strategy for single modality"""
        strategy = orchestrator_agent._select_fusion_strategy(
            "find videos", {"search_agent": "text"}
        )
        assert strategy == FusionStrategy.SIMPLE

    def test_fusion_strategy_selection_score_based(self, orchestrator_agent):
        """Test score-based strategy for multiple modalities"""
        strategy = orchestrator_agent._select_fusion_strategy(
            "find content",
            {"video_search_agent": "video", "image_search_agent": "image"},
        )
        assert strategy == FusionStrategy.SCORE_BASED

    def test_fusion_strategy_selection_hierarchical(self, orchestrator_agent):
        """Test hierarchical strategy for comparison queries"""
        strategy = orchestrator_agent._select_fusion_strategy(
            "compare video and image results",
            {"video_search_agent": "video"},
        )
        assert strategy == FusionStrategy.HIERARCHICAL

    def test_fusion_strategy_selection_temporal(self, orchestrator_agent):
        """Test temporal strategy for time-related multi-modal queries"""
        strategy = orchestrator_agent._select_fusion_strategy(
            "show timeline of events",
            {"video_search_agent": "video", "audio_analysis_agent": "audio"},
        )
        assert strategy == FusionStrategy.TEMPORAL

    def test_fuse_by_score(self, orchestrator_agent):
        """Test score-based fusion method"""
        task_results = {
            "video_search_agent": {
                "agent": "video_search_agent",
                "modality": "video",
                "result": "video result",
                "confidence": 0.9,
            },
            "image_search_agent": {
                "agent": "image_search_agent",
                "modality": "image",
                "result": "image result",
                "confidence": 0.3,
            },
        }

        fused = orchestrator_agent._fuse_by_score(task_results)

        assert fused["confidence"] > 0
        assert "VIDEO" in fused["content"]
        assert "IMAGE" in fused["content"]

    def test_fuse_hierarchically(self, orchestrator_agent):
        """Test hierarchical fusion method"""
        task_results = {
            "video_agent": {
                "agent": "video_agent",
                "modality": "video",
                "result": "video data",
                "confidence": 0.8,
            },
            "text_agent": {
                "agent": "text_agent",
                "modality": "text",
                "result": "text data",
                "confidence": 0.6,
            },
        }
        agent_modalities = {"video_agent": "video", "text_agent": "text"}

        fused = orchestrator_agent._fuse_hierarchically(task_results, agent_modalities)

        assert "VIDEO RESULTS" in fused["content"]
        assert "TEXT RESULTS" in fused["content"]
        assert fused["confidence"] > 0

    def test_fuse_simple(self, orchestrator_agent):
        """Test simple fusion method"""
        task_results = {
            "agent_a": {"result": "result A", "confidence": 0.8},
            "agent_b": {"result": "result B", "confidence": 0.6},
        }

        fused = orchestrator_agent._fuse_simple(task_results)

        assert "result A" in fused["content"]
        assert "result B" in fused["content"]
        assert fused["confidence"] == pytest.approx(0.7, abs=0.01)

    def test_detect_agent_modality(self, orchestrator_agent):
        """Test modality detection from agent name"""
        assert (
            orchestrator_agent._detect_agent_modality("video_search_agent") == "video"
        )
        assert (
            orchestrator_agent._detect_agent_modality("image_search_agent") == "image"
        )
        assert (
            orchestrator_agent._detect_agent_modality("audio_analysis_agent") == "audio"
        )
        assert orchestrator_agent._detect_agent_modality("document_agent") == "document"
        assert orchestrator_agent._detect_agent_modality("search_agent") == "text"


class TestOrchestratorIntelligence:
    """Test workflow intelligence integration"""

    @pytest.mark.asyncio
    async def test_templates_used_in_planning(self, mock_agent_registry):
        """Test that workflow intelligence templates are used in planning"""
        with patch("dspy.ChainOfThought"):
            mock_template = Mock()
            mock_template.name = "video_search_template"
            mock_template.task_sequence = [{"agent": "search_agent"}]

            mock_intelligence = Mock()
            mock_intelligence._find_matching_template = Mock(return_value=mock_template)
            mock_intelligence.record_workflow_execution = AsyncMock()

            agent = OrchestratorAgent(
                deps=OrchestratorDeps(),
                registry=mock_agent_registry,
                config_manager=_make_mock_config_manager(),
                port=8013,
                workflow_intelligence=mock_intelligence,
            )

        # Mock the DSPy module
        agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="search_agent",
                parallel_steps="",
                reasoning="Search",
            )
        )

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_cm = Mock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "cogniverse_agents.orchestrator_agent.httpx.AsyncClient",
            return_value=mock_cm,
        ):
            await agent._process_impl(
                OrchestratorInput(query="Show me ML videos", tenant_id="test:unit")
            )

        # Template matching was called
        mock_intelligence._find_matching_template.assert_called_once_with(
            "Show me ML videos"
        )
        # Execution was recorded
        mock_intelligence.record_workflow_execution.assert_called_once()

    @pytest.mark.asyncio
    async def test_cancel_workflow(self, orchestrator_agent):
        """Test cancelling an active workflow"""
        # Add a mock workflow
        orchestrator_agent.active_workflows["wf_123"] = Mock()

        result = orchestrator_agent.cancel_workflow("wf_123")

        assert result is True
        assert "wf_123" not in orchestrator_agent.active_workflows

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_workflow(self, orchestrator_agent):
        """Test cancelling a non-existent workflow returns False"""
        result = orchestrator_agent.cancel_workflow("wf_nonexistent")
        assert result is False

    def test_orchestration_span_emitted(self, orchestrator_agent):
        """The cogniverse.orchestration span records the workflow on the
        canonical input/output slots."""
        import json

        class _RecordingSpan:
            def __init__(self):
                self.attrs = {}

            def set_attribute(self, key, value):
                self.attrs[key] = value

        class _Ctx:
            def __init__(self, span):
                self._span = span

            def __enter__(self):
                return self._span

            def __exit__(self, *exc):
                return False

        class _RecordingTelemetry:
            def __init__(self):
                self.calls = []
                self.span_obj = _RecordingSpan()

            def span(self, *, name, tenant_id):
                self.calls.append({"name": name, "tenant_id": tenant_id})
                return _Ctx(self.span_obj)

        recorder = _RecordingTelemetry()
        orchestrator_agent.telemetry_manager = recorder
        orchestrator_agent._current_tenant_id = "acme:prod"

        orchestrator_agent._emit_orchestration_span(
            workflow_id="wf_test",
            query="q" * 300,
            agent_sequence=["search_agent", "summarizer_agent"],
            execution_time=1.5,
            success=True,
            tasks_completed=2,
        )

        assert len(recorder.calls) == 1
        assert recorder.calls[0]["name"] == "cogniverse.orchestration"
        assert recorder.calls[0]["tenant_id"] == "acme:prod"
        attrs = recorder.span_obj.attrs
        assert attrs["operation"] == "orchestration"
        assert attrs["input.value"] == "q" * 300
        out = json.loads(attrs["output.value"])
        assert out["workflow_id"] == "wf_test"
        assert out["agent_sequence"] == ["search_agent", "summarizer_agent"]
        assert out["execution_time"] == 1.5
        assert out["success"] is True
        assert out["tasks_completed"] == 2

    def test_orchestration_span_noop_without_telemetry(self, orchestrator_agent):
        """No telemetry_manager -> silent no-op (back-compat, must not raise)."""
        orchestrator_agent.telemetry_manager = None
        orchestrator_agent._emit_orchestration_span(
            workflow_id="w",
            query="q",
            agent_sequence=["a"],
            execution_time=0.1,
            success=True,
            tasks_completed=1,
        )

    def test_orchestration_span_requires_tenant(self, orchestrator_agent):
        """Telemetry set but no _current_tenant_id -> raise (guards callers that
        emit before _process_impl set the tenant)."""
        orchestrator_agent.telemetry_manager = object()  # truthy; span never reached
        orchestrator_agent._current_tenant_id = None
        with pytest.raises(RuntimeError, match="_current_tenant_id"):
            orchestrator_agent._emit_orchestration_span(
                workflow_id="w",
                query="q",
                agent_sequence=["a"],
                execution_time=0.1,
                success=True,
                tasks_completed=1,
            )

    def test_dynamic_agent_discovery(self, orchestrator_agent):
        """Test that agents come from registry, not hardcoded enum"""
        agents = orchestrator_agent.registry.list_agents()

        # Should include all agents registered in mock registry
        assert "entity_extraction_agent" in agents
        assert "profile_selection_agent" in agents
        assert "query_enhancement_agent" in agents
        assert "search_agent" in agents
        assert "video_search_agent" in agents

    @pytest.mark.asyncio
    async def test_event_queue_emit(self, mock_agent_registry):
        """Test that events are emitted to event queue"""
        with patch("dspy.ChainOfThought"):
            mock_queue = AsyncMock()
            mock_queue.enqueue = AsyncMock()

            agent = OrchestratorAgent(
                deps=OrchestratorDeps(),
                registry=mock_agent_registry,
                config_manager=_make_mock_config_manager(),
                port=8013,
                event_queue=mock_queue,
            )

        event = {"type": "test", "data": "test_data"}
        await agent._emit_event(event)
        mock_queue.enqueue.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_emit_event_noop_without_queue(self, orchestrator_agent):
        """Test _emit_event is a noop when no queue configured"""
        assert orchestrator_agent.event_queue is None
        # Should not raise
        await orchestrator_agent._emit_event({"type": "test"})


class TestOrchestratorArtifactLoading:
    @pytest.mark.asyncio
    async def test_loads_workflow_templates(self, mock_agent_registry):
        """OrchestratorAgent should delegate to workflow_intelligence.load_historical_data."""
        mock_wi = Mock()
        mock_wi.load_historical_data = AsyncMock()
        mock_wi.workflow_templates = {"tmpl_1": Mock()}

        with patch("dspy.ChainOfThought"):
            agent = OrchestratorAgent(
                deps=OrchestratorDeps(),
                registry=mock_agent_registry,
                config_manager=_make_mock_config_manager(),
                workflow_intelligence=mock_wi,
            )

        mock_tm = Mock()
        agent.telemetry_manager = mock_tm
        agent._load_artifact()

        mock_wi.load_historical_data.assert_called_once()
        assert agent.artifact_load_status == "loaded"

    def test_no_workflow_intelligence_skips(self, orchestrator_agent):
        """_load_artifact returns without touching telemetry when there is no
        workflow_intelligence."""
        assert orchestrator_agent.workflow_intelligence is None
        recording_tm = Mock()
        orchestrator_agent.telemetry_manager = recording_tm

        assert orchestrator_agent._load_artifact() is None
        # Returned before reaching telemetry — the wi guard short-circuits.
        recording_tm.span.assert_not_called()
        assert orchestrator_agent.artifact_load_status == "disabled"

    def test_no_telemetry_skips(self, mock_agent_registry):
        """_load_artifact does not attempt a load when telemetry is unset."""
        mock_wi = Mock()
        mock_wi.load_historical_data = AsyncMock()

        with patch("dspy.ChainOfThought"):
            agent = OrchestratorAgent(
                deps=OrchestratorDeps(),
                registry=mock_agent_registry,
                config_manager=_make_mock_config_manager(),
                workflow_intelligence=mock_wi,
            )

        agent.telemetry_manager = None
        agent._load_artifact()

        mock_wi.load_historical_data.assert_not_called()
        assert agent.artifact_load_status == "no_telemetry"

    @pytest.mark.asyncio
    async def test_artifact_load_failure_surfaces_error_status(
        self, mock_agent_registry, caplog
    ):
        """A workflow-store OUTAGE must not read as 'no templates yet': the
        agent keeps serving on defaults but records status 'error' and logs
        at WARNING instead of swallowing the failure at DEBUG."""
        import logging

        mock_wi = Mock()
        mock_wi.load_historical_data = AsyncMock(
            side_effect=RuntimeError("connection refused")
        )

        with patch("dspy.ChainOfThought"):
            agent = OrchestratorAgent(
                deps=OrchestratorDeps(),
                registry=mock_agent_registry,
                config_manager=_make_mock_config_manager(),
                workflow_intelligence=mock_wi,
            )

        mock_tm = Mock()
        agent.telemetry_manager = mock_tm
        with caplog.at_level(logging.WARNING):
            agent._load_artifact()

        # The load was attempted (failure path exercised), and the raise was
        # surfaced as a status + WARNING rather than propagating.
        mock_wi.load_historical_data.assert_awaited_once()
        assert agent.artifact_load_status == "error"
        assert (
            "OrchestratorAgent workflow artifact load failed; using defaults"
            in caplog.text
        )


class TestOrchestratorSemanticRouting:
    """The orchestrator routes its per-request DSPy LM through the semantic router."""

    def _agent_with_config(self, cfg):
        agent = OrchestratorAgent.__new__(OrchestratorAgent)
        agent._config_manager = MagicMock()
        patcher = patch(
            "cogniverse_foundation.config.utils.get_config", lambda **kw: cfg
        )
        patcher.start()
        return agent, patcher

    def test_disabled_semantic_router_yields_nullcontext(self):
        cfg = MagicMock()
        cfg.get_semantic_router.return_value = SemanticRouterConfig(enabled=False)
        agent, patcher = self._agent_with_config(cfg)
        try:
            ctx = agent._semantic_router_lm_context("acme:prod")
            assert isinstance(ctx, nullcontext)
        finally:
            patcher.stop()

    def test_enabled_semantic_router_routes_the_active_lm(self):
        cfg = MagicMock()
        cfg.get_semantic_router.return_value = SemanticRouterConfig(
            enabled=True,
            semantic_router_url="http://envoy:8801/v1",
            tenant_tiers={"acme:prod": "pro"},
            default_tier="free",
        )
        cfg.get_llm_config.return_value.resolve.return_value = LLMEndpointConfig(
            model="openai/planner", api_base="http://vllm:8101/v1"
        )
        agent, patcher = self._agent_with_config(cfg)
        try:
            with agent._semantic_router_lm_context("acme:prod"):
                active = dspy.settings.lm
            assert active.kwargs["api_base"] == "http://envoy:8801/v1"
            assert active.kwargs["extra_headers"] == {
                "x-authz-user-id": "acme:prod",
                "x-authz-user-groups": "pro",
            }
        finally:
            patcher.stop()

    def test_resolution_error_propagates(self):
        # No silent fallback: a broken config store surfaces rather than
        # quietly leaving the orchestrator on the ambient LM.
        cfg = MagicMock()
        cfg.get_semantic_router.side_effect = RuntimeError("config store down")
        agent, patcher = self._agent_with_config(cfg)
        try:
            with pytest.raises(RuntimeError, match="config store down"):
                agent._semantic_router_lm_context("acme:prod")
        finally:
            patcher.stop()


class TestEnsureMemoryForTenant:
    """Pin the orchestrator -> initialize_memory call contract.

    The lazy per-tenant init swallows every exception ("continue without
    memory"), so a kwarg that initialize_memory does not accept fails
    silently for every tenant. autospec binds the real signature, making
    a signature mismatch fail this test instead of only warning in logs.
    """

    def _system_config(self) -> SystemConfig:
        return SystemConfig(inference_service_urls={"denseon": "http://denseon:8000"})

    def test_initializes_memory_with_bare_model_name(self, orchestrator_agent):
        orchestrator_agent._config_manager.get_system_config = Mock(
            return_value=self._system_config()
        )
        cfg = MagicMock()
        cfg.get_llm_config.return_value.resolve.return_value = LLMEndpointConfig(
            model="openai/google/gemma-4-e4b-it",
            api_base="http://vllm-llm:8000/v1",
        )
        bootstrap = MagicMock(backend_url="http://vespa", backend_port=8080)
        with (
            patch("cogniverse_foundation.config.utils.get_config", return_value=cfg),
            patch(
                "cogniverse_foundation.config.bootstrap.BootstrapConfig"
                ".from_environment",
                return_value=bootstrap,
            ),
            patch.object(OrchestratorAgent, "initialize_memory", autospec=True) as init,
        ):
            init.return_value = True
            orchestrator_agent._ensure_memory_for_tenant("acme:prod")

        assert "acme:prod" in orchestrator_agent._memory_initialized_tenants, (
            "init must complete without raising — a swallowed TypeError from a "
            "bad kwarg leaves the tenant uninitialized on every request"
        )
        init.assert_called_once()
        kwargs = init.call_args.kwargs
        assert kwargs["llm_model"] == "google/gemma-4-e4b-it"
        assert kwargs["llm_base_url"] == "http://vllm-llm:8000/v1"
        assert kwargs["embedder_base_url"] == "http://denseon:8000"
        assert kwargs["tenant_id"] == "acme:prod"
        assert kwargs["agent_name"] == "orchestrator_agent"

    def test_second_call_for_same_tenant_is_a_noop(self, orchestrator_agent):
        orchestrator_agent._memory_initialized_tenants.add("acme:prod")
        with patch.object(
            OrchestratorAgent, "initialize_memory", autospec=True
        ) as init:
            orchestrator_agent._ensure_memory_for_tenant("acme:prod")
        init.assert_not_called()


class TestCoerceFloat:
    """coerce_float maps untrusted A2A/KG numeric fields to a finite float
    instead of raising. Sub-agent results and KG mention blobs carry
    ts_start/ts_end/score as arbitrary JSON — a real agent may emit "00:12",
    "high", "" or null, and a naked float() would crash the evidence walk."""

    def test_numeric_passthrough(self):
        from cogniverse_agents._coercion import coerce_float

        assert coerce_float(12.5) == 12.5
        assert coerce_float(5) == 5.0

    def test_numeric_string_parsed(self):
        from cogniverse_agents._coercion import coerce_float

        assert coerce_float("12.5") == 12.5
        assert coerce_float("3600") == 3600.0

    def test_non_numeric_string_falls_back(self):
        from cogniverse_agents._coercion import coerce_float

        assert coerce_float("00:12") == 0.0
        assert coerce_float("high") == 0.0
        assert coerce_float("") == 0.0

    def test_none_and_wrong_type_fall_back(self):
        from cogniverse_agents._coercion import coerce_float

        assert coerce_float(None) == 0.0
        assert coerce_float([1, 2]) == 0.0
        assert coerce_float({"a": 1}) == 0.0

    def test_custom_default(self):
        from cogniverse_agents._coercion import coerce_float

        assert coerce_float("nope", default=-1.0) == -1.0
        assert coerce_float(None, default=7.0) == 7.0

    def test_large_value_not_clamped(self):
        # Unlike parse_confidence, coerce_float must NOT clamp to [0,1] — a
        # 125.7s video timestamp and an unbounded ranking score survive whole.
        from cogniverse_agents._coercion import coerce_float

        assert coerce_float(125.7) == 125.7
        assert coerce_float(15.3) == 15.3

    def test_non_finite_falls_back(self):
        # NaN/inf in a sort key or interval check corrupt ordering — reject.
        from cogniverse_agents._coercion import coerce_float

        assert coerce_float(float("nan")) == 0.0
        assert coerce_float(float("inf")) == 0.0
        assert coerce_float("nan") == 0.0
        assert coerce_float("inf") == 0.0


class TestEvidenceCoercionCrashSafety:
    """A sub-agent A2A result carries ts_start/ts_end/score as arbitrary
    JSON. Non-numeric values degrade to 0.0 rather than crashing the
    evidence-extraction and ranking path."""

    def test_coerce_snippet_non_numeric_timestamp_does_not_crash(self):
        snippet = OrchestratorAgent._coerce_evidence_snippet(
            {
                "source_doc_id": "doc1",
                "ts_start": "00:12",
                "ts_end": "00:34",
                "text": "some evidence",
            }
        )
        assert snippet is not None
        assert snippet["ts_start"] == 0.0
        assert snippet["ts_end"] == 0.0
        assert snippet["source_doc_id"] == "doc1"

    def test_coerce_snippet_numeric_string_timestamp_parsed(self):
        snippet = OrchestratorAgent._coerce_evidence_snippet(
            {"source_doc_id": "doc1", "ts_start": "12.5", "ts_end": "34", "text": "x"}
        )
        assert snippet["ts_start"] == 12.5
        assert snippet["ts_end"] == 34.0

    def test_coerce_snippet_large_timestamp_not_clamped(self):
        snippet = OrchestratorAgent._coerce_evidence_snippet(
            {"source_doc_id": "doc1", "ts_start": 125.7, "ts_end": 130.2, "text": "x"}
        )
        assert snippet["ts_start"] == 125.7
        assert snippet["ts_end"] == 130.2

    def test_rank_evidence_non_numeric_score_does_not_crash_sort(self):
        ranked = OrchestratorAgent._rank_evidence_for_metadata(
            [
                {"source_doc_id": "str_score", "score": "high"},
                {"source_doc_id": "num_score", "score": 0.9},
            ]
        )
        # No ValueError in the sort; the numeric-score hit sorts first, the
        # string-score hit coerces to 0.0 for ordering and lands after it.
        assert [e["source_doc_id"] for e in ranked] == ["num_score", "str_score"]
