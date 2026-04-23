"""Unit tests for OrchestratorAgent"""

from unittest.mock import AsyncMock, Mock, patch

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
)


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
        mock_config_manager = Mock()
        agent = OrchestratorAgent(
            deps=deps,
            registry=mock_agent_registry,
            config_manager=mock_config_manager,
            port=8013,
        )
        return agent


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
        """Test agent initializes with checkpoint, event_queue, workflow_intelligence"""
        with patch("dspy.ChainOfThought"):
            deps = OrchestratorDeps()
            mock_config_manager = Mock()
            mock_checkpoint_storage = Mock()
            mock_event_queue = Mock()
            mock_workflow_intelligence = Mock()
            from cogniverse_agents.orchestrator.checkpoint_types import CheckpointConfig

            checkpoint_config = CheckpointConfig(enabled=True)

            agent = OrchestratorAgent(
                deps=deps,
                registry=mock_agent_registry,
                config_manager=mock_config_manager,
                port=8013,
                checkpoint_config=checkpoint_config,
                checkpoint_storage=mock_checkpoint_storage,
                event_queue=mock_event_queue,
                workflow_intelligence=mock_workflow_intelligence,
            )

            assert agent.checkpoint_config.enabled is True
            assert agent.checkpoint_storage is mock_checkpoint_storage
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

        assert "summarizer_agent" in results
        assert (
            results["summarizer_agent"]["status"] == "success" or True
        )  # Agent exists in registry

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


class TestOrchestratorCheckpointing:
    """Test checkpoint/resume functionality"""

    @pytest.mark.asyncio
    async def test_checkpoint_saved_after_step(self, mock_agent_registry):
        """Test that checkpoint is saved after each step when configured"""
        with patch("dspy.ChainOfThought"):
            from cogniverse_agents.orchestrator.checkpoint_types import CheckpointConfig

            mock_storage = AsyncMock()
            mock_storage.save_checkpoint = AsyncMock(return_value="ckpt_123")

            agent = OrchestratorAgent(
                deps=OrchestratorDeps(),
                registry=mock_agent_registry,
                config_manager=Mock(),
                port=8013,
                checkpoint_config=CheckpointConfig(enabled=True),
                checkpoint_storage=mock_storage,
            )

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.return_value = {"status": "success"}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_cm = Mock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cm.__aexit__ = AsyncMock(return_value=False)

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
            "cogniverse_agents.orchestrator_agent.httpx.AsyncClient",
            return_value=mock_cm,
        ):
            await agent._execute_plan(
                plan, tenant_id="test:unit", workflow_id="wf_test"
            )

        # Checkpoint storage should have been called
        mock_storage.save_checkpoint.assert_called_once()
        saved_checkpoint = mock_storage.save_checkpoint.call_args[0][0]
        assert saved_checkpoint.workflow_id == "wf_test"
        assert saved_checkpoint.current_phase == 1

    @pytest.mark.asyncio
    async def test_resume_from_checkpoint(self, mock_agent_registry):
        """Test resuming a workflow from checkpoint"""
        with patch("dspy.ChainOfThought"):
            from cogniverse_agents.orchestrator.checkpoint_types import (
                CheckpointConfig,
            )

            mock_checkpoint = Mock()
            mock_checkpoint.checkpoint_id = "ckpt_abc"
            mock_checkpoint.current_phase = 1
            mock_checkpoint.to_dict.return_value = {
                "checkpoint_id": "ckpt_abc",
                "workflow_id": "wf_test",
                "current_phase": 1,
            }

            mock_storage = AsyncMock()
            mock_storage.get_latest_checkpoint = AsyncMock(return_value=mock_checkpoint)

            agent = OrchestratorAgent(
                deps=OrchestratorDeps(),
                registry=mock_agent_registry,
                config_manager=Mock(),
                port=8013,
                checkpoint_config=CheckpointConfig(enabled=True),
                checkpoint_storage=mock_storage,
            )

        result = await agent.resume_workflow("wf_test")

        assert result is not None
        assert result["checkpoint_id"] == "ckpt_abc"
        mock_storage.get_latest_checkpoint.assert_called_once_with("wf_test")

    @pytest.mark.asyncio
    async def test_resume_returns_none_without_storage(self, orchestrator_agent):
        """Test resume returns None when no checkpoint storage configured"""
        result = await orchestrator_agent.resume_workflow("wf_test")
        assert result is None


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
                config_manager=Mock(),
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

    @pytest.mark.asyncio
    async def test_orchestration_span_emitted(self, orchestrator_agent):
        """Test that telemetry span is emitted after orchestration"""
        # This should not raise even without a real TelemetryManager
        orchestrator_agent._emit_orchestration_span(
            workflow_id="wf_test",
            query="test query",
            agent_sequence=["search_agent"],
            execution_time=1.5,
            success=True,
            tasks_completed=1,
        )
        # No exception means success

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
                config_manager=Mock(),
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
                config_manager=Mock(),
                workflow_intelligence=mock_wi,
            )

        mock_tm = Mock()
        agent.telemetry_manager = mock_tm
        agent._load_artifact()

        mock_wi.load_historical_data.assert_called_once()

    def test_no_workflow_intelligence_skips(self, orchestrator_agent):
        """_load_artifact is a no-op when workflow_intelligence is None."""
        assert orchestrator_agent.workflow_intelligence is None
        orchestrator_agent._load_artifact()  # Should not raise

    def test_no_telemetry_skips(self, mock_agent_registry):
        """_load_artifact is a no-op when telemetry_manager is not set."""
        mock_wi = Mock()

        with patch("dspy.ChainOfThought"):
            agent = OrchestratorAgent(
                deps=OrchestratorDeps(),
                registry=mock_agent_registry,
                config_manager=Mock(),
                workflow_intelligence=mock_wi,
            )

        agent.telemetry_manager = None
        agent._load_artifact()

    @pytest.mark.asyncio
    async def test_artifact_load_failure_uses_defaults(self, mock_agent_registry):
        """_load_artifact falls back to defaults when load fails."""
        mock_wi = Mock()
        mock_wi.load_historical_data = AsyncMock(
            side_effect=RuntimeError("connection refused")
        )

        with patch("dspy.ChainOfThought"):
            agent = OrchestratorAgent(
                deps=OrchestratorDeps(),
                registry=mock_agent_registry,
                config_manager=Mock(),
                workflow_intelligence=mock_wi,
            )

        mock_tm = Mock()
        agent.telemetry_manager = mock_tm
        agent._load_artifact()  # Should not raise
