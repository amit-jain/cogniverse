"""Unit tests for OrchestratorAgent"""

from unittest.mock import AsyncMock, Mock, patch

import dspy
import pytest
from cogniverse_agents.orchestrator_agent import (
    AgentStep,
    AgentType,
    OrchestrationModule,
    OrchestrationPlan,
    OrchestrationResult,
    OrchestratorAgent,
)


@pytest.fixture
def mock_dspy_lm():
    """Mock DSPy language model"""
    lm = Mock()
    lm.return_value = dspy.Prediction(
        agent_sequence="query_enhancement,entity_extraction,profile_selection,search",
        parallel_steps="0,1",
        reasoning="Enhance query and extract entities in parallel, then select profile and search sequentially",
    )
    return lm


@pytest.fixture
def mock_agent_registry():
    """Mock agent registry with test agents"""
    registry = {}

    # Create mock agents
    for agent_type in [
        AgentType.ENTITY_EXTRACTION,
        AgentType.PROFILE_SELECTION,
        AgentType.QUERY_ENHANCEMENT,
        AgentType.SEARCH,
    ]:
        mock_agent = Mock()
        mock_agent._process = AsyncMock(
            return_value={
                "status": "success",
                "agent": agent_type.value,
                "result": f"Mock result from {agent_type.value}",
            }
        )
        registry[agent_type] = mock_agent

    return registry


@pytest.fixture
def orchestrator_agent(mock_agent_registry):
    """Create OrchestratorAgent for testing"""
    with patch("dspy.ChainOfThought"):
        agent = OrchestratorAgent(
            tenant_id="test_tenant", agent_registry=mock_agent_registry, port=8013
        )
        return agent


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
            available_agents="query_enhancement,entity_extraction,profile_selection,search",
        )

        assert (
            result.agent_sequence
            == "query_enhancement,entity_extraction,profile_selection,search"
        )
        assert result.parallel_steps == "0,1"
        assert "parallel" in result.reasoning.lower()

    def test_forward_fallback(self):
        """Test fallback when DSPy fails"""
        module = OrchestrationModule()
        module.planner = Mock(side_effect=Exception("DSPy failed"))

        result = module.forward(
            query="Show me videos",
            available_agents="query_enhancement,entity_extraction,profile_selection,search",
        )

        # Fallback should have default sequence
        assert "query_enhancement" in result.agent_sequence
        assert "entity_extraction" in result.agent_sequence
        assert "search" in result.agent_sequence
        assert result.parallel_steps == "0,1"


class TestOrchestratorAgent:
    """Test OrchestratorAgent core functionality"""

    def test_agent_initialization(self, orchestrator_agent):
        """Test agent initializes with correct configuration"""
        assert orchestrator_agent.agent_name == "orchestrator_agent"
        assert orchestrator_agent.tenant_id == "test_tenant"
        assert "orchestration" in orchestrator_agent.capabilities
        assert "planning" in orchestrator_agent.capabilities
        assert len(orchestrator_agent.agent_registry) == 4

    @pytest.mark.asyncio
    async def test_create_plan(self, orchestrator_agent):
        """Test planning phase"""
        orchestrator_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="query_enhancement,profile_selection,search",
                parallel_steps="",
                reasoning="Sequential execution: enhance, select, search",
            )
        )

        plan = await orchestrator_agent._create_plan("Show me ML videos")

        assert isinstance(plan, OrchestrationPlan)
        assert plan.query == "Show me ML videos"
        assert len(plan.steps) == 3
        assert plan.steps[0].agent_type == AgentType.QUERY_ENHANCEMENT
        assert plan.steps[1].agent_type == AgentType.PROFILE_SELECTION
        assert plan.steps[2].agent_type == AgentType.SEARCH

    @pytest.mark.asyncio
    async def test_create_plan_with_parallel_groups(self, orchestrator_agent):
        """Test planning with parallel execution groups"""
        orchestrator_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="query_enhancement,entity_extraction,search",
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
                agent_sequence="query_enhancement,entity_extraction,profile_selection,search",
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
        """Test action phase execution"""
        plan = OrchestrationPlan(
            query="test query",
            steps=[
                AgentStep(
                    agent_type=AgentType.QUERY_ENHANCEMENT,
                    input_data={"query": "test query"},
                    depends_on=[],
                    reasoning="Enhance query",
                ),
                AgentStep(
                    agent_type=AgentType.SEARCH,
                    input_data={"query": "test query"},
                    depends_on=[0],
                    reasoning="Search",
                ),
            ],
            parallel_groups=[],
            reasoning="Test plan",
        )

        results = await orchestrator_agent._execute_plan(plan)

        assert len(results) == 2
        assert "query_enhancement" in results
        assert "search" in results
        assert results["query_enhancement"]["status"] == "success"
        assert results["search"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_execute_plan_agent_not_found(self, orchestrator_agent):
        """Test execution when agent not in registry"""
        plan = OrchestrationPlan(
            query="test query",
            steps=[
                AgentStep(
                    agent_type=AgentType.SUMMARIZER,  # Not in mock registry
                    input_data={"query": "test query"},
                    depends_on=[],
                    reasoning="Summarize",
                )
            ],
            parallel_groups=[],
            reasoning="Test plan",
        )

        results = await orchestrator_agent._execute_plan(plan)

        assert "summarizer" in results
        assert results["summarizer"]["status"] == "error"
        assert "not available" in results["summarizer"]["message"]

    @pytest.mark.asyncio
    async def test_execute_plan_agent_error(self, orchestrator_agent):
        """Test execution when agent raises exception"""
        # Make one agent fail
        orchestrator_agent.agent_registry[AgentType.SEARCH]._process = AsyncMock(
            side_effect=Exception("Agent failed")
        )

        plan = OrchestrationPlan(
            query="test query",
            steps=[
                AgentStep(
                    agent_type=AgentType.SEARCH,
                    input_data={"query": "test query"},
                    depends_on=[],
                    reasoning="Search",
                )
            ],
            parallel_groups=[],
            reasoning="Test plan",
        )

        results = await orchestrator_agent._execute_plan(plan)

        assert "search" in results
        assert results["search"]["status"] == "error"
        assert "Agent failed" in results["search"]["message"]

    @pytest.mark.asyncio
    async def test_process_full_workflow(self, orchestrator_agent):
        """Test complete orchestration workflow"""
        orchestrator_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="query_enhancement,search",
                parallel_steps="",
                reasoning="Enhance then search",
            )
        )

        result = await orchestrator_agent._process(
            {"query": "Show me machine learning videos"}
        )

        assert isinstance(result, OrchestrationResult)
        assert result.query == "Show me machine learning videos"
        assert len(result.plan.steps) == 2
        assert "query_enhancement" in result.agent_results
        assert "search" in result.agent_results
        assert result.final_output["status"] == "success"
        assert "Executed" in result.execution_summary

    @pytest.mark.asyncio
    async def test_process_empty_query(self, orchestrator_agent):
        """Test processing empty query"""
        result = await orchestrator_agent._process({"query": ""})

        assert result.query == ""
        assert len(result.plan.steps) == 0
        assert result.final_output["status"] == "error"
        assert "Empty query" in result.final_output["message"]

    def test_aggregate_results(self, orchestrator_agent):
        """Test result aggregation"""
        agent_results = {
            "query_enhancement": {
                "enhanced_query": "machine learning tutorials",
                "confidence": 0.9,
            },
            "search": {"results": [{"title": "ML Tutorial 1"}], "count": 1},
        }

        final_output = orchestrator_agent._aggregate_results(
            "test query", agent_results
        )

        assert final_output["query"] == "test query"
        assert final_output["status"] == "success"
        assert "query_enhancement" in final_output["results"]
        assert "search" in final_output["results"]

    def test_generate_summary(self, orchestrator_agent):
        """Test execution summary generation"""
        plan = OrchestrationPlan(
            query="test",
            steps=[
                AgentStep(
                    agent_type=AgentType.SEARCH,
                    input_data={},
                    depends_on=[],
                    reasoning="Search",
                ),
                AgentStep(
                    agent_type=AgentType.SUMMARIZER,
                    input_data={},
                    depends_on=[],
                    reasoning="Summarize",
                ),
            ],
            parallel_groups=[],
            reasoning="Test plan",
        )

        agent_results = {
            "search": {"status": "success"},
            "summarizer": {"status": "error", "message": "Failed"},
        }

        summary = orchestrator_agent._generate_summary(plan, agent_results)

        assert "1/2" in summary  # 1 successful out of 2
        assert "Test plan" in summary

    def test_dspy_to_a2a_output(self, orchestrator_agent):
        """Test conversion to A2A output format"""
        result = OrchestrationResult(
            query="test query",
            plan=OrchestrationPlan(
                query="test query",
                steps=[
                    AgentStep(
                        agent_type=AgentType.SEARCH,
                        input_data={"query": "test"},
                        depends_on=[],
                        reasoning="Search step",
                    )
                ],
                parallel_groups=[],
                reasoning="Test plan",
            ),
            agent_results={"search": {"status": "success"}},
            final_output={"status": "success"},
            execution_summary="Executed 1/1 steps",
        )

        a2a_output = orchestrator_agent._dspy_to_a2a_output(result)

        assert a2a_output["status"] == "success"
        assert a2a_output["agent"] == "orchestrator_agent"
        assert a2a_output["query"] == "test query"
        assert len(a2a_output["plan"]["steps"]) == 1
        assert a2a_output["plan"]["steps"][0]["agent_type"] == "search"
        assert a2a_output["execution_summary"] == "Executed 1/1 steps"

    def test_get_agent_skills(self, orchestrator_agent):
        """Test agent skills definition"""
        skills = orchestrator_agent._get_agent_skills()

        assert len(skills) == 1
        assert skills[0]["name"] == "orchestrate"
        assert "query" in skills[0]["input_schema"]
        assert "plan" in skills[0]["output_schema"]
        assert "agent_results" in skills[0]["output_schema"]
        assert len(skills[0]["examples"]) > 0


class TestOrchestratorAgentIntegration:
    """Integration tests for OrchestratorAgent"""

    @pytest.mark.asyncio
    async def test_full_orchestration_workflow(self, orchestrator_agent):
        """Test complete orchestration with multiple agents"""
        orchestrator_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="query_enhancement,entity_extraction,profile_selection,search",
                parallel_steps="0,1|2,3",
                reasoning="Enhance and extract in parallel, then select and search in parallel",
            )
        )

        result = await orchestrator_agent._process(
            {"query": "Show me detailed tutorials about deep learning architectures"}
        )

        # Verify complete workflow
        assert (
            result.query
            == "Show me detailed tutorials about deep learning architectures"
        )
        assert len(result.plan.steps) == 4
        assert len(result.plan.parallel_groups) == 2

        # Verify all agents executed
        assert "query_enhancement" in result.agent_results
        assert "entity_extraction" in result.agent_results
        assert "profile_selection" in result.agent_results
        assert "search" in result.agent_results

        # Verify results aggregated
        assert result.final_output["status"] == "success"
        assert len(result.final_output["results"]) == 4

        # Verify summary
        assert "4/4" in result.execution_summary

    @pytest.mark.asyncio
    async def test_partial_failure_handling(self, orchestrator_agent):
        """Test orchestration with partial agent failures"""
        # Make one agent fail
        orchestrator_agent.agent_registry[AgentType.ENTITY_EXTRACTION]._process = (
            AsyncMock(side_effect=Exception("Entity extraction failed"))
        )

        orchestrator_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="query_enhancement,entity_extraction,search",
                parallel_steps="",
                reasoning="Sequential execution",
            )
        )

        result = await orchestrator_agent._process({"query": "Test query"})

        # Verify partial execution
        assert result.query == "Test query"
        assert "query_enhancement" in result.agent_results
        assert result.agent_results["query_enhancement"]["status"] == "success"
        assert "entity_extraction" in result.agent_results
        assert result.agent_results["entity_extraction"]["status"] == "error"
        assert "search" in result.agent_results
        assert result.agent_results["search"]["status"] == "success"

        # Verify summary reflects partial failure
        assert "2/3" in result.execution_summary

    @pytest.mark.asyncio
    async def test_a2a_task_processing(self, orchestrator_agent):
        """Test processing via A2A task format"""
        orchestrator_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="profile_selection,search",
                parallel_steps="",
                reasoning="Select profile then search",
            )
        )

        # Simulate A2A task input
        dspy_input = orchestrator_agent._a2a_to_dspy_input(
            {
                "id": "test_task",
                "messages": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "text": "Find videos about Python programming",
                            }
                        ],
                    }
                ],
            }
        )

        result = await orchestrator_agent._process(dspy_input)
        a2a_output = orchestrator_agent._dspy_to_a2a_output(result)

        assert a2a_output["status"] == "success"
        assert a2a_output["query"] == "Find videos about Python programming"
        assert len(a2a_output["plan"]["steps"]) == 2
        assert a2a_output["plan"]["steps"][0]["agent_type"] == "profile_selection"
        assert a2a_output["plan"]["steps"][1]["agent_type"] == "search"

    @pytest.mark.asyncio
    async def test_dependency_chain_execution(self, orchestrator_agent):
        """Test that agents execute respecting dependencies"""
        execution_order = []

        # Track execution order
        def make_tracking_agent(agent_type):
            async def _process(input_data):
                execution_order.append(agent_type.value)
                return {"status": "success", "agent": agent_type.value}

            return _process

        for agent_type in orchestrator_agent.agent_registry.keys():
            orchestrator_agent.agent_registry[agent_type]._process = (
                make_tracking_agent(agent_type)
            )

        orchestrator_agent.dspy_module.forward = Mock(
            return_value=dspy.Prediction(
                agent_sequence="query_enhancement,entity_extraction,search",
                parallel_steps="0,1",  # 0 and 1 parallel, then 2
                reasoning="Test dependency chain",
            )
        )

        await orchestrator_agent._process({"query": "Test"})

        # Verify execution order
        assert len(execution_order) == 3
        # query_enhancement and entity_extraction can be in any order (parallel)
        assert set(execution_order[:2]) == {"query_enhancement", "entity_extraction"}
        # search must be last
        assert execution_order[2] == "search"
