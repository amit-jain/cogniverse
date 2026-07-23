"""
Unit tests for WorkflowIntelligence — read-only template loader
"""

from unittest.mock import Mock

import pytest

from cogniverse_agents.workflow.intelligence import (
    OptimizationStrategy,
    WorkflowIntelligence,
)
from cogniverse_agents.workflow_types import (
    TaskStatus,
    WorkflowPlan,
    WorkflowStatus,
    WorkflowTask,
)
from cogniverse_sdk.interfaces.workflow_store import AgentPerformance, WorkflowTemplate


def _make_intelligence(**kwargs) -> WorkflowIntelligence:
    """Create a WorkflowIntelligence instance with a mock telemetry_provider."""
    defaults = dict(
        telemetry_provider=Mock(),
        tenant_id="test_tenant",
    )
    defaults.update(kwargs)
    return WorkflowIntelligence(**defaults)


@pytest.mark.unit
class TestWorkflowIntelligence:
    """Test cases for WorkflowIntelligence class"""

    def test_workflow_intelligence_initialization(self):
        """Test WorkflowIntelligence initializes with required telemetry_provider and tenant_id"""
        intelligence = _make_intelligence(
            max_history_size=1000,
            optimization_strategy=OptimizationStrategy.BALANCED,
        )

        assert intelligence._store is not None
        # max_history_size caps the workflow-history deque (the behavior it drives).
        assert intelligence.workflow_history.maxlen == 1000
        assert intelligence.optimization_strategy == OptimizationStrategy.BALANCED

    def test_workflow_intelligence_requires_tenant_id(self):
        """Empty tenant_id must raise ValueError"""
        with pytest.raises(ValueError, match="tenant_id is required"):
            WorkflowIntelligence(telemetry_provider=Mock(), tenant_id="")

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_record_workflow_execution_is_noop(self):
        """record_workflow_execution is a no-op; records are telemetry spans."""
        intelligence = _make_intelligence()

        workflow_plan = WorkflowPlan(
            workflow_id="test-workflow",
            original_query="find AI videos",
            status=WorkflowStatus.COMPLETED,
            tasks=[
                WorkflowTask(
                    task_id="task1", agent_name="video_search", query="find AI videos"
                ),
            ],
        )

        # record_workflow_execution is a no-op on the per-request hot path.
        # Workflow records live in telemetry spans; batch optimization
        # rebuilds in-memory history via load_historical_data.
        assert len(intelligence.workflow_history) == 0
        await intelligence.record_workflow_execution(workflow_plan)
        assert len(intelligence.workflow_history) == 0

    @pytest.mark.ci_fast
    @pytest.mark.asyncio
    async def test_optimization_workflow_methods(self):
        """Test workflow optimization functionality"""
        intelligence = _make_intelligence()

        assert hasattr(intelligence, "optimize_workflow_plan")
        assert callable(getattr(intelligence, "optimize_workflow_plan"))

        workflow_plan = WorkflowPlan(
            workflow_id="test-workflow",
            original_query="test query",
            tasks=[
                WorkflowTask(task_id="task1", agent_name="agent1", query="test query")
            ],
        )

        optimized_plan = await intelligence.optimize_workflow_plan(
            "test query", workflow_plan
        )
        assert isinstance(optimized_plan, WorkflowPlan)

    @pytest.mark.asyncio
    async def test_get_agent_performance_metrics(self):
        """Test getting agent performance report (read-only from loaded data)"""
        intelligence = _make_intelligence()

        report = intelligence.get_agent_performance_report()
        assert isinstance(report, dict)
        # No data loaded, so report is empty
        assert len(report) == 0

    def test_query_type_classification(self):
        """Test query type classification functionality"""
        intelligence = _make_intelligence()

        assert hasattr(intelligence, "_classify_query_type")
        assert callable(getattr(intelligence, "_classify_query_type"))

        query_type = intelligence._classify_query_type(
            "find videos about machine learning"
        )
        assert isinstance(query_type, str)
        assert len(query_type) > 0
        assert query_type == "video_search"


@pytest.mark.unit
class TestSimplifiedWorkflowIntelligence:
    """Tests for the simplified read-only template loader."""

    def test_initialization_without_optimization(self):
        intelligence = _make_intelligence()
        assert intelligence is not None
        # No DSPy modules should be present
        assert not hasattr(intelligence, "workflow_optimizer")
        assert not hasattr(intelligence, "template_generator")

    @pytest.mark.asyncio
    async def test_load_templates(self):
        intelligence = _make_intelligence()
        templates = intelligence.get_workflow_templates()
        assert isinstance(templates, list)
        assert len(templates) == 0  # Nothing loaded

    @pytest.mark.asyncio
    async def test_find_matching_template(self):
        intelligence = _make_intelligence()
        template = WorkflowTemplate(
            template_id="t1",
            name="multi_modal_search",
            description="Multi-modal search workflow",
            query_patterns=["find videos and documents"],
            task_sequence=[
                {"agent": "search_agent"},
                {"agent": "document_agent"},
            ],
            expected_execution_time=3.0,
            success_rate=0.9,
        )
        intelligence.workflow_templates["t1"] = template

        match = intelligence._find_matching_template(
            "find videos and documents about AI"
        )
        assert match is not None
        assert match.template_id == "t1"

    def test_get_agent_performance_report(self):
        intelligence = _make_intelligence()
        report = intelligence.get_agent_performance_report()
        assert isinstance(report, dict)

    @pytest.mark.asyncio
    async def test_record_workflow_execution_is_noop_on_hot_path(self):
        """Per-request record_workflow_execution does not mutate in-memory history."""
        intelligence = _make_intelligence()
        plan = WorkflowPlan(
            workflow_id="test-wf",
            original_query="test",
            status=WorkflowStatus.COMPLETED,
            tasks=[],
        )
        await intelligence.record_workflow_execution(plan)
        # No-op — spans carry the record; in-memory history unchanged.
        assert len(intelligence.workflow_history) == 0

    @pytest.mark.asyncio
    async def test_record_execution_appends_to_history(self):
        """record_execution appends a WorkflowExecution to history."""
        from cogniverse_sdk.interfaces.workflow_store import WorkflowExecution

        intelligence = _make_intelligence()
        execution = WorkflowExecution(
            workflow_id="wf-1",
            query="test",
            query_type="general",
            execution_time=1.0,
            success=True,
            agent_sequence=["agent1"],
            task_count=1,
            parallel_efficiency=1.0,
            confidence_score=0.9,
        )
        await intelligence.record_execution(execution)
        assert len(intelligence.workflow_history) == 1
        assert intelligence.workflow_history[0].workflow_id == "wf-1"
        assert intelligence.workflow_history[0].success is True

    @pytest.mark.asyncio
    async def test_record_execution_respects_max_history_size(self):
        """record_execution should respect max_history_size limit."""
        from cogniverse_sdk.interfaces.workflow_store import WorkflowExecution

        intelligence = _make_intelligence(max_history_size=2)
        for i in range(3):
            execution = WorkflowExecution(
                workflow_id=f"wf-{i}",
                query="test",
                query_type="general",
                execution_time=1.0,
                success=True,
                agent_sequence=["agent1"],
                task_count=1,
                parallel_efficiency=1.0,
                confidence_score=0.9,
            )
            await intelligence.record_execution(execution)
        assert len(intelligence.workflow_history) == 2
        assert intelligence.workflow_history[0].workflow_id == "wf-1"
        assert intelligence.workflow_history[1].workflow_id == "wf-2"

    @pytest.mark.asyncio
    async def test_optimize_from_ground_truth_is_noop(self):
        """optimize_from_ground_truth returns skip status."""
        intelligence = _make_intelligence()
        result = await intelligence.optimize_from_ground_truth()
        assert result["status"] == "skipped"
        assert result["reason"] == "use_argo_batch_jobs"

    @pytest.mark.asyncio
    async def test_optimize_plan_without_templates_uses_strategy(self):
        """Without templates, optimize_workflow_plan falls through to strategy."""
        intelligence = _make_intelligence()
        plan = WorkflowPlan(
            workflow_id="wf-1",
            original_query="test query",
            tasks=[WorkflowTask(task_id="t1", agent_name="agent1", query="test query")],
        )
        result = await intelligence.optimize_workflow_plan("test query", plan)
        assert isinstance(result, WorkflowPlan)
        assert intelligence.optimization_stats["successful_optimizations"] == 1

    def test_get_workflow_templates_returns_values(self):
        """get_workflow_templates returns list of template objects."""
        intelligence = _make_intelligence()
        template = WorkflowTemplate(
            template_id="t1",
            name="test",
            description="test template",
            query_patterns=["test"],
            task_sequence=[],
            expected_execution_time=1.0,
            success_rate=0.9,
        )
        intelligence.workflow_templates["t1"] = template
        templates = intelligence.get_workflow_templates()
        assert len(templates) == 1
        assert templates[0].template_id == "t1"


@pytest.mark.unit
class TestExecutionOrder:
    """_calculate_execution_order groups tasks into dependency-layered phases
    (independent tasks share a phase) instead of forcing strict sequence."""

    def _task(self, tid, deps=()):
        return WorkflowTask(
            task_id=tid, agent_name="a", query="q", dependencies=set(deps)
        )

    def test_independent_tasks_share_one_phase(self):
        wi = _make_intelligence()
        order = wi._calculate_execution_order(
            [self._task("t1"), self._task("t2"), self._task("t3")]
        )
        assert order == [["t1", "t2", "t3"]]

    def test_linear_chain_one_task_per_phase(self):
        wi = _make_intelligence()
        order = wi._calculate_execution_order(
            [self._task("t1"), self._task("t2", {"t1"}), self._task("t3", {"t2"})]
        )
        assert order == [["t1"], ["t2"], ["t3"]]

    def test_diamond_runs_middle_tasks_in_parallel(self):
        wi = _make_intelligence()
        order = wi._calculate_execution_order(
            [
                self._task("t1"),
                self._task("t2", {"t1"}),
                self._task("t3", {"t1"}),
                self._task("t4", {"t2", "t3"}),
            ]
        )
        assert order == [["t1"], ["t2", "t3"], ["t4"]]

    def test_dependency_cycle_emitted_once_without_infinite_loop(self):
        wi = _make_intelligence()
        order = wi._calculate_execution_order(
            [self._task("t1", {"t2"}), self._task("t2", {"t1"})]
        )
        assert order == [["t1", "t2"]]

    def test_empty_tasks(self):
        wi = _make_intelligence()
        assert wi._calculate_execution_order([]) == []


@pytest.mark.unit
class TestGetReadyTasks:
    """get_ready_tasks returns WAITING tasks whose dependencies are complete."""

    def test_returns_waiting_tasks_with_met_dependencies(self):
        t1 = WorkflowTask(task_id="t1", agent_name="a", query="q")
        t2 = WorkflowTask(task_id="t2", agent_name="a", query="q", dependencies={"t1"})
        plan = WorkflowPlan(workflow_id="wf", original_query="q", tasks=[t1, t2])

        assert [t.task_id for t in plan.get_ready_tasks()] == ["t1"]

        t1.status = TaskStatus.COMPLETED
        assert [t.task_id for t in plan.get_ready_tasks()] == ["t2"]

    def test_completed_task_is_not_ready(self):
        t1 = WorkflowTask(task_id="t1", agent_name="a", query="q")
        t1.status = TaskStatus.COMPLETED
        plan = WorkflowPlan(workflow_id="wf", original_query="q", tasks=[t1])
        assert plan.get_ready_tasks() == []


@pytest.mark.unit
class TestPerformanceOptimization:
    """PERFORMANCE_BASED strategy writes a composite score into task metadata."""

    @pytest.mark.asyncio
    async def test_performance_score_written_to_task_metadata(self):
        intelligence = _make_intelligence(
            optimization_strategy=OptimizationStrategy.PERFORMANCE_BASED
        )
        intelligence.agent_performance["agent1"] = AgentPerformance(
            agent_name="agent1",
            total_executions=10,
            successful_executions=8,
            average_execution_time=1.0,
            average_confidence=0.6,
        )
        plan = WorkflowPlan(
            workflow_id="wf",
            original_query="unmatched query",
            tasks=[
                WorkflowTask(task_id="t1", agent_name="agent1", query="unmatched query")
            ],
        )

        result = await intelligence.optimize_workflow_plan("unmatched query", plan)

        # success_rate 0.8*0.4 + time_factor 0.5*0.3 + confidence 0.6*0.3
        assert result.tasks[0].metadata["performance_score"] == pytest.approx(0.65)
        assert result.metadata["performance_optimized"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@pytest.mark.unit
class TestLearnedQueryPatterns:
    """Successful executions feed query_type_patterns and template matching
    consults them — the learned corpus used to round-trip to storage without
    a single reader."""

    def _execution(self, query: str, success: bool = True):
        from cogniverse_sdk.interfaces.workflow_store import WorkflowExecution

        return WorkflowExecution(
            workflow_id="wf-1",
            query=query,
            query_type="",
            execution_time=1.0,
            success=success,
            agent_sequence=["video_search"],
            task_count=1,
            parallel_efficiency=1.0,
            confidence_score=0.9,
        )

    @pytest.mark.asyncio
    async def test_successful_execution_learns_classified_query(self):
        intel = _make_intelligence()

        await intel.record_execution(self._execution("show me sunset footage"))
        await intel.record_execution(
            self._execution("watch broken video", success=False)
        )
        await intel.record_execution(self._execution("SHOW ME SUNSET FOOTAGE"))

        assert intel.query_type_patterns["video_search"] == [
            "show me sunset footage"
        ]

    @pytest.mark.asyncio
    async def test_learned_patterns_capped_oldest_evicted(self):
        intel = _make_intelligence()
        cap = intel._MAX_LEARNED_PATTERNS_PER_TYPE

        for i in range(cap + 5):
            await intel.record_execution(self._execution(f"show clip number {i}"))

        learned = intel.query_type_patterns["video_search"]
        assert len(learned) == cap
        assert learned[0] == "show clip number 5"
        assert learned[-1] == f"show clip number {cap + 4}"

    @pytest.mark.asyncio
    async def test_learned_pattern_drives_template_match(self):
        from cogniverse_sdk.interfaces.workflow_store import WorkflowTemplate

        intel = _make_intelligence()
        template = WorkflowTemplate(
            template_id="tpl-video",
            name="video search",
            description="",
            query_patterns=["find video clips"],
            task_sequence=[],
            expected_execution_time=1.0,
            success_rate=0.9,
        )
        intel.workflow_templates[template.template_id] = template

        # No built-in pattern shares vocabulary with this phrasing.
        assert intel._find_matching_template("show me sunset footage") is None

        await intel.record_execution(self._execution("show me sunset footage"))

        assert (
            intel._find_matching_template("show me sunset footage") is template
        )

    @pytest.mark.asyncio
    async def test_other_type_patterns_do_not_leak_into_video_template(self):
        from cogniverse_sdk.interfaces.workflow_store import WorkflowTemplate

        intel = _make_intelligence()
        template = WorkflowTemplate(
            template_id="tpl-video",
            name="video search",
            description="",
            query_patterns=["find video clips"],
            task_sequence=[],
            expected_execution_time=1.0,
            success_rate=0.9,
        )
        intel.workflow_templates[template.template_id] = template

        await intel.record_execution(
            self._execution("compare quarterly revenue versus costs")
        )

        assert intel.query_type_patterns["comparison"] == [
            "compare quarterly revenue versus costs"
        ]
        assert (
            intel._find_matching_template("compare quarterly revenue versus costs")
            is None
        )
