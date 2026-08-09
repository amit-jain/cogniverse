"""
Unit tests for WorkflowIntelligence — read-only template loader
"""

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
from cogniverse_sdk.interfaces.workflow_store import (
    AgentPerformance,
    WorkflowExecution,
    WorkflowTemplate,
)


def _make_intelligence(**kwargs) -> WorkflowIntelligence:
    """Create a WorkflowIntelligence instance for the given tenant."""
    defaults = dict(
        tenant_id="test_tenant",
    )
    defaults.update(kwargs)
    return WorkflowIntelligence(**defaults)


def _observed_outcome_metadata():
    return {
        "_outcome_metadata": {
            "observed": True,
            "required_field_semantics": {
                "execution_time": "observed_duration_seconds",
                "success": "observed_execution_outcome",
                "parallel_efficiency": "observed_parallel_efficiency",
                "confidence_score": "observed_confidence_score",
            },
        }
    }


def _unobserved_outcome_metadata():
    return {
        "_outcome_metadata": {
            "observed": False,
            "required_field_semantics": {
                "execution_time": "unobserved_zero_sentinel",
                "success": "unobserved_false_sentinel",
                "parallel_efficiency": "unobserved_zero_sentinel",
                "confidence_score": "unobserved_zero_sentinel",
            },
        }
    }


def _observed_span():
    return {
        "attributes.input.value": "find the Curie laboratory footage",
        "attributes.output.value": {
            "workflow_id": "workflow-observed",
            "pattern": "sequential",
            "agent_sequence": ["search_agent"],
            "execution_time": 1.25,
            "success": True,
            "tasks_completed": 1,
            "confidence": 0.875,
        },
        "context.span_id": "span-observed",
    }


@pytest.mark.unit
class TestOrchestrationOutcomeExtraction:
    @staticmethod
    def _evaluator():
        from cogniverse_agents.routing.orchestration_evaluator import (
            OrchestrationEvaluator,
        )

        evaluator = object.__new__(OrchestrationEvaluator)
        evaluator.tenant_id = "acme:acme"
        return evaluator

    @pytest.mark.parametrize(
        ("location", "field"),
        [
            ("output", "workflow_id"),
            ("output", "pattern"),
            ("output", "agent_sequence"),
            ("output", "execution_time"),
            ("output", "success"),
            ("output", "tasks_completed"),
        ],
    )
    def test_missing_observed_field_rejects_span(self, location, field):
        span = _observed_span()
        target = span["attributes.output.value"] if location == "output" else span
        del target[field]

        with pytest.raises(
            ValueError,
            match=rf"^orchestration span requires observed field {field}$",
        ):
            self._evaluator()._extract_workflow_execution(span)

    @pytest.mark.parametrize("success", [1, "true", None])
    def test_success_must_be_an_explicit_boolean_outcome(self, success):
        span = _observed_span()
        span["attributes.output.value"]["success"] = success

        with pytest.raises(
            ValueError,
            match=r"^orchestration span success must be a bool$",
        ):
            self._evaluator()._extract_workflow_execution(span)

    def test_parallel_efficiency_is_normalized_to_unit_interval(self):
        span = _observed_span()
        span["attributes.output.value"].update(
            {
                "pattern": "parallel",
                "agent_sequence": ["search_agent", "summarizer_agent"],
                "tasks_completed": 2,
                "agent_times": "search_agent:1.0,summarizer_agent:1.0",
                "execution_time": 1.0,
            }
        )

        execution = self._evaluator()._extract_workflow_execution(span)

        assert execution is not None
        assert execution.parallel_efficiency == 1.0
        assert execution.metadata == {
            "orchestration_pattern": "parallel",
            "execution_order": [],
            "tasks_completed": 2,
            "span_id": "span-observed",
            "tenant_id": "acme:acme",
            "_outcome_metadata": {
                "observed": True,
                "required_field_semantics": {
                    "execution_time": "observed_duration_seconds",
                    "success": "observed_execution_outcome",
                    "parallel_efficiency": "observed_parallel_efficiency",
                    "confidence_score": "observed_confidence_score",
                },
            },
        }

    def test_parallel_span_without_agent_times_uses_declared_zero_sentinel(self):
        span = _observed_span()
        span["attributes.output.value"]["pattern"] = "parallel"

        execution = self._evaluator()._extract_workflow_execution(span)

        assert execution is not None
        assert execution.parallel_efficiency == 0.0
        assert execution.metadata["_outcome_metadata"] == {
            "observed": True,
            "required_field_semantics": {
                "execution_time": "observed_duration_seconds",
                "success": "observed_execution_outcome",
                "parallel_efficiency": "unobserved_zero_sentinel",
                "confidence_score": "observed_confidence_score",
            },
        }

    def test_missing_confidence_uses_declared_zero_sentinel(self):
        span = _observed_span()
        del span["attributes.output.value"]["confidence"]

        execution = self._evaluator()._extract_workflow_execution(span)

        assert execution is not None
        assert execution.confidence_score == 0.0
        assert execution.metadata["_outcome_metadata"] == {
            "observed": True,
            "required_field_semantics": {
                "execution_time": "observed_duration_seconds",
                "success": "observed_execution_outcome",
                "parallel_efficiency": "observed_parallel_efficiency",
                "confidence_score": "unobserved_zero_sentinel",
            },
        }

    def test_span_emitter_writes_actual_execution_fields(self):
        import json
        from contextlib import contextmanager

        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from cogniverse_agents.orchestrator_agent import OrchestratorAgent

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("workflow-outcome-test")

        class TelemetryManager:
            @contextmanager
            def span(self, name, tenant_id, require_export):
                assert tenant_id == "acme:acme"
                assert require_export is True
                with tracer.start_as_current_span(name) as span:
                    yield span

        agent = object.__new__(OrchestratorAgent)
        agent.telemetry_manager = TelemetryManager()
        agent._emit_orchestration_span(
            tenant_id="acme:acme",
            workflow_id="wf-emitted",
            query="find Curie footage",
            agent_sequence=["search_agent", "summarizer_agent"],
            execution_time=1.25,
            success=False,
            tasks_completed=1,
            pattern="parallel",
            execution_order=["search_agent", "summarizer_agent"],
            error_summary="ReadTimeout: summarizer exceeded 30s",
        )

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "cogniverse.orchestration"
        assert spans[0].attributes["input.value"] == "find Curie footage"
        assert json.loads(spans[0].attributes["output.value"]) == {
            "workflow_id": "wf-emitted",
            "agent_sequence": ["search_agent", "summarizer_agent"],
            "execution_order": ["search_agent", "summarizer_agent"],
            "pattern": "parallel",
            "execution_time": 1.25,
            "success": False,
            "tasks_completed": 1,
            "error_summary": "ReadTimeout: summarizer exceeded 30s",
        }
        assert spans[0].attributes["operation"] == "orchestration"
        assert spans[0].status.status_code.name == "ERROR"
        assert spans[0].status.description == "ReadTimeout: summarizer exceeded 30s"
        provider.shutdown()


@pytest.mark.unit
class TestUnobservedOutcomeSentinels:
    @staticmethod
    def _partially_observed_metadata():
        return {
            "_outcome_metadata": {
                "observed": True,
                "required_field_semantics": {
                    "execution_time": "observed_duration_seconds",
                    "success": "observed_execution_outcome",
                    "parallel_efficiency": "unobserved_zero_sentinel",
                    "confidence_score": "unobserved_zero_sentinel",
                },
            }
        }

    @pytest.mark.asyncio
    async def test_live_execution_accepts_declared_unobserved_metric_sentinels(self):
        execution = WorkflowExecution(
            workflow_id="live-with-unavailable-metrics",
            query="find source footage",
            query_type="VIDEO",
            execution_time=1.25,
            success=True,
            agent_sequence=["search_agent"],
            task_count=1,
            parallel_efficiency=0.0,
            confidence_score=0.0,
            metadata=self._partially_observed_metadata(),
        )
        intelligence = _make_intelligence()

        await intelligence.record_execution(execution)

        assert list(intelligence.workflow_history) == [execution]
        assert dict(intelligence.query_type_patterns) == {
            "video_search": ["find source footage"]
        }

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("field", "value"),
        [("parallel_efficiency", 0.75), ("confidence_score", 0.9)],
    )
    async def test_live_unobserved_metric_requires_its_zero_sentinel(
        self, field, value
    ):
        fields = {"parallel_efficiency": 0.0, "confidence_score": 0.0}
        fields[field] = value
        execution = WorkflowExecution(
            workflow_id="live-with-invalid-sentinel",
            query="find source footage",
            query_type="VIDEO",
            execution_time=1.25,
            success=True,
            agent_sequence=["search_agent"],
            task_count=1,
            parallel_efficiency=fields["parallel_efficiency"],
            confidence_score=fields["confidence_score"],
            metadata=self._partially_observed_metadata(),
        )
        intelligence = _make_intelligence()

        with pytest.raises(
            ValueError,
            match=(
                rf"^unobserved workflow execution field {field} must equal "
                r"its declared sentinel$"
            ),
        ):
            await intelligence.record_execution(execution)

        assert list(intelligence.workflow_history) == []
        assert dict(intelligence.query_type_patterns) == {}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("execution_time", 9.5),
            ("success", True),
            ("parallel_efficiency", 0.75),
            ("confidence_score", 0.9),
        ],
    )
    async def test_non_sentinel_generated_metric_rejects_before_mutation(
        self, field, value
    ):
        fields = {
            "execution_time": 0.0,
            "success": False,
            "parallel_efficiency": 0.0,
            "confidence_score": 0.0,
        }
        fields[field] = value
        execution = WorkflowExecution(
            workflow_id="generated-with-observed-value",
            query="find source footage",
            query_type="VIDEO",
            execution_time=fields["execution_time"],
            success=fields["success"],
            agent_sequence=["search_agent"],
            task_count=1,
            parallel_efficiency=fields["parallel_efficiency"],
            confidence_score=fields["confidence_score"],
            metadata=_unobserved_outcome_metadata(),
        )
        intelligence = _make_intelligence()

        with pytest.raises(
            ValueError,
            match=(
                rf"^unobserved workflow execution field {field} must equal "
                r"its declared sentinel$"
            ),
        ):
            await intelligence.record_execution(execution)

        assert list(intelligence.workflow_history) == []
        assert dict(intelligence.query_type_patterns) == {}


def test_agent_profiles_use_only_per_agent_observed_metrics():
    intelligence = _make_intelligence()
    first = WorkflowExecution(
        workflow_id="wf-agent-observations-1",
        query="find the exact Curie laboratory footage",
        query_type="VIDEO",
        execution_time=10.0,
        success=False,
        agent_sequence=["entity_agent", "search_agent", "summary_agent"],
        task_count=3,
        parallel_efficiency=0.0,
        confidence_score=0.1,
        error_details="ReadTimeout: search exceeded its deadline",
        metadata={
            **_observed_outcome_metadata(),
            "orchestration_pattern": "sequential",
            "agent_observations": [
                {
                    "agent_name": "entity_agent",
                    "execution_time": 0.1,
                    "success": True,
                    "confidence": 0.9,
                },
                {
                    "agent_name": "search_agent",
                    "execution_time": 0.4,
                    "success": False,
                },
                {
                    "agent_name": "summary_agent",
                    "execution_time": 0.2,
                    "success": True,
                },
            ],
        },
    )
    second = WorkflowExecution(
        workflow_id="wf-agent-observations-2",
        query="find another exact Curie source",
        query_type="VIDEO",
        execution_time=2.0,
        success=True,
        agent_sequence=["entity_agent", "search_agent", "summary_agent"],
        task_count=3,
        parallel_efficiency=0.0,
        confidence_score=0.99,
        metadata={
            **_observed_outcome_metadata(),
            "orchestration_pattern": "sequential",
            "agent_observations": [
                {
                    "agent_name": "entity_agent",
                    "execution_time": 0.2,
                    "success": True,
                    "confidence": 0.7,
                },
                {
                    "agent_name": "search_agent",
                    "execution_time": 1.6,
                    "success": True,
                    "confidence": 0.8,
                },
                {
                    "agent_name": "summary_agent",
                    "execution_time": 0.1,
                    "success": True,
                },
            ],
        },
    )

    profiles, _templates = intelligence.derive_learning_artifacts([first, second])

    assert [profile.agent_name for profile in profiles] == [
        "entity_agent",
        "search_agent",
    ]
    by_name = {profile.agent_name: profile for profile in profiles}
    assert (
        by_name["entity_agent"].total_executions,
        by_name["entity_agent"].successful_executions,
        by_name["entity_agent"].average_execution_time,
        by_name["entity_agent"].average_confidence,
        by_name["entity_agent"].error_rate,
        by_name["entity_agent"].preferred_query_types,
    ) == (2, 2, pytest.approx(0.15), pytest.approx(0.8), 0.0, ["VIDEO"])
    assert (
        by_name["search_agent"].total_executions,
        by_name["search_agent"].successful_executions,
        by_name["search_agent"].average_execution_time,
        by_name["search_agent"].average_confidence,
        by_name["search_agent"].error_rate,
        by_name["search_agent"].preferred_query_types,
    ) == (2, 1, 1.0, 0.8, 0.5, ["VIDEO"])


@pytest.mark.unit
class TestWorkflowIntelligence:
    """Test cases for WorkflowIntelligence class"""

    @pytest.mark.asyncio
    async def test_synthetic_generation_forwards_explicit_agent_configuration(
        self, monkeypatch
    ):
        from types import SimpleNamespace

        import cogniverse_synthetic

        captured = {}

        class RecordingSyntheticDataService:
            def __init__(self, **kwargs):
                captured["service_kwargs"] = kwargs

            async def generate(self, request):
                captured["request"] = request
                return SimpleNamespace(data=[])

        monkeypatch.setattr(
            cogniverse_synthetic,
            "SyntheticDataService",
            RecordingSyntheticDataService,
        )
        intelligence = _make_intelligence()
        agents_config = {"search_agent": {"description": "Searches indexed content"}}

        with pytest.raises(
            RuntimeError,
            match="Synthetic workflow response must contain exactly 7 plans",
        ):
            await intelligence.generate_synthetic_training_data(
                agents_config=agents_config,
                count=7,
                backend="backend-instance",
                backend_config="backend-config",
                generator_config="generator-config",
            )
        assert captured["service_kwargs"] == {
            "backend": "backend-instance",
            "backend_config": "backend-config",
            "generator_config": "generator-config",
            "agents_config": agents_config,
        }
        assert captured["request"].model_dump() == {
            "optimizer": "workflow",
            "count": 7,
            "vespa_sample_size": 200,
            "strategy": None,
            "max_profiles": 3,
            "tenant_id": "test_tenant:test_tenant",
        }

    def test_workflow_intelligence_initialization(self):
        """Test WorkflowIntelligence initializes with required tenant_id"""
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
            WorkflowIntelligence(tenant_id="")

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
            metadata=_observed_outcome_metadata(),
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
                metadata=_observed_outcome_metadata(),
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
            metadata=_observed_outcome_metadata(),
        )

    @pytest.mark.asyncio
    async def test_successful_execution_learns_classified_query(self):
        intel = _make_intelligence()

        await intel.record_execution(self._execution("show me sunset footage"))
        await intel.record_execution(
            self._execution("watch broken video", success=False)
        )
        await intel.record_execution(self._execution("SHOW ME SUNSET FOOTAGE"))

        assert intel.query_type_patterns["video_search"] == ["show me sunset footage"]

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

        assert intel._find_matching_template("show me sunset footage") is template

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
