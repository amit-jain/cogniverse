"""
Integration tests for local-storage-to-telemetry migration round-trips.

Verifies that every component migrated from local filesystem storage
to telemetry-backed storage via ArtifactManager correctly saves and
reloads data through a REAL Phoenix Docker instance.

Requires Docker to be running. Uses the ``phoenix_container`` and
``telemetry_manager_with_phoenix`` fixtures from tests/conftest.py.

Components tested:
1. ArtifactManager.save_blob / load_blob
2. RoutingOptimizer — checkpoint save/load
3. AdaptiveThresholdLearner — threshold state persist/reload
4. XGBoost meta-models — TrainingDecisionModel, TrainingStrategyModel, FusionBenefitModel
5. WorkflowIntelligence — execution persist/reload
6. MLflowIntegration — DSPy module save/load via telemetry blobs
7. PromptManager — no local fallback scanning
"""

import json
import uuid
from datetime import datetime

import pytest

from cogniverse_agents.optimizer.artifact_manager import ArtifactManager


@pytest.fixture
def real_provider(telemetry_manager_with_phoenix):
    """Get a real PhoenixProvider from the telemetry manager."""
    return telemetry_manager_with_phoenix.get_provider(tenant_id="migration-test")


# ---------------------------------------------------------------------------
# 1. ArtifactManager blob round-trip
# ---------------------------------------------------------------------------


class TestBlobRoundTrip:
    """Verify save_blob / load_blob produce identical data against real Phoenix."""

    @pytest.mark.asyncio
    async def test_blob_round_trip_string(self, real_provider):
        """Save a string blob, load it back, verify exact equality."""
        mgr = ArtifactManager(real_provider, tenant_id="blob-test")

        original = json.dumps({"model_weights": [0.1, 0.2, 0.3], "version": 2})
        dataset_id = await mgr.save_blob("checkpoint", "routing_optimizer", original)
        assert dataset_id

        loaded = await mgr.load_blob("checkpoint", "routing_optimizer")
        assert loaded == original
        assert json.loads(loaded) == json.loads(original)

    @pytest.mark.asyncio
    async def test_blob_round_trip_large_content(self, real_provider):
        """Save a large blob (simulating XGBoost JSON), verify round-trip."""
        mgr = ArtifactManager(real_provider, tenant_id="blob-large-test")

        # Simulate XGBoost model JSON (large payload)
        large_content = json.dumps(
            {
                "learner": {"feature_names": [f"f{i}" for i in range(100)]},
                "trees": [{"nodes": list(range(50))} for _ in range(10)],
            }
        )

        await mgr.save_blob("xgboost", "fusion_model", large_content)
        loaded = await mgr.load_blob("xgboost", "fusion_model")
        assert json.loads(loaded) == json.loads(large_content)

    @pytest.mark.asyncio
    async def test_blob_load_nonexistent_returns_none(self, real_provider):
        """Loading a blob that was never saved returns None."""
        mgr = ArtifactManager(real_provider, tenant_id="blob-missing-test")
        result = await mgr.load_blob("nonexistent_kind", "nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_blob_overwrite(self, real_provider):
        """Saving a blob twice overwrites — latest version is loaded."""
        mgr = ArtifactManager(real_provider, tenant_id="blob-overwrite-test")

        await mgr.save_blob("threshold", "states", "version_1")
        await mgr.save_blob("threshold", "states", "version_2")

        loaded = await mgr.load_blob("threshold", "states")
        assert loaded == "version_2"

    @pytest.mark.asyncio
    async def test_blob_tenant_isolation(self, telemetry_manager_with_phoenix):
        """Blobs from tenant A are invisible to tenant B."""
        provider_a = telemetry_manager_with_phoenix.get_provider(tenant_id="blob-iso-a")
        provider_b = telemetry_manager_with_phoenix.get_provider(tenant_id="blob-iso-b")

        mgr_a = ArtifactManager(provider_a, tenant_id="blob-iso-a")
        mgr_b = ArtifactManager(provider_b, tenant_id="blob-iso-b")

        await mgr_a.save_blob("model", "key1", "tenant_a_data")

        # Tenant B should get None
        result_b = await mgr_b.load_blob("model", "key1")
        assert result_b is None

        # Tenant A should get its own data
        result_a = await mgr_a.load_blob("model", "key1")
        assert result_a == "tenant_a_data"


# ---------------------------------------------------------------------------
# 2. RoutingOptimizer checkpoint round-trip
# ---------------------------------------------------------------------------


class TestRoutingOptimizerCheckpoint:
    """Verify AutoTuningOptimizer checkpoint save/load through real Phoenix.

    save_checkpoint/load_checkpoint live on AutoTuningOptimizer (subclass),
    which requires a strategy instance.
    """

    @pytest.mark.asyncio
    async def test_checkpoint_round_trip(self, real_provider):
        """Save a checkpoint via blob, load it in a new optimizer, verify state."""
        from cogniverse_agents.routing.base import RoutingStrategy
        from cogniverse_agents.routing.optimizer import AutoTuningOptimizer

        # Minimal strategy stub for test
        class StubStrategy(RoutingStrategy):
            config = {"strategy_type": "stub"}

            async def route(self, query, context=None):
                return None

            def get_confidence(self, query, context=None):
                return 0.5

        strategy = StubStrategy()

        optimizer = AutoTuningOptimizer(
            strategy=strategy,
            telemetry_provider=real_provider,
            tenant_id="checkpoint-test",
        )

        # Set state that gets checkpointed
        optimizer.optimization_attempts = 7
        optimizer.best_performance = 0.91
        optimizer.best_params = {"learning_rate": 0.01, "batch_size": 32}

        await optimizer.save_checkpoint()

        # Create a new optimizer and load the checkpoint
        optimizer2 = AutoTuningOptimizer(
            strategy=StubStrategy(),
            telemetry_provider=real_provider,
            tenant_id="checkpoint-test",
        )

        await optimizer2.load_checkpoint()

        assert optimizer2.optimization_attempts == 7
        assert optimizer2.best_performance == 0.91
        assert optimizer2.best_params == {"learning_rate": 0.01, "batch_size": 32}


# ---------------------------------------------------------------------------
# 3. AdaptiveThresholdLearner state round-trip
# ---------------------------------------------------------------------------


class TestAdaptiveThresholdRoundTrip:
    """Verify AdaptiveThresholdLearner persist/reload through real Phoenix."""

    @pytest.mark.asyncio
    async def test_threshold_state_round_trip(self, real_provider):
        """Persist threshold states, reload in new learner, verify deques."""
        from cogniverse_agents.routing.adaptive_threshold_learner import (
            AdaptiveThresholdLearner,
            ThresholdParameter,
        )

        learner = AdaptiveThresholdLearner(
            telemetry_provider=real_provider,
            tenant_id="threshold-test",
        )

        # Simulate some threshold updates
        if ThresholdParameter.ROUTING_CONFIDENCE in learner.threshold_states:
            state = learner.threshold_states[ThresholdParameter.ROUTING_CONFIDENCE]
            state.current_value = 0.82
            state.best_value = 0.85

        await learner._persist_state()

        # Create new learner and reload
        learner2 = AdaptiveThresholdLearner(
            telemetry_provider=real_provider,
            tenant_id="threshold-test",
        )

        await learner2.load_stored_state()

        if ThresholdParameter.ROUTING_CONFIDENCE in learner2.threshold_states:
            state2 = learner2.threshold_states[ThresholdParameter.ROUTING_CONFIDENCE]
            assert state2.current_value == 0.82
            assert state2.best_value == 0.85


# ---------------------------------------------------------------------------
# 4. XGBoost meta-model round-trips
# ---------------------------------------------------------------------------


class TestXGBoostMetaModelRoundTrip:
    """Verify XGBoost model save/load via telemetry for all 3 model classes."""

    @pytest.mark.asyncio
    async def test_training_decision_model_round_trip(self, real_provider):
        """Train a TrainingDecisionModel, save, load in new instance, verify."""
        xgb = pytest.importorskip("xgboost")
        import numpy as np

        from cogniverse_agents.routing.xgboost_meta_models import (
            TrainingDecisionModel,
        )

        model_wrapper = TrainingDecisionModel(
            telemetry_provider=real_provider,
            tenant_id="xgb-decision-test",
        )

        # Directly create and train the XGBoost model
        X = np.array([[0.5, 100, 0.8], [0.3, 50, 0.6], [0.9, 200, 0.95]])
        y = np.array([1, 0, 1])
        model_wrapper.model = xgb.XGBClassifier(
            n_estimators=5, max_depth=2, use_label_encoder=False, eval_metric="logloss"
        )
        model_wrapper.model.fit(X, y)
        model_wrapper.is_trained = True

        await model_wrapper.save_to_telemetry()

        # Load in new instance
        model_wrapper2 = TrainingDecisionModel(
            telemetry_provider=real_provider,
            tenant_id="xgb-decision-test",
        )

        loaded = await model_wrapper2.load_from_telemetry()
        assert loaded is True

        predictions_original = model_wrapper.model.predict(X)
        predictions_loaded = model_wrapper2.model.predict(X)
        np.testing.assert_array_equal(predictions_original, predictions_loaded)

    @pytest.mark.asyncio
    async def test_training_strategy_model_round_trip(self, real_provider):
        """Train a TrainingStrategyModel, save, load, verify predictions."""
        xgb = pytest.importorskip("xgboost")
        import numpy as np

        from cogniverse_agents.routing.xgboost_meta_models import (
            TrainingStrategyModel,
        )

        model_wrapper = TrainingStrategyModel(
            telemetry_provider=real_provider,
            tenant_id="xgb-strategy-test",
        )

        X = np.array([[0.5, 100, 0.8], [0.3, 50, 0.6], [0.9, 200, 0.95]])
        y = np.array([0, 1, 2])
        model_wrapper.model = xgb.XGBClassifier(
            n_estimators=5,
            max_depth=2,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
        )
        model_wrapper.model.fit(X, y)
        model_wrapper.is_trained = True

        await model_wrapper.save_to_telemetry()

        model_wrapper2 = TrainingStrategyModel(
            telemetry_provider=real_provider,
            tenant_id="xgb-strategy-test",
        )

        loaded = await model_wrapper2.load_from_telemetry()
        assert loaded is True

        predictions_original = model_wrapper.model.predict(X)
        predictions_loaded = model_wrapper2.model.predict(X)
        np.testing.assert_array_equal(predictions_original, predictions_loaded)

    @pytest.mark.asyncio
    async def test_fusion_benefit_model_round_trip(self, real_provider):
        """Train a FusionBenefitModel, save, load, verify predictions."""
        xgb = pytest.importorskip("xgboost")
        import numpy as np

        from cogniverse_agents.routing.xgboost_meta_models import FusionBenefitModel

        model_wrapper = FusionBenefitModel(
            telemetry_provider=real_provider,
            tenant_id="xgb-fusion-test",
        )

        X = np.array([[0.5, 0.7, 0.2], [0.3, 0.4, 0.8], [0.9, 0.1, 0.5]])
        y = np.array([0.82, 0.45, 0.73])
        model_wrapper.model = xgb.XGBRegressor(n_estimators=5, max_depth=2)
        model_wrapper.model.fit(X, y)
        model_wrapper.is_trained = True

        await model_wrapper.save_to_telemetry()

        model_wrapper2 = FusionBenefitModel(
            telemetry_provider=real_provider,
            tenant_id="xgb-fusion-test",
        )

        loaded = await model_wrapper2.load_from_telemetry()
        assert loaded is True

        predictions_original = model_wrapper.model.predict(X)
        predictions_loaded = model_wrapper2.model.predict(X)
        np.testing.assert_allclose(predictions_original, predictions_loaded, rtol=1e-5)


# ---------------------------------------------------------------------------
# 5. WorkflowIntelligence round-trip
# ---------------------------------------------------------------------------


class TestWorkflowIntelligenceRoundTrip:
    """Verify WorkflowIntelligence persist/reload through real Phoenix."""

    @pytest.mark.asyncio
    async def test_execution_persist_and_reload(self, real_provider):
        """Persist a workflow execution, reload in new instance, verify."""
        from cogniverse_agents.workflow.intelligence import (
            WorkflowExecution,
            WorkflowIntelligence,
        )

        wi = WorkflowIntelligence(
            telemetry_provider=real_provider,
            tenant_id="wi-exec-test",
        )

        # Create and record a WorkflowExecution
        execution = WorkflowExecution(
            workflow_id=f"wf-{uuid.uuid4().hex[:8]}",
            query="find videos about quantum computing",
            query_type="video_search",
            execution_time=2.5,
            success=True,
            agent_sequence=["routing_agent", "video_search_agent"],
            task_count=2,
            parallel_efficiency=0.85,
            confidence_score=0.92,
            timestamp=datetime.now(),
        )

        await wi.record_execution(execution)

        # Create new instance and reload
        wi2 = WorkflowIntelligence(
            telemetry_provider=real_provider,
            tenant_id="wi-exec-test",
        )

        await wi2.load_historical_data()

        assert len(wi2.workflow_history) >= 1
        latest = wi2.workflow_history[-1]
        assert latest.query_type == "video_search"
        assert latest.success is True
        assert latest.confidence_score == 0.92

    @pytest.mark.asyncio
    async def test_template_persist_and_reload(self, real_provider):
        """Persist a workflow template via blob, reload, verify."""
        from cogniverse_agents.workflow.intelligence import (
            WorkflowIntelligence,
            WorkflowTemplate,
        )

        wi = WorkflowIntelligence(
            telemetry_provider=real_provider,
            tenant_id="wi-template-test",
        )

        # Manually create and persist a template
        template = WorkflowTemplate(
            template_id="template_1",
            name="video_analysis",
            description="Standard video analysis workflow",
            query_patterns=["video_search"],
            task_sequence=[
                {"agent": "routing_agent"},
                {"agent": "video_search_agent"},
                {"agent": "summarizer"},
            ],
            expected_execution_time=3.0,
            success_rate=0.95,
        )
        wi.workflow_templates["template_1"] = template
        await wi._persist_template(template)

        # Create new instance and reload
        wi2 = WorkflowIntelligence(
            telemetry_provider=real_provider,
            tenant_id="wi-template-test",
        )

        await wi2.load_historical_data()

        assert "template_1" in wi2.workflow_templates
        loaded_template = wi2.workflow_templates["template_1"]
        assert loaded_template.name == "video_analysis"
        assert loaded_template.success_rate == 0.95


# ---------------------------------------------------------------------------
# 6. MLflowIntegration DSPy model round-trip (telemetry blob path)
# ---------------------------------------------------------------------------


class TestMLflowIntegrationRoundTrip:
    """Verify MLflowIntegration save/load DSPy models via telemetry blobs.

    This tests the telemetry blob storage path, NOT the MLflow server.
    """

    @pytest.mark.asyncio
    async def test_dspy_model_blob_round_trip(self, real_provider):
        """Save DSPy module JSON via artifact manager blob, load it back."""
        import tempfile

        import dspy

        # Create a simple DSPy module
        class SimpleRouter(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predict = dspy.ChainOfThought("query -> agent")

        model = SimpleRouter()

        # Save via DSPy's .save() -> read JSON -> save_blob
        mgr = ArtifactManager(real_provider, tenant_id="mlflow-blob-test")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/module.json"
            model.save(model_path)
            with open(model_path) as f:
                model_json = f.read()

        dataset_id = await mgr.save_blob("model", "test_router", model_json)
        assert dataset_id

        # Load it back
        loaded_json = await mgr.load_blob("model", "test_router")
        assert loaded_json is not None
        assert json.loads(loaded_json) == json.loads(model_json)

        # Verify we can reconstruct the module from the loaded JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = f"{tmpdir}/module.json"
            with open(model_path, "w") as f:
                f.write(loaded_json)
            loaded_model = SimpleRouter()
            loaded_model.load(model_path)

        assert hasattr(loaded_model, "predict")


# ---------------------------------------------------------------------------
# 7. PromptManager — no local fallback
# ---------------------------------------------------------------------------


class TestPromptManagerNoLocalFallback:
    """Verify PromptManager does not scan local filesystem for artifacts."""

    def test_no_standard_paths_scanning(self):
        """PromptManager initializes with artifacts=None (no filesystem scanning)."""
        from unittest.mock import MagicMock

        from cogniverse_core.common.utils.prompt_manager import PromptManager

        mock_config_manager = MagicMock()
        mock_config_manager.get_all.return_value = {}

        # Mock get_config to return a config object
        import cogniverse_core.common.utils.prompt_manager as pm_module

        original_get_config = pm_module.get_config

        def mock_get_config(tenant_id, config_manager):
            mock = MagicMock()
            mock.get_all.return_value = {"prompts": {}}
            return mock

        pm_module.get_config = mock_get_config

        try:
            pm = PromptManager(
                config_manager=mock_config_manager,
                tenant_id="test",
            )

            # No artifacts_path provided, no config path -> should be None
            assert pm.artifacts is None
            assert pm.get_status()["using_defaults"] is True
        finally:
            pm_module.get_config = original_get_config


# ---------------------------------------------------------------------------
# 8. QueryEnhancementPipeline SIMBA round-trip
# ---------------------------------------------------------------------------


class TestQueryEnhancementPipelineSIMBA:
    """Verify SIMBA patterns persist and reload through real Phoenix."""

    @pytest.mark.asyncio
    async def test_simba_pattern_persist_and_reload(self, real_provider):
        """Record an enhancement pattern via SIMBA, reload in new pipeline."""
        from cogniverse_agents.routing.query_enhancement_engine import (
            QueryEnhancementPipeline,
        )
        from cogniverse_agents.routing.simba_query_enhancer import (
            QueryEnhancementPattern,
        )

        tenant = f"simba-rt-{uuid.uuid4().hex[:8]}"
        pipeline = QueryEnhancementPipeline(
            enable_simba=True,
            telemetry_provider=real_provider,
            tenant_id=tenant,
        )

        # Manually inject a pattern into SIMBA's memory
        pattern = QueryEnhancementPattern(
            original_query="find videos about quantum computing",
            enhanced_query="quantum computing tutorial video demonstrations",
            entities=[{"text": "quantum computing", "label": "TOPIC"}],
            relationships=[
                {
                    "subject": "quantum computing",
                    "relation": "domain",
                    "object": "physics",
                }
            ],
            enhancement_strategy="entity_expansion",
            search_quality_improvement=0.35,
            routing_confidence_improvement=0.15,
            success_rate=0.9,
            usage_count=3,
            avg_improvement=0.25,
            pattern_confidence=0.85,
        )
        pipeline.simba_enhancer.enhancement_patterns.append(pattern)

        # Persist via SIMBA's internal method
        await pipeline.simba_enhancer._persist_data()

        # Create a fresh pipeline with the same tenant and reload
        pipeline2 = QueryEnhancementPipeline(
            enable_simba=True,
            telemetry_provider=real_provider,
            tenant_id=tenant,
        )
        await pipeline2.simba_enhancer.load_stored_data()

        assert len(pipeline2.simba_enhancer.enhancement_patterns) >= 1
        loaded = pipeline2.simba_enhancer.enhancement_patterns[0]
        assert loaded.original_query == "find videos about quantum computing"
        assert (
            loaded.enhanced_query == "quantum computing tutorial video demonstrations"
        )
        assert loaded.enhancement_strategy == "entity_expansion"
        assert loaded.search_quality_improvement == 0.35
        assert loaded.success_rate == 0.9

    @pytest.mark.asyncio
    async def test_simba_requires_telemetry_provider(self):
        """Constructing with enable_simba=True and no provider raises ValueError."""
        from cogniverse_agents.routing.query_enhancement_engine import (
            QueryEnhancementPipeline,
        )

        with pytest.raises(ValueError, match="telemetry_provider is required"):
            QueryEnhancementPipeline(enable_simba=True, telemetry_provider=None)


# ---------------------------------------------------------------------------
# 9. DSPyAgentOptimizerPipeline prompt round-trip
# ---------------------------------------------------------------------------


class TestDSPyOptimizerPipelineRoundTrip:
    """Verify DSPyAgentOptimizerPipeline save/load prompts through real Phoenix."""

    @pytest.mark.asyncio
    async def test_prompt_save_and_load_round_trip(self, real_provider):
        """Save compiled module prompts, load them back, verify equality."""
        from unittest.mock import Mock

        from cogniverse_agents.optimizer.dspy_agent_optimizer import (
            DSPyAgentOptimizerPipeline,
            DSPyAgentPromptOptimizer,
        )

        tenant = f"dspy-opt-rt-{uuid.uuid4().hex[:8]}"
        optimizer = DSPyAgentPromptOptimizer()
        pipeline = DSPyAgentOptimizerPipeline(optimizer)

        # Create mock compiled modules with extractable artifacts
        for module_name in ["query_analysis", "agent_routing"]:
            mock_module = Mock(spec=[])
            mock_module.demos = []

            # Add the component that _extract_artifacts_from_module looks for
            component = Mock()
            component.signature = f"Optimized {module_name} signature v1"
            if module_name == "query_analysis":
                mock_module.generate_analysis = component
            else:
                mock_module.generate_routing = component

            pipeline.compiled_modules[module_name] = mock_module

        # Save via telemetry
        await pipeline.save_optimized_prompts(
            tenant_id=tenant,
            telemetry_provider=real_provider,
        )

        # Load back via ArtifactManager
        mgr = ArtifactManager(real_provider, tenant_id=tenant)

        qa_prompts = await mgr.load_prompts("query_analysis")
        assert qa_prompts is not None
        assert "signature" in qa_prompts
        assert "query_analysis" in qa_prompts["signature"]

        ar_prompts = await mgr.load_prompts("agent_routing")
        assert ar_prompts is not None
        assert "signature" in ar_prompts
        assert "agent_routing" in ar_prompts["signature"]

    @pytest.mark.asyncio
    async def test_prompt_tenant_isolation(self, telemetry_manager_with_phoenix):
        """Prompts saved for tenant A are invisible to tenant B."""
        from unittest.mock import Mock

        from cogniverse_agents.optimizer.dspy_agent_optimizer import (
            DSPyAgentOptimizerPipeline,
            DSPyAgentPromptOptimizer,
        )

        provider_a = telemetry_manager_with_phoenix.get_provider(tenant_id="dspy-iso-a")
        provider_b = telemetry_manager_with_phoenix.get_provider(tenant_id="dspy-iso-b")

        optimizer = DSPyAgentPromptOptimizer()
        pipeline = DSPyAgentOptimizerPipeline(optimizer)

        # Set up a single module
        mock_module = Mock(spec=[])
        mock_module.demos = []
        component = Mock()
        component.signature = "Tenant A signature"
        mock_module.generate_analysis = component
        pipeline.compiled_modules["query_analysis"] = mock_module

        # Save for tenant A
        await pipeline.save_optimized_prompts(
            tenant_id="dspy-iso-a",
            telemetry_provider=provider_a,
        )

        # Tenant B should not see tenant A's prompts
        mgr_b = ArtifactManager(provider_b, tenant_id="dspy-iso-b")
        result_b = await mgr_b.load_prompts("query_analysis")
        assert result_b is None

        # Tenant A should see its own prompts
        mgr_a = ArtifactManager(provider_a, tenant_id="dspy-iso-a")
        result_a = await mgr_a.load_prompts("query_analysis")
        assert result_a is not None
        assert "Tenant A" in result_a["signature"]
