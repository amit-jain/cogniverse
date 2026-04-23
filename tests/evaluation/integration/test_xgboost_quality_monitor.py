"""
Integration test for XGBoost TrainingDecisionModel wired into QualityMonitor.

Uses real Phoenix for telemetry provider (artifact persistence).
Trains the XGBoost model on sample data, then verifies the quality monitor
uses it to make smarter optimization timing decisions.
"""

import json
import logging
from datetime import datetime

import pytest

from cogniverse_evaluation.quality_monitor import (
    AgentEvalResult,
    AgentType,
    LiveEvalResult,
    QualityMonitor,
    Verdict,
)
from tests.utils.llm_config import get_llm_model

logger = logging.getLogger(__name__)


@pytest.fixture
def monitor_with_xgboost(phoenix_container, tmp_path):
    """QualityMonitor with real telemetry provider for XGBoost."""
    from cogniverse_foundation.telemetry.manager import TelemetryManager
    from cogniverse_foundation.telemetry.registry import get_telemetry_registry

    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()

    import cogniverse_foundation.telemetry.manager as tm_module
    from cogniverse_foundation.telemetry.config import (
        BatchExportConfig,
        TelemetryConfig,
    )

    phoenix_url = "http://localhost:16006"

    config = TelemetryConfig(
        otlp_endpoint="localhost:14317",
        provider_config={
            "http_endpoint": phoenix_url,
            "grpc_endpoint": "http://localhost:14317",
        },
        batch_config=BatchExportConfig(use_sync_export=True),
    )
    manager = TelemetryManager(config=config)
    tm_module._telemetry_manager = manager

    provider = manager.get_provider(tenant_id="xgboost_test")

    golden_queries = [
        {
            "query": "test",
            "expected_videos": ["v1"],
            "ground_truth": "test",
            "query_type": "test",
            "source": "test",
        },
    ]
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden_queries))

    m = QualityMonitor(
        tenant_id="xgboost_test",
        runtime_url="http://testserver",
        phoenix_http_endpoint=phoenix_url,
        llm_base_url="http://localhost:11434",
        llm_model=get_llm_model(),
        golden_dataset_path=str(golden_path),
        telemetry_provider=provider,
    )

    yield m

    import asyncio

    try:
        loop = asyncio.get_running_loop()
        loop.run_until_complete(m.close())
    except RuntimeError:
        asyncio.run(m.close())
    TelemetryManager.reset()
    get_telemetry_registry().clear_cache()


@pytest.mark.integration
class TestXGBoostQualityMonitorIntegration:
    """Test XGBoost meta-model integration with real Phoenix telemetry."""

    def test_xgboost_model_initializes_with_real_provider(self, monitor_with_xgboost):
        """TrainingDecisionModel creates with real telemetry provider."""
        model = monitor_with_xgboost._get_training_decision_model()
        assert model is not None
        assert not model.is_trained

    def test_untrained_model_uses_fallback_heuristic(self, monitor_with_xgboost):
        """Untrained XGBoost model uses fallback rules, not crash."""
        live = LiveEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="xgboost_test",
            agent_results={
                AgentType.SEARCH: AgentEvalResult(
                    agent=AgentType.SEARCH,
                    score=0.3,
                    baseline_score=0.8,
                    degradation_pct=0.6,
                    sample_count=20,
                ),
            },
        )

        verdicts = monitor_with_xgboost.check_thresholds(None, live)
        # Fallback heuristic: 20 samples < 50 required → should_train=False
        # XGBoost overrides OPTIMIZE → SKIP (not enough data for training)
        assert verdicts[AgentType.SEARCH] == Verdict.SKIP

    def test_trained_model_makes_informed_decision(self, monitor_with_xgboost):
        """Train XGBoost on sample data, verify it makes a decision."""
        from cogniverse_agents.routing.xgboost_meta_models import ModelingContext
        from cogniverse_agents.search.multi_modal_reranker import QueryModality

        model = monitor_with_xgboost._get_training_decision_model()

        # Create training data with clear signal:
        # Low success_rate (<0.7) → training helped (improvement > 0.02)
        # High success_rate (>=0.7) → training didn't help (improvement < 0.02)
        contexts = []
        outcomes = []
        import random

        random.seed(42)
        for _ in range(50):
            sr = random.uniform(0.3, 0.95)
            contexts.append(
                ModelingContext(
                    modality=QueryModality.VIDEO,
                    real_sample_count=random.randint(50, 300),
                    synthetic_sample_count=random.randint(0, 100),
                    success_rate=sr,
                    avg_confidence=sr + random.uniform(-0.1, 0.1),
                    days_since_last_training=random.randint(1, 30),
                    current_performance_score=sr,
                )
            )
            outcomes.append(0.15 if sr < 0.7 else 0.005)

        result = model.train(contexts, outcomes)
        assert result["status"] == "success"
        assert model.is_trained

        # Now check_thresholds should use the trained model
        live = LiveEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="xgboost_test",
            agent_results={
                AgentType.SEARCH: AgentEvalResult(
                    agent=AgentType.SEARCH,
                    score=0.4,
                    baseline_score=0.8,
                    degradation_pct=0.5,
                    sample_count=100,
                ),
            },
        )

        # Also verify directly: low success_rate + enough data → should_train=True
        from cogniverse_agents.routing.xgboost_meta_models import ModelingContext

        low_perf_context = ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=100,
            synthetic_sample_count=50,
            success_rate=0.4,
            avg_confidence=0.4,
            days_since_last_training=14,
            current_performance_score=0.4,
        )
        should_train, expected_improvement = model.should_train(low_perf_context)
        assert should_train is True, (
            f"Trained model should recommend training for low success_rate=0.4, "
            f"got should_train={should_train}, improvement={expected_improvement}"
        )

        # High success_rate → should_train=False
        high_perf_context = ModelingContext(
            modality=QueryModality.VIDEO,
            real_sample_count=100,
            synthetic_sample_count=50,
            success_rate=0.9,
            avg_confidence=0.9,
            days_since_last_training=3,
            current_performance_score=0.9,
        )
        should_skip, _ = model.should_train(high_perf_context)
        assert should_skip is False, (
            "Trained model should skip training for high success_rate=0.9"
        )

        # Now verify through check_thresholds
        verdicts = monitor_with_xgboost.check_thresholds(None, live)
        # score=0.4 < floor 0.5 → naive says OPTIMIZE
        # XGBoost trained on data where low success_rate benefited → confirms OPTIMIZE
        assert verdicts[AgentType.SEARCH] == Verdict.OPTIMIZE

    def test_modeling_context_built_from_eval_results(self, monitor_with_xgboost):
        """_build_modeling_context extracts correct values from eval results."""
        live = LiveEvalResult(
            timestamp=datetime.utcnow(),
            tenant_id="xgboost_test",
            agent_results={
                AgentType.SEARCH: AgentEvalResult(
                    agent=AgentType.SEARCH,
                    score=0.65,
                    baseline_score=0.80,
                    degradation_pct=0.19,
                    sample_count=50,
                ),
            },
        )

        context = monitor_with_xgboost._build_modeling_context(
            AgentType.SEARCH, None, live
        )

        assert context is not None
        assert context.real_sample_count == 50
        assert context.success_rate == 0.65
        assert context.current_performance_score == 0.65
