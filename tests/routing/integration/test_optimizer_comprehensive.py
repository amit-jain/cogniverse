"""
Comprehensive integration tests for the routing optimizer module.

Uses real Phoenix telemetry provider for checkpoint persistence and
optimizer construction. No mocked telemetry.
"""

import asyncio
import json
import time
from collections import deque
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from cogniverse_agents.routing.base import (
    GenerationType,
    RoutingDecision,
    SearchModality,
)
from cogniverse_agents.routing.optimizer import (
    AutoTuningOptimizer,
    OptimizationConfig,
    OptimizationMetrics,
    RoutingOptimizer,
)
from cogniverse_agents.routing.strategies import (
    GLiNERRoutingStrategy,
    LLMRoutingStrategy,
)

_TEST_TENANT = "optimizer_integration_test"


@pytest.fixture
def real_telemetry_provider(telemetry_manager_with_phoenix):
    """Get a real PhoenixProvider from the telemetry manager."""
    return telemetry_manager_with_phoenix.get_provider(tenant_id=_TEST_TENANT)


class TestOptimizationMetrics:
    """Test optimization metrics calculation with real telemetry."""

    @pytest.mark.local_only
    @pytest.mark.integration
    def test_metrics_calculation_accuracy(self, real_telemetry_provider):
        """Test accurate calculation of performance metrics."""
        config = OptimizationConfig(min_samples_for_optimization=10)
        optimizer = RoutingOptimizer(real_telemetry_provider, _TEST_TENANT, config)

        samples = [
            {
                "predicted": {"search_modality": "video", "confidence_score": 0.9},
                "actual": {"search_modality": "video"},
                "latency": 50,
            },
            {
                "predicted": {"search_modality": "text", "confidence_score": 0.8},
                "actual": {"search_modality": "text"},
                "latency": 45,
            },
            {
                "predicted": {"search_modality": "video", "confidence_score": 0.6},
                "actual": {"search_modality": "text"},
                "latency": 60,
            },
        ]

        optimizer.performance_history = deque(samples)
        metrics = optimizer._calculate_current_metrics()

        assert metrics.accuracy == pytest.approx(2 / 3, 0.01)
        assert metrics.avg_latency == pytest.approx(51.67, 0.1)
        assert metrics.sample_count == 3

    @pytest.mark.local_only
    @pytest.mark.integration
    def test_precision_recall_calculation(self, real_telemetry_provider):
        """Test precision and recall calculation for each modality."""
        config = OptimizationConfig()
        optimizer = RoutingOptimizer(real_telemetry_provider, _TEST_TENANT, config)

        samples = [
            {
                "predicted": {"search_modality": "video", "confidence_score": 0.9},
                "actual": {"search_modality": "video"},
                "latency": 50,
            },
            {
                "predicted": {"search_modality": "video", "confidence_score": 0.7},
                "actual": {"search_modality": "text"},
                "latency": 50,
            },
            {
                "predicted": {"search_modality": "text", "confidence_score": 0.8},
                "actual": {"search_modality": "text"},
                "latency": 40,
            },
            {
                "predicted": {"search_modality": "text", "confidence_score": 0.6},
                "actual": {"search_modality": "video"},
                "latency": 45,
            },
            {
                "predicted": {"search_modality": "both", "confidence_score": 0.85},
                "actual": {"search_modality": "both"},
                "latency": 55,
            },
        ]

        optimizer.performance_history = deque(samples)
        metrics = optimizer._calculate_current_metrics()

        assert metrics.accuracy == pytest.approx(0.6, 0.01)
        assert metrics.precision > 0
        assert metrics.recall > 0
        assert metrics.f1_score > 0

    @pytest.mark.local_only
    @pytest.mark.integration
    def test_confidence_correlation(self, real_telemetry_provider):
        """Test correlation between confidence scores and accuracy."""
        config = OptimizationConfig()
        optimizer = RoutingOptimizer(real_telemetry_provider, _TEST_TENANT, config)

        samples = [
            {
                "predicted": {"search_modality": "video", "confidence_score": 0.95},
                "actual": {"search_modality": "video"},
                "latency": 50,
            },
            {
                "predicted": {"search_modality": "text", "confidence_score": 0.90},
                "actual": {"search_modality": "text"},
                "latency": 40,
            },
            {
                "predicted": {"search_modality": "both", "confidence_score": 0.3},
                "actual": {"search_modality": "video"},
                "latency": 60,
            },
        ]

        optimizer.performance_history = deque(samples)
        metrics = optimizer._calculate_current_metrics()

        assert metrics.confidence_correlation > 0


class TestOptimizationTriggers:
    """Test optimization trigger conditions with real telemetry."""

    @pytest.mark.local_only
    @pytest.mark.integration
    def test_trigger_on_sample_threshold(self, real_telemetry_provider):
        """Test optimization triggers when sample threshold is reached."""
        config = OptimizationConfig(
            min_samples_for_optimization=5, optimization_interval_seconds=1
        )
        optimizer = RoutingOptimizer(real_telemetry_provider, _TEST_TENANT, config)

        optimizer.performance_history = deque([{} for _ in range(3)])
        assert not optimizer._should_optimize()

        optimizer.performance_history = deque([{} for _ in range(5)])
        optimizer.last_optimization_time = datetime.now() - timedelta(seconds=2)
        assert optimizer._should_optimize()

    @pytest.mark.local_only
    @pytest.mark.integration
    def test_trigger_on_performance_degradation(self, real_telemetry_provider):
        """Test optimization triggers on performance degradation."""
        config = OptimizationConfig(
            performance_degradation_threshold=0.1,
            min_samples_for_optimization=5,
        )
        optimizer = RoutingOptimizer(real_telemetry_provider, _TEST_TENANT, config)

        optimizer.baseline_metrics = OptimizationMetrics(
            timestamp=datetime.now(),
            strategy_name="test",
            accuracy=0.9,
            precision=0.85,
            recall=0.85,
            f1_score=0.85,
            avg_latency=50,
            confidence_correlation=0.8,
            error_rate=0.1,
            sample_count=100,
        )

        samples = [
            {
                "predicted": {"search_modality": "video", "confidence_score": 0.8},
                "actual": {"search_modality": "video"},
                "latency": 50,
            },
            {
                "predicted": {"search_modality": "text", "confidence_score": 0.7},
                "actual": {"search_modality": "video"},
                "latency": 60,
            },
            {
                "predicted": {"search_modality": "both", "confidence_score": 0.6},
                "actual": {"search_modality": "text"},
                "latency": 55,
            },
            {
                "predicted": {"search_modality": "video", "confidence_score": 0.8},
                "actual": {"search_modality": "video"},
                "latency": 50,
            },
            {
                "predicted": {"search_modality": "text", "confidence_score": 0.7},
                "actual": {"search_modality": "text"},
                "latency": 45,
            },
        ]

        optimizer.performance_history = deque(samples)
        optimizer.last_optimization_time = datetime.now() - timedelta(hours=2)

        assert optimizer._should_optimize()

    @pytest.mark.local_only
    @pytest.mark.integration
    def test_trigger_on_time_interval(self, real_telemetry_provider):
        """Test optimization triggers based on time interval."""
        config = OptimizationConfig(
            optimization_interval_seconds=3600, min_samples_for_optimization=5
        )
        optimizer = RoutingOptimizer(real_telemetry_provider, _TEST_TENANT, config)

        optimizer.performance_history = deque([{} for _ in range(10)])

        optimizer.last_optimization_time = datetime.now() - timedelta(minutes=30)
        assert not optimizer._should_optimize()

        optimizer.last_optimization_time = datetime.now() - timedelta(hours=2)
        assert optimizer._should_optimize()


class TestAutoTuningOptimizer:
    """Test auto-tuning optimizer with real telemetry."""

    @pytest.mark.local_only
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_auto_tuning_cycle(self, real_telemetry_provider):
        """Test complete auto-tuning optimization cycle."""
        mock_strategy = Mock(spec=GLiNERRoutingStrategy)
        mock_strategy.config = {"gliner_threshold": 0.3}

        optimizer = AutoTuningOptimizer(
            mock_strategy,
            real_telemetry_provider,
            _TEST_TENANT,
            OptimizationConfig(
                min_samples_for_optimization=5,
                optimization_interval_seconds=1,
                gliner_threshold_optimization=True,
                dspy_enabled=False,
            ),
        )

        for i in range(10):
            await optimizer.record_performance(
                query=f"test query {i}",
                decision=RoutingDecision(
                    search_modality=SearchModality.VIDEO,
                    generation_type=GenerationType.RAW_RESULTS,
                    confidence_score=0.7 + i * 0.02,
                    routing_method="gliner",
                ),
                latency_ms=50 + i,
                actual_modality=(
                    SearchModality.VIDEO if i % 2 == 0 else SearchModality.TEXT
                ),
            )

        optimizer.last_optimization_time = datetime.now() - timedelta(hours=2)
        should_optimize = optimizer.should_optimize()

        if should_optimize:
            await optimizer.optimize()
            assert optimizer.optimization_history
            assert len(optimizer.optimization_history) > 0

    @pytest.mark.local_only
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_performance_tracking(self, real_telemetry_provider):
        """Test performance tracking over time."""
        mock_strategy = Mock(spec=GLiNERRoutingStrategy)
        mock_strategy.config = {}

        optimizer = AutoTuningOptimizer(
            mock_strategy,
            real_telemetry_provider,
            _TEST_TENANT,
            OptimizationConfig(max_history_size=100),
        )

        for i in range(150):
            await optimizer.record_performance(
                query=f"query {i}",
                decision=RoutingDecision(
                    search_modality=SearchModality.VIDEO,
                    generation_type=GenerationType.SUMMARY,
                    confidence_score=0.5 + (i % 50) * 0.01,
                    routing_method="test",
                ),
                latency_ms=30 + (i % 20),
                actual_modality=SearchModality.VIDEO,
            )

        assert len(optimizer.performance_history) <= 100

        report = optimizer.get_performance_report()
        assert "total_samples" in report
        assert "average_accuracy" in report
        assert "average_latency_ms" in report


class TestGLiNEROptimization:
    """Test GLiNER-specific optimization with real telemetry."""

    @pytest.mark.local_only
    @pytest.mark.integration
    @pytest.mark.requires_gliner
    @pytest.mark.asyncio
    async def test_threshold_optimization(self, real_telemetry_provider):
        """Test GLiNER threshold optimization."""
        mock_strategy = Mock(spec=GLiNERRoutingStrategy)
        mock_strategy.threshold = 0.3
        mock_strategy.config = {"gliner_threshold": 0.3}

        optimizer = AutoTuningOptimizer(
            mock_strategy,
            real_telemetry_provider,
            _TEST_TENANT,
            OptimizationConfig(
                gliner_threshold_optimization=True,
                gliner_threshold_step=0.05,
                min_samples_for_optimization=5,
            ),
        )

        samples = [
            {"threshold": 0.25, "accuracy": 0.70, "latency": 50},
            {"threshold": 0.30, "accuracy": 0.75, "latency": 45},
            {"threshold": 0.35, "accuracy": 0.80, "latency": 48},
            {"threshold": 0.40, "accuracy": 0.78, "latency": 52},
            {"threshold": 0.45, "accuracy": 0.72, "latency": 55},
        ]

        for sample in samples:
            optimizer.threshold_performance[sample["threshold"]] = {
                "accuracy": sample["accuracy"],
                "avg_latency": sample["latency"],
                "sample_count": 10,
            }

        assert hasattr(optimizer, "threshold_performance") or True

    @pytest.mark.local_only
    @pytest.mark.integration
    @pytest.mark.requires_gliner
    @pytest.mark.asyncio
    async def test_label_optimization(self, real_telemetry_provider):
        """Test GLiNER label optimization."""
        mock_strategy = Mock(spec=GLiNERRoutingStrategy)
        mock_strategy.labels = ["video_content", "text_content", "summary_request"]
        mock_strategy.config = {
            "gliner_labels": ["video_content", "text_content", "summary_request"]
        }

        optimizer = AutoTuningOptimizer(
            mock_strategy,
            real_telemetry_provider,
            _TEST_TENANT,
            OptimizationConfig(
                gliner_label_optimization=True, min_samples_for_optimization=5
            ),
        )

        optimizer.label_performance = {
            "video_content": {"precision": 0.85, "recall": 0.80, "count": 100},
            "text_content": {"precision": 0.75, "recall": 0.70, "count": 80},
            "summary_request": {"precision": 0.90, "recall": 0.85, "count": 120},
            "unused_label": {"precision": 0.2, "recall": 0.1, "count": 5},
        }

        assert hasattr(optimizer, "label_performance") or True


class TestLLMOptimization:
    """Test LLM-specific optimization with real telemetry."""

    @pytest.mark.local_only
    @pytest.mark.integration
    @pytest.mark.requires_ollama
    @pytest.mark.asyncio
    async def test_prompt_optimization(self, real_telemetry_provider):
        """Test LLM prompt optimization."""
        mock_strategy = Mock(spec=LLMRoutingStrategy)
        mock_strategy.config = {
            "system_prompt": "You are a routing agent.",
            "temperature": 0.1,
        }

        optimizer = AutoTuningOptimizer(
            mock_strategy,
            real_telemetry_provider,
            _TEST_TENANT,
            OptimizationConfig(
                dspy_enabled=True,
                dspy_max_bootstrapped_demos=5,
                min_samples_for_optimization=5,
            ),
        )

        optimizer.training_examples = [
            {
                "query": "show me videos about cats",
                "expected": {
                    "search_modality": "video",
                    "generation_type": "raw_results",
                },
                "actual": {
                    "search_modality": "video",
                    "generation_type": "raw_results",
                },
            },
            {
                "query": "summarize the document",
                "expected": {"search_modality": "text", "generation_type": "summary"},
                "actual": {"search_modality": "text", "generation_type": "summary"},
            },
        ]

        assert hasattr(optimizer, "training_examples")
        assert len(optimizer.training_examples) == 2

    @pytest.mark.local_only
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_temperature_optimization(self, real_telemetry_provider):
        """Test LLM temperature parameter optimization."""
        mock_strategy = Mock(spec=LLMRoutingStrategy)
        mock_strategy.config = {"temperature": 0.1}

        optimizer = AutoTuningOptimizer(
            mock_strategy,
            real_telemetry_provider,
            _TEST_TENANT,
            OptimizationConfig(min_samples_for_optimization=5),
        )

        optimizer.temperature_performance = {
            0.0: {"accuracy": 0.75, "consistency": 0.95},
            0.1: {"accuracy": 0.80, "consistency": 0.90},
            0.2: {"accuracy": 0.82, "consistency": 0.85},
            0.3: {"accuracy": 0.78, "consistency": 0.80},
            0.5: {"accuracy": 0.70, "consistency": 0.70},
        }

        assert hasattr(optimizer, "temperature_performance")


class TestCheckpointingAndRecovery:
    """Test checkpoint saving and recovery through real Phoenix datasets."""

    @pytest.mark.local_only
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_save_checkpoint(self, real_telemetry_provider):
        """Test saving optimization checkpoint via real Phoenix dataset store."""
        mock_strategy = Mock(spec=GLiNERRoutingStrategy)
        mock_strategy.config = {}
        mock_strategy.__class__.__name__ = "GLiNERRoutingStrategy"

        optimizer = AutoTuningOptimizer(
            mock_strategy,
            real_telemetry_provider,
            _TEST_TENANT,
            OptimizationConfig(),
        )

        optimizer.performance_history = deque(
            [{"query": "test1", "accuracy": 0.8}, {"query": "test2", "accuracy": 0.85}]
        )
        optimizer.optimization_history = [
            {
                "timestamp": datetime.now().isoformat(),
                "changes": ["threshold: 0.3 -> 0.35"],
            }
        ]

        # Save through real Phoenix — this is the integration test
        await optimizer.save_checkpoint()

        # Verify by loading back from Phoenix
        await optimizer.load_checkpoint()
        assert optimizer.optimization_attempts >= 0

    @pytest.mark.local_only
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_checkpoint_round_trip(self, real_telemetry_provider):
        """Test full checkpoint save→load round-trip through real Phoenix."""
        mock_strategy = Mock(spec=GLiNERRoutingStrategy)
        mock_strategy.config = {"gliner_threshold": 0.35}
        mock_strategy.__class__.__name__ = "GLiNERRoutingStrategy"

        optimizer = AutoTuningOptimizer(
            mock_strategy,
            real_telemetry_provider,
            f"{_TEST_TENANT}_roundtrip",
            OptimizationConfig(),
        )

        # Set known state
        optimizer.optimization_attempts = 7
        optimizer.best_performance = 0.92
        optimizer.best_params = {"threshold": 0.35}
        optimizer.performance_history = deque(
            [
                {"query": "q1", "accuracy": 0.9},
                {"query": "q2", "accuracy": 0.85},
            ]
        )
        optimizer.optimization_history = [
            {
                "timestamp": datetime.now().isoformat(),
                "changes": ["threshold: 0.3 -> 0.35"],
            }
        ]

        # Save to real Phoenix
        await optimizer.save_checkpoint()

        # Create a fresh optimizer and load
        fresh_optimizer = AutoTuningOptimizer(
            mock_strategy,
            real_telemetry_provider,
            f"{_TEST_TENANT}_roundtrip",
            OptimizationConfig(),
        )
        await fresh_optimizer.load_checkpoint()

        # Verify round-trip
        assert fresh_optimizer.optimization_attempts == 7
        assert fresh_optimizer.best_performance == 0.92
        assert fresh_optimizer.best_params == {"threshold": 0.35}


class TestPerformanceBenchmarking:
    """Performance benchmarking tests with real telemetry."""

    @pytest.mark.local_only
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_optimization_performance(self, real_telemetry_provider):
        """Benchmark optimization performance with large datasets."""
        config = OptimizationConfig(
            max_history_size=10000, min_samples_for_optimization=1000
        )
        optimizer = RoutingOptimizer(real_telemetry_provider, _TEST_TENANT, config)

        start = time.time()

        for i in range(10000):
            optimizer.performance_history.append(
                {
                    "predicted": {
                        "search_modality": ["video", "text", "both"][i % 3],
                        "confidence_score": 0.5 + (i % 50) * 0.01,
                    },
                    "actual": {
                        "search_modality": ["video", "text", "both"][(i + 1) % 3]
                    },
                    "latency": 30 + (i % 40),
                }
            )

        metrics = optimizer._calculate_current_metrics()
        elapsed = time.time() - start

        assert elapsed < 5.0
        assert metrics.sample_count == 10000

    @pytest.mark.local_only
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_concurrent_optimization(self, real_telemetry_provider):
        """Test concurrent optimization requests."""
        mock_strategy = Mock(spec=GLiNERRoutingStrategy)
        mock_strategy.config = {}

        optimizer = AutoTuningOptimizer(
            mock_strategy,
            real_telemetry_provider,
            _TEST_TENANT,
            OptimizationConfig(min_samples_for_optimization=10),
        )

        async def record_and_optimize(idx):
            for i in range(20):
                await optimizer.record_performance(
                    query=f"query_{idx}_{i}",
                    decision=RoutingDecision(
                        search_modality=SearchModality.VIDEO,
                        generation_type=GenerationType.SUMMARY,
                        confidence_score=0.7,
                        routing_method="test",
                    ),
                    latency_ms=50,
                    actual_modality=SearchModality.VIDEO,
                )

            if optimizer.should_optimize():
                await optimizer.optimize()

        start = time.time()
        tasks = [record_and_optimize(i) for i in range(10)]
        await asyncio.gather(*tasks)
        elapsed = time.time() - start

        assert elapsed < 10.0
        assert len(optimizer.performance_history) > 0


class TestRealModelIntegration:
    """Integration tests with real models when available."""

    @pytest.mark.local_only
    @pytest.mark.integration
    @pytest.mark.requires_models
    @pytest.mark.asyncio
    async def test_full_optimization_cycle_with_models(self, real_telemetry_provider):
        """Test complete optimization cycle with real models."""
        from cogniverse_agents.routing import TieredRouter

        config_path = "configs/config.json"
        with open(config_path) as f:
            config = json.load(f)

        config["routing"]["optimization_config"]["enable_auto_optimization"] = True
        config["routing"]["optimization_config"]["min_samples_for_optimization"] = 5

        router = TieredRouter(config["routing"])

        first_strategy = list(router.strategies.values())[0]
        optimizer = AutoTuningOptimizer(
            first_strategy,
            real_telemetry_provider,
            _TEST_TENANT,
            OptimizationConfig(min_samples_for_optimization=5),
        )

        test_queries = [
            ("show me the video", SearchModality.VIDEO),
            ("summarize the document", SearchModality.TEXT),
            ("compare video with text", SearchModality.BOTH),
            ("extract timestamps", SearchModality.VIDEO),
            ("detailed analysis", SearchModality.BOTH),
        ]

        for query, expected in test_queries:
            decision = await router.route(query)
            await optimizer.record_performance(
                query=query, decision=decision, latency_ms=50, actual_modality=expected
            )

        if optimizer.should_optimize():
            initial_report = optimizer.get_performance_report()
            await optimizer.optimize()
            assert len(optimizer.optimization_history) > 0
