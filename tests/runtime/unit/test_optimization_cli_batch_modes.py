"""Unit tests for optimization_cli batch modes: simba, workflow, gateway-thresholds, profile.

Tests:
1. CLI argument parser recognizes all new modes
2. Each optimization function handles empty span data gracefully
3. Each function produces expected artifact types when given mock span data
"""

import argparse
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pandas as pd
import pytest

# Patch targets: these are imported locally inside each function,
# so we patch at the source module.
_PATCH_CONFIG = "cogniverse_foundation.config.utils.create_default_config_manager"
_PATCH_TELEMETRY = "cogniverse_foundation.telemetry.manager.get_telemetry_manager"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeTelemetryConfig:
    """Minimal config with get_project_name."""

    def get_project_name(self, tenant_id: str, service: Optional[str] = None) -> str:
        if service:
            return f"cogniverse-{tenant_id}-{service}"
        return f"cogniverse-{tenant_id}"


class FakeTraceStore:
    """In-memory trace store returning canned DataFrames."""

    def __init__(self, spans_df: pd.DataFrame | None = None):
        self._spans_df = spans_df if spans_df is not None else pd.DataFrame()

    async def get_spans(self, **kwargs) -> pd.DataFrame:
        return self._spans_df


class FakeDatasetStore:
    """Records calls to create_dataset and get_dataset."""

    def __init__(self):
        self.created: List[Dict[str, Any]] = []

    async def create_dataset(self, name, data, metadata=None):
        self.created.append({"name": name, "data": data, "metadata": metadata})
        return f"dataset-{len(self.created)}"

    async def get_dataset(self, name):
        raise KeyError(f"No dataset {name}")


class FakeExperimentStore:
    async def create_experiment(self, name, metadata=None):
        return name

    async def log_run(self, experiment_id, inputs, outputs, metadata=None):
        return "run-1"


class FakeTelemetryProvider:
    """Minimal TelemetryProvider stand-in with trace + dataset stores."""

    def __init__(self, spans_df: pd.DataFrame | None = None):
        self._trace_store = FakeTraceStore(spans_df)
        self._dataset_store = FakeDatasetStore()
        self._experiment_store = FakeExperimentStore()

    @property
    def traces(self):
        return self._trace_store

    @property
    def datasets(self):
        return self._dataset_store

    @property
    def experiments(self):
        return self._experiment_store


class FakeTelemetryManager:
    def __init__(self, provider):
        self._provider = provider
        self.config = FakeTelemetryConfig()

    def get_provider(self, tenant_id):
        return self._provider


@pytest.fixture
def empty_provider():
    return FakeTelemetryProvider(spans_df=pd.DataFrame())


@pytest.fixture
def fake_telemetry_manager(empty_provider):
    return FakeTelemetryManager(empty_provider)


def _patch_infra(fake_mgr):
    """Return a combined context manager patching config + telemetry."""
    return (
        patch(_PATCH_CONFIG),
        patch(_PATCH_TELEMETRY, return_value=fake_mgr),
    )


# ---------------------------------------------------------------------------
# Test: CLI argument parser recognizes all new modes
# ---------------------------------------------------------------------------


class TestCliArgumentParser:
    """Verify the argparse config accepts all new modes."""

    @pytest.fixture
    def parser(self):
        """Build the parser matching the real CLI choices."""
        parser = argparse.ArgumentParser(description="Test")
        parser.add_argument(
            "--mode",
            choices=[
                "cleanup",
                "triggered",
                "simba",
                "workflow",
                "gateway-thresholds",
                "profile",
                "entity-extraction",
                "routing",
                "synthetic",
            ],
            required=True,
        )
        parser.add_argument("--tenant-id", default="default")
        parser.add_argument("--lookback-hours", type=int, default=24)
        parser.add_argument("--agents")
        parser.add_argument("--trigger-dataset")
        return parser

    @pytest.mark.parametrize(
        "mode",
        [
            "simba",
            "workflow",
            "gateway-thresholds",
            "profile",
            "entity-extraction",
            "routing",
            "synthetic",
        ],
    )
    def test_new_mode_accepted(self, parser, mode):
        args = parser.parse_args(["--mode", mode])
        assert args.mode == mode
        assert args.tenant_id == "default"
        assert args.lookback_hours == 24

    @pytest.mark.parametrize(
        "mode",
        [
            "simba",
            "workflow",
            "gateway-thresholds",
            "profile",
            "entity-extraction",
            "routing",
            "synthetic",
        ],
    )
    def test_new_mode_with_tenant_and_lookback(self, parser, mode):
        args = parser.parse_args(
            ["--mode", mode, "--tenant-id", "acme", "--lookback-hours", "48"]
        )
        assert args.mode == mode
        assert args.tenant_id == "acme"
        assert args.lookback_hours == 48

    def test_invalid_mode_rejected(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["--mode", "nonexistent"])


# ---------------------------------------------------------------------------
# Test: each mode handles empty span data gracefully
# ---------------------------------------------------------------------------


class TestEmptySpanHandling:
    """Each optimization function returns no_data when Phoenix has no matching spans."""

    @pytest.mark.asyncio
    async def test_simba_no_data(self, fake_telemetry_manager):
        from cogniverse_runtime.optimization_cli import run_simba_optimization

        p1, p2 = _patch_infra(fake_telemetry_manager)
        with p1, p2:
            result = await run_simba_optimization(
                tenant_id="test:unit", lookback_hours=1
            )
        assert result["status"] == "no_data"
        assert result["spans_found"] == 0

    @pytest.mark.asyncio
    async def test_workflow_no_data(self, fake_telemetry_manager):
        from cogniverse_runtime.optimization_cli import run_workflow_optimization

        p1, p2 = _patch_infra(fake_telemetry_manager)
        with p1, p2:
            result = await run_workflow_optimization(
                tenant_id="test:unit", lookback_hours=1
            )
        assert result["status"] == "no_data"
        assert result["spans_found"] == 0

    @pytest.mark.asyncio
    async def test_gateway_thresholds_no_data(self, fake_telemetry_manager):
        from cogniverse_runtime.optimization_cli import (
            run_gateway_thresholds_optimization,
        )

        p1, p2 = _patch_infra(fake_telemetry_manager)
        with p1, p2:
            result = await run_gateway_thresholds_optimization(
                tenant_id="test:unit", lookback_hours=1
            )
        assert result["status"] == "no_data"
        assert result["spans_found"] == 0

    @pytest.mark.asyncio
    async def test_profile_no_data(self, fake_telemetry_manager):
        from cogniverse_runtime.optimization_cli import run_profile_optimization

        p1, p2 = _patch_infra(fake_telemetry_manager)
        with p1, p2:
            result = await run_profile_optimization(
                tenant_id="test:unit", lookback_hours=1
            )
        assert result["status"] == "no_data"
        assert result["spans_found"] == 0


# ---------------------------------------------------------------------------
# Test: functions handle spans with no extractable training examples
# ---------------------------------------------------------------------------


def _make_spans_df(span_name: str, rows: list[dict]) -> pd.DataFrame:
    """Build a spans DataFrame with the given name and attribute columns."""
    df = pd.DataFrame(rows)
    df["name"] = span_name
    return df


class TestSpansWithNoExamples:
    """Spans exist but contain no usable training data (missing attributes)."""

    @pytest.mark.asyncio
    async def test_simba_spans_missing_attributes(self):
        spans_df = _make_spans_df(
            "cogniverse.query_enhancement",
            [
                {
                    "attributes.query_enhancement.original_query": "",
                    "attributes.query_enhancement.enhanced_query": "",
                }
            ],
        )
        provider = FakeTelemetryProvider(spans_df)
        mgr = FakeTelemetryManager(provider)

        from cogniverse_runtime.optimization_cli import run_simba_optimization

        p1, p2 = _patch_infra(mgr)
        with p1, p2:
            result = await run_simba_optimization(
                tenant_id="test:unit", lookback_hours=1
            )
        assert result["status"] == "no_data"
        assert result["spans_found"] == 1
        assert result["examples"] == 0

    @pytest.mark.asyncio
    async def test_profile_spans_low_confidence(self):
        """Profile optimization skips examples with confidence < 0.5."""
        spans_df = _make_spans_df(
            "cogniverse.profile_selection",
            [
                {
                    "attributes.profile_selection.query": "find videos",
                    "attributes.profile_selection.selected_profile": "video_colpali_smol500_mv_frame",
                    "attributes.profile_selection.modality": "video",
                    "attributes.profile_selection.complexity": "simple",
                    "attributes.profile_selection.intent": "video_search",
                    "attributes.profile_selection.confidence": 0.2,
                },
            ],
        )
        provider = FakeTelemetryProvider(spans_df)
        mgr = FakeTelemetryManager(provider)

        from cogniverse_runtime.optimization_cli import run_profile_optimization

        p1, p2 = _patch_infra(mgr)
        with p1, p2:
            result = await run_profile_optimization(
                tenant_id="test:unit", lookback_hours=1
            )
        assert result["status"] == "no_data"
        assert result["spans_found"] == 1
        assert result["examples"] == 0


# ---------------------------------------------------------------------------
# Test: gateway threshold analysis with mock span data
# ---------------------------------------------------------------------------


class TestGatewayThresholdAnalysis:
    """Verify threshold tuning logic with synthetic gateway spans."""

    @pytest.mark.asyncio
    async def test_high_simple_error_rate_raises_threshold(self):
        """When simple-routed queries fail often, threshold should increase."""
        rows = []
        # 5 simple queries, 3 with ERROR status
        for i in range(5):
            rows.append(
                {
                    "attributes.gateway": {
                        "complexity": "simple",
                        "confidence": 0.8,
                        "modality": "video",
                        "generation_type": "raw_results",
                        "routed_to": "search_agent",
                    },
                    "status_code": "ERROR" if i < 3 else "OK",
                }
            )
        # 2 complex queries, both OK
        for _ in range(2):
            rows.append(
                {
                    "attributes.gateway": {
                        "complexity": "complex",
                        "confidence": 0.4,
                        "modality": "video",
                        "generation_type": "raw_results",
                        "routed_to": "orchestrator_agent",
                    },
                    "status_code": "OK",
                }
            )

        spans_df = _make_spans_df("cogniverse.gateway", rows)
        provider = FakeTelemetryProvider(spans_df)
        mgr = FakeTelemetryManager(provider)

        from cogniverse_runtime.optimization_cli import (
            run_gateway_thresholds_optimization,
        )

        p1, p2 = _patch_infra(mgr)
        with p1, p2:
            result = await run_gateway_thresholds_optimization(
                tenant_id="test:unit", lookback_hours=1
            )

        assert result["status"] == "success"
        thresholds = result["thresholds"]
        # Threshold should have been raised from default 0.4
        assert thresholds["fast_path_confidence_threshold"] > 0.4
        assert "artifact_id" in result

    @pytest.mark.asyncio
    async def test_all_ok_keeps_threshold_stable(self):
        """When error rates are low, threshold stays near default."""
        rows = []
        for _ in range(10):
            rows.append(
                {
                    "attributes.gateway": {
                        "complexity": "simple",
                        "confidence": 0.75,
                        "modality": "video",
                        "generation_type": "raw_results",
                        "routed_to": "search_agent",
                    },
                    "status_code": "OK",
                }
            )

        spans_df = _make_spans_df("cogniverse.gateway", rows)
        provider = FakeTelemetryProvider(spans_df)
        mgr = FakeTelemetryManager(provider)

        from cogniverse_runtime.optimization_cli import (
            run_gateway_thresholds_optimization,
        )

        p1, p2 = _patch_infra(mgr)
        with p1, p2:
            result = await run_gateway_thresholds_optimization(
                tenant_id="test:unit", lookback_hours=1
            )

        assert result["status"] == "success"
        # Threshold should stay at default (0.4) since no high error rates
        threshold = result["thresholds"]["fast_path_confidence_threshold"]
        assert 0.3 <= threshold <= 0.5


# ---------------------------------------------------------------------------
# Test: workflow optimization with mock orchestration spans
# ---------------------------------------------------------------------------


class TestWorkflowOptimization:
    """Verify workflow optimization extracts executions and saves artifacts."""

    @pytest.mark.asyncio
    async def test_workflow_with_orchestration_spans(self):
        """Workflow mode processes orchestration spans through the evaluator.

        OrchestrationEvaluator._extract_workflow_execution expects nested dicts
        at ``attributes.orchestration`` and ``attributes.routing`` — that's how
        Phoenix returns span rows when the OTel attributes use a dotted prefix.
        """
        rows = [
            {
                "name": "cogniverse.orchestration",
                "context.span_id": f"span-{i}",
                "attributes.orchestration": {
                    "workflow_id": f"wf-{i}",
                    "query": f"test query {i}",
                    "pattern": "sequential",
                    "agents_used": "search_agent,summarizer_agent",
                    "execution_time": "2.5",
                    "tasks_completed": "2",
                    "execution_order": "search_agent,summarizer_agent",
                },
                "attributes.routing": {"confidence": "0.8"},
                "status_code": "OK",
                "status_message": None,
            }
            for i in range(3)
        ]
        spans_df = pd.DataFrame(rows)
        provider = FakeTelemetryProvider(spans_df)
        mgr = FakeTelemetryManager(provider)

        from cogniverse_runtime.optimization_cli import run_workflow_optimization

        p1, p2 = _patch_infra(mgr)
        with p1, p2:
            result = await run_workflow_optimization(
                tenant_id="test:unit", lookback_hours=1
            )

        assert result["status"] == "success"
        assert result["spans_found"] == 3
        assert result["workflows_extracted"] == 3


# ---------------------------------------------------------------------------
# Test: entity-extraction mode
# ---------------------------------------------------------------------------


class TestEntityExtractionOptimization:
    """Entity extraction optimization handles missing/empty span data."""

    @pytest.mark.asyncio
    async def test_entity_extraction_no_spans(self, fake_telemetry_manager):
        from cogniverse_runtime.optimization_cli import (
            run_entity_extraction_optimization,
        )

        p1, p2 = _patch_infra(fake_telemetry_manager)
        with p1, p2:
            result = await run_entity_extraction_optimization(
                tenant_id="test:unit", lookback_hours=1
            )
        assert result["status"] == "no_data"
        assert result["spans_found"] == 0

    @pytest.mark.asyncio
    async def test_entity_extraction_spans_no_entities(self):
        """Spans with entity_count == 0 produce no training examples."""
        spans_df = _make_spans_df(
            "cogniverse.entity_extraction",
            [
                {
                    "attributes.entity_extraction": {
                        "query": "find something",
                        "entity_count": 0,
                        "entities": "[]",
                    }
                }
            ],
        )
        provider = FakeTelemetryProvider(spans_df)
        mgr = FakeTelemetryManager(provider)

        from cogniverse_runtime.optimization_cli import (
            run_entity_extraction_optimization,
        )

        p1, p2 = _patch_infra(mgr)
        with p1, p2:
            result = await run_entity_extraction_optimization(
                tenant_id="test:unit", lookback_hours=1
            )
        assert result["status"] == "no_data"
        assert result["spans_found"] == 1
        assert result["examples"] == 0


# ---------------------------------------------------------------------------
# Test: routing mode
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test: synthetic data merge helper
# ---------------------------------------------------------------------------


class TestSyntheticDataMerge:
    @pytest.mark.asyncio
    async def test_load_approved_synthetic_no_data(self):
        """Returns empty list when no synthetic data exists."""
        from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data

        provider = FakeTelemetryProvider()
        result = await _load_approved_synthetic_data(provider, "default", "simba")
        assert result == []

    @pytest.mark.asyncio
    async def test_load_approved_synthetic_filters_by_status(self):
        """Only returns demos with approved/auto_approved status."""
        from unittest.mock import AsyncMock, patch

        from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data

        approved_demo = {
            "input": '{"query": "test"}',
            "output": "enhanced",
            "metadata": {"approval_status": "approved"},
        }
        pending_demo = {
            "input": '{"query": "other"}',
            "output": "out",
            "metadata": {"approval_status": "pending"},
        }
        auto_approved = {
            "input": '{"query": "auto"}',
            "output": "out2",
            "metadata": {"approval_status": "auto_approved"},
        }

        with patch(
            "cogniverse_agents.optimizer.artifact_manager.ArtifactManager"
        ) as MockAM:
            mock_am = MockAM.return_value
            mock_am.load_demonstrations = AsyncMock(
                return_value=[approved_demo, pending_demo, auto_approved]
            )

            provider = FakeTelemetryProvider()
            result = await _load_approved_synthetic_data(provider, "default", "simba")

        assert len(result) == 2
        assert approved_demo in result
        assert auto_approved in result
        assert pending_demo not in result


# ---------------------------------------------------------------------------
# Test: _create_teleprompter optimizer selection
# ---------------------------------------------------------------------------


class TestCreateTeleprompter:
    """Verify optimizer selection based on training set size."""

    def test_small_trainset_uses_bootstrap(self):
        """< 50 examples should use BootstrapFewShot."""
        from dspy.teleprompt import BootstrapFewShot

        from cogniverse_runtime.optimization_cli import _create_teleprompter

        tp = _create_teleprompter(10)
        assert isinstance(tp, BootstrapFewShot), (
            f"Expected BootstrapFewShot for 10 examples, got {type(tp).__name__}"
        )

    def test_49_uses_bootstrap(self):
        """Boundary: 49 examples should still use BootstrapFewShot."""
        from dspy.teleprompt import BootstrapFewShot

        from cogniverse_runtime.optimization_cli import _create_teleprompter

        tp = _create_teleprompter(49)
        assert isinstance(tp, BootstrapFewShot), (
            f"Expected BootstrapFewShot for 49 examples, got {type(tp).__name__}"
        )

    def test_50_uses_scaled_bootstrap(self):
        """Boundary: >= 50 examples should use scaled BootstrapFewShot."""
        from dspy.teleprompt import BootstrapFewShot

        from cogniverse_runtime.optimization_cli import _create_teleprompter

        tp = _create_teleprompter(50)
        assert isinstance(tp, BootstrapFewShot)
        assert tp.max_bootstrapped_demos == 8
        assert tp.max_labeled_demos == 16

    def test_large_trainset_uses_scaled_bootstrap(self):
        """200 examples should use scaled BootstrapFewShot with more demos."""
        from dspy.teleprompt import BootstrapFewShot

        from cogniverse_runtime.optimization_cli import _create_teleprompter

        tp = _create_teleprompter(200)
        assert isinstance(tp, BootstrapFewShot)
        assert tp.max_bootstrapped_demos == 8
        assert tp.max_labeled_demos == 16

    def test_zero_uses_bootstrap(self):
        """Edge case: 0 examples should use BootstrapFewShot."""
        from dspy.teleprompt import BootstrapFewShot

        from cogniverse_runtime.optimization_cli import _create_teleprompter

        tp = _create_teleprompter(0)
        assert isinstance(tp, BootstrapFewShot)


# ---------------------------------------------------------------------------
# Test: synthetic generation mode
# ---------------------------------------------------------------------------


def _gateway_spans(rows: list[dict]) -> pd.DataFrame:
    """Build a ``cogniverse.gateway`` spans DataFrame with ``attributes.gateway``
    populated from ``rows``. Each row is ``{"complexity": ..., "confidence":
    ..., "status_code": ...}``; ``status_code`` defaults to ``OK`` if absent.
    The DataFrame shape matches what Phoenix's ``get_spans`` returns."""
    records = []
    for r in rows:
        records.append(
            {
                "attributes.gateway": {
                    "complexity": r.get("complexity"),
                    "confidence": r.get("confidence"),
                },
                "status_code": r.get("status_code", "OK"),
            }
        )
    df = pd.DataFrame(records)
    df["name"] = "cogniverse.gateway"
    return df


class TestComputeGatewayThresholdsAlgorithm:
    """Tight assertions on every output field of ``_compute_gateway_thresholds``.

    The calibration has three branches:
      (1) simple_error_rate > 0.2        → optimized = min(0.4 + 0.1, 0.95) = 0.5
      (2) complex_err < 0.05 AND mean > 0.8 → optimized = max(0.4 - 0.05, 0.5) = 0.5
      (3) otherwise                       → optimized = 0.4 (default)

    ``gliner_threshold`` is always ``round(max(0.15, min(p25 * 0.8, 0.5)), 3)``.
    Tests cover each branch plus degenerate inputs.
    """

    def test_empty_df_reports_no_data(self):
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        result = _compute_gateway_thresholds(pd.DataFrame())
        assert result == {"status": "no_data", "spans_found": 0}

    def test_missing_attributes_gateway_column(self):
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        df = pd.DataFrame([{"name": "cogniverse.gateway", "status_code": "OK"}])
        result = _compute_gateway_thresholds(df)
        assert result == {
            "status": "no_data",
            "spans_found": 1,
            "reason": "no_gateway_attributes",
        }

    def test_no_confidence_values_across_spans(self):
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        df = _gateway_spans(
            [
                {"complexity": "simple", "confidence": None},
                {"complexity": "complex", "confidence": None},
            ]
        )
        result = _compute_gateway_thresholds(df)
        assert result == {
            "status": "no_data",
            "spans_found": 2,
            "reason": "no_confidence_data",
        }

    def test_high_simple_error_rate_raises_threshold(self):
        """Branch (1): 5 of 10 simple spans are errors → rate = 0.5 > 0.2.
        Optimizer raises fast_path threshold from 0.4 → 0.5."""
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        rows = []
        # 10 simple spans: 5 with status=ERROR (high error rate), all conf=0.5.
        for i in range(10):
            rows.append(
                {
                    "complexity": "simple",
                    "confidence": 0.5,
                    "status_code": "ERROR" if i < 5 else "OK",
                }
            )
        # 2 complex spans, no errors.
        rows += [{"complexity": "complex", "confidence": 0.5} for _ in range(2)]

        result = _compute_gateway_thresholds(_gateway_spans(rows))
        assert result["status"] == "ready"
        assert result["spans_found"] == 12

        t = result["thresholds"]
        assert t["fast_path_confidence_threshold"] == 0.5
        # All confidences = 0.5 → p25 = 0.5 → gliner = round(min(0.5*0.8, 0.5), 3)
        assert t["gliner_threshold"] == 0.4

        a = t["analysis"]
        assert a["total_spans"] == 12
        assert a["simple_count"] == 10
        assert a["complex_count"] == 2
        assert a["simple_error_rate"] == 0.5
        assert a["complex_error_rate"] == 0.0
        assert a["mean_confidence"] == 0.5
        assert a["p25_confidence"] == 0.5

    def test_high_confidence_low_complex_errors_lowers_threshold(self):
        """Branch (2): complex_error_rate = 0, mean_confidence = 0.9 > 0.8,
        simple_error_rate = 0 (not > 0.2). Optimizer lowers threshold from
        0.4 → max(0.35, 0.5) = 0.5."""
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        rows = [{"complexity": "simple", "confidence": 0.9} for _ in range(10)] + [
            {"complexity": "complex", "confidence": 0.9} for _ in range(5)
        ]

        result = _compute_gateway_thresholds(_gateway_spans(rows))
        assert result["status"] == "ready"

        t = result["thresholds"]
        assert t["fast_path_confidence_threshold"] == 0.5
        # p25 = 0.9 → gliner = round(max(0.15, min(0.72, 0.5)), 3) = 0.5
        assert t["gliner_threshold"] == 0.5

        a = t["analysis"]
        assert a["mean_confidence"] == 0.9
        assert a["p25_confidence"] == 0.9
        assert a["simple_error_rate"] == 0.0
        assert a["complex_error_rate"] == 0.0

    def test_moderate_signal_keeps_default_threshold(self):
        """Branch (3): simple_error_rate = 0.1 (not > 0.2), mean_confidence =
        0.55 (not > 0.8). Neither branch fires; threshold stays at 0.4."""
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        rows = []
        for i in range(10):
            rows.append(
                {
                    "complexity": "simple",
                    "confidence": 0.6 if i < 5 else 0.5,
                    "status_code": "ERROR" if i == 0 else "OK",
                }
            )
        rows += [{"complexity": "complex", "confidence": 0.5} for _ in range(2)]

        result = _compute_gateway_thresholds(_gateway_spans(rows))
        t = result["thresholds"]
        assert t["fast_path_confidence_threshold"] == 0.4

        a = t["analysis"]
        # 1 of 10 simple = 0.1; doesn't trigger branch 1.
        assert a["simple_error_rate"] == 0.1
        # Mean of 5x 0.6 + 5x 0.5 + 2x 0.5 over 12 = 6.5 / 12 ≈ 0.5417
        assert a["mean_confidence"] == 0.5417

    def test_gliner_floor_at_0_15(self):
        """When p25 * 0.8 < 0.15, gliner_threshold floors at 0.15 (prevents
        the GLiNER model from being effectively disabled by a near-zero
        threshold derived from low-confidence training data)."""
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        rows = [{"complexity": "simple", "confidence": 0.05} for _ in range(4)]
        result = _compute_gateway_thresholds(_gateway_spans(rows))
        t = result["thresholds"]
        # p25 = 0.05, p25*0.8 = 0.04, below the 0.15 floor.
        assert t["gliner_threshold"] == 0.15

    def test_gliner_ceiling_at_0_5(self):
        """When p25 * 0.8 > 0.5, gliner_threshold caps at 0.5 (preserves
        recall — too high a threshold means GLiNER misses valid entities)."""
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        rows = [{"complexity": "simple", "confidence": 0.95} for _ in range(4)]
        result = _compute_gateway_thresholds(_gateway_spans(rows))
        t = result["thresholds"]
        # p25 = 0.95, p25*0.8 = 0.76, caps at 0.5.
        assert t["gliner_threshold"] == 0.5

    def test_status_col_absent_means_zero_error_rate(self):
        """Spans without a ``status_code`` column count as all-OK — the
        optimizer must not crash on minimal Phoenix schemas that lack it."""
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        df = _gateway_spans([{"complexity": "simple", "confidence": 0.5}])
        df = df.drop(columns=["status_code"])
        result = _compute_gateway_thresholds(df)
        a = result["thresholds"]["analysis"]
        assert a["simple_error_rate"] == 0.0
        assert a["complex_error_rate"] == 0.0

    def test_malformed_attributes_dict_treated_as_missing(self):
        """Defensive: an ``attributes.gateway`` value that's not a dict (e.g.
        a stray string from a malformed write) must not crash the compute."""
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        df = pd.DataFrame(
            [
                {
                    "name": "cogniverse.gateway",
                    "attributes.gateway": "not-a-dict",
                    "status_code": "OK",
                }
            ]
        )
        result = _compute_gateway_thresholds(df)
        # No complexity/confidence extractable → no confidence data.
        assert result["status"] == "no_data"
        assert result["reason"] == "no_confidence_data"


class TestSyntheticGeneration:
    """Verify synthetic generation CLI mode."""

    @pytest.mark.asyncio
    async def test_synthetic_no_backend_returns_failed(self, fake_telemetry_manager):
        """Synthetic generation without backend config should fail gracefully."""
        from cogniverse_runtime.optimization_cli import run_synthetic_generation

        p1, p2 = _patch_infra(fake_telemetry_manager)
        with p1, p2:
            result = await run_synthetic_generation(
                tenant_id="test:unit",
                optimizer_types=["simba"],
                count=5,
            )

        # Should fail (no real backend) but not crash
        assert "results" in result
        assert "simba" in result["results"]
        assert result["results"]["simba"]["status"] in ("success", "failed", "no_data")
