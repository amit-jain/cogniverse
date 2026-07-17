"""Unit tests for optimization_cli batch modes: simba, workflow, gateway-thresholds, profile.

Tests:
1. CLI argument parser recognizes all new modes
2. Each optimization function handles empty span data gracefully
3. Each function produces expected artifact types when given mock span data
"""

import json
from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pandas as pd
import pytest

from cogniverse_runtime.optimization_cli import build_parser

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
    """Records calls to create_dataset, delete_dataset, and get_dataset."""

    def __init__(self):
        self.created: List[Dict[str, Any]] = []
        self.deleted: List[str] = []

    async def create_dataset(self, name, data, metadata=None):
        self.created.append({"name": name, "data": data, "metadata": metadata})
        return f"dataset-{len(self.created)}"

    async def delete_dataset(self, name) -> bool:
        # Blobs are last-write-wins: the artifact store deletes before create.
        self.deleted.append(name)
        return True

    async def get_dataset(self, name):
        raise KeyError(f"No dataset {name}")


class FakeTelemetryProvider:
    """Minimal TelemetryProvider stand-in with trace + dataset stores."""

    def __init__(self, spans_df: pd.DataFrame | None = None):
        self._trace_store = FakeTraceStore(spans_df)
        self._dataset_store = FakeDatasetStore()

    @property
    def traces(self):
        return self._trace_store

    @property
    def datasets(self):
        return self._dataset_store


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


@contextmanager
def _patch_telemetry(fake_mgr):
    """Patch get_telemetry_manager at BOTH lookup sites: the source module
    (optimization_cli imports it at call time) and the orchestration evaluator
    (which binds it at module import, so the source patch doesn't reach it)."""
    with (
        patch(_PATCH_TELEMETRY, return_value=fake_mgr),
        patch(
            "cogniverse_agents.routing.orchestration_evaluator.get_telemetry_manager",
            return_value=fake_mgr,
        ),
    ):
        yield


def _patch_infra(fake_mgr):
    """Return a combined context manager patching config + telemetry."""
    return (
        patch(_PATCH_CONFIG),
        _patch_telemetry(fake_mgr),
    )


# ---------------------------------------------------------------------------
# Test: CLI argument parser recognizes all new modes
# ---------------------------------------------------------------------------


_REAL_MODES = [
    "cleanup",
    "triggered",
    "simba",
    "workflow",
    "gateway-thresholds",
    "online-routing-eval",
    "profile",
    "entity-extraction",
    "synthetic",
    "rollback",
    "ab-compare",
    "egress-netpol",
    "monthly-reports",
]


class TestCliArgumentParser:
    """Drive the REAL CLI parser (build_parser) so the test can't drift from
    production the way the old hand-built parser had (it listed a phantom
    'routing' mode, omitted 5 real modes, and used a wrong tenant default)."""

    @pytest.fixture
    def parser(self):
        return build_parser()

    @pytest.mark.parametrize("mode", _REAL_MODES)
    def test_real_mode_accepted(self, parser, mode):
        assert parser.parse_args(["--mode", mode]).mode == mode

    def test_online_routing_eval_is_a_mode(self, parser):
        assert (
            parser.parse_args(["--mode", "online-routing-eval"]).mode
            == "online-routing-eval"
        )

    def test_routing_is_not_a_mode(self, parser):
        # 'routing' is the router family, NOT an optimization CLI mode.
        with pytest.raises(SystemExit):
            parser.parse_args(["--mode", "routing"])

    def test_cleanup_tenant_defaults_to_none(self, parser):
        # cleanup + monthly-reports run globally; tenant_id default is None so
        # the no-tenant CronWorkflows don't exit 2 on argparse.
        assert parser.parse_args(["--mode", "cleanup"]).tenant_id is None

    def test_tenant_and_lookback_hours(self, parser):
        args = parser.parse_args(
            ["--mode", "simba", "--tenant-id", "acme:prod", "--lookback-hours", "48"]
        )
        assert args.tenant_id == "acme:prod"
        assert args.lookback_hours == 48.0

    def test_lookback_hours_default(self, parser):
        assert parser.parse_args(["--mode", "simba"]).lookback_hours == 24.0

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


def _gateway_row(complexity: str, confidence: float, status_code: str) -> dict:
    """A canonical cogniverse.gateway span row (decision on output.value).

    Only the calibration MATH needs controlled complexity/status inputs (a real
    gateway won't emit ERROR spans on demand); the real producer->reader
    contract is covered by the real-Phoenix gateway test.
    """
    return {
        "attributes.output.value": json.dumps(
            {
                "complexity": complexity,
                "confidence": confidence,
                "modality": "video",
                "generation_type": "raw_results",
                "routed_to": "search_agent"
                if complexity == "simple"
                else "orchestrator_agent",
            }
        ),
        "status_code": status_code,
    }


class TestSpansWithNoExamples:
    """Spans exist but contain no usable training data (missing attributes)."""

    @pytest.mark.asyncio
    async def test_simba_spans_missing_attributes(self):
        # Canonical span whose enhancement is empty -> no usable training pair.
        spans_df = _make_spans_df(
            "cogniverse.query_enhancement",
            [
                {
                    "attributes.input.value": "robots",
                    "attributes.output.value": json.dumps({"enhanced_query": ""}),
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
                    "attributes.input.value": "find videos",
                    "attributes.output.value": json.dumps(
                        {
                            "selected_profile": "video_colpali_smol500_mv_frame",
                            "modality": "video",
                            "complexity": "simple",
                            "intent": "video_search",
                            "confidence": 0.2,
                        }
                    ),
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
        rows = [
            # 5 simple queries, 3 with ERROR status
            _gateway_row("simple", 0.8, "ERROR" if i < 3 else "OK")
            for i in range(5)
        ]
        # 2 complex queries, both OK
        rows += [_gateway_row("complex", 0.4, "OK") for _ in range(2)]

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
        rows = [_gateway_row("simple", 0.75, "OK") for _ in range(10)]

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

        OrchestrationEvaluator._extract_workflow_execution reads the workflow
        off the canonical input.value (query) and output.value (the decision).
        """
        rows = [
            {
                "name": "cogniverse.orchestration",
                "context.span_id": f"span-{i}",
                "attributes.input.value": f"test query {i}",
                "attributes.output.value": json.dumps(
                    {
                        "workflow_id": f"wf-{i}",
                        "pattern": "sequential",
                        "agent_sequence": ["search_agent", "summarizer_agent"],
                        "execution_time": 2.5,
                        "tasks_completed": 2,
                        "confidence": 0.8,
                    }
                ),
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
        """Spans with no entities produce no training examples."""
        # Canonical span whose entity list is empty -> no usable training pair.
        spans_df = _make_spans_df(
            "cogniverse.entity_extraction",
            [
                {
                    "attributes.input.value": "find something",
                    "attributes.output.value": json.dumps({"entities": []}),
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

    def test_teacher_settings_forwarded_to_bootstrap(self):
        """The configured teacher LM must reach BootstrapFewShot — DSPy runs
        the bootstrap teacher inside dspy.context(**teacher_settings), so an
        unforwarded teacher means the student silently teaches itself."""
        from cogniverse_runtime.optimization_cli import _create_teleprompter

        sentinel = object()
        small = _create_teleprompter(10, teacher_settings={"lm": sentinel})
        assert small.teacher_settings == {"lm": sentinel}
        assert small.max_bootstrapped_demos == 4

        scaled = _create_teleprompter(50, teacher_settings={"lm": sentinel})
        assert scaled.teacher_settings == {"lm": sentinel}
        assert scaled.max_bootstrapped_demos == 8

    def test_teacher_settings_default_empty(self):
        from cogniverse_runtime.optimization_cli import _create_teleprompter

        tp = _create_teleprompter(10)
        assert tp.teacher_settings == {}

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
    """Build a ``cogniverse.gateway`` spans DataFrame with the canonical
    ``output.value`` decision populated from ``rows``. Each row is
    ``{"complexity": ..., "confidence": ..., "status_code": ...}``;
    ``status_code`` defaults to ``OK`` if absent. The DataFrame shape matches
    what Phoenix's ``get_spans`` returns."""
    records = []
    for r in rows:
        records.append(
            {
                "attributes.output.value": json.dumps(
                    {
                        "complexity": r.get("complexity"),
                        "confidence": r.get("confidence"),
                    }
                ),
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
      (2) complex_err < 0.05 AND mean > 0.8 → optimized = max(0.4 - 0.05, 0.3) = 0.35
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
        simple_error_rate = 0 (not > 0.2). Optimizer lowers the threshold from
        0.4 → max(0.35, 0.3) = 0.35 so MORE queries stay on the fast path — the
        floor must be below the 0.4 default, not above it."""
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        rows = [{"complexity": "simple", "confidence": 0.9} for _ in range(10)] + [
            {"complexity": "complex", "confidence": 0.9} for _ in range(5)
        ]

        result = _compute_gateway_thresholds(_gateway_spans(rows))
        assert result["status"] == "ready"

        t = result["thresholds"]
        # Genuinely lowered from the 0.4 default (the pre-fix 0.5 floor RAISED it).
        assert t["fast_path_confidence_threshold"] == pytest.approx(0.35)
        assert t["fast_path_confidence_threshold"] < 0.4
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
        """Defensive: an ``output.value`` that parses to a non-dict (e.g. a
        stray string from a malformed write) must not crash the compute."""
        from cogniverse_runtime.optimization_cli import _compute_gateway_thresholds

        df = pd.DataFrame(
            [
                {
                    "name": "cogniverse.gateway",
                    "attributes.output.value": json.dumps("not-a-dict"),
                    "status_code": "OK",
                }
            ]
        )
        result = _compute_gateway_thresholds(df)
        # No decision dict extractable → treated as missing, no crash.
        assert result["status"] == "no_data"
        assert result["reason"] == "no_gateway_attributes"


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


class TestOptimizeAgentPersistence:
    """_optimize_agent must construct ArtifactManager(provider, tenant_id) and
    persist the compiled module via save_blob(kind="model", ...). The prior code
    called ArtifactManager(telemetry_provider=...) (missing the required
    tenant_id) and a non-existent store_artifact() — so every triggered
    optimization failed. The fake ArtifactManager below enforces the real
    interface, so the old code would raise (TypeError / AttributeError) here."""

    @pytest.mark.asyncio
    async def test_optimize_agent_persists_compiled_module(self):
        from unittest.mock import MagicMock

        from cogniverse_runtime.optimization_cli import _optimize_agent

        captured: Dict[str, Any] = {}

        class _FakeArtifactManager:
            def __init__(self, telemetry_provider, tenant_id):  # both REQUIRED
                captured["tenant_id"] = tenant_id

            async def save_blob(self, kind, key, content):
                captured["kind"] = kind
                captured["key"] = key
                return "artifact-xyz"

        class _FakeOptimizer:
            optimization_settings = {
                "max_bootstrapped_demos": 1,
                "max_labeled_demos": 1,
                "max_rounds": 1,
                "max_errors": 1,
                "teacher_settings": {},
            }

            def initialize_language_model(self, endpoint, teacher_endpoint_config=None):
                self.lm = MagicMock()  # consumed by dspy.context(lm=optimizer.lm)

            def create_query_analysis_signature(self):
                return object()

        class _FakeCompiled:
            def dump_state(self):
                return {"demos": []}

        class _FakeTeleprompter:
            def __init__(self, *a, **k):
                pass

            def compile(self, module, trainset=None):
                return _FakeCompiled()

        high_df = pd.DataFrame([{"query": "find cats", "output": "{}", "score": 0.9}])

        with (
            patch(
                "cogniverse_agents.optimizer.dspy_agent_optimizer.DSPyAgentPromptOptimizer",
                _FakeOptimizer,
            ),
            patch("dspy.ChainOfThought", lambda sig: object()),
            patch("dspy.teleprompt.BootstrapFewShot", _FakeTeleprompter),
            patch(
                "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
                _FakeArtifactManager,
            ),
        ):
            result = await _optimize_agent(
                "search",
                pd.DataFrame([]),
                high_df,
                "http://lm",
                config_manager=MagicMock(),
                telemetry_provider=MagicMock(),
                tenant_id="acme:prod",
            )

        assert result["status"] == "success"
        assert result["training_examples"] == 1
        assert captured["tenant_id"] == "acme:prod"
        # The compile reaches traffic through the versioned-prompts serving
        # path only; no side blob is written and no artifact id is reported.
        assert "key" not in captured, captured
        assert "artifact_id" not in result, result

    @pytest.mark.asyncio
    async def test_optimize_agent_threads_teacher_into_bootstrap(self):
        """_optimize_agent must hand the teacher endpoint to the real optimizer
        and forward the resulting teacher_settings into BootstrapFewShot —
        DSPy runs the bootstrap teacher inside dspy.context(**teacher_settings)."""
        from unittest.mock import MagicMock

        from cogniverse_foundation.config.unified_config import LLMEndpointConfig
        from cogniverse_runtime.optimization_cli import _optimize_agent

        captured: Dict[str, Any] = {}

        class _FakeArtifactManager:
            def __init__(self, telemetry_provider, tenant_id):
                pass

            async def save_blob(self, kind, key, content):
                return "artifact-teacher"

        class _FakeCompiled:
            def dump_state(self):
                return {"demos": []}

        class _CapturingTeleprompter:
            def __init__(self, *a, **k):
                captured["teleprompter_kwargs"] = k

            def compile(self, module, trainset=None):
                return _FakeCompiled()

        student = LLMEndpointConfig(
            model="hosted_vllm/org/Student", api_base="http://student:8000/v1"
        )
        teacher = LLMEndpointConfig(
            model="hosted_vllm/org/Teacher", api_base="http://teacher:9000/v1"
        )
        high_df = pd.DataFrame([{"query": "find cats", "output": "{}", "score": 0.9}])

        with (
            patch("dspy.teleprompt.BootstrapFewShot", _CapturingTeleprompter),
            patch(
                "cogniverse_agents.optimizer.artifact_manager.ArtifactManager",
                _FakeArtifactManager,
            ),
        ):
            result = await _optimize_agent(
                "search",
                pd.DataFrame([]),
                high_df,
                student,
                config_manager=MagicMock(),
                telemetry_provider=MagicMock(),
                tenant_id="acme:prod",
                teacher_endpoint=teacher,
            )

        assert result["status"] == "success"
        teacher_settings = captured["teleprompter_kwargs"]["teacher_settings"]
        assert teacher_settings["lm"].model == "hosted_vllm/org/Teacher"
        assert teacher_settings["lm"].kwargs["api_base"] == "http://teacher:9000/v1"


class FailingTraceStore:
    """Trace store whose get_spans always raises (Phoenix down/slow)."""

    def __init__(self):
        self.calls = 0

    async def get_spans(self, **kwargs) -> pd.DataFrame:
        self.calls += 1
        raise TimeoutError("phoenix query timed out")


class TestQuerySpansFailureIsNotNoData:
    """A failed Phoenix query must raise, not return an empty frame.

    Flattening the exception to an empty DataFrame made every batch mode
    report status=no_data during a Phoenix timeout — indistinguishable
    from a genuinely empty optimization window.
    """

    @pytest.mark.asyncio
    async def test_query_failure_raises_after_retries(self, monkeypatch):
        import asyncio as _asyncio

        from cogniverse_runtime import optimization_cli as cli

        provider = FakeTelemetryProvider()
        store = FailingTraceStore()
        provider._trace_store = store
        manager = FakeTelemetryManager(provider)

        monkeypatch.setattr(_asyncio, "sleep", _instant_sleep)
        with patch(_PATCH_TELEMETRY, return_value=manager):
            with pytest.raises(RuntimeError, match="after 3 attempts"):
                await cli._query_spans_by_name(
                    provider, "acme:prod", "cogniverse.entity_extraction", 1.0
                )
        assert store.calls == 3

    @pytest.mark.asyncio
    async def test_transient_failure_recovers_on_retry(self, monkeypatch):
        import asyncio as _asyncio

        from cogniverse_runtime import optimization_cli as cli

        df = pd.DataFrame([{"name": "cogniverse.entity_extraction", "x": 1}])

        class FlakyStore:
            def __init__(self):
                self.calls = 0

            async def get_spans(self, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError("first attempt times out")
                return df

        provider = FakeTelemetryProvider()
        store = FlakyStore()
        provider._trace_store = store
        manager = FakeTelemetryManager(provider)

        monkeypatch.setattr(_asyncio, "sleep", _instant_sleep)
        with patch(_PATCH_TELEMETRY, return_value=manager):
            out = await cli._query_spans_by_name(
                provider, "acme:prod", "cogniverse.entity_extraction", 1.0
            )
        assert store.calls == 2
        assert len(out) == 1


async def _instant_sleep(_seconds):
    return None
