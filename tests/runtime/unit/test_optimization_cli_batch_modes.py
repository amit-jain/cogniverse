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
                "once",
                "full",
                "dspy",
                "cleanup",
                "triggered",
                "simba",
                "workflow",
                "gateway-thresholds",
                "profile",
                "entity-extraction",
                "routing",
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
        ["simba", "workflow", "gateway-thresholds", "profile", "entity-extraction", "routing"],
    )
    def test_new_mode_accepted(self, parser, mode):
        args = parser.parse_args(["--mode", mode])
        assert args.mode == mode
        assert args.tenant_id == "default"
        assert args.lookback_hours == 24

    @pytest.mark.parametrize(
        "mode",
        ["simba", "workflow", "gateway-thresholds", "profile", "entity-extraction", "routing"],
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
                tenant_id="default", lookback_hours=1
            )
        assert result["status"] == "no_data"
        assert result["spans_found"] == 0

    @pytest.mark.asyncio
    async def test_workflow_no_data(self, fake_telemetry_manager):
        from cogniverse_runtime.optimization_cli import run_workflow_optimization

        p1, p2 = _patch_infra(fake_telemetry_manager)
        with p1, p2:
            result = await run_workflow_optimization(
                tenant_id="default", lookback_hours=1
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
                tenant_id="default", lookback_hours=1
            )
        assert result["status"] == "no_data"
        assert result["spans_found"] == 0

    @pytest.mark.asyncio
    async def test_profile_no_data(self, fake_telemetry_manager):
        from cogniverse_runtime.optimization_cli import run_profile_optimization

        p1, p2 = _patch_infra(fake_telemetry_manager)
        with p1, p2:
            result = await run_profile_optimization(
                tenant_id="default", lookback_hours=1
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
                tenant_id="default", lookback_hours=1
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
                tenant_id="default", lookback_hours=1
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
                tenant_id="default", lookback_hours=1
            )

        assert result["status"] == "success"
        thresholds = result["thresholds"]
        # Threshold should have been raised from default 0.7
        assert thresholds["fast_path_confidence_threshold"] > 0.7
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
                tenant_id="default", lookback_hours=1
            )

        assert result["status"] == "success"
        # Threshold should stay at default since no high error rates
        threshold = result["thresholds"]["fast_path_confidence_threshold"]
        assert 0.6 <= threshold <= 0.75


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
                tenant_id="default", lookback_hours=1
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
                tenant_id="default", lookback_hours=1
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
                tenant_id="default", lookback_hours=1
            )
        assert result["status"] == "no_data"
        assert result["spans_found"] == 1
        assert result["examples"] == 0


# ---------------------------------------------------------------------------
# Test: routing mode
# ---------------------------------------------------------------------------


class TestRoutingOptimization:
    """Routing optimization handles missing/empty span data."""

    @pytest.mark.asyncio
    async def test_routing_no_spans(self, fake_telemetry_manager):
        from cogniverse_runtime.optimization_cli import run_routing_optimization

        p1, p2 = _patch_infra(fake_telemetry_manager)
        with p1, p2:
            result = await run_routing_optimization(
                tenant_id="default", lookback_hours=1
            )
        assert result["status"] == "no_data"
        assert result["spans_found"] == 0

    @pytest.mark.asyncio
    async def test_routing_spans_low_confidence(self):
        """Routing optimization skips examples with confidence < 0.5."""
        spans_df = _make_spans_df(
            "cogniverse.routing",
            [
                {
                    "attributes.routing": {
                        "query": "search for videos",
                        "recommended_agent": "search_agent",
                        "primary_intent": "video_search",
                        "confidence": 0.3,
                    }
                }
            ],
        )
        provider = FakeTelemetryProvider(spans_df)
        mgr = FakeTelemetryManager(provider)

        from cogniverse_runtime.optimization_cli import run_routing_optimization

        p1, p2 = _patch_infra(mgr)
        with p1, p2:
            result = await run_routing_optimization(
                tenant_id="default", lookback_hours=1
            )
        assert result["status"] == "no_data"
        assert result["spans_found"] == 1
        assert result["examples"] == 0
