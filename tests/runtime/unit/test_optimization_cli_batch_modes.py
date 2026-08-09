"""Unit tests for optimization_cli batch modes: simba, workflow, gateway-thresholds, profile.

Tests:
1. CLI argument parser recognizes all new modes
2. Each optimization function handles empty span data gracefully
3. Each function produces expected artifact types when given mock span data
"""

import asyncio
import hashlib
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pandas as pd
import pytest

from cogniverse_runtime.optimization_cli import build_parser
from cogniverse_sdk.interfaces.workflow_store import WorkflowLearningState

# Patch targets: these are imported locally inside each function,
# so we patch at the source module.
_PATCH_CONFIG = "cogniverse_foundation.config.utils.create_default_config_manager"
_PATCH_TELEMETRY = "cogniverse_foundation.telemetry.manager.get_telemetry_manager"


def _signed_approved_record(record: dict[str, Any]) -> dict[str, Any]:
    signed = {
        "confidence": 0.9,
        "created_at": "2026-08-05T01:00:00+00:00",
        "reviewed_at": "2026-08-05T01:01:00+00:00",
        **record,
    }
    decision = signed.get("metadata.decision")
    decision_intent = dict(decision) if isinstance(decision, dict) else decision
    if isinstance(decision_intent, dict):
        decision_intent.pop("timestamp", None)
    identity = {
        "item_id": signed.get("item_id"),
        "status": signed.get("status"),
        "decision": decision_intent,
    }
    signed["metadata.approval_decision_sha256"] = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    signed["metadata.approval_decision_timestamp"] = signed["reviewed_at"]
    canonical_json = json.dumps(
        signed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    signed["metadata.approval_record_json"] = canonical_json
    signed["metadata.approval_record_sha256"] = hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()
    return signed


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
        self.calls: List[Dict[str, Any]] = []

    async def get_spans(self, **kwargs) -> pd.DataFrame:
        return self._spans_df

    async def get_all_spans(self, **kwargs) -> pd.DataFrame:
        self.calls.append(kwargs)
        return self._spans_df.copy(deep=True)


class FakeDatasetStore:
    """Records calls to create_dataset, delete_dataset, and get_dataset."""

    def __init__(self):
        self.created: List[Dict[str, Any]] = []
        self.deleted: List[str] = []
        self.datasets: Dict[str, pd.DataFrame] = {}

    async def replace_dataset(self, name, data, metadata=None):
        return await self.create_dataset(name=name, data=data, metadata=metadata)

    async def create_dataset(self, name, data, metadata=None):
        self.created.append({"name": name, "data": data, "metadata": metadata})
        self.datasets[name] = data.copy(deep=True)
        return f"dataset-{len(self.created)}"

    async def delete_dataset(self, name) -> bool:
        # Blobs are last-write-wins: the artifact store deletes before create.
        self.deleted.append(name)
        return self.datasets.pop(name, None) is not None

    async def get_dataset(self, name):
        from cogniverse_foundation.telemetry.providers.base import (
            DatasetNotFoundError,
        )

        if name not in self.datasets:
            raise DatasetNotFoundError(f"No dataset {name}")
        return self.datasets[name].copy(deep=True)


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


class FakeWorkflowStore:
    """Canonical in-memory workflow-state boundary for CLI unit tests."""

    def __init__(self):
        self.states = {}

    async def replace_learning_state(
        self, tenant_id, executions, profiles, patterns, templates
    ):
        self.states[tenant_id] = {
            "executions": list(executions),
            "profiles": list(profiles),
            "patterns": dict(patterns),
            "templates": list(templates),
        }

    def _state(self, tenant_id):
        return self.states.get(
            tenant_id,
            {"executions": [], "profiles": [], "patterns": {}, "templates": []},
        )

    async def load_learning_state(self, tenant_id):
        state = self._state(tenant_id)
        return WorkflowLearningState(
            executions=list(state["executions"]),
            profiles=list(state["profiles"]),
            patterns=dict(state["patterns"]),
            templates=list(state["templates"]),
        )


class FakeTelemetryManager:
    def __init__(self, provider):
        self._provider = provider
        self.config = FakeTelemetryConfig()

    def get_provider(self, tenant_id):
        return self._provider


@pytest.fixture(autouse=True)
def _fresh_workflow_store_registry():
    """WorkflowStoreRegistry caches store instances process-wide, and
    TelemetryWorkflowStore caches per-tenant ArtifactManagers bound to the
    provider resolved at first touch. A store cached by an earlier test would
    bypass this file's telemetry patches (and a store built under the patches
    must not hand a fake provider to later callers) — clear on both sides so
    every get() rebuilds against whatever is patched right now."""
    from cogniverse_core.registries import WorkflowStoreRegistry

    WorkflowStoreRegistry.clear_cache()
    yield
    WorkflowStoreRegistry.clear_cache()


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


def _expected_primary_lm():
    """``llm_config.primary`` from the active config — the endpoint the CLI
    modes resolve for their DSPy work."""
    from cogniverse_foundation.config.unified_config import LLMConfig
    from tests.utils.llm_config import _load_config

    return LLMConfig.from_dict(_load_config()["llm_config"]).primary


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

    @pytest.mark.asyncio
    async def test_non_numeric_confidence_dropped_not_fatal(self):
        """One span with a string confidence must not abort the recompute;
        thresholds come from the numeric rows only."""
        rows = [_gateway_row("simple", c, "OK") for c in (0.5, 0.6, 0.7, 0.8)]
        rows.append(_gateway_row("simple", "high", "OK"))

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
        # Numeric rows [0.5, 0.6, 0.7, 0.8]: mean 0.65 keeps the default
        # fast path; p25 0.575 -> gliner 0.575 * 0.8 = 0.46.
        assert thresholds["fast_path_confidence_threshold"] == 0.4
        assert thresholds["gliner_threshold"] == 0.46
        analysis = thresholds["analysis"]
        assert analysis["total_spans"] == 5
        assert analysis["mean_confidence"] == 0.65
        assert analysis["p25_confidence"] == 0.575


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
                "start_time": datetime.now(timezone.utc) - timedelta(seconds=3 - i),
                "attributes.input.value": f"test query {i}",
                "attributes.output.value": json.dumps(
                    {
                        "workflow_id": f"wf-{i}",
                        "pattern": "sequential",
                        "agent_sequence": ["search_agent", "summarizer_agent"],
                        "execution_order": ["search_agent", "summarizer_agent"],
                        "execution_time": 2.5,
                        "success": True,
                        "tasks_completed": 2,
                        "confidence": 0.8,
                        "agent_observations": [
                            {
                                "agent_name": "search_agent",
                                "execution_time": 1.0,
                                "success": True,
                                "confidence": 0.9,
                            },
                            {
                                "agent_name": "summarizer_agent",
                                "execution_time": 1.5,
                                "success": True,
                                "confidence": 0.7,
                            },
                        ],
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

        workflow_store = FakeWorkflowStore()
        p1, p2 = _patch_infra(mgr)
        workflow_store_patch = patch(
            "cogniverse_core.registries.WorkflowStoreRegistry.get",
            return_value=workflow_store,
        )
        with p1, p2, workflow_store_patch:
            result = await run_workflow_optimization(
                tenant_id="test:unit", lookback_hours=1
            )

        assert result["status"] == "success"
        assert result["spans_found"] == 3
        assert result["workflows_extracted"] == 3

    @pytest.mark.asyncio
    async def test_drains_more_than_one_batch_and_persists_serving_artifacts(self):
        query = "find exact aurora video"
        evaluation_base = datetime.now(timezone.utc) - timedelta(minutes=2)
        rows = [
            {
                "name": "cogniverse.orchestration",
                "context.span_id": f"span-page-{index:02d}",
                "start_time": evaluation_base + timedelta(milliseconds=index),
                "attributes.input.value": query,
                "attributes.output.value": json.dumps(
                    {
                        "workflow_id": f"wf-page-{index:02d}",
                        "pattern": "sequential",
                        "agent_sequence": ["search_agent", "summarizer_agent"],
                        "execution_order": ["search_agent", "summarizer_agent"],
                        "execution_time": 2.5,
                        "success": True,
                        "tasks_completed": 2,
                        "confidence": 0.8,
                        "agent_observations": [
                            {
                                "agent_name": "search_agent",
                                "execution_time": 1.0,
                                "success": True,
                                "confidence": 0.9,
                            },
                            {
                                "agent_name": "summarizer_agent",
                                "execution_time": 1.5,
                                "success": True,
                                "confidence": 0.7,
                            },
                        ],
                    }
                ),
                "status_code": "OK",
                "status_message": None,
            }
            for index in range(55)
        ]
        provider = FakeTelemetryProvider(pd.DataFrame(rows))
        manager = FakeTelemetryManager(provider)

        from cogniverse_agents.workflow.intelligence import WorkflowIntelligence
        from cogniverse_runtime.optimization_cli import run_workflow_optimization

        workflow_store = FakeWorkflowStore()
        config_patch, telemetry_patch = _patch_infra(manager)
        workflow_store_patch = patch(
            "cogniverse_core.registries.WorkflowStoreRegistry.get",
            return_value=workflow_store,
        )
        with config_patch, telemetry_patch, workflow_store_patch:
            result = await run_workflow_optimization(
                tenant_id="test:workflow-pagination",
                lookback_hours=1,
            )
            fresh_intelligence = WorkflowIntelligence(
                tenant_id="test:workflow-pagination"
            )
            await fresh_intelligence.load_historical_data()

        assert result == {
            "status": "success",
            "spans_found": 55,
            "workflows_extracted": 55,
            "execution_demos_saved": 55,
            "agent_profiles_saved": 2,
            "workflow_templates_saved": 1,
        }
        assert len(provider.traces.calls) == 2
        assert {call["end_time"] for call in provider.traces.calls} == {
            provider.traces.calls[0]["end_time"]
        }
        assert len(fresh_intelligence.workflow_history) == 55
        assert set(fresh_intelligence.agent_performance) == {
            "search_agent",
            "summarizer_agent",
        }
        search_profile = fresh_intelligence.agent_performance["search_agent"]
        assert (
            search_profile.total_executions,
            search_profile.successful_executions,
            search_profile.average_execution_time,
            search_profile.average_confidence,
            search_profile.error_rate,
            search_profile.preferred_query_types,
        ) == (55, 55, 1.0, 0.9, 0.0, ["sequential_query"])

        summarizer_profile = fresh_intelligence.agent_performance["summarizer_agent"]
        assert (
            summarizer_profile.total_executions,
            summarizer_profile.successful_executions,
            summarizer_profile.average_execution_time,
            summarizer_profile.average_confidence,
            summarizer_profile.error_rate,
            summarizer_profile.preferred_query_types,
        ) == (55, 55, 1.5, 0.7, 0.0, ["sequential_query"])

        template = fresh_intelligence._find_matching_template(query)
        assert template is not None
        assert template.query_patterns == [query]
        assert template.task_sequence == [
            {"agent": "search_agent", "task": "process", "dependencies": []},
            {
                "agent": "summarizer_agent",
                "task": "process",
                "dependencies": ["template_task_0"],
            },
        ]
        assert template.expected_execution_time == 2.5
        assert template.success_rate == 1.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure_point", ["template", "corpus"])
    async def test_learning_state_helper_forwards_exact_state_and_store_failure(
        self, failure_point
    ):
        from cogniverse_runtime.optimization_cli import (
            _save_workflow_learning_state,
        )
        from cogniverse_sdk.interfaces.workflow_store import WorkflowTemplate

        def template(template_id, agent):
            return WorkflowTemplate(
                template_id=template_id,
                name=template_id,
                description=f"template for {agent}",
                query_patterns=[f"query for {agent}"],
                task_sequence=[{"agent": agent, "task": "process", "dependencies": []}],
                expected_execution_time=1.0,
                success_rate=1.0,
            )

        previous = template("previous", "search_agent")
        replacements = [
            template("replacement-a", "search_agent"),
            template("replacement-b", "summarizer_agent"),
        ]
        failure = ConnectionError(f"{failure_point} store unavailable")

        class Store:
            def __init__(self):
                self.templates = {previous.template_id: previous}
                self.calls = []

            async def replace_learning_state(
                self, tenant_id, executions, profiles, patterns, templates
            ):
                self.calls.append(
                    (tenant_id, executions, profiles, patterns, templates)
                )
                raise failure

        store = Store()

        with pytest.raises(ConnectionError) as exc_info:
            await _save_workflow_learning_state(
                store,
                tenant_id="acme:prod",
                executions=[],
                profiles=[],
                patterns={},
                templates=replacements,
            )

        assert exc_info.value is failure
        assert store.templates == {"previous": previous}
        assert store.calls == [("acme:prod", [], [], {}, replacements)]

    @pytest.mark.asyncio
    async def test_learning_state_helper_awaits_store_owned_serialization(self):
        from cogniverse_runtime.optimization_cli import (
            _save_workflow_learning_state,
        )

        class Store:
            def __init__(self):
                self.active = 0
                self.max_active = 0
                self.calls = []
                self.lock = asyncio.Lock()

            async def replace_learning_state(
                self, tenant_id, executions, profiles, patterns, templates
            ):
                self.calls.append(
                    (tenant_id, executions, profiles, patterns, templates)
                )
                async with self.lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                    await asyncio.sleep(0.02)
                    self.active -= 1

        store = Store()

        await asyncio.gather(
            *(
                _save_workflow_learning_state(
                    store,
                    tenant_id="acme:prod",
                    executions=[],
                    profiles=[],
                    patterns={},
                    templates=[],
                )
                for _ in range(2)
            )
        )

        assert store.max_active == 1
        assert store.active == 0
        assert store.calls == [
            ("acme:prod", [], [], {}, []),
            ("acme:prod", [], [], {}, []),
        ]


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
# Test: the compile modes bind their LM task-locally
# ---------------------------------------------------------------------------


def _expected_optimization_lm():
    """``llm_config.resolve("optimization")`` from the active config — the
    endpoint the simba / profile / entity-extraction compiles run against."""
    from cogniverse_foundation.config.unified_config import LLMConfig
    from tests.utils.llm_config import _load_config

    return LLMConfig.from_dict(_load_config()["llm_config"]).resolve("optimization")


def _expected_teacher_lm():
    """``llm_config.resolve_teacher()`` — the bootstrap teacher endpoint."""
    from cogniverse_foundation.config.unified_config import LLMConfig
    from tests.utils.llm_config import _load_config

    return LLMConfig.from_dict(_load_config()["llm_config"]).resolve_teacher()


async def _foreign_task_owns_dspy(monkeypatch):
    """Hand DSPy's ambient binding to an async task that has finished by the
    time the mode under test runs, and return that task's LM.

    Ownership is recorded in module globals and never released, so this is the
    state any long-lived process is in once something else configured DSPy
    first (the runtime lifespan, an ingest job, an earlier request)."""
    import asyncio
    from importlib import import_module

    import dspy

    # The module, not the ``settings`` singleton the package re-exports —
    # ambient ownership lives in module-level globals.
    dspy_settings = import_module("dspy.dsp.utils.settings")
    monkeypatch.setattr(dspy_settings, "config_owner_async_task", None)
    monkeypatch.setitem(dspy_settings.main_thread_config, "lm", None)

    owner_lm = dspy.LM(
        model="openai/ambient-owner",
        api_base="http://127.0.0.1:29071/v1",
        api_key="not-required",
    )

    async def _claim():
        dspy.configure(lm=owner_lm)

    await asyncio.create_task(_claim())
    return owner_lm


def _ambient_lm():
    """The process-wide LM binding, read the way ``dspy.configure`` writes it."""
    from importlib import import_module

    return import_module("dspy.dsp.utils.settings").main_thread_config["lm"]


def _record_compile(monkeypatch, seen: Dict[str, Any], key_of):
    """Capture the LM DSPy resolves inside ``BootstrapFewShot.compile``.

    The compile is the one step that needs a live LM endpoint (one call per
    trainset row per bootstrap round, plus the teacher's); everything around it
    — span extraction, the synthetic merge, ``dump_state`` serialization, the
    ArtifactManager dataset round-trip — runs for real. The student module is
    returned unchanged so the saved artifact holds real DSPy state.

    ``key_of(trainset)`` names the slot in ``seen`` so concurrent runs can be
    told apart."""
    import dspy
    from dspy.teleprompt import BootstrapFewShot

    def _compile(self, student, *, teacher=None, trainset):
        seen[key_of(trainset)] = {
            "student_lm": dspy.settings.lm,
            "teacher_lm": self.teacher_settings["lm"],
            "trainset_size": len(trainset),
            "max_bootstrapped_demos": self.max_bootstrapped_demos,
        }
        return student

    monkeypatch.setattr(BootstrapFewShot, "compile", _compile)


def _saved_blob(provider, dataset_name: str) -> dict:
    """The DSPy state persisted under ``dataset_name``, parsed."""
    created = [c for c in provider.datasets.created if c["name"] == dataset_name]
    assert len(created) == 1
    return json.loads(created[0]["data"]["content"].iloc[0])


_QUERY_ENHANCEMENT_SPANS = [
    {
        "attributes.input.value": "robot arms",
        "attributes.output.value": json.dumps(
            {"enhanced_query": "robotic arms assembling cars", "confidence": 0.9}
        ),
    },
    {
        "attributes.input.value": "solar panels",
        "attributes.output.value": json.dumps(
            {"enhanced_query": "photovoltaic panel rooftop install", "confidence": 0.7}
        ),
    },
]


class TestCompileModesBindLmTaskLocally:
    """simba / profile / entity-extraction compile under the resolved
    ``optimization`` endpoint even when another async task owns DSPy's ambient
    binding. That binding can only be written by the task that claimed it
    first, so writing it here aborted the whole mode before the compile."""

    @pytest.mark.asyncio
    async def test_simba_compile_reaches_a_live_endpoint(self, monkeypatch, tmp_path):
        """The unstubbed BootstrapFewShot compile, over a real HTTP LM socket,
        with a foreign async task holding DSPy's ambient binding: the bootstrap
        call reaches the configured endpoint and the answer it returns is what
        lands in the saved artifact."""
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        from cogniverse_agents.query_enhancement_agent import QueryEnhancementModule
        from cogniverse_runtime.optimization_cli import run_simba_optimization
        from tests.utils.llm_config import _load_config

        answer = {
            "enhanced_query": "robotic manipulator arms on an assembly line",
            "expansion_terms": "manipulator,assembly line,industrial robot",
            "synonyms": "robot arm,manipulator",
            "context": "manufacturing,factory automation",
            "confidence": "0.83",
            "reasoning": "Broadened the query with manufacturing vocabulary.",
        }
        # The served fields come from the signature the module actually runs,
        # reasoning field included, in the order the adapter expects them.
        served = "".join(
            f"[[ ## {name} ## ]]\n{answer[name]}\n\n"
            for name in QueryEnhancementModule().enhancer.predict.signature.output_fields
        )
        received: List[Dict[str, Any]] = []

        class _ChatCompletions(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append({"path": self.path, "body": body})
                payload = json.dumps(
                    {
                        "id": "chatcmpl-stub",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body["model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": served + "[[ ## completed ## ]]",
                                },
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatCompletions)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            blob = _load_config()
            base = f"http://127.0.0.1:{port}/v1"
            # The model ids carry the port so DSPy's response cache cannot
            # answer this run from an earlier session's entry.
            blob["llm_config"]["primary"] = {
                **blob["llm_config"]["primary"],
                "model": f"openai/stub-student-{port}",
                "api_base": base,
                "api_key": "not-required",
            }
            blob["llm_config"]["teacher"] = {
                **blob["llm_config"]["teacher"],
                "model": f"openai/stub-teacher-{port}",
                "api_base": base,
                "api_key": "not-required",
            }
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps(blob))
            monkeypatch.setenv("COGNIVERSE_CONFIG", str(config_path))

            provider = FakeTelemetryProvider(
                _make_spans_df(
                    "cogniverse.query_enhancement", _QUERY_ENHANCEMENT_SPANS[:1]
                )
            )
            owner_lm = await _foreign_task_owns_dspy(monkeypatch)

            p1, p2 = _patch_infra(FakeTelemetryManager(provider))
            with p1, p2:
                result = await run_simba_optimization(
                    tenant_id="test:unit", lookback_hours=1
                )
        finally:
            server.shutdown()
            server.server_close()

        assert result == {
            "status": "success",
            "spans_found": 1,
            "training_examples": 1,
            "artifact_id": "dataset-1",
        }
        # BootstrapFewShot runs the bootstrap on the teacher endpoint.
        assert [r["path"] for r in received] == ["/v1/chat/completions"]
        assert received[0]["body"]["model"] == f"stub-teacher-{port}"

        state = _saved_blob(provider, "dspy-model-test:unit-simba_query_enhancement")
        demos = state["enhancer.predict"]["demos"]
        assert len(demos) == 1
        assert demos[0]["query"] == "robot arms"
        assert demos[0]["enhanced_query"] == answer["enhanced_query"]
        assert demos[0]["confidence"] == answer["confidence"]
        assert _ambient_lm() is owner_lm

    @pytest.mark.asyncio
    async def test_simba_compiles_under_optimization_lm(self, monkeypatch):
        from cogniverse_runtime.optimization_cli import run_simba_optimization

        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.query_enhancement", _QUERY_ENHANCEMENT_SPANS)
        )
        owner_lm = await _foreign_task_owns_dspy(monkeypatch)
        seen: Dict[str, Any] = {}
        _record_compile(monkeypatch, seen, lambda trainset: "simba")

        p1, p2 = _patch_infra(FakeTelemetryManager(provider))
        with p1, p2:
            result = await run_simba_optimization(
                tenant_id="test:unit", lookback_hours=1
            )

        assert result == {
            "status": "success",
            "spans_found": 2,
            "training_examples": 2,
            "artifact_id": "dataset-1",
        }

        expected = _expected_optimization_lm()
        compiled_with = seen["simba"]
        assert compiled_with["student_lm"].model == expected.model
        assert compiled_with["student_lm"].kwargs["api_base"] == expected.api_base
        assert compiled_with["student_lm"] is not owner_lm
        assert compiled_with["teacher_lm"].model == _expected_teacher_lm().model
        assert compiled_with["trainset_size"] == 2
        assert compiled_with["max_bootstrapped_demos"] == 4

        state = _saved_blob(provider, "dspy-model-test:unit-simba_query_enhancement")
        assert sorted(state) == ["enhancer.predict"]
        # The binding is contextual, so the artifact carries no LM of its own.
        assert state["enhancer.predict"]["lm"] is None
        assert provider.datasets.deleted == [
            "dspy-model-test:unit-simba_query_enhancement"
        ]
        assert _ambient_lm() is owner_lm

    @pytest.mark.asyncio
    async def test_profile_compiles_under_optimization_lm(self, monkeypatch):
        from cogniverse_runtime.optimization_cli import run_profile_optimization

        provider = FakeTelemetryProvider(
            _make_spans_df(
                "cogniverse.profile_selection",
                [
                    {
                        "attributes.input.value": "show me the factory floor",
                        "attributes.output.value": json.dumps(
                            {
                                "selected_profile": "video_colpali_smol500_mv_frame",
                                "modality": "video",
                                "complexity": "simple",
                                "intent": "video_search",
                                "confidence": 0.9,
                            }
                        ),
                    }
                ],
            )
        )
        owner_lm = await _foreign_task_owns_dspy(monkeypatch)
        seen: Dict[str, Any] = {}
        _record_compile(monkeypatch, seen, lambda trainset: "profile")

        p1, p2 = _patch_infra(FakeTelemetryManager(provider))
        with p1, p2:
            result = await run_profile_optimization(
                tenant_id="test:unit", lookback_hours=1
            )

        assert result == {
            "status": "success",
            "spans_found": 1,
            "training_examples": 1,
            "artifact_id": "dataset-1",
        }

        expected = _expected_optimization_lm()
        compiled_with = seen["profile"]
        assert compiled_with["student_lm"].model == expected.model
        assert compiled_with["student_lm"].kwargs["api_base"] == expected.api_base
        assert compiled_with["student_lm"] is not owner_lm
        assert compiled_with["teacher_lm"].model == _expected_teacher_lm().model
        assert compiled_with["trainset_size"] == 1

        state = _saved_blob(provider, "dspy-model-test:unit-profile_selection")
        assert sorted(state) == ["selector.predict"]
        assert state["selector.predict"]["lm"] is None
        assert _ambient_lm() is owner_lm

    @pytest.mark.asyncio
    async def test_entity_extraction_compiles_under_optimization_lm(self, monkeypatch):
        from cogniverse_runtime.optimization_cli import (
            run_entity_extraction_optimization,
        )

        provider = FakeTelemetryProvider(
            _make_spans_df(
                "cogniverse.entity_extraction",
                [
                    {
                        "attributes.input.value": "videos of Boston Dynamics robots",
                        "attributes.output.value": json.dumps(
                            {
                                "entities": [
                                    {
                                        "text": "Boston Dynamics",
                                        "type": "ORG",
                                        "confidence": 0.9,
                                    }
                                ]
                            }
                        ),
                    }
                ],
            )
        )
        owner_lm = await _foreign_task_owns_dspy(monkeypatch)
        seen: Dict[str, Any] = {}
        _record_compile(monkeypatch, seen, lambda trainset: "entity_extraction")

        p1, p2 = _patch_infra(FakeTelemetryManager(provider))
        with p1, p2:
            result = await run_entity_extraction_optimization(
                tenant_id="test:unit", lookback_hours=1
            )

        assert result == {
            "status": "success",
            "spans_found": 1,
            "training_examples": 1,
            "artifact_id": "dataset-1",
        }

        expected = _expected_optimization_lm()
        compiled_with = seen["entity_extraction"]
        assert compiled_with["student_lm"].model == expected.model
        assert compiled_with["student_lm"].kwargs["api_base"] == expected.api_base
        assert compiled_with["student_lm"] is not owner_lm
        assert compiled_with["teacher_lm"].model == _expected_teacher_lm().model
        assert compiled_with["trainset_size"] == 1

        state = _saved_blob(provider, "dspy-model-test:unit-entity_extraction")
        assert sorted(state) == ["extractor.predict"]
        assert state["extractor.predict"]["lm"] is None
        assert _ambient_lm() is owner_lm

    @pytest.mark.asyncio
    async def test_compile_failure_unwinds_the_lm_binding(self, monkeypatch):
        """A compile that dies against the LM endpoint reports the failure,
        writes no artifact, and releases the LM it bound — the ambient binding
        the foreign task owns must come back untouched."""
        import dspy
        from dspy.teleprompt import BootstrapFewShot

        from cogniverse_runtime.optimization_cli import run_simba_optimization

        provider = FakeTelemetryProvider(
            _make_spans_df("cogniverse.query_enhancement", _QUERY_ENHANCEMENT_SPANS)
        )
        owner_lm = await _foreign_task_owns_dspy(monkeypatch)

        bound_during_compile = []

        def _dead_endpoint(self, student, *, teacher=None, trainset):
            bound_during_compile.append(dspy.settings.lm)
            raise ConnectionError("optimization LM endpoint refused the connection")

        monkeypatch.setattr(BootstrapFewShot, "compile", _dead_endpoint)

        p1, p2 = _patch_infra(FakeTelemetryManager(provider))
        with p1, p2:
            result = await run_simba_optimization(
                tenant_id="test:unit", lookback_hours=1
            )

        assert result == {
            "status": "failed",
            "error": "optimization LM endpoint refused the connection",
        }
        assert bound_during_compile[0].model == _expected_optimization_lm().model
        assert provider.datasets.created == []
        assert provider.datasets.deleted == []
        assert _ambient_lm() is owner_lm
        assert dspy.settings.lm is owner_lm

    @pytest.mark.asyncio
    async def test_concurrent_simba_runs_keep_their_own_lm_binding(self, monkeypatch):
        """Two simba runs in flight at once each compile under the LM they
        built. A process-wide binding would hand both compiles whichever LM
        was written last (and leave that LM behind for the rest of the
        process); the task-local one cannot bleed across runs."""
        import asyncio
        import itertools

        import dspy

        from cogniverse_runtime.optimization_cli import run_simba_optimization

        both_in_flight = asyncio.Barrier(2)

        class _BarrierTraceStore(FakeTraceStore):
            """Holds each run inside its span query until both are running."""

            async def get_spans(self, **kwargs):
                await both_in_flight.wait()
                return await super().get_spans(**kwargs)

        def _provider() -> FakeTelemetryProvider:
            spans_df = _make_spans_df(
                "cogniverse.query_enhancement", _QUERY_ENHANCEMENT_SPANS
            )
            provider = FakeTelemetryProvider(spans_df)
            provider._trace_store = _BarrierTraceStore(spans_df)
            return provider

        providers = {"test:one": _provider(), "test:two": _provider()}

        class _PerTenantTelemetryManager:
            def __init__(self):
                self.config = FakeTelemetryConfig()

            def get_provider(self, tenant_id):
                return providers[tenant_id]

        tags = itertools.count(1)

        def _tagged_lm(config):
            return dspy.LM(
                model=f"openai/tagged-{next(tags)}",
                api_base=config.api_base,
                api_key="not-required",
            )

        monkeypatch.setattr(
            "cogniverse_foundation.config.llm_factory.create_dspy_lm", _tagged_lm
        )
        owner_lm = await _foreign_task_owns_dspy(monkeypatch)

        seen: Dict[str, Any] = {}
        # Both runs compile the same trainset, so each compile is keyed by the
        # LM tag it was handed.
        _record_compile(monkeypatch, seen, lambda trainset: dspy.settings.lm.model)

        p1, p2 = _patch_infra(_PerTenantTelemetryManager())
        with p1, p2:
            results = await asyncio.gather(
                run_simba_optimization(tenant_id="test:one", lookback_hours=1),
                run_simba_optimization(tenant_id="test:two", lookback_hours=1),
            )

        def _tag(lm) -> int:
            return int(lm.model.rsplit("-", 1)[1])

        assert sorted(seen) == ["openai/tagged-1", "openai/tagged-3"]
        first, second = seen["openai/tagged-1"], seen["openai/tagged-3"]
        assert first["student_lm"] is not second["student_lm"]
        # Each run's teacher is the LM it built right after its own student —
        # a shared binding would cross the pairs.
        assert _tag(first["teacher_lm"]) == 2
        assert _tag(second["teacher_lm"]) == 4
        assert [r["status"] for r in results] == ["success", "success"]
        assert [c["name"] for c in providers["test:one"].datasets.created] == [
            "dspy-model-test:one-simba_query_enhancement"
        ]
        assert [c["name"] for c in providers["test:two"].datasets.created] == [
            "dspy-model-test:two-simba_query_enhancement"
        ]
        assert _ambient_lm() is owner_lm


# ---------------------------------------------------------------------------
# Test: routing mode
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test: synthetic data merge helper
# ---------------------------------------------------------------------------


class TestSyntheticDataMerge:
    def test_approved_dataset_name_canonicalizes_tenant(self):
        from cogniverse_core.approval.interfaces import (
            approved_synthetic_dataset_name,
        )

        assert (
            approved_synthetic_dataset_name("acme")
            == "approved_synthetic_data-acme:acme"
        )
        assert (
            approved_synthetic_dataset_name("acme:production")
            == "approved_synthetic_data-acme:production"
        )
        with pytest.raises(ValueError, match="tenant_id is required"):
            approved_synthetic_dataset_name("")

    @pytest.mark.asyncio
    async def test_load_approved_synthetic_no_data(self):
        """Returns empty list when no synthetic data exists."""
        from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data

        provider = FakeTelemetryProvider()
        result = await _load_approved_synthetic_data(
            provider, "default", "query_enhancement"
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_load_approved_synthetic_isolates_tenant_rows_and_order(self):
        """Concurrent optimizers consume only their tenant's ordered records."""
        from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data

        class ApprovedDatasetStore:
            def __init__(self):
                self.names = []

            async def get_dataset(self, name):
                self.names.append(name)
                frames = {
                    "approved_synthetic_data-acme:alpha": pd.DataFrame(
                        [
                            {
                                "input": {
                                    "item_id": "alpha-approved-1",
                                    "confidence": 0.91,
                                    "status": "approved",
                                    "created_at": "2026-08-05T01:00:00+00:00",
                                    "reviewed_at": "2026-08-05T01:01:00+00:00",
                                    "query": "Find exact PyTorch tutorials",
                                    "enhanced_query": "Find exact PyTorch framework tutorials",
                                    "expansion_terms": ["framework"],
                                    "synonyms": ["torch"],
                                    "context": "document_text",
                                    "reasoning": "Framework disambiguates the library.",
                                    "metadata.batch_id": "batch-a",
                                    "metadata.agent_type": "query_enhancement",
                                    "context.optimizer": "query_enhancement",
                                    "context.purpose": "optimizer training",
                                }
                            },
                            {
                                "input": {
                                    "item_id": "pending",
                                    "status": "pending_review",
                                    "query": "Do not consume",
                                    "context.optimizer": "query_enhancement",
                                }
                            },
                            {
                                "input": {
                                    "item_id": "alpha-approved-2",
                                    "confidence": 0.88,
                                    "status": "approved",
                                    "created_at": "2026-08-05T01:02:00+00:00",
                                    "reviewed_at": "2026-08-05T01:03:00+00:00",
                                    "query": "Find exact JAX tutorials",
                                    "enhanced_query": "Find exact JAX framework tutorials",
                                    "expansion_terms": ["framework"],
                                    "synonyms": [],
                                    "context": "document_text",
                                    "reasoning": "Framework distinguishes JAX documentation.",
                                    "metadata.batch_id": "batch-b",
                                    "metadata.agent_type": "query_enhancement",
                                    "context.optimizer": "query_enhancement",
                                    "context.purpose": "optimizer training",
                                }
                            },
                        ]
                    ),
                    "approved_synthetic_data-acme:beta": pd.DataFrame(
                        [
                            {
                                "input": {
                                    "item_id": "beta-approved-1",
                                    "confidence": 0.95,
                                    "status": "approved",
                                    "query": "Find exact Vespa tutorials",
                                    "enhanced_query": "Find exact Vespa search tutorials",
                                    "expansion_terms": ["search"],
                                    "synonyms": [],
                                    "context": "document_text",
                                    "reasoning": "Search identifies the Vespa platform.",
                                    "metadata.agent_type": "query_enhancement",
                                    "context.optimizer": "query_enhancement",
                                }
                            }
                        ]
                    ),
                }
                frame = frames[name]
                frame["input"] = frame["input"].map(_signed_approved_record)
                return frame

        provider = FakeTelemetryProvider()
        dataset_store = ApprovedDatasetStore()
        provider._dataset_store = dataset_store

        alpha, beta = await asyncio.gather(
            _load_approved_synthetic_data(provider, "acme:alpha", "query_enhancement"),
            _load_approved_synthetic_data(provider, "acme:beta", "query_enhancement"),
        )

        assert alpha == [
            {
                "query": "Find exact PyTorch tutorials",
                "enhanced_query": "Find exact PyTorch framework tutorials",
                "expansion_terms": ["framework"],
                "synonyms": ["torch"],
                "context": "document_text",
                "reasoning": "Framework disambiguates the library.",
            },
            {
                "query": "Find exact JAX tutorials",
                "enhanced_query": "Find exact JAX framework tutorials",
                "expansion_terms": ["framework"],
                "synonyms": [],
                "context": "document_text",
                "reasoning": "Framework distinguishes JAX documentation.",
            },
        ]
        assert beta == [
            {
                "query": "Find exact Vespa tutorials",
                "enhanced_query": "Find exact Vespa search tutorials",
                "expansion_terms": ["search"],
                "synonyms": [],
                "context": "document_text",
                "reasoning": "Search identifies the Vespa platform.",
            }
        ]
        assert dataset_store.names == [
            "approved_synthetic_data-acme:alpha",
            "approved_synthetic_data-acme:beta",
        ]

    @pytest.mark.asyncio
    async def test_load_approved_synthetic_filters_nonapproved_and_other_optimizer(
        self,
    ):
        from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data

        class ApprovedDatasetStore:
            async def get_dataset(self, name):
                assert name == "approved_synthetic_data-acme:alpha"
                records = [
                    {
                        "item_id": "pending",
                        "status": "pending_review",
                        "query": "Do not consume",
                        "context.optimizer": "query_enhancement",
                    },
                    {
                        "item_id": "other-optimizer",
                        "status": "approved",
                        "query": "Wrong optimizer",
                        "context.optimizer": "profile",
                    },
                ]
                return pd.DataFrame(
                    [{"input": _signed_approved_record(record)} for record in records]
                )

        provider = FakeTelemetryProvider()
        provider._dataset_store = ApprovedDatasetStore()

        result = await _load_approved_synthetic_data(
            provider, "acme:alpha", "query_enhancement"
        )

        assert result == []

    @pytest.mark.asyncio
    async def test_load_approved_synthetic_requires_canonical_agent_owner(self):
        from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data

        class ApprovedDatasetStore:
            async def get_dataset(self, name):
                record = {
                    "item_id": "wrong-owner",
                    "status": "approved",
                    "query": "Find exact PyTorch tutorials",
                    "enhanced_query": "Find exact PyTorch framework tutorials",
                    "expansion_terms": ["framework"],
                    "synonyms": [],
                    "context": "document_text",
                    "reasoning": "Framework disambiguates the library.",
                    "metadata.agent_type": "profile_selection",
                    "context.optimizer": "query_enhancement",
                }
                return pd.DataFrame([{"input": _signed_approved_record(record)}])

        provider = FakeTelemetryProvider()
        provider._dataset_store = ApprovedDatasetStore()

        with pytest.raises(
            ValueError,
            match=(
                "Approved synthetic dataset row 0 for optimizer=query_enhancement "
                "requires metadata.agent_type='query_enhancement', got "
                "'profile_selection'"
            ),
        ):
            await _load_approved_synthetic_data(
                provider, "acme:alpha", "query_enhancement"
            )

    @pytest.mark.asyncio
    async def test_load_approved_synthetic_validates_values_before_returning_them(self):
        from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data

        class ApprovedDatasetStore:
            async def get_dataset(self, name):
                record = {
                    "item_id": "invalid-profile",
                    "status": "approved",
                    "query": "find transformer lectures",
                    "available_profiles": "video_colpali,document_colpali",
                    "selected_profile": "missing_profile",
                    "reasoning": "A selector response with an invalid target.",
                    "query_intent": "document_search",
                    "modality": "document",
                    "complexity": "complex",
                    "metadata.agent_type": "profile_selection",
                    "context.optimizer": "profile",
                }
                return pd.DataFrame([{"input": _signed_approved_record(record)}])

        provider = FakeTelemetryProvider()
        provider._dataset_store = ApprovedDatasetStore()

        with pytest.raises(
            ValueError,
            match=(
                "Approved synthetic dataset row 0 for optimizer=profile "
                "selected_profile 'missing_profile' is absent from "
                "available_profiles"
            ),
        ):
            await _load_approved_synthetic_data(provider, "acme:alpha", "profile")

    @pytest.mark.asyncio
    async def test_load_approved_synthetic_surfaces_dataset_outage(self):
        """A Phoenix outage cannot masquerade as an empty approved dataset."""
        from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data

        class UnavailableDatasetStore:
            async def get_dataset(self, name):
                raise ConnectionError("Phoenix refused the dataset request")

        provider = FakeTelemetryProvider()
        provider._dataset_store = UnavailableDatasetStore()

        with pytest.raises(
            RuntimeError,
            match=(
                "Failed to load approved synthetic data for "
                "tenant=acme:production optimizer=query_enhancement "
                "dataset=approved_synthetic_data-acme:production"
            ),
        ) as error:
            await _load_approved_synthetic_data(
                provider, "acme:production", "query_enhancement"
            )

        assert isinstance(error.value.__cause__, ConnectionError)
        assert str(error.value.__cause__) == "Phoenix refused the dataset request"

    @pytest.mark.asyncio
    async def test_load_approved_synthetic_rejects_missing_provider_frame(self):
        """Only the provider's typed not-found result represents absence."""
        from cogniverse_runtime.optimization_cli import _load_approved_synthetic_data

        class MissingFrameDatasetStore:
            async def get_dataset(self, name):
                assert name == "approved_synthetic_data-acme:production"
                return None

        provider = FakeTelemetryProvider()
        provider._dataset_store = MissingFrameDatasetStore()

        with pytest.raises(
            RuntimeError,
            match=(
                "Approved synthetic dataset provider returned no frame for "
                "tenant=acme:production optimizer=query_enhancement "
                "dataset=approved_synthetic_data-acme:production"
            ),
        ):
            await _load_approved_synthetic_data(
                provider, "acme:production", "query_enhancement"
            )


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
        """Synthetic generation with no reachable backend reports the failure
        per optimizer type instead of raising out of the CLI mode."""
        from cogniverse_runtime.optimization_cli import run_synthetic_generation

        p1, p2 = _patch_infra(fake_telemetry_manager)
        with p1, p2:
            result = await run_synthetic_generation(
                tenant_id="test:unit",
                optimizer_types=["simba"],
                count=5,
            )

        assert result["status"] == "failed"
        assert list(result["results"]) == ["simba"]
        assert result["results"]["simba"]["status"] == "failed"
        assert result["results"]["simba"]["error"] != ""

    @pytest.mark.asyncio
    async def test_synthetic_generation_binds_lm_when_another_task_owns_dspy(
        self, fake_telemetry_manager, monkeypatch
    ):
        """The generators must see the configured LM even when a DIFFERENT
        async task already owns DSPy's ambient binding — writing that binding
        (``dspy.configure`` or ``dspy.settings.lm = ...``) raises for every
        task but the owner, which aborted the whole synthetic run.

        The search backend the generators sample content from needs a Vespa
        container, which this in-process test has none of, so the registry
        hands back an inert stand-in; the LM binding under test is read inside
        ``generate``, where production reads it."""
        import asyncio
        from importlib import import_module

        import dspy

        from cogniverse_core.registries.backend_registry import BackendRegistry
        from cogniverse_runtime.optimization_cli import run_synthetic_generation
        from cogniverse_synthetic.schemas import SyntheticDataResponse
        from cogniverse_synthetic.service import SyntheticDataService

        # The module, not the ``settings`` singleton the package re-exports —
        # ambient ownership lives in module-level globals.
        dspy_settings = import_module("dspy.dsp.utils.settings")
        monkeypatch.setattr(dspy_settings, "config_owner_async_task", None)
        monkeypatch.setitem(dspy_settings.main_thread_config, "lm", None)

        owner_lm = dspy.LM(
            model="openai/ambient-owner",
            api_base="http://127.0.0.1:29071/v1",
            api_key="not-required",
        )

        async def _claim_ambient():
            dspy.configure(lm=owner_lm)

        await asyncio.create_task(_claim_ambient())

        seen: Dict[str, Any] = {}

        async def _record_bound_lm(self, request):
            seen["lm"] = dspy.settings.lm
            return SyntheticDataResponse(
                optimizer=request.optimizer,
                schema_name="RoutingExampleSchema",
                count=0,
                selected_profiles=[],
                profile_selection_reasoning="",
                data=[],
                metadata={},
            )

        monkeypatch.setattr(SyntheticDataService, "generate", _record_bound_lm)
        monkeypatch.setattr(
            BackendRegistry,
            "get_search_backend",
            lambda *a, **k: object(),
        )

        p1, p2 = _patch_infra(fake_telemetry_manager)
        with p1, p2:
            result = await run_synthetic_generation(
                tenant_id="test:unit",
                optimizer_types=["simba"],
                count=5,
            )

        expected = _expected_primary_lm()
        assert seen["lm"].model == expected.model
        assert seen["lm"].kwargs["api_base"] == expected.api_base
        assert seen["lm"] is not owner_lm
        assert result["results"]["simba"] == {
            "status": "no_data",
            "examples_generated": 0,
        }
        # The ambient binding still belongs to the task that claimed it.
        assert dspy_settings.main_thread_config["lm"] is owner_lm

    @pytest.mark.asyncio
    async def test_concurrent_synthetic_runs_keep_their_own_lm_binding(
        self, fake_telemetry_manager, monkeypatch
    ):
        """Two synthetic runs in flight at once each generate under their own
        LM. A process-wide binding would hand both runs whichever LM was
        written last; the task-local one cannot bleed across runs.

        The search backend needs a Vespa container this in-process test has
        none of, so the registry hands back an inert stand-in."""
        import asyncio
        import itertools

        import dspy

        from cogniverse_core.registries.backend_registry import BackendRegistry
        from cogniverse_runtime.optimization_cli import run_synthetic_generation
        from cogniverse_synthetic.schemas import SyntheticDataResponse
        from cogniverse_synthetic.service import SyntheticDataService

        tags = itertools.count(1)

        def _tagged_lm(config):
            return dspy.LM(
                model=f"openai/tagged-{next(tags)}",
                api_base=config.api_base,
                api_key="not-required",
            )

        monkeypatch.setattr(
            "cogniverse_foundation.config.llm_factory.create_dspy_lm", _tagged_lm
        )
        monkeypatch.setattr(
            BackendRegistry, "get_search_backend", lambda *a, **k: object()
        )

        both_inside = asyncio.Barrier(2)
        seen: Dict[str, Any] = {}

        async def _record_bound_lm(self, request):
            seen[request.tenant_id] = dspy.settings.lm
            # Hold both runs inside generate so their bindings are live at once.
            await both_inside.wait()
            return SyntheticDataResponse(
                optimizer=request.optimizer,
                schema_name="RoutingExampleSchema",
                count=0,
                selected_profiles=[],
                profile_selection_reasoning="",
                data=[],
                metadata={},
            )

        monkeypatch.setattr(SyntheticDataService, "generate", _record_bound_lm)

        p1, p2 = _patch_infra(fake_telemetry_manager)
        with p1, p2:
            results = await asyncio.gather(
                run_synthetic_generation(
                    tenant_id="test:one", optimizer_types=["simba"], count=1
                ),
                run_synthetic_generation(
                    tenant_id="test:two", optimizer_types=["simba"], count=1
                ),
            )

        assert sorted(seen) == ["test:one", "test:two"]
        assert seen["test:one"] is not seen["test:two"]
        assert {seen["test:one"].model, seen["test:two"].model} == {
            "openai/tagged-1",
            "openai/tagged-2",
        }
        assert [r["results"]["simba"]["status"] for r in results] == [
            "no_data",
            "no_data",
        ]


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
    from a genuinely empty optimization window. The retry budget is bounded:
    2 attempts with a 60s per-attempt timeout, so a persistently down or
    hung Phoenix costs at most ~125s per call site (this runs in a per-agent
    loop; the previous 3x120s budget hung a cycle for 370s per agent).
    """

    def test_retry_budget_constants(self):
        from cogniverse_runtime import optimization_cli as cli

        assert cli._SPAN_QUERY_ATTEMPTS == 2
        assert cli._SPAN_QUERY_TIMEOUT_S == 60

    @pytest.mark.asyncio
    async def test_query_failure_raises_after_exactly_two_attempts(self, monkeypatch):
        import asyncio as _asyncio

        from cogniverse_runtime import optimization_cli as cli

        provider = FakeTelemetryProvider()
        store = FailingTraceStore()
        provider._trace_store = store
        manager = FakeTelemetryManager(provider)

        monkeypatch.setattr(_asyncio, "sleep", _instant_sleep)
        with patch(_PATCH_TELEMETRY, return_value=manager):
            with pytest.raises(RuntimeError, match="after 2 attempts"):
                await cli._query_spans_by_name(
                    provider, "acme:prod", "cogniverse.entity_extraction", 1.0
                )
        assert store.calls == 2

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


class TestQuerySpansHungPhoenixIsCancelled:
    """A get_spans call that hangs forever must be cancelled per attempt.

    A dead Phoenix raises promptly; a hung one never returns — only
    asyncio.wait_for's cancellation bounds the retry budget. The wall-clock
    band proves each attempt was cut at the per-attempt timeout instead of
    hanging the cycle.
    """

    @pytest.mark.asyncio
    async def test_hung_query_cancelled_each_attempt_then_raises(self, monkeypatch):
        import asyncio as _asyncio
        import time

        from cogniverse_runtime import optimization_cli as cli

        class HangingTraceStore:
            def __init__(self):
                self.calls = 0
                self.cancelled = 0

            async def get_spans(self, **kwargs):
                self.calls += 1
                try:
                    await _asyncio.Event().wait()
                except _asyncio.CancelledError:
                    self.cancelled += 1
                    raise

        provider = FakeTelemetryProvider()
        store = HangingTraceStore()
        provider._trace_store = store
        manager = FakeTelemetryManager(provider)

        monkeypatch.setattr(cli, "_SPAN_QUERY_TIMEOUT_S", 0.2)
        monkeypatch.setattr(cli, "_SPAN_QUERY_ATTEMPTS", 2)
        monkeypatch.setattr(_asyncio, "sleep", _instant_sleep)

        start = time.monotonic()
        with patch(_PATCH_TELEMETRY, return_value=manager):
            with pytest.raises(RuntimeError, match="after 2 attempts"):
                await cli._query_spans_by_name(
                    provider, "acme:prod", "cogniverse.entity_extraction", 1.0
                )
        elapsed = time.monotonic() - start

        assert store.calls == 2
        assert store.cancelled == 2
        # Two 0.2s attempt timeouts must have elapsed; anything near 2s or
        # beyond means a hung attempt was not cancelled.
        assert 0.35 <= elapsed < 2.0, elapsed


class TestGoldenSetCandidates:
    """Golden-set growth skips rows whose score cannot coerce to float."""

    def test_junk_score_row_skipped_valid_rows_survive(self):
        from cogniverse_runtime.optimization_cli import _golden_set_candidates

        df = pd.DataFrame(
            [
                {"category": "high_scoring", "query": "good one", "score": 0.9},
                {"category": "high_scoring", "query": "junk score", "score": "great"},
                {"category": "high_scoring", "query": "good two", "score": 0.85},
                {"category": "high_scoring", "query": "none score", "score": None},
                {"category": "high_scoring", "query": "below cut", "score": 0.5},
                {"category": "low_scoring", "query": "wrong category", "score": 0.95},
            ]
        )

        candidate = {
            "expected_videos": [],
            "ground_truth": "",
            "query_type": "live_traffic",
            "source": "quality_monitor",
        }
        assert _golden_set_candidates(df) == [
            {"query": "good one", **candidate},
            {"query": "good two", **candidate},
        ]


class TestRunFailed:
    """_run_failed maps a mode result to the failed/ok exit decision."""

    def test_top_level_failed(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert _run_failed({"status": "failed", "error": "phoenix down"}) is True

    def test_top_level_error(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert _run_failed({"status": "error"}) is True

    def test_top_level_success_wins_over_nested(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert (
            _run_failed({"status": "success", "results": {"a": {"status": "failed"}}})
            is False
        )

    def test_batch_shape_nested_failure(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert (
            _run_failed(
                {
                    "search": {"status": "failed", "error": "lm down"},
                    "summary": {"status": "success"},
                }
            )
            is True
        )

    def test_batch_shape_skips_and_nonfatal_eval_error_ok(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert (
            _run_failed(
                {
                    "search": {"status": "skipped", "reason": "no_data"},
                    "post_optimization_eval": {"error": "eval unavailable"},
                    "baseline_updated": True,
                }
            )
            is False
        )

    def test_no_data_is_ok(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert _run_failed({"status": "no_data"}) is False

    def test_non_dict_is_ok(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert _run_failed(None) is False

    def test_failed_string_marker_fails(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert _run_failed("failed: Vespa connection refused") is True
        assert _run_failed("error: boom") is True

    def test_completed_string_marker_ok(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        assert _run_failed("completed: {'fact': 3}") is False
        assert _run_failed("skipped: path /logs is not a directory") is False

    def test_failed_key_dict_fails(self):
        from cogniverse_runtime.optimization_cli import _run_failed

        # config_vacuum encodes an outage as {"failed": <exc>}, no status key.
        assert _run_failed({"config_vacuum": {"failed": "Vespa refused"}}) is True
        # A zero failed-count is not a failure.
        assert _run_failed({"failed": 0, "succeeded": 5}) is False

    def test_cleanup_total_outage_shape_fails(self):
        """The exact run_cleanup result under a total mem0/Vespa outage: the
        per-tenant memory_cleanup entry is a 'failed: ...' string and
        config_vacuum is {'failed': ...}, neither carrying a top-level status.
        The old .get('status')-only check returned False here → exit 0 =
        SUCCESS while the cron did nothing."""
        from cogniverse_runtime.optimization_cli import _run_failed

        outage_result = {
            "log_retention_days": 7,
            "memory_retention_days": 30,
            "memory_cleanup": {"acme:acme": "failed: Vespa connection refused"},
            "tenants_processed": 1,
            "log_cleanup": {"path": "/logs", "scanned": 0, "deleted": 0, "errors": []},
            "temp_cleanup": {"path": "/tmp", "scanned": 0, "deleted": 0, "errors": []},
            "config_vacuum": {"failed": "Vespa connection refused"},
        }
        assert _run_failed(outage_result) is True

    def test_cleanup_healthy_shape_ok(self):
        """The happy run_cleanup result — completed per-tenant strings, a
        dropped-count vacuum, empty prune errors — must NOT trip the exit."""
        from cogniverse_runtime.optimization_cli import _run_failed

        healthy_result = {
            "log_retention_days": 7,
            "memory_retention_days": 30,
            "memory_cleanup": {"acme:acme": "completed: {'fact': 3}"},
            "tenants_processed": 1,
            "log_cleanup": {"path": "/logs", "scanned": 5, "deleted": 2, "errors": []},
            "temp_cleanup": {"path": "/tmp", "scanned": 0, "deleted": 0, "errors": []},
            "config_vacuum": {"dropped": 4, "keep_versions": 10},
        }
        assert _run_failed(healthy_result) is False


class TestMainExitCode:
    """The exit code is the only success signal Argo sees for a workflow
    step — a failed run must exit non-zero, not print-and-exit-0."""

    def _run_main(self, monkeypatch, mode_result) -> int:
        import sys

        from cogniverse_runtime import optimization_cli as cli

        async def fake_run(*args, **kwargs):
            return mode_result

        monkeypatch.setattr(cli, "run_triggered_optimization", fake_run)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "optimization_cli",
                "--mode",
                "triggered",
                "--tenant-id",
                "t1",
                "--agents",
                "search",
                "--trigger-dataset",
                "trigger-ds",
            ],
        )
        with pytest.raises(SystemExit) as exc:
            cli.main()
        return exc.value.code

    def test_failed_result_exits_nonzero(self, monkeypatch):
        code = self._run_main(
            monkeypatch, {"status": "failed", "error": "phoenix down"}
        )
        assert code == 1

    def test_success_result_exits_zero(self, monkeypatch):
        code = self._run_main(
            monkeypatch, {"status": "success", "training_examples": 3}
        )
        assert code == 0
